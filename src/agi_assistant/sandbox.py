"""命令校验与 Docker、Local、Mock 三种沙箱执行器。"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass

from .config import SandboxConfig, SecurityConfig


@dataclass(slots=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    backend: str = "mock"
    blocked: bool = False


class CommandValidator:
    BLOCKED = {"rm", "mkfs", "shutdown", "reboot", "dd", "sudo"}

    def __init__(self, config: SecurityConfig) -> None:
        self.config = config

    def validate(self, command: str) -> tuple[bool, str]:
        if not command.strip() or len(command) > self.config.max_command_length:
            return False, "命令为空或超过长度限制"
        try:
            executable = shlex.split(command)[0]
        except ValueError:
            return False, "命令引号不完整"
        if executable in self.BLOCKED:
            return False, f"高风险命令 {executable} 已阻止"
        if self.config.allowlist_mode and executable not in self.config.allowlist:
            return False, f"命令 {executable} 不在白名单"
        return True, ""


class Sandbox:
    def __init__(self, config: SandboxConfig, security: SecurityConfig) -> None:
        self.config = config
        self.validator = CommandValidator(security)

    async def execute(self, command: str) -> ExecutionResult:
        allowed, reason = self.validator.validate(command)
        if not allowed:
            return ExecutionResult(stderr=reason, exit_code=126, backend=self.config.backend, blocked=True)
        if not self.config.enabled or self.config.backend == "mock":
            return ExecutionResult(stdout=f'[mock] 命令 "{command}" 在模拟沙箱中执行', backend="mock")
        if self.config.backend == "local":
            return await self._run_local(command)
        return await self._run_docker(command)

    async def _run_local(self, command: str) -> ExecutionResult:
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), self.config.timeout_ms / 1000)
        except TimeoutError:
            process.kill()
            return ExecutionResult(stderr="执行超时", exit_code=124, backend="local")
        limit = self.config.max_output_bytes
        return ExecutionResult(
            stdout=stdout[:limit].decode(errors="replace"),
            stderr=stderr[:limit].decode(errors="replace"),
            exit_code=process.returncode or 0,
            backend="local",
        )

    async def _run_docker(self, command: str) -> ExecutionResult:
        args = [
            "docker",
            "run",
            "--rm",
            f"--memory={self.config.memory_limit_mb}m",
            f"--pids-limit={self.config.max_pids}",
        ]
        if self.config.network_disabled:
            args += ["--network=none"]
        if self.config.readonly_rootfs:
            args += ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"]
        args += [self.config.image, "sh", "-lc", command]
        try:
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), self.config.timeout_ms / 1000)
            limit = self.config.max_output_bytes
            return ExecutionResult(
                stdout=stdout[:limit].decode(errors="replace"),
                stderr=stderr[:limit].decode(errors="replace"),
                exit_code=process.returncode or 0,
                backend="docker",
            )
        except (FileNotFoundError, TimeoutError):
            return ExecutionResult(stdout=f'[mock] Docker 不可用，模拟执行 "{command}"', backend="mock")
