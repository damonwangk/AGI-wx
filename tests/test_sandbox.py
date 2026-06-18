"""沙箱安全测试。"""

import pytest

from agi_assistant.config import SandboxConfig, SecurityConfig
from agi_assistant.sandbox import Sandbox


@pytest.mark.asyncio
async def test_block_destructive_command() -> None:
    sandbox = Sandbox(SandboxConfig(backend="mock"), SecurityConfig())
    result = await sandbox.execute("rm -rf /")
    assert result.blocked is True
    assert result.exit_code == 126


@pytest.mark.asyncio
async def test_mock_execution() -> None:
    sandbox = Sandbox(SandboxConfig(backend="mock"), SecurityConfig())
    result = await sandbox.execute("echo hello")
    assert result.backend == "mock"
