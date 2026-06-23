"""ReAct 计划生成与 DAG 并行执行运行时。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from .config import AppConfig
from .infrastructure import Infrastructure
from .llm import LLMClient, parse_json_payload
from .models import ReactStep, StepType, StreamEvent, TaskNode, TaskState
from .tools import ToolRegistry

EventCallback = Callable[[StreamEvent], Awaitable[None]]


class GraphRuntime:
    def __init__(self, config: AppConfig, llm: LLMClient, tools: ToolRegistry, infra: Infrastructure) -> None:
        self.config, self.llm, self.tools, self.infra = config, llm, tools, infra

    async def plan(self, query: str, selected: dict[str, Any]) -> TaskState:
        descriptions = [{"name": tool.name, "description": tool.description} for tool in selected.values()]
        raw = await self.llm.chat(
            "把任务规划为 JSON 数组。每项字段：id,name,tool,params,depends_on,race_group。只使用给定工具。",
            [
                {
                    "role": "user",
                    "content": f"任务：{query}\n工具：{json.dumps(descriptions, ensure_ascii=False)}",
                }
            ],
        )
        parsed = parse_json_payload(raw, [])
        nodes: list[TaskNode] = []
        if isinstance(parsed, list):
            for index, item in enumerate(parsed):
                if isinstance(item, dict) and item.get("tool") in selected:
                    # LLM 可能返回数字 ID；统一转换为任务模型要求的字符串。
                    normalized = {
                        **item,
                        "id": str(item.get("id", f"step-{index + 1}")),
                        "name": str(item.get("name", f"执行 {item['tool']}")),
                        "tool": str(item["tool"]),
                        "params": item.get("params") if isinstance(item.get("params"), dict) else {},
                        "depends_on": [str(value) for value in item.get("depends_on", [])],
                        "race_group": str(item.get("race_group") or ""),
                    }
                    try:
                        nodes.append(TaskNode.model_validate(normalized))
                    except ValidationError:
                        # 单个规划节点格式错误时跳过，最终使用规则规划兜底。
                        continue
        if not nodes:
            nodes = self._rule_plan(query, selected)
        return TaskState(id=str(uuid.uuid4()), query=query, nodes=nodes)

    def _rule_plan(self, query: str, selected: dict[str, Any]) -> list[TaskNode]:
        names: list[str] = []
        rules = [
            ("rag_search", ("知识库", "文档", "资料")),
            ("search_web", ("搜索", "最新", "联网")),
            ("current_time", ("时间", "几点", "日期")),
            ("weather", ("天气", "气温")),
            ("exec_command", ("执行", "命令", "终端")),
            ("write_document", ("写文档", "报告", "保存")),
        ]
        for name, keywords in rules:
            if name in selected and any(keyword in query for keyword in keywords):
                names.append(name)
        if not names and selected:
            names.append(next(iter(selected)))
        return [
            TaskNode(
                id=f"step-{index + 1}",
                name=f"调用 {name}",
                tool=name,
                params=self._params(name, query),
            )
            for index, name in enumerate(names)
        ]

    @staticmethod
    def _params(tool: str, query: str) -> dict[str, Any]:
        key = "command" if tool == "exec_command" else "city" if tool == "weather" else "query"
        return {key: query}

    async def execute(
        self, task: TaskState, callback: EventCallback | None = None
    ) -> tuple[list[str], list[ReactStep]]:
        """逐拓扑层执行，竞速组只保留首个成功结果。"""
        observations: list[str] = []
        steps: list[ReactStep] = []
        pending = {node.id: node for node in task.nodes}
        completed: set[str] = set()
        semaphore = asyncio.Semaphore(self.config.graph_runtime.max_parallel)
        while pending:
            ready = [node for node in pending.values() if set(node.depends_on) <= completed]
            if not ready:
                raise ValueError("任务图存在环或悬空依赖")
            groups: dict[str, list[TaskNode]] = defaultdict(list)
            for node in ready:
                groups[node.race_group or f"single:{node.id}"].append(node)
            results = await asyncio.gather(
                *(self._execute_group(group, semaphore, callback) for group in groups.values())
            )
            # 当前层的所有节点都已经得到终态；竞速败者也必须从待执行集合移除。
            for node in ready:
                completed.add(node.id)
                pending.pop(node.id, None)
            for group_results in results:
                for _node, result, node_steps in group_results:
                    steps.extend(node_steps)
                    if result:
                        observations.append(result)
            await self.infra.save_snapshot(task.id, task.model_dump(mode="json"))
        return observations, steps

    async def _execute_group(
        self, nodes: list[TaskNode], semaphore: asyncio.Semaphore, callback: EventCallback | None
    ) -> list[tuple[TaskNode, str, list[ReactStep]]]:
        if len(nodes) == 1 or not self.config.graph_runtime.enable_racing:
            return await asyncio.gather(*(self._execute_node(node, semaphore, callback) for node in nodes))
        tasks = [asyncio.create_task(self._execute_node(node, semaphore, callback)) for node in nodes]
        pending = set(tasks)
        winner: tuple[TaskNode, str, list[ReactStep]] | None = None
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                result = await future
                if result[1]:
                    winner = result
                    break
        for future in pending:
            future.cancel()
        output = [winner] if winner else [await future for future in tasks if future.done() and not future.cancelled()]
        winner_id = winner[0].id if winner else ""
        for node in nodes:
            if node.id != winner_id:
                node.status = "skipped"
        return output

    async def _execute_node(
        self, node: TaskNode, semaphore: asyncio.Semaphore, callback: EventCallback | None
    ) -> tuple[TaskNode, str, list[ReactStep]]:
        thought = ReactStep(type=StepType.THOUGHT, content=node.name)
        action = ReactStep(type=StepType.ACTION, content=f"调用 {node.tool}", tool=node.tool, params=node.params)
        steps = [thought, action]
        if callback:
            await callback(StreamEvent(type="step", data=thought.model_dump(mode="json")))
            await callback(StreamEvent(type="step", data=action.model_dump(mode="json")))
        tool = self.tools.tools.get(node.tool)
        if not tool:
            node.status, node.error = "failed", f"工具 {node.tool} 不存在"
            return node, "", steps
        async with semaphore:
            for attempt in range(self.config.harness.max_retries):
                try:
                    result = await asyncio.wait_for(
                        tool.execute(node.params), self.config.harness.step_timeout_ms / 1000
                    )
                    node.status, node.result = "done", result
                    observation = ReactStep(type=StepType.OBSERVATION, content=result, tool=node.tool)
                    steps.append(observation)
                    if callback:
                        await callback(StreamEvent(type="step", data=observation.model_dump(mode="json")))
                    return node, result, steps
                except Exception as exc:  # noqa: BLE001
                    node.error = str(exc)
                    if attempt + 1 < self.config.harness.max_retries:
                        await asyncio.sleep(self.config.harness.retry_delay_ms / 1000)
            node.status = "failed"
            return node, "", steps
