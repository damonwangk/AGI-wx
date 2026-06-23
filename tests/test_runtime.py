"""DAG 调度与竞速状态测试。"""

import pytest

from agi_assistant.config import AppConfig
from agi_assistant.documents import DocumentLibrary
from agi_assistant.infrastructure import Infrastructure
from agi_assistant.llm import LLMClient
from agi_assistant.models import TaskNode, TaskState
from agi_assistant.rag import RAGEngine
from agi_assistant.runtime import GraphRuntime
from agi_assistant.sandbox import Sandbox
from agi_assistant.tools import ToolRegistry


@pytest.mark.asyncio
async def test_race_marks_losers_skipped(tmp_path) -> None:
    config = AppConfig()
    infra = Infrastructure(config)
    llm = LLMClient(config)
    rag = RAGEngine(config.rag, llm, infra)
    tools = ToolRegistry(
        rag,
        Sandbox(config.sandbox.model_copy(update={"backend": "mock"}), config.security),
        DocumentLibrary(tmp_path),
        {},
    )
    runtime = GraphRuntime(config, llm, tools, infra)
    task = TaskState(
        query="时间",
        id="task",
        nodes=[
            TaskNode(id="a", name="A", tool="current_time", race_group="clock"),
            TaskNode(id="b", name="B", tool="current_time", race_group="clock"),
        ],
    )
    observations, _ = await runtime.execute(task)
    assert len(observations) == 1
    assert sum(node.status == "skipped" for node in task.nodes) == 1
    await llm.close()


@pytest.mark.asyncio
async def test_plan_normalizes_numeric_node_fields(tmp_path, monkeypatch) -> None:
    """DeepSeek 返回数字节点 ID 时，规划仍应能被任务模型接受。"""
    config = AppConfig()
    infra = Infrastructure(config)
    llm = LLMClient(config)
    rag = RAGEngine(config.rag, llm, infra)
    tools = ToolRegistry(
        rag,
        Sandbox(config.sandbox.model_copy(update={"backend": "mock"}), config.security),
        DocumentLibrary(tmp_path),
        {},
    )
    runtime = GraphRuntime(config, llm, tools, infra)

    async def numeric_plan(*_args, **_kwargs) -> str:
        return '[{"id":1,"name":"查询天气","tool":"weather","params":{"city":"北京"},"depends_on":[],"race_group":1}]'

    monkeypatch.setattr(llm, "chat", numeric_plan)
    task = await runtime.plan("北京天气怎么样？", {"weather": tools.tools["weather"]})

    assert task.nodes[0].id == "1"
    assert task.nodes[0].race_group == "1"
    await llm.close()
