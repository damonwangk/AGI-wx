"""三层记忆测试。"""

from agi_assistant.config import MemoryConfig
from agi_assistant.memory import LongTermMemory, ShortTermMemory


def test_short_term_window() -> None:
    memory = ShortTermMemory(max_turns=1)
    memory.add("user", "第一轮")
    memory.add("assistant", "回答")
    memory.add("user", "第二轮")
    assert [item["content"] for item in memory.snapshot()] == ["回答", "第二轮"]


def test_long_term_recall_and_dedup() -> None:
    memory = LongTermMemory(MemoryConfig())
    first = memory.store("用户喜欢 Python", 0.8)
    second = memory.store("用户喜欢 Python", 0.5)
    assert first.id == second.id
    assert memory.recall("Python")
