"""短期、长期、偏好三层记忆。"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import MemoryConfig


@dataclass(slots=True)
class MemoryItem:
    id: int
    content: str
    importance: float
    embedding: list[float] = field(default_factory=list)
    category: str = "general"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))


class ShortTermMemory:
    def __init__(self, max_turns: int) -> None:
        self.messages: deque[dict[str, str]] = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def snapshot(self) -> list[dict[str, str]]:
        return list(self.messages)


class LongTermMemory:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.items: list[MemoryItem] = []
        self._next_id = 1

    def store(
        self,
        content: str,
        importance: float = 0.5,
        embedding: list[float] | None = None,
        category: str = "general",
    ) -> MemoryItem:
        normalized = content.strip()
        for item in self.items:
            if item.content == normalized:
                item.importance = max(item.importance, importance)
                return item
        item = MemoryItem(self._next_id, normalized, importance, embedding or [], category)
        self._next_id += 1
        self.items.append(item)
        return item

    def recall(self, query: str, embedding: list[float] | None = None, top_k: int | None = None) -> list[MemoryItem]:
        """优先向量召回；没有向量时使用字符词频余弦。"""
        top_k = top_k or self.config.long_term_top_k
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.items:
            if embedding and item.embedding and len(embedding) == len(item.embedding):
                similarity = _cosine(embedding, item.embedding)
            else:
                similarity = _counter_cosine(Counter(query), Counter(item.content))
            score = similarity * 0.7 + item.importance * 0.3
            if score >= 0.25:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for _, item in scored[:top_k]:
            item.last_accessed = datetime.now(UTC)
        return [item for _, item in scored[:top_k]]

    def consolidate(self) -> None:
        """执行重要性衰减和过期淘汰，避免长期记忆无限增长。"""
        policy = self.config.consolidation
        now = datetime.now(UTC)
        kept: list[MemoryItem] = []
        for item in self.items:
            age_days = max((now - item.created_at).days, 0)
            item.importance *= policy.decay_rate**age_days
            if age_days <= policy.ttl_days or item.importance >= policy.min_importance:
                kept.append(item)
        self.items = kept


class PreferenceMemory:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def save_batch(self, values: dict[str, str]) -> None:
        self.values.update(values)


class MemoryStack:
    def __init__(self, config: MemoryConfig) -> None:
        self.stm = ShortTermMemory(config.short_term_max_turns)
        self.ltm = LongTermMemory(config)
        self.preferences = PreferenceMemory()

    def context(self, recalled: list[MemoryItem]) -> str:
        sections: list[str] = []
        if self.preferences.values:
            sections.append("用户偏好：" + "；".join(f"{k}={v}" for k, v in self.preferences.values.items()))
        if recalled:
            sections.append("相关记忆：" + "；".join(item.content for item in recalled))
        return "\n".join(sections)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right)) or 1.0
    return dot / norm


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right[key] for key, value in left.items())
    norm = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values())) or 1.0
    return dot / norm
