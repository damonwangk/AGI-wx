"""RAG 切分与本地检索测试。"""

import pytest

from agi_assistant.config import AppConfig
from agi_assistant.infrastructure import Infrastructure
from agi_assistant.llm import LLMClient
from agi_assistant.rag import RAGEngine


@pytest.mark.asyncio
async def test_ingest_and_search_without_external_services() -> None:
    config = AppConfig()
    infra = Infrastructure(config)
    llm = LLMClient(config)
    rag = RAGEngine(config.rag, llm, infra)
    result = await rag.ingest("Python 是一种编程语言。\n\nMilvus 是向量数据库。")
    hits = await rag.search_multi(["Python 编程"], 2)
    assert result["indexed_count"] >= 1
    assert hits and "Python" in str(hits[0])
    await llm.close()


def test_split_preserves_content() -> None:
    chunks = RAGEngine._split("第一段\n\n第二段", 20, 2)
    assert "第一段" in chunks[0]
