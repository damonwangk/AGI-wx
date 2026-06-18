"""文档切分、多查询改写、混合检索、RRF 与回答合成。"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass

from .config import RAGConfig
from .infrastructure import Infrastructure
from .llm import LLMClient, parse_json_payload
from .memory import _cosine


@dataclass(slots=True)
class Chunk:
    id: int
    doc_hash: str
    index: int
    content: str
    parent: str
    embedding: list[float]


class RAGEngine:
    def __init__(self, config: RAGConfig, llm: LLMClient, infra: Infrastructure) -> None:
        self.config = config
        self.llm = llm
        self.infra = infra
        self.chunks: list[Chunk] = []

    @property
    def loaded(self) -> bool:
        return bool(self.chunks)

    def restore(self, rows: list[tuple[object, ...]]) -> None:
        """从 PostgreSQL 行恢复内存检索索引。"""
        self.chunks = []
        for row in rows:
            embedding = row[5] if isinstance(row[5], list) else []
            self.chunks.append(Chunk(int(row[0]), str(row[1]), int(row[2]), str(row[3]), str(row[4] or ""), embedding))

    async def ingest(self, content: str) -> dict[str, object]:
        doc_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        parents = self._split(content, self.config.chunk_size * 3, 0)
        children: list[tuple[str, str]] = []
        for parent in parents:
            children.extend(
                (child, parent) for child in self._split(parent, self.config.chunk_size, self.config.chunk_overlap)
            )
        indexed = 0
        for index, (child, parent) in enumerate(children):
            embedding = await self.llm.embed(child)
            pg_id = await self.infra.save_chunk(doc_hash, index, child, parent, embedding)
            chunk = Chunk(pg_id, doc_hash, index, child, parent, embedding)
            self.chunks.append(chunk)
            await self.infra.index_external(pg_id, doc_hash, index, child, embedding)
            indexed += 1
        return {
            "doc_hash": doc_hash,
            "chunk_count": len(children),
            "parent_count": len(parents),
            "indexed_count": indexed,
            "chunk_preview": [child[:120] for child, _ in children[:3]],
        }

    async def query(
        self, query: str, history: list[dict[str, str]] | None = None
    ) -> tuple[str, list[dict[str, object]]]:
        queries = await self._rewrite(query, history or [])
        results = await self.search_multi(queries, self.config.top_k)
        context = "\n\n".join(str(item["parent"] or item["content"]) for item in results)
        if not context:
            return "知识库中没有找到相关内容。", []
        answer = await self.llm.chat(
            "你是知识库问答助手。只能基于给定上下文回答；信息不足时明确说明。",
            [{"role": "user", "content": f"上下文：\n{context}\n\n问题：{query}"}],
        )
        return answer, results

    async def search_multi(self, queries: list[str], top_k: int) -> list[dict[str, object]]:
        result_sets = await asyncio.gather(*(self._search_one(query, max(top_k * 4, 10)) for query in queries))
        scores: dict[int, float] = {}
        by_id: dict[int, Chunk] = {}
        k = self.config.rrf_constant_k
        for results in result_sets:
            for rank, chunk in enumerate(results, 1):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank)
                by_id[chunk.id] = chunk
        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            {
                "id": item_id,
                "content": by_id[item_id].content,
                "parent": by_id[item_id].parent,
                "score": scores[item_id],
                "source": "hybrid",
                "doc_hash": by_id[item_id].doc_hash,
            }
            for item_id in ranked
        ]

    async def _search_one(self, query: str, limit: int) -> list[Chunk]:
        embedding = await self.llm.embed(query)
        semantic = sorted(self.chunks, key=lambda chunk: _cosine(embedding, chunk.embedding), reverse=True)[:limit]
        terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        keyword = sorted(
            self.chunks,
            key=lambda chunk: sum(term in chunk.content.lower() for term in terms),
            reverse=True,
        )[:limit]
        es_ids, milvus_ids, graph_ids = await asyncio.gather(
            self.infra.search_es(query, limit),
            self.infra.search_milvus(embedding, limit),
            self.infra.search_graph(query, limit),
        )
        es_ranked = [chunk for item_id in es_ids for chunk in self.chunks if chunk.id == item_id]
        milvus_ranked = [chunk for item_id in milvus_ids for chunk in self.chunks if chunk.id == item_id]
        graph_ranked = [chunk for item_id in graph_ids for chunk in self.chunks if chunk.id == item_id]
        scores: dict[int, float] = {}
        by_id = {chunk.id: chunk for chunk in self.chunks}
        for source in (semantic, keyword, es_ranked, milvus_ranked):
            for rank, chunk in enumerate(source, 1):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (self.config.rrf_constant_k + rank)
        for rank, chunk in enumerate(graph_ranked, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + self.infra.config.neo4j.weight / (
                self.config.rrf_constant_k + rank
            )
        return [by_id[item_id] for item_id in sorted(scores, key=scores.get, reverse=True)[:limit]]

    async def _rewrite(self, query: str, history: list[dict[str, str]]) -> list[str]:
        cfg = self.config.rewrite
        if not cfg.enabled or cfg.num_queries <= 1 or not self.llm.is_real:
            return [query]
        raw = await self.llm.chat(
            "结合最近对话消解指代，输出 JSON 字符串数组；第一项保留原问题语义。",
            [{"role": "user", "content": f"历史：{history[-4:]}\n问题：{query}"}],
        )
        parsed = parse_json_payload(raw, [query])
        values = [str(value).strip() for value in parsed if str(value).strip()] if isinstance(parsed, list) else [query]
        return list(dict.fromkeys([query, *values]))[: cfg.num_queries]

    @staticmethod
    def _split(text: str, size: int, overlap: int) -> list[str]:
        """优先按段落切分，超长段落再按字符窗口处理。"""
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if len(candidate) <= size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= size:
                current = paragraph
                continue
            step = max(size - overlap, 1)
            chunks.extend(paragraph[index : index + size] for index in range(0, len(paragraph), step))
            current = ""
        if current:
            chunks.append(current)
        return chunks or ([text] if text else [])

    async def delete(self, doc_hash: str) -> int:
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.doc_hash != doc_hash]
        await self.infra.delete_chunks(doc_hash)
        return before - len(self.chunks)
