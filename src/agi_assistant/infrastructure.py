"""PostgreSQL、Milvus、Elasticsearch、Neo4j、Kafka 的可降级适配器。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .config import AppConfig

logger = logging.getLogger(__name__)


class Infrastructure:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pg: Any = None
        self.es: Any = None
        self.milvus: Any = None
        self.neo4j: Any = None
        self.kafka: Any = None
        self.status = {name: "disconnected" for name in ("pg", "elasticsearch", "milvus", "neo4j", "kafka")}

    async def connect(self) -> None:
        """并行连接所有外部服务，每一路失败都只记录状态。"""
        await asyncio.gather(
            self._connect_pg(),
            self._connect_es(),
            self._connect_milvus(),
            self._connect_neo4j(),
            self._connect_kafka(),
        )

    async def close(self) -> None:
        if self.pg:
            await self.pg.close()
        if self.es:
            await self.es.close()
        if self.neo4j:
            await self.neo4j.close()
        if self.kafka:
            await self.kafka.stop()

    async def _connect_pg(self) -> None:
        try:
            import psycopg

            self.pg = await psycopg.AsyncConnection.connect(self.config.postgres.dsn, connect_timeout=2)
            await self._bootstrap_schema()
            self.status["pg"] = "connected"
        except Exception as exc:  # noqa: BLE001
            logger.warning("PostgreSQL 不可用：%s", exc)

    async def _bootstrap_schema(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS chat_history (
                id BIGSERIAL PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY(user_id, key))""",
            """CREATE TABLE IF NOT EXISTS long_term_memory (
                id BIGSERIAL PRIMARY KEY, content TEXT NOT NULL, importance DOUBLE PRECISION,
                embedding JSONB, category TEXT DEFAULT 'general', created_at TIMESTAMPTZ DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS rag_chunks (
                id BIGSERIAL PRIMARY KEY, doc_hash TEXT NOT NULL, chunk_idx INTEGER NOT NULL,
                content TEXT NOT NULL, parent_content TEXT, embedding JSONB,
                UNIQUE(doc_hash, chunk_idx))""",
            """CREATE TABLE IF NOT EXISTS snapshots (
                task_id TEXT PRIMARY KEY, state_json JSONB NOT NULL, updated_at TIMESTAMPTZ DEFAULT now())""",
        ]
        async with self.pg.cursor() as cursor:
            for statement in statements:
                await cursor.execute(statement)
        await self.pg.commit()

    async def _connect_es(self) -> None:
        try:
            from elasticsearch import AsyncElasticsearch

            cfg = self.config.elasticsearch
            kwargs = {"hosts": cfg.addresses, "request_timeout": 2}
            if cfg.username:
                kwargs["basic_auth"] = (cfg.username, cfg.password)
            self.es = AsyncElasticsearch(**kwargs)
            if not await self.es.ping():
                raise ConnectionError("ping 失败")
            if not await self.es.indices.exists(index="rag_chunks"):
                await self.es.indices.create(index="rag_chunks")
            self.status["elasticsearch"] = "connected"
        except Exception as exc:  # noqa: BLE001
            self.es = None
            logger.warning("Elasticsearch 不可用：%s", exc)

    async def _connect_milvus(self) -> None:
        try:
            from pymilvus import MilvusClient

            uri = f"http://{self.config.milvus.host}:{self.config.milvus.port}"
            self.milvus = await asyncio.to_thread(MilvusClient, uri=uri)
            exists = await asyncio.to_thread(self.milvus.has_collection, collection_name="rag_chunks")
            if not exists:
                await asyncio.to_thread(
                    self.milvus.create_collection,
                    collection_name="rag_chunks",
                    dimension=self.config.rag.rag_milvus_dim,
                    metric_type="COSINE",
                )
            self.status["milvus"] = "connected"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 不可用：%s", exc)

    async def _connect_neo4j(self) -> None:
        if not self.config.neo4j.enabled:
            return
        try:
            from neo4j import AsyncGraphDatabase

            cfg = self.config.neo4j
            self.neo4j = AsyncGraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
            await self.neo4j.verify_connectivity()
            self.status["neo4j"] = "connected"
        except Exception as exc:  # noqa: BLE001
            self.neo4j = None
            logger.warning("Neo4j 不可用：%s", exc)

    async def _connect_kafka(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer

            self.kafka = AIOKafkaProducer(bootstrap_servers=self.config.kafka.brokers)
            await asyncio.wait_for(self.kafka.start(), timeout=3)
            self.status["kafka"] = "connected"
        except Exception as exc:  # noqa: BLE001
            self.kafka = None
            logger.warning("Kafka 不可用：%s", exc)

    async def save_chat(self, role: str, content: str) -> None:
        if not self.pg:
            return
        async with self.pg.cursor() as cursor:
            await cursor.execute("INSERT INTO chat_history(role, content) VALUES (%s, %s)", (role, content))
        await self.pg.commit()

    async def save_preference(self, user_id: str, key: str, value: str) -> None:
        if not self.pg:
            return
        async with self.pg.cursor() as cursor:
            await cursor.execute(
                """INSERT INTO preferences(user_id,key,value) VALUES (%s,%s,%s)
                ON CONFLICT(user_id,key) DO UPDATE SET value=EXCLUDED.value""",
                (user_id, key, value),
            )
        await self.pg.commit()

    async def load_state(self) -> dict[str, list[Any]]:
        """从 PostgreSQL 恢复聊天、偏好、长期记忆和 RAG 块。"""
        state: dict[str, list[Any]] = {"chat": [], "preferences": [], "memories": [], "chunks": []}
        if not self.pg:
            return state
        queries = {
            "chat": "SELECT role,content FROM chat_history ORDER BY id DESC LIMIT 20",
            "preferences": "SELECT key,value FROM preferences WHERE user_id='default'",
            "memories": "SELECT id,content,importance,embedding,category FROM long_term_memory ORDER BY id",
            "chunks": "SELECT id,doc_hash,chunk_idx,content,parent_content,embedding FROM rag_chunks ORDER BY id",
        }
        async with self.pg.cursor() as cursor:
            for key, query in queries.items():
                await cursor.execute(query)
                state[key] = list(await cursor.fetchall())
        state["chat"].reverse()
        return state

    async def save_memory(self, content: str, importance: float, embedding: list[float], category: str) -> None:
        if not self.pg:
            return
        async with self.pg.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO long_term_memory(content,importance,embedding,category) VALUES (%s,%s,%s,%s)",
                (content, importance, json.dumps(embedding), category),
            )
        await self.pg.commit()

    async def save_chunk(self, doc_hash: str, index: int, content: str, parent: str, embedding: list[float]) -> int:
        """以 PostgreSQL 为真相源；不可用时返回本地负 ID。"""
        if not self.pg:
            return -(index + 1)
        async with self.pg.cursor() as cursor:
            await cursor.execute(
                """INSERT INTO rag_chunks(doc_hash,chunk_idx,content,parent_content,embedding)
                VALUES (%s,%s,%s,%s,%s) ON CONFLICT(doc_hash,chunk_idx) DO UPDATE
                SET content=EXCLUDED.content,parent_content=EXCLUDED.parent_content,
                embedding=EXCLUDED.embedding RETURNING id""",
                (doc_hash, index, content, parent, json.dumps(embedding)),
            )
            row = await cursor.fetchone()
        await self.pg.commit()
        return int(row[0])

    async def index_external(self, pg_id: int, doc_hash: str, index: int, content: str, embedding: list[float]) -> None:
        if self.es:
            try:
                await self.es.index(
                    index="rag_chunks",
                    id=str(pg_id),
                    document={
                        "pg_id": pg_id,
                        "doc_hash": doc_hash,
                        "chunk_idx": index,
                        "content": content,
                    },
                    refresh=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ES 写入失败：%s", exc)
        if self.milvus and pg_id > 0 and embedding:
            try:
                await asyncio.to_thread(
                    self.milvus.insert,
                    collection_name="rag_chunks",
                    data=[{"id": pg_id, "vector": embedding, "content": content}],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Milvus 写入失败：%s", exc)
        if self.neo4j and pg_id > 0:
            try:
                async with self.neo4j.session() as session:
                    await session.run(
                        "MERGE (c:RAGChunk {pg_id:$id}) SET c.content=$content, c.doc_hash=$hash, c.chunk_idx=$idx",
                        id=pg_id,
                        content=content,
                        hash=doc_hash,
                        idx=index,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neo4j 写入失败：%s", exc)

    async def search_es(self, query: str, limit: int) -> list[int]:
        if not self.es:
            return []
        try:
            result = await self.es.search(index="rag_chunks", query={"match": {"content": query}}, size=limit)
            return [int(hit["_source"]["pg_id"]) for hit in result["hits"]["hits"]]
        except Exception:  # noqa: BLE001
            return []

    async def search_milvus(self, embedding: list[float], limit: int) -> list[int]:
        if not self.milvus or len(embedding) != self.config.rag.rag_milvus_dim:
            return []
        try:
            result = await asyncio.to_thread(
                self.milvus.search,
                collection_name="rag_chunks",
                data=[embedding],
                limit=limit,
                output_fields=["id"],
            )
            return [int(hit["id"]) for hit in result[0]]
        except Exception:  # noqa: BLE001
            return []

    async def search_graph(self, query: str, limit: int) -> list[int]:
        """Neo4j 第三路召回；使用 chunk 文本包含匹配并返回 PG 主键。"""
        if not self.neo4j:
            return []
        try:
            async with self.neo4j.session() as session:
                result = await session.run(
                    "MATCH (c:RAGChunk) WHERE toLower(c.content) CONTAINS toLower($query) "
                    "RETURN c.pg_id AS id LIMIT $limit",
                    query=query,
                    limit=limit,
                )
                return [int(record["id"]) async for record in result]
        except Exception:  # noqa: BLE001
            return []

    async def delete_chunks(self, doc_hash: str) -> None:
        """按文档哈希级联删除 PostgreSQL、ES、Milvus 和 Neo4j 数据。"""
        ids: list[int] = []
        if self.pg:
            async with self.pg.cursor() as cursor:
                await cursor.execute("DELETE FROM rag_chunks WHERE doc_hash=%s RETURNING id", (doc_hash,))
                ids = [int(row[0]) for row in await cursor.fetchall()]
            await self.pg.commit()
        if self.es:
            await self.es.delete_by_query(
                index="rag_chunks", query={"term": {"doc_hash.keyword": doc_hash}}, conflicts="proceed"
            )
        if self.milvus and ids:
            await asyncio.to_thread(self.milvus.delete, collection_name="rag_chunks", ids=ids)
        if self.neo4j:
            async with self.neo4j.session() as session:
                await session.run("MATCH (c:RAGChunk {doc_hash:$hash}) DETACH DELETE c", hash=doc_hash)

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.kafka:
            logger.info("事件降级到日志：%s %s", event_type, payload)
            return
        await self.kafka.send_and_wait(
            self.config.kafka.topic,
            json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False).encode(),
        )

    async def save_snapshot(self, task_id: str, state: dict[str, Any]) -> None:
        if not self.pg:
            return
        async with self.pg.cursor() as cursor:
            await cursor.execute(
                """INSERT INTO snapshots(task_id,state_json) VALUES (%s,%s)
                ON CONFLICT(task_id) DO UPDATE SET state_json=EXCLUDED.state_json,updated_at=now()""",
                (task_id, json.dumps(state, ensure_ascii=False, default=str)),
            )
        await self.pg.commit()
