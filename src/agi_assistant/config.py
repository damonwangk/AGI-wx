"""配置加载：兼容原项目的 config/config.yaml 与环境变量占位符。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    api_url: str = "https://api.deepseek.com/chat/completions"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.7


class EmbeddingConfig(BaseModel):
    api_url: str = ""
    api_key: str = ""
    model: str = ""


class ServiceConfig(BaseModel):
    host: str = "localhost"
    port: int = 0


class PostgresConfig(ServiceConfig):
    port: int = 5432
    user: str = "aiagent"
    password: str = "aiagent123"
    database: str = "aiagent"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ElasticsearchConfig(BaseModel):
    addresses: list[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    username: str = ""
    password: str = ""


class KafkaConfig(BaseModel):
    brokers: list[str] = Field(default_factory=lambda: ["localhost:29092"])
    topic: str = "agent-events"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"
    max_hops: int = 2
    weight: float = 0.3
    enabled: bool = True


class RewriteConfig(BaseModel):
    enabled: bool = True
    num_queries: int = 3


class RerankConfig(BaseModel):
    enabled: bool = True
    preview_len: int = 200


class RAGConfig(BaseModel):
    chunk_size: int = 200
    chunk_overlap: int = 50
    top_k: int = 3
    rrf_constant_k: int = 60
    semantic_weight: float = 0.7
    enable_hybrid_search: bool = True
    rag_milvus_dim: int = 2048
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)


class ConsolidationConfig(BaseModel):
    similarity_threshold: float = 0.8
    dedup_threshold: float = 0.95
    ttl_days: int = 30
    decay_rate: float = 0.995
    min_importance: float = 0.3
    trigger_interval: int = 5


class MemoryConfig(BaseModel):
    short_term_max_turns: int = 5
    long_term_top_k: int = 3
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)


class HarnessConfig(BaseModel):
    max_retries: int = 3
    retry_delay_ms: int = 200
    step_timeout_ms: int = 5000
    max_iterations: int = 5


class SandboxConfig(BaseModel):
    enabled: bool = True
    backend: str = "docker"
    image: str = "ubuntu:22.04"
    timeout_ms: int = 30000
    max_output_bytes: int = 65536
    memory_limit_mb: int = 256
    cpu_percent: int = 50
    max_pids: int = 64
    network_disabled: bool = True
    readonly_rootfs: bool = True


class SecurityConfig(BaseModel):
    max_command_length: int = 500
    allowlist_mode: bool = False
    allowlist: list[str] = Field(default_factory=list)


class GraphRuntimeConfig(BaseModel):
    max_parallel: int = 2
    race_timeout_ms: int = 30000
    enable_racing: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    milvus: ServiceConfig = Field(default_factory=lambda: ServiceConfig(port=19530))
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    graph_runtime: GraphRuntimeConfig = Field(default_factory=GraphRuntimeConfig)
    server: dict[str, Any] = Field(default_factory=lambda: {"port": "8090"})
    search: dict[str, str] = Field(default_factory=dict)

    @property
    def server_port(self) -> int:
        return int(self.server.get("port", 8090))


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)}")


def _expand_env(value: Any) -> Any:
    """递归展开 YAML 中的 ${ENV_NAME}，未设置时返回空字符串。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载配置文件；AGI_CONFIG 可覆盖默认路径。"""
    config_path = Path(path or os.getenv("AGI_CONFIG", "config/config.yaml"))
    if not config_path.exists():
        return AppConfig()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(_expand_env(raw))
