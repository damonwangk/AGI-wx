# AGI Assistant（Python）

基于 Python 3.12、FastAPI 和 asyncio 的多能力智能体系统。支持普通对话、RAG、工具调用、ReAct 任务图、三层记忆、知识图谱、沙箱和 MCP 工具。

## 核心能力

- 四模式智能路由：`chat`、`tool`、`rag`、`react`
- SSE 流式 token、工具步骤、任务图和最终结果
- Milvus + Elasticsearch + 本地检索的 RRF 混合召回
- 多查询改写、父子切块、Rerank 接口和引用结果
- 短期记忆、长期向量记忆、用户偏好
- DAG 拓扑调度、并行执行、竞速、重试、取消和快照
- Docker、Local、Mock 沙箱与命令安全校验
- PostgreSQL、Milvus、Elasticsearch、Neo4j、Kafka 独立降级
- 文档上传、PDF 文本提取、版本化文档库
- MCP JSON-RPC HTTP 工具动态注册

## 架构

```text
src/agi_assistant/
├── api.py             FastAPI 路由和 SSE
├── agent.py           UnifiedAgent 主编排
├── runtime.py         ReAct Planner 与 DAG 运行时
├── rag.py             切分、索引、混合检索、RRF
├── memory.py          三层记忆
├── tools.py           内置工具和 MCP 注册
├── sandbox.py         命令校验与三种执行后端
├── infrastructure.py  外部服务适配器
├── documents.py       文档解析和版本化存储
├── llm.py             对话、流式和 Embedding 客户端
├── config.py          YAML 配置
└── models.py          API 与运行时模型
```

## Conda 环境运行

项目当前使用：

```bash
conda activate test1-rag
python -m pip install -e '.[dev]'
agi-assistant
```

访问：<http://localhost:8090>

也可以直接运行：

```bash
conda run -n test1-rag python -m agi_assistant.main
```

## 配置

默认读取 `config/config.yaml`。密钥支持环境变量：

```yaml
llm:
  api_key: "${DEEPSEEK_API_KEY}"
```

可通过 `AGI_CONFIG` 指定其他配置文件：

```bash
AGI_CONFIG=config/config.docker.yaml agi-assistant
```

未配置 LLM 或外部数据库时，系统使用本地 Mock、内存检索和日志事件降级，HTTP 服务仍可启动。

## 测试

```bash
conda run -n test1-rag python -m pytest
conda run -n test1-rag python -m ruff check src tests
```

## Docker Compose

```bash
docker compose up --build
```

Compose 会启动应用、PostgreSQL、Milvus、Elasticsearch、Neo4j 和 Kafka。容器使用 `config/config.docker.yaml`。

## API

- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/chat/cancel`
- `POST /api/upload`
- `POST /api/docs/delete`
- `GET /api/documents`
- `GET|DELETE /api/documents/{id}`
- `POST /api/documents/{id}/ingest`
- `GET /api/memory`
- `GET /api/tools`
- `POST /api/tools/mcp`
- `GET /api/snapshots`
- `GET /api/status`
