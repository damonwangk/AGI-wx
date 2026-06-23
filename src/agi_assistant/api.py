"""FastAPI 接口层：完整保留原 Go 服务的 API 路径。"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import UnifiedAgent
from .config import AppConfig, load_config
from .documents import parse_document
from .infrastructure import Infrastructure
from .models import ChatOptions, ChatRequest, StreamEvent

logger = logging.getLogger(__name__)


def create_app(config: AppConfig | None = None, connect_infrastructure: bool = True) -> FastAPI:
    cfg = config or load_config()
    infra = Infrastructure(cfg)
    agent = UnifiedAgent(cfg, infra)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if connect_infrastructure:
            await infra.connect()
            await agent.bootstrap()
        yield
        await agent.close()
        await infra.close()

    app = FastAPI(title="AGI-wx", version="1.0.0", lifespan=lifespan)
    app.state.config, app.state.infra, app.state.agent = cfg, infra, agent

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        options = ChatOptions(use_rag=body.use_rag, selected_tools=body.selected_tools, explicit=body.explicit)
        return await agent.process(body.message, options)

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest):
        async def events():
            queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

            async def emit(event: StreamEvent) -> None:
                await queue.put(event)

            options = ChatOptions(use_rag=body.use_rag, selected_tools=body.selected_tools, explicit=body.explicit)

            async def run_worker() -> None:
                """确保后台异常也会通知 SSE 消费端结束等待。"""
                try:
                    await agent.process(body.message, options, emit)
                except Exception:  # noqa: BLE001 - 对外统一返回安全错误信息
                    logger.exception("流式对话处理失败")
                    await queue.put(StreamEvent(type="error", data={"message": "任务处理失败，请稍后重试"}))
                finally:
                    await queue.put(None)

            worker = asyncio.create_task(run_worker())
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"event: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False, default=str)}\n\n"
                if event.type == "done":
                    break
            await worker

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/chat/cancel")
    async def chat_cancel():
        agent.cancel()
        return {"ok": True, "message": "已发送取消信号"}

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile | None = File(default=None)):
        if file:
            data, filename, content_type = (
                await file.read(),
                file.filename or "upload.txt",
                file.content_type or "text/plain",
            )
        else:
            data, filename, content_type = (
                await request.body(),
                "upload.txt",
                request.headers.get("content-type", "text/plain"),
            )
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(413, "文件超过 64MB")
        content, pages, needs_ocr = parse_document(filename, content_type, data)
        if needs_ocr:
            return {
                "filename": filename,
                "pages": pages,
                "text_chars": len(content),
                "needs_ocr": True,
                "chunk_count": 0,
            }
        result = await agent.rag.ingest(content)
        return {
            "filename": filename,
            "content_type": content_type,
            "parser": "python",
            "pages": pages,
            "text_chars": len(content),
            "needs_ocr": False,
            **result,
            "chunks": [{"id": c.id, "content": c.content, "doc_hash": c.doc_hash} for c in agent.rag.chunks],
        }

    @app.post("/api/docs/delete")
    async def docs_delete(request: Request):
        payload = await request.json()
        doc_hash = str(payload.get("doc_hash", ""))
        return {"ok": True, "deleted": await agent.rag.delete(doc_hash), "doc_hash": doc_hash}

    @app.get("/api/documents")
    async def documents():
        return [record.model_dump(mode="json", exclude={"content"}) for record in agent.documents.list()]

    @app.api_route("/api/documents/{doc_id}", methods=["GET", "DELETE"])
    async def document_by_id(doc_id: str, request: Request):
        if request.method == "DELETE":
            return {"ok": agent.documents.delete(doc_id)}
        record = agent.documents.get(doc_id)
        if not record:
            raise HTTPException(404, "文档不存在")
        return record

    @app.post("/api/documents/{doc_id}/ingest")
    async def ingest_document(doc_id: str):
        record = agent.documents.get(doc_id)
        if not record:
            raise HTTPException(404, "文档不存在")
        return await agent.rag.ingest(record.content)

    @app.get("/api/memory")
    async def memory():
        return {
            "short_term": agent.memory.stm.snapshot(),
            "long_term": [asdict(item) for item in agent.memory.ltm.items],
            "preferences": agent.memory.preferences.values,
        }

    @app.get("/api/tools")
    async def tools():
        return agent.tools.list()

    @app.post("/api/tools/mcp")
    async def register_mcp(payload: dict[str, Any]):
        tool = await agent.tools.register_mcp(
            str(payload["name"]),
            str(payload.get("description", "MCP 工具")),
            str(payload["endpoint"]),
            list(payload.get("parameters", [])),
        )
        return tool.model_dump()

    @app.get("/api/snapshots")
    async def snapshots():
        return agent.snapshots

    @app.get("/api/status")
    async def status():
        return {
            "status": "ok",
            "infrastructure": infra.status,
            "llm_model": cfg.llm.model,
            "embedding_model": cfg.embedding.model,
            "is_mock": not agent.llm.is_real,
            "rag_loaded": agent.rag.loaded,
            "tool_count": len(agent.tools.tools),
        }

    frontend = Path("frontend")
    if frontend.exists():
        app.mount("/assets", StaticFiles(directory=frontend), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def frontend_fallback(path: str):
            candidate = frontend / path
            return FileResponse(candidate if candidate.is_file() else frontend / "index.html")

    @app.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
