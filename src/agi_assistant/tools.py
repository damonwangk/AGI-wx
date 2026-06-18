"""内置工具、文档工具与 MCP HTTP 工具注册表。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .documents import DocumentLibrary
from .rag import RAGEngine
from .sandbox import Sandbox

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


class Tool(BaseModel):
    name: str
    description: str
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    handler: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, params: dict[str, Any]) -> str:
        return await self.handler(params)


class ToolRegistry:
    def __init__(
        self,
        rag: RAGEngine,
        sandbox: Sandbox,
        documents: DocumentLibrary,
        search_config: dict[str, str],
    ) -> None:
        self.tools: dict[str, Tool] = {}
        self.rag, self.sandbox, self.documents = rag, sandbox, documents
        self.search_config = search_config
        self._register_defaults()

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list(self) -> list[dict[str, Any]]:
        return [tool.model_dump() for tool in self.tools.values()]

    async def register_mcp(self, name: str, description: str, endpoint: str, parameters: list[dict[str, Any]]) -> Tool:
        """注册远程 MCP 风格 HTTP 工具，调用时发送 JSON-RPC tools/call。"""

        async def invoke(params: dict[str, Any]) -> str:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": params},
                    },
                )
                response.raise_for_status()
                return str(response.json().get("result", response.json()))

        tool = Tool(name=name, description=description, parameters=parameters, handler=invoke)
        self.register(tool)
        return tool

    def _register_defaults(self) -> None:
        async def current_time(_: dict[str, Any]) -> str:
            return datetime.now().astimezone().isoformat(timespec="seconds")

        async def weather(params: dict[str, Any]) -> str:
            city = str(params.get("city", "当前城市"))
            return f"{city}天气工具未配置实时数据源，请使用 search_web 查询。"

        async def search_web(params: dict[str, Any]) -> str:
            query = str(params.get("query", ""))
            api_key = self.search_config.get("api_key", "")
            if not api_key or "填入" in api_key or "你的" in api_key:
                return f"搜索服务未配置，查询词：{query}"
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": 5},
                )
                response.raise_for_status()
                return str(response.json().get("results", []))

        async def rag_search(params: dict[str, Any]) -> str:
            answer, _ = await self.rag.query(str(params.get("query", "")))
            return answer

        async def exec_command(params: dict[str, Any]) -> str:
            result = await self.sandbox.execute(str(params.get("command", "")))
            return result.stdout or result.stderr

        async def write_document(params: dict[str, Any]) -> str:
            record = self.documents.write(str(params.get("name", "untitled.md")), str(params.get("content", "")))
            return record.model_dump_json()

        async def list_documents(_: dict[str, Any]) -> str:
            return str([record.model_dump(mode="json", exclude={"content"}) for record in self.documents.list()])

        definitions = [
            ("current_time", "获取当前时间", [], current_time),
            (
                "weather",
                "查询城市天气",
                [{"name": "city", "type": "string", "required": True}],
                weather,
            ),
            (
                "search_web",
                "搜索互联网",
                [{"name": "query", "type": "string", "required": True}],
                search_web,
            ),
            (
                "rag_search",
                "检索个人知识库",
                [{"name": "query", "type": "string", "required": True}],
                rag_search,
            ),
            (
                "exec_command",
                "在安全沙箱执行命令",
                [{"name": "command", "type": "string", "required": True}],
                exec_command,
            ),
            ("write_document", "写入版本化文档", [], write_document),
            ("list_documents", "列出文档库", [], list_documents),
        ]
        for name, description, parameters, handler in definitions:
            self.register(Tool(name=name, description=description, parameters=parameters, handler=handler))
