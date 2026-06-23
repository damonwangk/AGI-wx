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


def _clean_city_query(value: str) -> str:
    """从自然语言天气问题中提取城市名称。"""
    city = value.strip().rstrip("？?。！!")
    for suffix in ("天气怎么样", "天气如何", "的天气", "天气", "气温怎么样", "气温如何", "气温"):
        if city.endswith(suffix):
            city = city[: -len(suffix)].strip()
            break
    return city or "北京"


def _weather_description(code: int) -> str:
    """将 Open-Meteo 的 WMO 天气代码转换为中文描述。"""
    descriptions = {
        0: "晴",
        1: "晴间多云",
        2: "多云",
        3: "阴",
        45: "雾",
        48: "雾凇",
        51: "小毛毛雨",
        53: "中等毛毛雨",
        55: "强毛毛雨",
        56: "轻度冻毛毛雨",
        57: "强冻毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        66: "轻度冻雨",
        67: "强冻雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        77: "米雪",
        80: "小阵雨",
        81: "中阵雨",
        82: "强阵雨",
        85: "小阵雪",
        86: "大阵雪",
        95: "雷暴",
        96: "雷暴伴轻度冰雹",
        99: "雷暴伴强冰雹",
    }
    return descriptions.get(code, f"未知天气代码 {code}")


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
            city = _clean_city_query(str(params.get("city", "北京")))
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    location_response = await client.get(
                        "https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1, "language": "zh", "format": "json"},
                    )
                    location_response.raise_for_status()
                    locations = location_response.json().get("results", [])
                    if not locations:
                        return f"未找到城市“{city}”，请检查城市名称。"
                    location = locations[0]
                    weather_response = await client.get(
                        "https://api.open-meteo.com/v1/forecast",
                        params={
                            "latitude": location["latitude"],
                            "longitude": location["longitude"],
                            "current": (
                                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                                "precipitation,weather_code,wind_speed_10m"
                            ),
                            "timezone": "auto",
                        },
                    )
                    weather_response.raise_for_status()
                payload = weather_response.json()
                current = payload["current"]
                units = payload.get("current_units", {})
                description = _weather_description(int(current["weather_code"]))
                return (
                    f"{location['name']}当前天气（{current['time']}）：{description}，"
                    f"温度 {current['temperature_2m']}{units.get('temperature_2m', '°C')}，"
                    f"体感 {current['apparent_temperature']}{units.get('apparent_temperature', '°C')}，"
                    f"湿度 {current['relative_humidity_2m']}{units.get('relative_humidity_2m', '%')}，"
                    f"降水 {current['precipitation']}{units.get('precipitation', 'mm')}，"
                    f"风速 {current['wind_speed_10m']}{units.get('wind_speed_10m', 'km/h')}。"
                    "数据来源：Open-Meteo。"
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError):
                return "实时天气服务暂时不可用，请稍后重试。"

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
