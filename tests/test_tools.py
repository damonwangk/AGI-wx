"""内置工具测试。"""

import httpx
import pytest

from agi_assistant.config import AppConfig
from agi_assistant.documents import DocumentLibrary
from agi_assistant.infrastructure import Infrastructure
from agi_assistant.llm import LLMClient
from agi_assistant.rag import RAGEngine
from agi_assistant.sandbox import Sandbox
from agi_assistant.tools import ToolRegistry


@pytest.mark.asyncio
async def test_weather_returns_realtime_data(tmp_path, monkeypatch) -> None:
    """天气工具应查询城市坐标并返回实时天气。"""
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "北京"
            return httpx.Response(
                200,
                json={"results": [{"name": "北京市", "latitude": 39.9042, "longitude": 116.4074}]},
            )
        return httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-06-20T23:45",
                    "temperature_2m": 27.1,
                    "apparent_temperature": 28.0,
                    "relative_humidity_2m": 61,
                    "precipitation": 0.0,
                    "weather_code": 1,
                    "wind_speed_10m": 8.2,
                },
                "current_units": {
                    "temperature_2m": "°C",
                    "apparent_temperature": "°C",
                    "relative_humidity_2m": "%",
                    "precipitation": "mm",
                    "wind_speed_10m": "km/h",
                },
            },
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout")),
    )
    config = AppConfig()
    infra = Infrastructure(config)
    llm = LLMClient(config)
    registry = ToolRegistry(
        RAGEngine(config.rag, llm, infra),
        Sandbox(config.sandbox.model_copy(update={"backend": "mock"}), config.security),
        DocumentLibrary(tmp_path),
        {},
    )

    result = await registry.tools["weather"].execute({"city": "北京天气怎么样？"})

    assert "北京市当前天气" in result
    assert "27.1°C" in result
    assert "晴间多云" in result
    await llm.close()
