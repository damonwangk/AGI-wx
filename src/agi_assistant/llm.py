"""OpenAI 兼容的对话与向量客户端，失败时提供确定性降级。"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import AppConfig

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=60)

    @property
    def is_real(self) -> bool:
        return bool(self.config.llm.api_key and "你的" not in self.config.llm.api_key)

    async def close(self) -> None:
        await self.client.aclose()

    async def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        """调用兼容 Chat Completions 的接口；故障时返回可识别的降级回答。"""
        if not self.is_real:
            return self._mock(messages)
        payload = {
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "messages": self._normalize_messages(system, messages),
        }
        try:
            response = await self.client.post(
                self.config.llm.api_url,
                headers={"Authorization": f"Bearer {self.config.llm.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - 外部 API 必须统一降级
            logger.warning("LLM 调用失败，切换到 Mock：%s", exc)
            return self._mock(messages)

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """逐 token 输出；不支持流式或请求失败时按文本片段降级。"""
        if not self.is_real:
            for token in self._mock(messages).split(" "):
                yield token + " "
            return
        payload = {
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "stream": True,
            "messages": self._normalize_messages(system, messages),
        }
        try:
            async with self.client.stream(
                "POST",
                self.config.llm.api_url,
                headers={"Authorization": f"Bearer {self.config.llm.api_key}"},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        yield content
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 流式调用失败：%s", exc)
            yield self._mock(messages)

    async def embed(self, text: str) -> list[float]:
        """生成文本向量；无真实服务时使用稳定哈希向量支持本地召回。"""
        cfg = self.config.embedding
        if cfg.api_key and "你的" not in cfg.api_key and cfg.api_url:
            try:
                response = await self.client.post(
                    cfg.api_url,
                    headers={"Authorization": f"Bearer {cfg.api_key}"},
                    json={"model": cfg.model, "input": text},
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding 调用失败，使用本地向量：%s", exc)
        vector = [0.0] * 256
        for index, char in enumerate(text):
            vector[(ord(char) + index * 31) % len(vector)] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    async def extract_preferences(self, text: str) -> dict[str, str]:
        """使用规则即时提取常见偏好；保持无 LLM 时也可工作。"""
        rules = [("姓名", "我叫"), ("喜欢", "我喜欢"), ("语言", "请用")]
        for key, marker in rules:
            if marker in text:
                value = text.split(marker, 1)[1].strip(" ，。,.！!")
                if value:
                    return {key: value[:100]}
        return {}

    @staticmethod
    def _mock(messages: list[dict[str, str]]) -> str:
        latest = messages[-1]["content"] if messages else ""
        return f"[Mock 模式] 已收到：{latest}"

    @staticmethod
    def _normalize_messages(system: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """将内部工具观察转换为 DeepSeek 可接受的普通消息。"""
        normalized = [{"role": "system", "content": system}]
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "tool":
                # 这里没有对应的 tool_call_id，因此不能直接发送 OpenAI tool 角色。
                normalized.append({"role": "user", "content": f"工具观察：{content}"})
            else:
                normalized.append({"role": role, "content": content})
        return normalized


def parse_json_payload(text: str, fallback: Any) -> Any:
    """解析 LLM 常见的 Markdown JSON 输出。"""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return fallback
