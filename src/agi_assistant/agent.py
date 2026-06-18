"""统一 Agent：路由、上下文装配、四种模式分发与记忆回写。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .config import AppConfig
from .documents import DocumentLibrary
from .infrastructure import Infrastructure
from .llm import LLMClient
from .memory import MemoryStack
from .models import ChatOptions, ChatResponse, StreamEvent
from .rag import RAGEngine
from .runtime import GraphRuntime
from .sandbox import Sandbox
from .tools import ToolRegistry

EventCallback = Callable[[StreamEvent], Awaitable[None]]


class UnifiedAgent:
    def __init__(self, config: AppConfig, infra: Infrastructure) -> None:
        self.config, self.infra = config, infra
        self.llm = LLMClient(config)
        self.memory = MemoryStack(config.memory)
        self.documents = DocumentLibrary()
        self.rag = RAGEngine(config.rag, self.llm, infra)
        self.sandbox = Sandbox(config.sandbox, config.security)
        self.tools = ToolRegistry(self.rag, self.sandbox, self.documents, config.search)
        self.runtime = GraphRuntime(config, self.llm, self.tools, infra)
        self.snapshots: list[dict[str, Any]] = []
        self._tasks: set[asyncio.Task[Any]] = set()

    async def close(self) -> None:
        await self.llm.close()

    async def bootstrap(self) -> None:
        """连接完成后恢复持久化状态。"""
        state = await self.infra.load_state()
        for role, content in state["chat"]:
            self.memory.stm.add(str(role), str(content))
        self.memory.preferences.save_batch({str(key): str(value) for key, value in state["preferences"]})
        for _id, content, importance, embedding, category in state["memories"]:
            values = embedding if isinstance(embedding, list) else []
            self.memory.ltm.store(str(content), float(importance), values, str(category))
        self.rag.restore(state["chunks"])

    def cancel(self) -> None:
        """取消所有当前正在处理的请求。"""
        for task in tuple(self._tasks):
            task.cancel()

    async def process(self, query: str, options: ChatOptions, callback: EventCallback | None = None) -> ChatResponse:
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        response = ChatResponse(query=query)
        try:
            await self._prepare(query, options, response)
            if callback:
                await callback(StreamEvent(type="route", data={"mode": response.mode}))
                if response.extracted_info:
                    await callback(StreamEvent(type="memory", data={"extracted_info": response.extracted_info}))
            await self._dispatch(query, options, response, callback)
            await self._finalize(query, response)
        except asyncio.CancelledError:
            response.interrupted = True
            response.answer = "[已中断] 用户取消了当前任务"
        finally:
            if task:
                self._tasks.discard(task)
        response.short_term_count = len(self.memory.stm.messages)
        response.long_term_count = len(self.memory.ltm.items)
        response.preferences = dict(self.memory.preferences.values)
        if callback:
            await callback(StreamEvent(type="done", data=response.model_dump(mode="json")))
        return response

    async def _prepare(self, query: str, options: ChatOptions, response: ChatResponse) -> None:
        self.memory.stm.add("user", query)
        await self.infra.save_chat("user", query)
        preferences = await self.llm.extract_preferences(query)
        if preferences:
            self.memory.preferences.save_batch(preferences)
            response.extracted_info = "；".join(f"已记住：{key} = {value}" for key, value in preferences.items())
            await asyncio.gather(
                *(self.infra.save_preference("default", key, value) for key, value in preferences.items())
            )
        response.mode = self._route(query, options)

    def _route(self, query: str, options: ChatOptions) -> str:
        if options.explicit:
            if options.selected_tools:
                return "react" if len(options.selected_tools) > 1 or self._is_complex(query) else "tool"
            return "rag" if options.use_rag and self.rag.loaded else "chat"
        if self._is_complex(query):
            return "react"
        if any(word in query for word in ("时间", "天气", "搜索", "执行命令", "写文档")):
            return "tool"
        if self.rag.loaded and any(word in query for word in ("知识库", "文档", "资料", "根据上传")):
            return "rag"
        return "chat"

    @staticmethod
    def _is_complex(query: str) -> bool:
        return any(word in query for word in ("然后", "并且", "分别", "多步", "综合", "对比"))

    async def _context(self, query: str, mode: str) -> str:
        embedding = await self.llm.embed(query)
        recalled = self.memory.ltm.recall(query, embedding)
        constraints = "安全约束：不得执行破坏性命令，不得泄露密钥。"
        memory = self.memory.context(recalled)
        mode_hint = {
            "chat": "当前模式：普通对话。",
            "tool": "当前模式：单工具调用。",
            "rag": "当前模式：知识库问答。",
            "react": "当前模式：多步骤任务规划。",
        }[mode]
        return "\n".join(part for part in (constraints, mode_hint, memory) if part)

    async def _dispatch(
        self,
        query: str,
        options: ChatOptions,
        response: ChatResponse,
        callback: EventCallback | None,
    ) -> None:
        context = await self._context(query, response.mode)
        history = self.memory.stm.snapshot()
        if response.mode == "rag":
            response.answer, response.search_results = await self.rag.query(query, history)
            if callback:
                await callback(StreamEvent(type="rag_result", data={"search_results": response.search_results}))
                await callback(StreamEvent(type="token", data={"content": response.answer}))
            return
        selected = {
            name: tool
            for name, tool in self.tools.tools.items()
            if not options.selected_tools or name in options.selected_tools
        }
        if response.mode == "tool":
            tool = self._pick_tool(query, selected)
            params = self.runtime._params(tool.name, query)
            result = await tool.execute(params)
            response.tool_call = {"tool": tool.name, "params": params, "result": result}
            if callback:
                await callback(StreamEvent(type="tool_call", data=response.tool_call))
            response.answer = await self.llm.chat(context, [*history, {"role": "tool", "content": result}])
            return
        if response.mode == "react":
            task = await self.runtime.plan(query, selected)
            response.task = task
            if callback:
                await callback(StreamEvent(type="graph_ready", data=task.model_dump(mode="json")))
            observations, response.steps = await self.runtime.execute(task, callback)
            response.answer = await self.llm.chat(
                context + "\n请根据工具观察合成最终答案。",
                [
                    {"role": "user", "content": query},
                    {"role": "tool", "content": "\n".join(observations)},
                ],
            )
            self.snapshots.append(task.model_dump(mode="json"))
            return
        response.answer = ""
        if callback:
            async for token in self.llm.stream(context, history):
                response.answer += token
                await callback(StreamEvent(type="token", data={"content": token}))
        else:
            response.answer = await self.llm.chat(context, history)

    def _pick_tool(self, query: str, selected: dict[str, Any]):
        rules = [
            ("current_time", ("时间", "几点", "日期")),
            ("weather", ("天气", "气温")),
            ("search_web", ("搜索", "联网", "最新")),
            ("rag_search", ("知识库", "文档", "资料")),
            ("exec_command", ("命令", "终端", "执行")),
            ("write_document", ("写文档", "报告", "保存")),
        ]
        for name, keywords in rules:
            if name in selected and any(keyword in query for keyword in keywords):
                return selected[name]
        return next(iter(selected.values()))

    async def _finalize(self, query: str, response: ChatResponse) -> None:
        self.memory.stm.add("assistant", response.answer)
        await self.infra.save_chat("assistant", response.answer)
        embedding = await self.llm.embed(response.answer)
        item = self.memory.ltm.store(response.answer[:1000], 0.5, embedding, "episodic")
        await self.infra.save_memory(item.content, item.importance, item.embedding, item.category)
        if len(self.memory.ltm.items) % self.config.memory.consolidation.trigger_interval == 0:
            self.memory.ltm.consolidate()
        await self.infra.publish("agent.chat", {"query": query, "mode": response.mode})
