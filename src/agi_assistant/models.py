"""跨模块共享的数据模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChatOptions(BaseModel):
    use_rag: bool = False
    selected_tools: list[str] = Field(default_factory=list)
    explicit: bool = False


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False
    selected_tools: list[str] = Field(default_factory=list)
    explicit: bool = False


class StepType(StrEnum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"


class ReactStep(BaseModel):
    type: StepType
    content: str
    tool: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class TaskNode(BaseModel):
    id: str
    name: str
    tool: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    race_group: str = ""
    status: str = "pending"
    result: str = ""
    error: str = ""


class TaskState(BaseModel):
    id: str
    query: str
    nodes: list[TaskNode] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    interrupted: bool = False


class ChatResponse(BaseModel):
    query: str
    answer: str = ""
    mode: str = "chat"
    extracted_info: str = ""
    tool_call: dict[str, Any] | None = None
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[ReactStep] = Field(default_factory=list)
    task: TaskState | None = None
    interrupted: bool = False
    short_term_count: int = 0
    long_term_count: int = 0
    preferences: dict[str, str] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    type: str
    data: Any


class DocumentRecord(BaseModel):
    id: str
    name: str
    content: str
    content_type: str = "text/plain"
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ingested: bool = False
