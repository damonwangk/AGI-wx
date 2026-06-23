"""HTTP 契约冒烟测试。"""

from fastapi.testclient import TestClient

from agi_assistant.api import create_app
from agi_assistant.config import AppConfig


def test_status_tools_and_chat() -> None:
    app = create_app(AppConfig(), connect_infrastructure=False)
    with TestClient(app) as client:
        assert client.get("/api/status").status_code == 200
        assert client.get("/api/tools").json()
        response = client.post("/api/chat", json={"message": "你好"})
        assert response.status_code == 200
        assert response.json()["mode"] == "chat"


def test_upload_and_rag() -> None:
    app = create_app(AppConfig(), connect_infrastructure=False)
    with TestClient(app) as client:
        response = client.post("/api/upload", files={"file": ("note.txt", "Python 是编程语言", "text/plain")})
        assert response.status_code == 200
        assert response.json()["indexed_count"] >= 1


def test_stream_finishes_when_agent_raises(monkeypatch) -> None:
    """后台任务异常时 SSE 必须返回错误并关闭，不能让前端永久等待。"""
    app = create_app(AppConfig(), connect_infrastructure=False)

    async def fail_process(*_args, **_kwargs):
        raise ValueError("测试异常")

    monkeypatch.setattr(app.state.agent, "process", fail_process)
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"message": "北京天气怎么样？"})

    assert response.status_code == 200
    assert "event: error" in response.text
