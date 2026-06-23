"""配置兼容性测试。"""

from pathlib import Path

from agi_assistant.config import AppConfig, load_config


def test_default_config() -> None:
    config = AppConfig()
    assert config.server_port == 8090
    assert config.rag.rewrite.enabled is True


def test_load_existing_yaml() -> None:
    config = load_config("config/config.yaml")
    assert config.postgres.port == 5432
    assert config.rag.top_k == 3


def test_load_deepseek_key_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    """验证本地 .env 能启用真实 DeepSeek 配置。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        'llm:\n  api_key: "${DEEPSEEK_API_KEY}"\n  model: deepseek-chat\n',
        encoding="utf-8",
    )

    config = load_config("config.yaml")

    assert config.llm.api_key == "test-key"
    assert config.llm.model == "deepseek-chat"
