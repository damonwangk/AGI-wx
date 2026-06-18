"""配置兼容性测试。"""

from agi_assistant.config import AppConfig, load_config


def test_default_config() -> None:
    config = AppConfig()
    assert config.server_port == 8090
    assert config.rag.rewrite.enabled is True


def test_load_existing_yaml() -> None:
    config = load_config("config/config.yaml")
    assert config.postgres.port == 5432
    assert config.rag.top_k == 3
