"""LLM 消息兼容性测试。"""

from agi_assistant.llm import LLMClient


def test_tool_observation_is_normalized_for_deepseek() -> None:
    """无 tool_call_id 的内部观察不能直接使用 tool 角色。"""
    messages = LLMClient._normalize_messages(
        "系统提示",
        [{"role": "tool", "content": "北京天气工具未配置实时数据源"}],
    )

    assert messages[1] == {
        "role": "user",
        "content": "工具观察：北京天气工具未配置实时数据源",
    }
