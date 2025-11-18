from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ChatMessage:
    """
    对话消息：兼容 DeepSeek / OpenAI 风格。
    role: "system" / "user" / "assistant"
    """
    role: str
    content: str


@dataclass
class LLMResponse:
    """
    模型返回结果的封装。
    """
    text: str
    raw: Dict[str, Any] | None = None


class LLMClient:
    """
    大模型客户端的抽象基类。
    所有具体模型（DeepSeek、本地 LLM）都应该实现 generate 方法。
    """

    def generate(self, messages: List[ChatMessage]) -> LLMResponse:
        raise NotImplementedError("LLMClient.generate() must be implemented by subclasses.")
