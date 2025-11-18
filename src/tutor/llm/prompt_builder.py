from __future__ import annotations

from typing import List

from ..schemas import RAGConfig
from .base import ChatMessage


class PromptBuilder:
    """
    负责把 RAGConfig 中的 system_prompt 和 user_prompt_template
    填上 question 与 context，生成发送给 LLM 的 messages 列表。
    """

    def __init__(self, rag_cfg: RAGConfig) -> None:
        self.rag_cfg = rag_cfg

    def build_messages(self, question: str, context: str) -> List[ChatMessage]:
        """
        根据模板构造 system + user 消息。
        """
        messages: List[ChatMessage] = []

        system_prompt = (self.rag_cfg.system_prompt or "").strip()
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        user_prompt = (self.rag_cfg.user_prompt_template or "").format(
            question=question,
            context=context,
        )

        messages.append(ChatMessage(role="user", content=user_prompt.strip()))

        return messages
