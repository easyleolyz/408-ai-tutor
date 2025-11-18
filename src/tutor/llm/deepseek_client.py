from __future__ import annotations

import os
from typing import List

import httpx

from ..schemas import ModelConfig
from .base import ChatMessage, LLMResponse, LLMClient


class DeepSeekClient(LLMClient):
    """
    DeepSeek Chat Completion 的简单封装。
    使用 /chat/completions 接口，非流式。
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        self.model_cfg = model_cfg

        # base_url & model_name
        self.base_url = (model_cfg.base_url or "https://api.deepseek.com").rstrip("/")
        # DeepSeek 官方文档里是 POST https://api.deepseek.com/chat/completions
        self.endpoint = self.base_url + "/chat/completions"

        # 模型名，例如 "deepseek-chat" 或其他模型
        self.model_name = model_cfg.model_name or "deepseek-chat"

        # 采样参数
        self.temperature = model_cfg.temperature
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout

        # 从环境变量中读取 API Key
        env_name = model_cfg.api_key_env or "DEEPSEEK_API_KEY"
        api_key = os.getenv(env_name)

        if not api_key:
            raise RuntimeError(
                f"DeepSeek API key not found in environment variable '{env_name}'. "
                f"Please set it, e.g. in PowerShell: `set {env_name}=YOUR_KEY`"
            )

        self.api_key = api_key

    def generate(self, messages: List[ChatMessage]) -> LLMResponse:
        """
        调用 DeepSeek /chat/completions 接口，返回 LLMResponse。
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            # 出错时返回一个简单的错误信息，方便调试
            return LLMResponse(
                text=f"[DeepSeek 调用失败] {e}",
                raw={"error": str(e)},
            )

        # 解析返回结果
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
        except Exception as e:  # noqa: BLE001
            return LLMResponse(
                text=f"[DeepSeek 返回格式异常] {e}",
                raw=data,
            )

        return LLMResponse(text=content, raw=data)


# 简单自测入口
if __name__ == "__main__":
    from ..config import load_model_config

    cfg = load_model_config("config/model.deepseek.yaml")
    client = DeepSeekClient(cfg)

    msgs = [
        ChatMessage(role="system", content="你是一个友好的中文助手。"),
        ChatMessage(role="user", content="简单介绍一下你自己。"),
    ]

    resp = client.generate(msgs)
    print("模型回答：")
    print(resp.text)
