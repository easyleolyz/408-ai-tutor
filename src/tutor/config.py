from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .schemas import ModelConfig, IndexConfig, RAGConfig


# 项目根目录：.../408-ai-tutor
# 当前文件在 src/tutor/config.py，所以往上两级就是项目根目录
BASE_DIR: Path = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """
    返回项目根目录路径，方便其他模块使用。
    """
    return BASE_DIR


def resolve_path(path: Union[str, Path]) -> Path:
    """
    将相对路径转成以项目根目录为基准的绝对路径。
    例如 "config/index.yaml" -> "D:/dev/408-ai-tutor/config/index.yaml"
    """
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取 YAML 文件并返回字典。
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping (object).")

    return data


def load_model_config(path: Union[str, Path]) -> ModelConfig:
    """
    加载模型配置（既可以是 DeepSeek API，也可以是本地模型配置）。
    """
    data = load_yaml(path)

    # 从 YAML 字典构造 ModelConfig
    cfg = ModelConfig(
        backend=data.get("backend", "api"),
        provider=data.get("provider", "deepseek"),
        api_key_env=data.get("api_key_env"),
        base_url=data.get("base_url"),
        model_name=data.get("model_name"),
        model_name_or_path=data.get("model_name_or_path"),
        device=data.get("device", "cpu"),
        dtype=data.get("dtype", "float16"),
        temperature=float(data.get("temperature", 0.2)),
        max_tokens=int(data.get("max_tokens", 1024)),
        timeout=int(data.get("timeout", 30)),
        extra={},  # 先置空，下面填充
    )

    # 把没列出的字段放到 extra 里，便于以后扩展
    known_keys = {
        "backend",
        "provider",
        "api_key_env",
        "base_url",
        "model_name",
        "model_name_or_path",
        "device",
        "dtype",
        "temperature",
        "max_tokens",
        "timeout",
    }
    for k, v in data.items():
        if k not in known_keys:
            cfg.extra[k] = v

    return cfg


def load_index_config(path: Union[str, Path] = "config/index.yaml") -> IndexConfig:
    """
    加载索引构建相关配置。
    """
    data = load_yaml(path)

    return IndexConfig(
        index_dir=data["index_dir"],
        data_dir=data["data_dir"],
        file_patterns=data.get("file_patterns", ["*.md", "*.txt"]),

        embedding_dim=int(data.get("embedding_dim", 1024)),
        faiss_index_type=data.get("faiss_index_type", "Flat"),

        chunk_size=int(data.get("chunk_size", 400)),
        chunk_overlap=int(data.get("chunk_overlap", 50)),

        embedding_model=data.get("embedding_model", "BAAI/bge-m3"),
        embedding_device=data.get("embedding_device", "auto"),
        embedding_batch_size=int(data.get("embedding_batch_size", 16)),
        normalize_embeddings=bool(data.get("normalize_embeddings", True)),
    )


def load_rag_config(path: Union[str, Path] = "config/rag.yaml") -> RAGConfig:
    """
    加载 RAG 流程相关配置。
    """
    data = load_yaml(path)

    return RAGConfig(
        model_config_path=data["model_config_path"],
        top_k=int(data.get("top_k", 5)),
        max_context_chars=int(data.get("max_context_chars", 2000)),
        system_prompt=data.get("system_prompt", ""),
        user_prompt_template=data.get("user_prompt_template", ""),
    )


# 方便你在命令行测试这个模块
if __name__ == "__main__":
    from pprint import pprint

    print("Project root:", project_root())

    print("\nIndex config:")
    pprint(load_index_config())

    print("\nRAG config:")
    rag_cfg = load_rag_config()
    pprint(rag_cfg)

    print("\nModel config (from rag.model_config_path):")
    model_cfg = load_model_config(rag_cfg.model_config_path)
    pprint(model_cfg)
