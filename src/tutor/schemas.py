from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ========== 基础数据结构：文档 & 切分块 ==========

@dataclass
class Document:
    """
    原始文档，代表一份教材 / 笔记 / 真题文件中的整体内容。
    """
    id: str                     # 文档唯一 ID（可以是文件名）
    text: str                   # 文本内容（未切分）
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    文本块（chunk），用于向量化和检索的最小单位。
    比如从一个 md 文件中切出一个小节的文字。
    """
    id: str                     # chunk 唯一 ID
    document_id: str            # 来自哪个 Document
    text: str                   # chunk 文本内容
    metadata: Dict[str, Any] = field(default_factory=dict)


# ========== RAG 输出结果结构 ==========

@dataclass
class Answer:
    """
    RAG Pipeline 最终返回给用户的结果。
    """
    question: str                               # 用户问题
    text: str                                   # 大模型生成的回答
    used_chunks: List[Chunk] = field(default_factory=list)  # 用到的上下文片段
    raw_model_output: Optional[Dict[str, Any]] = None       # 保留原始模型返回（可选）


# ========== 配置相关的数据结构 ==========

@dataclass
class ModelConfig:
    """
    模型配置：既可以表示 DeepSeek 等 API 型，也可以表示本地模型。
    """
    backend: str                # "api" 或 "local"
    provider: str               # "deepseek" / "huggingface" / "llama.cpp" 等

    # API 型模型常用字段
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None

    # 本地模型常用字段
    model_name_or_path: Optional[str] = None
    device: str = "cpu"         # "cpu" / "cuda"
    dtype: str = "float16"      # float16 / bfloat16 / float32

    # 通用生成参数
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 30

    # 预留扩展字段（配置里出现但没单独列出的）
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexConfig:
    """
    向量索引 & 文本切分相关配置。
    """
    index_dir: str              # 索引存放目录，如 "data/index"
    data_dir: str               # 原始数据目录，如 "data/raw"
    file_patterns: List[str]    # 需要扫描的文件模式，如 ["*.md", "*.txt"]

    embedding_dim: int = 768
    faiss_index_type: str = "Flat"

    chunk_size: int = 400
    chunk_overlap: int = 50


@dataclass
class RAGConfig:
    """
    RAG 流程相关配置。
    """
    model_config_path: str      # 模型配置文件路径，如 "config/model.deepseek.yaml"

    top_k: int = 5
    max_context_chars: int = 2000

    system_prompt: str = ""
    user_prompt_template: str = ""
