from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from ..schemas import IndexConfig

# 全局缓存，避免每次都重新加载模型
_tokenizer_cache: Optional[AutoTokenizer] = None
_model_cache: Optional[AutoModel] = None
_model_name_cache: Optional[str] = None
_device_cache: Optional[str] = None


def _resolve_device(cfg: IndexConfig) -> str:
    """
    根据配置决定使用的设备：
    - "auto": 有 CUDA 就用 "cuda"，否则 "cpu"
    - 其他：直接返回（如 "cpu" / "cuda")
    """
    if cfg.embedding_device == "auto":
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return cfg.embedding_device


def get_embedding_model(cfg: IndexConfig) -> tuple[AutoTokenizer, AutoModel]:
    """
    加载 HuggingFace 模型和分词器，并做缓存。
    """
    global _tokenizer_cache, _model_cache, _model_name_cache, _device_cache

    model_name = cfg.embedding_model
    device = _resolve_device(cfg)

    if (
        _model_cache is not None
        and _tokenizer_cache is not None
        and _model_name_cache == model_name
        and _device_cache == device
    ):
        return _tokenizer_cache, _model_cache

    print(f"[Embeddings] Loading HF model: {model_name} on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    _tokenizer_cache = tokenizer
    _model_cache = model
    _model_name_cache = model_name
    _device_cache = device

    return tokenizer, model


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    标准 mean pooling：对非 padding token 的 hidden state 取平均。
    last_hidden_state: [batch, seq_len, hidden]
    attention_mask:    [batch, seq_len]
    """
    # 扩展维度，使得可以进行逐元素相乘
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch, seq_len, 1]
    masked = last_hidden_state * mask  # 把 padding 部分置 0

    sum_embeddings = masked.sum(dim=1)  # [batch, hidden]
    lengths = mask.sum(dim=1).clamp(min=1)  # [batch, 1]
    return sum_embeddings / lengths


def embed_texts(texts: List[str], cfg: IndexConfig) -> np.ndarray:
    """
    使用 HuggingFace Transformers 对一批文本做编码，返回 numpy.float32 数组。
    默认会对向量做归一化（cfg.normalize_embeddings）。
    """
    if not texts:
        return np.zeros((0, cfg.embedding_dim), dtype="float32")

    tokenizer, model = get_embedding_model(cfg)
    device = next(model.parameters()).device

    all_embeddings: List[np.ndarray] = []
    batch_size = cfg.embedding_batch_size

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,  # 对于我们 400 字左右的 chunk 足够
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
            pooled = _mean_pooling(last_hidden, encoded["attention_mask"])  # [batch, hidden]

            if cfg.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu().numpy().astype("float32"))

    embeddings = np.vstack(all_embeddings)

    # 防御性检查：如果 dim 不一致，提醒用户配对 embedding_dim
    if embeddings.shape[1] != cfg.embedding_dim:
        print(
            f"[Embeddings][警告] 模型输出维度为 {embeddings.shape[1]}，"
            f"但 index.yaml 中配置 embedding_dim={cfg.embedding_dim}。"
            " 建议把 embedding_dim 改成实际输出维度。"
        )

    return embeddings
