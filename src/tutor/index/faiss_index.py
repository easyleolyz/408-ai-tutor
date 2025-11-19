from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ..config import load_index_config, resolve_path
from ..schemas import Chunk, IndexConfig
from .embeddings import embed_texts


class FaissIndex:
    """
    基于 FAISS 的向量索引封装：
    - build(...)      用一批 Chunk 构建索引（现在使用真实 embedding）
    - save()          把索引和 chunks 元数据保存到磁盘
    - load_from_disk  从已有文件加载索引
    - search(...)     根据向量查询相似的 chunks（RAG 会用到）
    """

    def __init__(self, index_cfg: IndexConfig) -> None:
        self.index_cfg = index_cfg
        self.dim = index_cfg.embedding_dim
        self.index: faiss.Index = None  # type: ignore
        self.chunks: List[Chunk] = []

    # ========== 构建与保存 ==========

    def build(self, chunks: List[Chunk]) -> None:
        """
        根据给定的 chunks 构建 FAISS 索引。
        使用 BGE-M3 等真实 embedding 模型。
        """
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        self.chunks = chunks

        embeddings = self._embed_chunks(chunks)

        # 校验维度
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: got {embeddings.shape[1]}, "
                f"but index_cfg.embedding_dim = {self.dim}"
            )

        # 使用最简单的 L2 距离索引
        if self.index_cfg.faiss_index_type == "Flat":
            index = faiss.IndexFlatL2(self.dim)
        else:
            raise NotImplementedError(
                f"faiss_index_type={self.index_cfg.faiss_index_type} is not supported yet."
            )

        index.add(embeddings)
        self.index = index

    def _embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        对所有 chunk.text 做 embedding。
        """
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts, self.index_cfg)
        return embeddings

    def save(self) -> None:
        """
        将 FAISS 索引和 chunks 元数据保存到 index_dir 目录。
        """
        if self.index is None:
            raise ValueError("Index has not been built yet. Call build() first.")

        index_dir = resolve_path(self.index_cfg.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "chunks.jsonl"

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(index_path))

        # 保存 chunks 元数据（jsonl，每行一个 chunk）
        with meta_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                data = {
                    "id": c.id,
                    "document_id": c.document_id,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"[FaissIndex] 索引已保存到: {index_path}")
        print(f"[FaissIndex] Chunk 元数据已保存到: {meta_path}")

    # ========== 加载与检索 ==========

    @classmethod
    def load_from_disk(cls, index_cfg: IndexConfig | None = None) -> "FaissIndex":
        """
        从磁盘加载已有索引和 chunks 元数据。
        """
        if index_cfg is None:
            index_cfg = load_index_config()

        index_dir = resolve_path(index_cfg.index_dir)
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "chunks.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Chunk metadata file not found: {meta_path}")

        obj = cls(index_cfg)
        obj.index = faiss.read_index(str(index_path))
        obj.chunks = []

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                chunk = Chunk(
                    id=data["id"],
                    document_id=data["document_id"],
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                )
                obj.chunks.append(chunk)

        print(f"[FaissIndex] 已从 {index_path} 和 {meta_path} 加载索引。")
        return obj

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Chunk]]:
        """
        使用给定的 query 向量检索最相似的 top_k 个 chunk。
        返回 (distance, Chunk) 列表。
        当前使用 L2 距离：向量经过归一化后，L2 与余弦距离等价。
        """
        if self.index is None:
            raise ValueError("Index is not loaded or built.")

        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]

        D, I = self.index.search(query_vector.astype("float32"), top_k)
        results: List[Tuple[float, Chunk]] = []

        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((float(dist), self.chunks[idx]))

        return results


# 方便命令行测试
if __name__ == "__main__":
    from ..ingestion.loaders import load_documents_from_config
    from ..ingestion.splitters import split_documents_to_chunks
    from ..config import load_index_config
    from pprint import pprint

    cfg = load_index_config()
    docs = load_documents_from_config(cfg)
    chunks = split_documents_to_chunks(docs, cfg)

    print(f"文档数量: {len(docs)}")
    print(f"chunk 数量: {len(chunks)}")

    index = FaissIndex(cfg)
    index.build(chunks)
    index.save()

    # 再从磁盘加载一次测试
    loaded = FaissIndex.load_from_disk(cfg)
    print(f"加载后的 chunk 数量: {len(loaded.chunks)}")

    # 用第一个 chunk 的文本做一个 query 测试
    from .embeddings import embed_texts

    q_vec = embed_texts([loaded.chunks[0].text], cfg)[0]
    res = loaded.search(q_vec, top_k=3)
    print("\nSearch 结果预览:")
    for dist, ch in res:
        print("-" * 40)
        print("distance:", dist)
        print("chunk id:", ch.id)
        print("text preview:", ch.text[:80].replace("\n", " ") + "...")
