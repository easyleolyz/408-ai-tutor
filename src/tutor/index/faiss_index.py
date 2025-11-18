from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ..config import load_index_config, resolve_path
from ..schemas import Chunk, IndexConfig


class FaissIndex:
    """
    简单封装的 FAISS 索引：
    - build(...)      用一批 Chunk 构建索引（当前用随机向量占位）
    - save()          把索引和 chunks 元数据保存到磁盘
    - load_from_disk  从已有文件加载索引
    - search(...)     根据向量查询相似的 chunks（后面 RAG 会用）
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
        当前版本使用“随机向量”作为占位，保证流程通。
        后续会替换成真实的 embedding 模型。
        """
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        self.chunks = chunks

        # TODO: 替换为真实的文本向量化逻辑
        embeddings = self._fake_embed_chunks(chunks)

        # 使用最简单的 L2 距离索引
        if self.index_cfg.faiss_index_type == "Flat":
            index = faiss.IndexFlatL2(self.dim)
        else:
            # 目前只支持 Flat，其他类型留作扩展
            raise NotImplementedError(
                f"faiss_index_type={self.index_cfg.faiss_index_type} is not supported yet."
            )

        index.add(embeddings)
        self.index = index

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
        当前构建用的是 L2 距离：距离越小越相似。
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

    # ========== 内部：占位 embedding（随机向量） ==========

    def _fake_embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        临时使用随机向量来代替真实 embedding。
        这样可以先把索引构建和检索的流程跑通。

        之后你可以：
        - 写一个真正的 embedding 模型函数替换这里
        - 比如使用 HuggingFace / sentence-transformers / OpenAI embedding 等
        """
        n = len(chunks)
        # 为了每次运行结果不同，这里不固定随机种子
        embeddings = np.random.randn(n, self.dim).astype("float32")
        return embeddings


# 方便命令行快速测试
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

    # 随便拿第一个向量当 query 搜索，测试 search 功能
    q = loaded._fake_embed_chunks([loaded.chunks[0]])[0]
    res = loaded.search(q, top_k=3)
    print("\nSearch 结果预览:")
    for dist, ch in res:
        print("-" * 40)
        print("distance:", dist)
        print("chunk id:", ch.id)
        print("text preview:", ch.text[:80].replace("\n", " ") + "...")
