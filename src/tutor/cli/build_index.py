from __future__ import annotations

import argparse

from ..config import load_index_config
from ..ingestion.loaders import load_documents_from_config
from ..ingestion.splitters import split_documents_to_chunks
from ..index.faiss_index import FaissIndex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 408-AI-Tutor 的向量索引（目前使用随机向量占位）。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/index.yaml",
        help="索引配置文件路径，默认 config/index.yaml",
    )

    args = parser.parse_args()

    # 1. 读取索引配置
    index_cfg = load_index_config(args.config)
    print(f"[build_index] 使用配置文件: {args.config}")
    print(f"[build_index] data_dir = {index_cfg.data_dir}")
    print(f"[build_index] index_dir = {index_cfg.index_dir}")

    # 2. 加载文档
    documents = load_documents_from_config(index_cfg)
    print(f"[build_index] 发现文档数量: {len(documents)}")
    if not documents:
        print("[build_index] 未找到任何文档，请检查 data/raw 目录和 file_patterns 配置。")
        return

    # 3. 切分为 chunks
    chunks = split_documents_to_chunks(documents, index_cfg)
    print(f"[build_index] 切分得到 chunk 数量: {len(chunks)}")

    # 4. 构建 FAISS 索引（当前使用随机向量作为占位）
    index = FaissIndex(index_cfg)
    index.build(chunks)
    index.save()

    print("[build_index] 索引构建完成 ✅")


if __name__ == "__main__":
    main()
