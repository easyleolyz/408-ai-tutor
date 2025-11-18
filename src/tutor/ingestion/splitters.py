from __future__ import annotations

from typing import List

from ..schemas import Document, Chunk, IndexConfig
from ..config import load_index_config
from .loaders import load_documents_from_config


def simple_char_window_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    最简单的按字符滑动窗口切分：
    - 每个 chunk 最多 chunk_size 个字符
    - 相邻 chunk 之间有 chunk_overlap 个字符的重叠
    先跑通流程，后面可以换成更智能的按句子 / 段落切分。
    """
    chunks: List[str] = []
    n = len(text)

    if n == 0:
        return chunks

    # 防止 overlap >= chunk_size 导致死循环
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        # 下一段从 end - overlap 开始
        if end == n:
            break
        start = end - chunk_overlap

    return chunks


def split_documents_to_chunks(
    documents: List[Document],
    index_cfg: IndexConfig | None = None,
) -> List[Chunk]:
    """
    将一批 Document 切分成若干 Chunk。
    """
    if index_cfg is None:
        index_cfg = load_index_config()

    all_chunks: List[Chunk] = []

    for doc in documents:
        raw_chunks = simple_char_window_split(
            doc.text,
            chunk_size=index_cfg.chunk_size,
            chunk_overlap=index_cfg.chunk_overlap,
        )

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc.id}::chunk_{i}"
            chunk_metadata = {
                **doc.metadata,     # 继承文档元数据
                "chunk_index": i,
            }
            all_chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=doc.id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    return all_chunks


# 方便直接在命令行测试整个 ingestion 流程
if __name__ == "__main__":
    from pprint import pprint

    cfg = load_index_config()
    docs = load_documents_from_config(cfg)
    chunks = split_documents_to_chunks(docs, cfg)

    print(f"文档数量: {len(docs)}")
    print(f"总 chunk 数量: {len(chunks)}")

    # 打印前几个 chunk 看看效果
    for c in chunks[:3]:
        print("-" * 40)
        print(f"Chunk ID: {c.id}")
        print(f"From Document: {c.document_id}")
        print(f"Metadata: {c.metadata}")
        print("Text preview:")
        print(c.text[:200].replace("\n", " ") + "...")
