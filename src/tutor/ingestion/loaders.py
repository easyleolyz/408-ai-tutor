from __future__ import annotations

from pathlib import Path
from typing import List

from ..config import load_index_config, resolve_path
from ..schemas import Document, IndexConfig


def _discover_files(cfg: IndexConfig) -> List[Path]:
    """
    根据 IndexConfig 中的 data_dir 和 file_patterns 找到所有原始文件。
    目前默认只在 data_dir 的第一层目录下用 glob 搜索，
    如果以后需要递归子目录，可以把 glob 改成 rglob。
    """
    root = resolve_path(cfg.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    files: List[Path] = []
    for pattern in cfg.file_patterns:
        files.extend(root.glob(pattern))

    # 只保留真正的文件
    files = [f for f in files if f.is_file()]

    return files


def load_documents_from_config(index_cfg: IndexConfig | None = None) -> List[Document]:
    """
    从配置中指定的 data_dir 读取所有文本文件，返回 Document 列表。
    """
    if index_cfg is None:
        index_cfg = load_index_config()

    files = _discover_files(index_cfg)
    documents: List[Document] = []

    root = resolve_path(index_cfg.data_dir)

    for path in files:
        # 用相对路径作为文档 id，避免绝对路径太长
        try:
            doc_id = path.relative_to(root).as_posix()
        except ValueError:
            # 如果 relative_to 失败，就退回用名字
            doc_id = path.name

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 如果不是 utf-8，就尽量用 ignore 读
            text = path.read_text(encoding="utf-8", errors="ignore")

        metadata = {
            "source_path": str(path),
            "file_name": path.name,
            "subject": _infer_subject_from_name(path.name),
        }

        documents.append(Document(id=doc_id, text=text, metadata=metadata))

    return documents


def _infer_subject_from_name(file_name: str) -> str:
    """
    根据文件名简单猜一下是哪个科目（很粗糙的启发规则，后面可以改进）。
    """
    lower = file_name.lower()
    if "ds" in lower or "data_struct" in lower or "数据结构" in file_name:
        return "data_structures"
    if "os" in lower or "操作系统" in file_name:
        return "operating_systems"
    if "network" in lower or "计网" in file_name or "网络" in file_name:
        return "computer_networks"
    if "co" in lower or "组成原理" in file_name or "计组" in file_name:
        return "computer_organization"
    return "unknown"


# 方便在命令行里直接测试
if __name__ == "__main__":
    from pprint import pprint

    cfg = load_index_config()
    docs = load_documents_from_config(cfg)

    print(f"发现文档数量: {len(docs)}")
    for d in docs:
        print("-" * 40)
        print(f"Document ID: {d.id}")
        print(f"Metadata: {d.metadata}")
        print("Text preview:")
        print(d.text[:200].replace("\n", " ") + "...")
