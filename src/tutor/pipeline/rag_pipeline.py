from __future__ import annotations

from typing import List

import numpy as np

from ..config import load_rag_config, load_index_config, load_model_config
from ..index.faiss_index import FaissIndex
from ..index.embeddings import embed_texts
from ..schemas import (
    Answer,
    Chunk,
    IndexConfig,
    ModelConfig,
    RAGConfig,
)
from ..llm.base import LLMClient
from ..llm.deepseek_client import DeepSeekClient
from ..llm.prompt_builder import PromptBuilder


class RAGPipeline:
    """
    最小可用 RAG Pipeline：
    - 从磁盘加载 FAISS 索引和 chunks
    - 用真实 embedding 检索 top_k chunks
    - 拼接成 context
    - 调用 DeepSeek 返回答案
    """

    def __init__(
        self,
        rag_cfg: RAGConfig,
        index_cfg: IndexConfig,
        llm_client: LLMClient,
        faiss_index: FaissIndex,
    ) -> None:
        self.rag_cfg = rag_cfg
        self.index_cfg = index_cfg
        self.llm_client = llm_client
        self.faiss_index = faiss_index
        self.prompt_builder = PromptBuilder(rag_cfg)

    # ===== 工厂方法：从默认配置创建 =====

    @classmethod
    def from_configs(
        cls,
        rag_cfg_path: str = "config/rag.yaml",
        index_cfg_path: str = "config/index.yaml",
    ) -> "RAGPipeline":
        rag_cfg = load_rag_config(rag_cfg_path)
        index_cfg = load_index_config(index_cfg_path)

        model_cfg = load_model_config(rag_cfg.model_config_path)

        # 目前只实现 DeepSeek API
        if model_cfg.backend == "api" and model_cfg.provider.lower() == "deepseek":
            llm_client: LLMClient = DeepSeekClient(model_cfg)
        else:
            raise NotImplementedError(
                f"Unsupported model backend/provider: backend={model_cfg.backend}, provider={model_cfg.provider}"
            )

        faiss_index = FaissIndex.load_from_disk(index_cfg)

        return cls(
            rag_cfg=rag_cfg,
            index_cfg=index_cfg,
            llm_client=llm_client,
            faiss_index=faiss_index,
        )

    # ===== 检索相关（使用真实 embedding） =====

    def _embed_query(self, question: str) -> np.ndarray:
        """
        使用与索引相同的 embedding 模型，对 query 做编码。
        如果以后你用的是 BGE 系列，可以考虑在这里加上 query 的指令前缀。
        """
        vecs = embed_texts([question], self.index_cfg)
        return vecs[0]

    def retrieve_chunks(self, question: str) -> List[Chunk]:
        """
        根据问题，检索 top_k 个 chunk。
        """
        query_vec = self._embed_query(question)
        results = self.faiss_index.search(query_vec, top_k=self.rag_cfg.top_k)
        chunks = [c for _, c in results]
        return chunks

    def _build_context_text(self, chunks: List[Chunk]) -> str:
        """
        将若干 chunk 拼接成 context 字符串，并限制总长度。
        同时加上简单的“来源”信息，方便模型回答时引用。
        """
        pieces: List[str] = []
        total_len = 0

        for i, ch in enumerate(chunks, 1):
            header = f"[{i}] 来源: {ch.metadata.get('file_name', ch.document_id)}\n"
            text = header + ch.text
            if total_len + len(text) > self.rag_cfg.max_context_chars:
                break
            pieces.append(text)
            total_len += len(text)

        return "\n\n".join(pieces)

    # ===== 对外主接口 =====

    def answer(self, question: str) -> Answer:
        """
        对外主入口：传入问题字符串，返回 Answer。
        """
        chunks = self.retrieve_chunks(question)
        context = self._build_context_text(chunks)

        messages = self.prompt_builder.build_messages(question, context)
        llm_resp = self.llm_client.generate(messages)

        return Answer(
            question=question,
            text=llm_resp.text,
            used_chunks=chunks,
            raw_model_output=llm_resp.raw,
        )


# 方便命令行快速测试
if __name__ == "__main__":
    pipeline = RAGPipeline.from_configs()
    q = "请用通俗的语言讲讲什么是栈，它有哪些典型应用？"
    ans = pipeline.answer(q)
    print("问题：", q)
    print("\nAI 回答：")
    print(ans.text)
    print("\n使用到的片段来源：")
    for i, ch in enumerate(ans.used_chunks, 1):
        print(f"- [{i}] {ch.metadata.get('file_name', ch.document_id)}")
