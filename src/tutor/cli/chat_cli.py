from __future__ import annotations

import argparse

from ..pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="408-AI-Tutor 命令行对话（RAG + DeepSeek）"
    )
    parser.add_argument(
        "--rag-config",
        type=str,
        default="config/rag.yaml",
        help="RAG 配置文件路径（默认：config/rag.yaml）",
    )
    parser.add_argument(
        "--index-config",
        type=str,
        default="config/index.yaml",
        help="索引配置文件路径（默认：config/index.yaml）",
    )

    args = parser.parse_args()

    # 初始化 RAG Pipeline
    try:
        pipeline = RAGPipeline.from_configs(
            rag_cfg_path=args.rag_config,
            index_cfg_path=args.index_config,
        )
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print("请先构建索引，例如：")
        print("  python -m tutor.cli.build_index")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 初始化 RAGPipeline 失败：{e}")
        return

    print("====================================")
    print("  408-AI-Tutor 命令行对话已启动 ✅")
    print("  输入 408 相关问题，我会尽量结合资料回答。")
    print("  输入 'exit' / 'quit' / 'q' / '退出' 结束对话。")
    print("====================================")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAI：好的，下次再一起刷 408 ~")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"} or question in {"退出", "拜拜", "再见"}:
            print("AI：好的，下次再一起刷 408 ~")
            break

        try:
            answer = pipeline.answer(question)
        except Exception as e:  # noqa: BLE001
            print(f"[错误] 生成回答失败：{e}")
            continue

        print("\nAI：")
        print(answer.text.strip())

        # 简单打印一下使用到的 chunk 来源
        if answer.used_chunks:
            print("\n[参考片段来源]")
            for i, ch in enumerate(answer.used_chunks, 1):
                fname = ch.metadata.get("file_name", ch.document_id)
                print(f"- [{i}] {fname}")


if __name__ == "__main__":
    main()
