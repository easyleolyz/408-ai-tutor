from tutor.pipeline.rag_pipeline import RAGPipeline

def main():
    pipeline = RAGPipeline.from_configs()

    print("RAG 测试模式（输入空行回车退出）")
    while True:
        try:
            q = input("\n问题 > ").strip()
        except EOFError:
            # 比如你按了 Ctrl+D，也优雅退出
            print("\n收到 EOF，退出。")
            break

        if not q:
            # 空行退出
            print("已退出。")
            break

        ans = pipeline.answer(q)

        print("\n=== 回答 ===")
        print(ans.text)

        print("\n=== 使用到的片段来源 ===")
        for i, ch in enumerate(ans.used_chunks, 1):
            print(f"- [{i}] {ch.metadata.get('file_name', ch.document_id)}")

if __name__ == "__main__":
    main()
