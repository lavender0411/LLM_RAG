from rag_pipeline import RAGPipeline
from config import MAX_LEN

def main():
    rag = RAGPipeline()

    while True:
        # 使用者輸入的查詢內容
        query = input("\n請輸入您的問題（按 q 離開）： \n")

        if query.lower() == "q":
            break

        # 使用者輸入處理產出
        answer = rag.process_query(query)

        print("\n--- 回覆 ---")
        print(answer)

if __name__ == "__main__":
    main()