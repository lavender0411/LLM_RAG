
from config import (DB_NAME, FAQ_PATH, CHROMA_PATH)
import chromadb
import json
from rag_pipeline import RAGPipeline

data = json.load(open(FAQ_PATH, "r"))
chroma_client = chromadb.PersistentClient(CHROMA_PATH)

def build_db():
    print("正在建立資料庫")

    # 建立 ChromaDB
    if DB_NAME not in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.create_collection(
            name=DB_NAME,
            metadata={"hnsw:space": "cosine"} 
        )

    collection = chroma_client.get_collection(DB_NAME)
    
    rag = RAGPipeline()

    # 原始資料加入 Chroma DB
    for i, item in enumerate(data):
        # 先檢查 ID 是否已存在
        existing = collection.get(ids=[str(i)])
        if len(existing["ids"]) > 0:
            continue

        qa = f"Q:{item['question']}\nA:{item['answer']}"
        qa_embedding = rag.embed_model(qa)[0]

        # 加入 DB
        collection.add(
            ids=[str(i)],
            documents=[qa],
            embeddings=[qa_embedding],
            metadatas=[{
                "question": item["question"],
                "answer": item["answer"]
            }]
        )
    print("Chroma DB 已建立完成！")



if __name__ == "__main__":
    build_db()
