# --------------------------------
# import
# --------------------------------
# 導入 config
from config import (
    EMBED_MODEL,
    LLM_MODEL,
    DB_NAME,
    DEDUP_THRES,
    TOP_K,
    OPENAI_API_KEY,
    CHROMA_PATH,
    MAX_LEN
)

# 導入 OpenAI
from openai import OpenAI
# 導入 chromadb 向量資料庫
import chromadb
# 導入 cosine 相似工具
from sklearn.metrics.pairwise import cosine_similarity
# 導入 torch
import torch


# --------------------------------
# 初始化
# --------------------------------
# 初始化 OpenAI
client = OpenAI(api_key = OPENAI_API_KEY)
# 初始化 ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# --------------------------------
# 建立 RAG 流程
# --------------------------------
class RAGPipeline:
    def __init__(self):
        self.client = client
        self.collection = chroma_client.get_collection(DB_NAME)
        
    # 向量轉換
    def embed_model(self, texts, model=EMBED_MODEL): 
        if isinstance(texts, str): 
            texts = [texts] 
        resp = self.client.embeddings.create( model=model, input=texts ) 
        vectors = [item.embedding for item in resp.data] 
        return vectors if len(vectors) <= 50 else torch.tensor(vectors)
    
    # 查詢 LLM
    def llm(self, prompt, model=LLM_MODEL):
        resp =  self.client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content
        
    # 檢索 DB
    def retrieve(self, query, top_k=TOP_K):
        query_vec = self.embed_model(query)[0]
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )
        # 僅用在測試 auto_test
        # ids = results["ids"][0]
        # return results["documents"][0], ids

        return results["documents"][0]

    # 語意去重複
    def semantic_dedup(self, chunks, threshold=DEDUP_THRES):
        if len(chunks) <= 1:
            return chunks

        embeddings = self.embed_model(chunks)
        sim_matrix = cosine_similarity(embeddings)

        kept = []
        removed = set()
        for i in range(len(chunks)):
            if i in removed:
                continue
            for j in range(i+1, len(chunks)):
                if sim_matrix[i, j] > threshold:
                    removed.add(j)
            kept.append(chunks[i])
        return kept

    # 依重要度重新排序 LLM
    def rerank(self, query, chunks, top_k=TOP_K):
        query_vec = self.embed_model(query)[0]
        chunk_vecs = self.embed_model(chunks)

        # 用 cosine similarity 重新排序
        sims = cosine_similarity([query_vec], chunk_vecs)[0]

        # 排序後取前 K
        ranked = [x for _, x in sorted(zip(sims, chunks), reverse=True)]
        return ranked[:top_k]

    # 語意合併 LLM
    def merge(self, chunks):
        prompt = f"""
以下段落有部分重複或相似，請合併成最小必要資訊。

段落：
{chunks}

只輸出合併後的內容，不需要額外說明。
"""
        out =  self.llm(prompt)
        return [out]
    
    # 處理 pipeline 流程
    def process_context(self, query, chunks):
        x = self.semantic_dedup(chunks)
        x = self.rerank(query, chunks)
        x =  self.merge(x)
        return x


    # 生成回答
    def answer(self, query, context):
        prompt = f"""
你是客服人員，請用客服語氣回應。
你只能依照資料來源回答問題，不可以使用外部知識。
如果資料不足，請回答「我是客服，無法回答不相關問題」。

使用者問題：
{query}

資料來源（Context）：
{context}

請根據資料來源回答。
"""
        return  self.llm(prompt)
    
    # 處理使用者輸入 query
    def process_query(self, query):
        # 判斷輸入長度
        if (0 < len(query) <= MAX_LEN):
            # 進 DB 調資料
            chunks, ids = self.retrieve(query)
            # chunk 處理
            context = self.process_context(query, chunks)
            # 生成答案
            answer = self.answer(query, context)
        elif not len(query): 
            answer = f"您沒有輸入問題。"
        else:
            answer = f"您的問題過長，請限制在 {MAX_LEN} 字以內。"
        return answer
