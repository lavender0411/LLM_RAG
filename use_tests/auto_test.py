import csv
from rag_pipeline import RAGPipeline
from config import MAX_LEN

test_queries = [
    # ---- 1) FAQ 核心查詢（10）
    "我要修改聯絡電話怎麼做？",
    "會員資料更新在哪裡操作？",
    "如何申請帳號？",
    "密碼忘記了怎麼辦？",
    "如何查詢過去的訂單紀錄？",
    "要怎麼更改寄送地址？",
    "付款方式有哪些？",
    "我想取消訂單可以嗎？",
    "客服聯絡方式在哪裡？",
    "我要如何變更電子郵件？",

    # ---- 2) 模糊提問（5）
    "我資料想改一下，有什麼方式？",
    "可以幫我調整我的資訊嗎？",
    "哪裡能整理會員設定？",
    "我的帳戶內容想動一下，要去哪？",
    "怎樣調整個人內容？",

    # ---- 3) 無關問題（5）
    "你覺得明天會下雨嗎？",
    "AI 是不是會取代人類？",
    "給我一句勵志名言。",
    "你喜歡吃什麼？",
    "股票要怎麼買比較賺？",

    # ---- 4) 邊界測試（3）
    "",  # 空字串
    "改",  # 一字
    "我想要修改我的基本資料但我不太確定是哪裡出問題，我想請你幫我檢查一下這整段流程是不是有什麼問題並且告訴我要從哪個頁面開始操作，最好再順便提醒我可能會遺漏的地方",  # 超長

    # ---- 5) 錯字 / 語病（2）
    "我想秀改資料在哪裡？",
    "要改個人自料怎麼按？"
]


results = []
wrong = []

rag = RAGPipeline()
count = 0

for query in test_queries:
    try:
        print('start', count)
        count += 1
        if (0 < len(query) <= MAX_LEN):
            # 進 DB 調資料
            chunks, ids = rag.retrieve(query)
            # chunk 處理
            context = rag.process_context(query, chunks)
            # 生成答案
            answer = rag.answer(query, context)
            results.append({
            "count": count, 
            "query": query,
            "context": [f"Q{i}" for i in ids],
            "answer": answer})
        else:
            if not len(query):
                answer = f"您沒有輸入問題。"
            else:
                answer = f"您的問題過長，請限制在 {MAX_LEN} 字以內。"
            
            results.append({
                "count": count, 
                "query": query,
                "context": [],
                "answer": answer
        })
    except:
        wrong.append(count)


print(wrong)

# 輸出 CSV
with open("test_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["count","query","context","answer"])
    writer.writeheader()
    writer.writerows(results)



