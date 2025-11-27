from rag_pipeline import RAGPipeline
from config import MAX_LEN
import gradio as gr

# 初始化 RAG
rag = RAGPipeline()

# 定義回答函數
def answer_fn(query):

    # 使用者輸入處理產出
    answer = rag.process_query(query)

    return answer

# 建立 Gradio 網頁介面
iface = gr.Interface(
    fn=answer_fn,
    inputs=gr.Textbox(lines=2, placeholder="輸入您的問題..."),
    outputs=gr.Textbox(lines=20),
    title="客服 RAG 系統",
    description="輸入問題即可得到客服回答",
)

# 啟動網頁
if __name__ == "__main__":
    iface.launch(share=True) 