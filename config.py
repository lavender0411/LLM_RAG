import os
from dotenv import load_dotenv

# 讀取 .env
load_dotenv()

# ENUM
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"
DB_NAME = "faq"
DEDUP_THRES = 0.9
TOP_K = 5
MAX_LEN = 100

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PATH
FAQ_PATH = "./faq.json"
CHROMA_PATH = "./chroma_db"
