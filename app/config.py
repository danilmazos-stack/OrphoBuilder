import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen2.5:3b")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./textfixer.db")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))