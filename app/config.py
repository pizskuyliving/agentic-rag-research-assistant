"""Semua pengaturan dibaca dari .env di satu tempat."""
import os

from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

MODEL_RINGAN = os.getenv("MODEL_RINGAN", "groq/openai/gpt-oss-20b")
MODEL_BERAT = os.getenv("MODEL_BERAT", "groq/openai/gpt-oss-120b")
# Untuk model reasoning (gpt-oss): low = hemat token & cepat; medium/high = lebih teliti
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

# Model untuk node PENILAIAN (grade & evaluasi): "ringan" atau "berat".
# Tes menunjukkan model ringan sering salah menilai relevansi, jadi default "berat".
PERAN_GRADE = os.getenv("PERAN_GRADE", "berat")
PERAN_EVALUASI = os.getenv("PERAN_EVALUASI", "berat")
MODEL_CADANGAN_GEMINI = os.getenv("MODEL_CADANGAN_GEMINI", "gemini/gemini-3.5-flash-lite")
MODEL_CADANGAN_MISTRAL = os.getenv("MODEL_CADANGAN_MISTRAL", "mistral/mistral-small-latest")

# Embedding & Qdrant
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jurnal")

# Chunking & retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))        # dalam karakter
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
