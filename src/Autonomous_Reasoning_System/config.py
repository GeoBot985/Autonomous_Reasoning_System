import os
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Database ---
MEMORY_DB_PATH = os.getenv("ARS_MEMORY_DB_PATH", str(DATA_DIR / "memory.duckdb"))

# --- Ollama (LLM) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_BASE = f"{OLLAMA_HOST}/api"
#LLM_MODEL = os.getenv("ARS_LLM_MODEL", "gemma3:1b")
LLM_MODEL = os.getenv("ARS_LLM_MODEL", "granite4:3b")


# --- Embeddings (FastEmbed) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Official, 384-dim, ONNX auto-download, ~500 MB but fast on CPU
VECTOR_DIMENSION = 384
