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
LLM_MODEL = os.getenv("ARS_LLM_MODEL", "granite4:1b")

# --- Timeouts ---
TAGS_TIMEOUT = 5
WARMUP_TIMEOUT = 15
REQUEST_TIMEOUT = 90
LLM_RETRIES = 2
LLM_TIMEOUT = 15  # For individual calls, maybe longer if needed? brain.py used 90s REQUEST_TIMEOUT.
                  # llm/engine.py used 15s. 15s is very short for 3b model.
                  # Let's standardize to something reasonable like 60s for gen.

# --- Embeddings (FastEmbed) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Official, 384-dim, ONNX auto-download, ~500 MB but fast on CPU
VECTOR_DIMENSION = 384
