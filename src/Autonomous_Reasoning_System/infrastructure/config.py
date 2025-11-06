"""
Global configuration for MyAssistant.
You can swap out the LLM provider later without changing other modules.
"""

# LLM provider placeholder (e.g., "ollama", "lmstudio", "openai", "groq", etc.)
LLM_PROVIDER = "ollama"

# Default model name for local inference (only used if provider supports it)
DEFAULT_MODEL = "gemma3:1b"

# Memory storage placeholders (wired later)
DUCKDB_PATH = "data/memory.db"
PARQUET_DIR = "data/parquet"

MEMORY_DB_PATH = "data/memory.db"
MEMORY_PARQUET_PATH = "data/memory.parquet"


# WhatsApp, RAG, or other services can go here later
