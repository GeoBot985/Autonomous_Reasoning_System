"""
Global configuration for MyAssistant.
Loads values from environment variables or uses defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base data directory (relative to repo unless overridden)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))

# LLM provider placeholder (e.g., "ollama", "lmstudio", "openai", "groq", etc.)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Default model name for local inference
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3:1b")

# Ollama Base URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Memory storage
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", str(DATA_DIR / "memory.duckdb"))
MEMORY_PARQUET_PATH = os.getenv("MEMORY_PARQUET_PATH", str(DATA_DIR / "memory.parquet"))

# WhatsApp Configuration
WA_USER_DATA_DIR = os.getenv("WA_USER_DATA_DIR", None)
WA_SELF_CHAT_URL = os.getenv("WA_SELF_CHAT_URL", "https://web.whatsapp.com")
WA_SELF_NAME = os.getenv("WA_SELF_NAME", "User")
WA_POLL_INTERVAL = int(os.getenv("WA_POLL_INTERVAL", "2"))
