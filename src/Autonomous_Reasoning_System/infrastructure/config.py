"""
Global configuration for MyAssistant.
Loads values from environment variables or uses defaults.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM provider placeholder (e.g., "ollama", "lmstudio", "openai", "groq", etc.)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Default model name for local inference
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3:1b")

# Ollama Base URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Memory storage
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memory.duckdb")
MEMORY_PARQUET_PATH = os.getenv("MEMORY_PARQUET_PATH", "data/memory.parquet")

# WhatsApp Configuration
WA_USER_DATA_DIR = os.getenv("WA_USER_DATA_DIR", r"C:\Users\GeorgeC\AppData\Local\Google\Chrome\User Data\Profile 2")
WA_SELF_CHAT_URL = os.getenv("WA_SELF_CHAT_URL", "https://web.whatsapp.com/send/?phone=27796995695")
WA_SELF_NAME = os.getenv("WA_SELF_NAME", "GeorgeC")
WA_POLL_INTERVAL = int(os.getenv("WA_POLL_INTERVAL", "2"))
