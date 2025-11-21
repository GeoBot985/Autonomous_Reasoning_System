import requests
import json
from ..infrastructure import config
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_MODEL = config.DEFAULT_MODEL or "gemma3:1b"
BASE_URL = config.OLLAMA_BASE_URL or "http://localhost:11434"

class LLMEngine:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or DEFAULT_MODEL

    def generate_response(self, prompt: str) -> str:
        """
        Dummy fallback. Replace later.
        """
        return f"[DUMMY RESPONSE from {self.provider} - model={self.model}] You said: {prompt}"

    def classify_memory(self, text: str):
        lower = text.lower()
        if "remind" in lower or "meeting" in lower or "schedule" in lower:
            return {"type": "task", "importance": 0.5}
        if "squash" in lower or "appointment" in lower or "call" in lower:
            return {"type": "event", "importance": 0.5}
        return {"type": "misc", "importance": 0.5}

    def embed_text(self, text: str):
        return None


# ✅ MODULE-LEVEL function (not inside class!)
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Wraps the LLM call using HTTP requests to Ollama.
    """
    merged_system = system_prompt
    full_prompt = f"SYSTEM:\n{merged_system}\n\nUSER:\n{user_prompt}"

    url = f"{BASE_URL}/api/generate"
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        raw_out = data.get("response", "").strip()

        if not raw_out:
            return f"(no response content)"
        return raw_out

    except requests.exceptions.ConnectionError:
        print(f"[LLM ERROR] Could not connect to Ollama at {url}. Is it running?")
        return "Error: AI service unavailable."
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return f"LLM call failed: {e}"
