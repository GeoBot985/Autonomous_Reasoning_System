import requests
import json
import time
import logging
from ..infrastructure import config
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_MODEL = config.DEFAULT_MODEL or "gemma3:1b"
BASE_URL = config.OLLAMA_BASE_URL or "http://localhost:11434"

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or DEFAULT_MODEL

    def generate_response(self, prompt: str) -> str:
        """
        Dummy fallback. Replace later.
        """
        return call_llm("You are a helpful assistant.", prompt)

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
def call_llm(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    """
    Wraps the LLM call using HTTP requests to Ollama.
    Includes retry logic and timeouts for reliability.
    """
    merged_system = system_prompt
    full_prompt = f"SYSTEM:\n{merged_system}\n\nUSER:\n{user_prompt}"

    url = f"{BASE_URL}/api/generate"
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False
    }

    attempt = 0
    backoff = 1

    while attempt < retries:
        try:
            attempt += 1
            response = requests.post(url, json=payload, timeout=30) # 30s timeout
            response.raise_for_status()

            data = response.json()
            raw_out = data.get("response", "").strip()

            if not raw_out:
                logger.warning(f"[LLM] Empty response from model (attempt {attempt}).")
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return f"(no response content)"

            return raw_out

        except requests.exceptions.ConnectTimeout:
            logger.warning(f"[LLM] Timeout connecting to Ollama (attempt {attempt}).")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[LLM] Could not connect to Ollama at {url}. Is it running? (attempt {attempt})")
        except requests.exceptions.ReadTimeout:
             logger.warning(f"[LLM] Read timeout waiting for response (attempt {attempt}).")
        except json.JSONDecodeError:
            logger.warning(f"[LLM] Failed to decode JSON response (attempt {attempt}).")
        except Exception as e:
            logger.error(f"[LLM] Unexpected error: {e}")

        if attempt < retries:
            time.sleep(backoff)
            backoff *= 2

    return "Error: AI service unavailable after retries."
