import requests
import json
import time
import logging
from ..infrastructure import config
from ..infrastructure.observability import Metrics
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_MODEL = config.DEFAULT_MODEL or "gemma3:1b"
BASE_URL = config.OLLAMA_BASE_URL or "http://localhost:11434"

logger = logging.getLogger(__name__)

# Circuit Breaker State
_CIRCUIT_OPEN = False
_CIRCUIT_OPEN_UNTIL = 0
_CONSECUTIVE_FAILURES = 0
_FAILURE_THRESHOLD = 3
_RESET_TIMEOUT = 30

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
def call_llm(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
    """
    Wraps the LLM call using HTTP requests to Ollama.
    Includes retry logic, timeouts, and circuit breaker for reliability.
    """
    global _CIRCUIT_OPEN, _CIRCUIT_OPEN_UNTIL, _CONSECUTIVE_FAILURES

    # Check Circuit Breaker
    if _CIRCUIT_OPEN:
        if time.time() < _CIRCUIT_OPEN_UNTIL:
            logger.warning("[LLM] Circuit breaker open. Skipping call.")
            return "Error: AI service temporarily suspended due to repeated failures."
        else:
            logger.info("[LLM] Circuit breaker reset timeout expired. Retrying...")
            _CIRCUIT_OPEN = False
            _CONSECUTIVE_FAILURES = 0

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
    # Retries is number of retries, so total attempts = retries + 1
    total_attempts = retries + 1

    while attempt < total_attempts:
        start_time = time.time()
        try:
            attempt += 1
            # Timeout reduced to 15s as per requirement
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()

            latency = time.time() - start_time
            Metrics().record_time("llm_latency", latency)

            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.warning(f"[LLM] Failed to decode JSON response (attempt {attempt}).")
                raise

            raw_out = data.get("response", "").strip()

            if not raw_out:
                logger.warning(f"[LLM] Empty response from model (attempt {attempt}).")
                if attempt < total_attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # Treat empty response as failure for circuit breaker?
                # Usually empty response is better than crash, but if persistent...
                # Let's treat it as success but empty content to avoid breaking circuit on logic issues.
                _CONSECUTIVE_FAILURES = 0
                return f"(no response content)"

            # Success
            _CONSECUTIVE_FAILURES = 0
            return raw_out

        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
            logger.warning(f"[LLM] Timeout waiting for response (attempt {attempt}).")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[LLM] Could not connect to Ollama at {url}. Is it running? (attempt {attempt})")
        except Exception as e:
            logger.error(f"[LLM] Unexpected error: {e}")

        if attempt < total_attempts:
            time.sleep(backoff)
            backoff *= 2

    # If we reach here, all attempts failed
    _CONSECUTIVE_FAILURES += 1
    logger.error(f"[LLM] Call failed after {total_attempts} attempts. Consecutive failures: {_CONSECUTIVE_FAILURES}")

    if _CONSECUTIVE_FAILURES >= _FAILURE_THRESHOLD:
        _CIRCUIT_OPEN = True
        _CIRCUIT_OPEN_UNTIL = time.time() + _RESET_TIMEOUT
        logger.critical(f"[LLM] Circuit breaker ACTIVATED. Pausing LLM calls for {_RESET_TIMEOUT}s.")

    return "Error: AI service unavailable after retries."
