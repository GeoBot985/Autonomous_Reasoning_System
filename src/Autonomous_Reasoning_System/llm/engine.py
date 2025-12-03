import requests
import json
import time
import logging
from typing import Optional

# Fix imports to use config properly
from .. import config
from ..infrastructure.observability import Metrics

logger = logging.getLogger(__name__)

# Circuit Breaker State
_CIRCUIT_OPEN = False
_CIRCUIT_OPEN_UNTIL = 0
_CONSECUTIVE_FAILURES = 0
_FAILURE_THRESHOLD = 3
_RESET_TIMEOUT = 30

class LLMEngine:
    """
    Unified LLM Engine.
    Handles connection to Ollama, retries, circuit breaking, and session management.
    """
    def __init__(self, model: str = None, api_base: str = None):
        self.model = model or config.LLM_MODEL
        self.api_base = (api_base or config.OLLAMA_API_BASE).rstrip('/')
        self.generate_url = f"{self.api_base}/generate"
        self.tags_url = f"{self.api_base}/tags"

        # Use a persistent session for better connection handling
        self.session = requests.Session()
        self.session.trust_env = False  # avoid proxy/env interference for local Ollama

        self._check_model_exists()
        self._warmup()

    def _check_model_exists(self) -> None:
        logger.info(f"Checking if model '{self.model}' exists locally...")
        try:
            resp = self.session.get(self.tags_url, timeout=config.TAGS_TIMEOUT)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if any(self.model in m for m in models):
                    logger.info(f"Model '{self.model}' found.")
                else:
                    logger.warning(f"Model '{self.model}' not found in Ollama!")
        except Exception as e:
            logger.warning(f"Could not list models: {e}")

    def _warmup(self) -> None:
        logger.info("Warming up LLM...")
        try:
            self.session.post(
                self.generate_url,
                json={"model": self.model, "prompt": "hi", "stream": False, "keep_alive": "5m"},
                timeout=config.WARMUP_TIMEOUT
            )
            logger.info("LLM Warmed up.")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def ping(self, timeout: int = 2) -> bool:
        """Quick health check against the tags endpoint."""
        try:
            self.session.get(self.tags_url, timeout=timeout).raise_for_status()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, format: str = None) -> str:
        """
        Generates a response from the LLM.
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

        full_prompt = prompt
        if system:
            full_prompt = f"SYSTEM: {system}\n\nUSER: {prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature
        }
        if format:
            payload["format"] = format

        logger.debug(f"Sending request to Ollama ({len(full_prompt)} chars)...")

        attempt = 0
        backoff = 1
        total_attempts = config.LLM_RETRIES + 1

        while attempt < total_attempts:
            start_time = time.time()
            try:
                attempt += 1
                resp = self.session.post(self.generate_url, json=payload, timeout=config.REQUEST_TIMEOUT)
                resp.raise_for_status()

                latency = time.time() - start_time
                Metrics().record_time("llm_latency", latency)

                data = resp.json()
                raw_out = data.get("response", "").strip()

                if not raw_out:
                    logger.warning(f"[LLM] Empty response from model (attempt {attempt}).")
                    if attempt < total_attempts:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    # Reset failures on successful request even if empty?
                    # Prefer to treat empty as non-critical failure here but return something.
                    _CONSECUTIVE_FAILURES = 0
                    return ""

                # Success
                _CONSECUTIVE_FAILURES = 0
                return raw_out

            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
                logger.warning(f"[LLM] Timeout waiting for response (attempt {attempt}).")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[LLM] Could not connect to Ollama. Is it running? (attempt {attempt})")
            except Exception as e:
                logger.error(f"[LLM] Unexpected error: {e}")
                # Try to log detailed error info if response exists
                if 'resp' in locals() and hasattr(resp, 'status_code'):
                    logger.error(f"Ollama status={resp.status_code}, body={resp.text[:500]}")

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

        return f"[Error: LLM unavailable]"

    # Backwards compatibility methods if needed
    def generate_response(self, prompt: str) -> str:
        return self.generate(prompt)

    def classify_memory(self, text: str):
        # This was in the old engine.py, keeping for compatibility if used elsewhere
        lower = text.lower()
        if "remind" in lower or "meeting" in lower or "schedule" in lower:
            return {"type": "task", "importance": 0.5}
        if "squash" in lower or "appointment" in lower or "call" in lower:
            return {"type": "event", "importance": 0.5}
        return {"type": "misc", "importance": 0.5}

    def embed_text(self, text: str):
        return None

# Global instance for module-level usage (legacy support)
_default_engine = None

def get_default_engine():
    global _default_engine
    if _default_engine is None:
        _default_engine = LLMEngine()
    return _default_engine

def call_llm(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
    """
    Wrapper for backward compatibility.
    """
    engine = get_default_engine()
    # Ignoring retries arg as it's handled by config/engine internally now,
    # or we could pass it if we refactored generate to take it.
    return engine.generate(user_prompt, system=system_prompt)
