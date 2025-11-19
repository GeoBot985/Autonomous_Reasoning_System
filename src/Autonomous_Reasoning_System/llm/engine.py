import subprocess
import json
from ..infrastructure import config
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_MODEL = "deepseek-r1:14b"

class LLMEngine:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.DEFAULT_MODEL

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
    Wraps the LLM call. Now injects today's date dynamically so
    the model can calculate ages and interpret time properly.
    """
    # ✅ Get current date in local timezone
        # Date injection removed — personal facts override everything
    merged_system = system_prompt

    # ✅ Example call to local LLM via Ollama
    full_input = f"SYSTEM:\n{merged_system}\n\nUSER:\n{user_prompt}"

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma3:1b"],
            input=full_input.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )

        raw_out = result.stdout.decode("utf-8", errors="replace").strip()
        raw_err = result.stderr.decode("utf-8", errors="replace").strip()

        print("───────── LLM DEBUG ─────────")
        print("STDOUT:\n", raw_out[:500])
        print("STDERR:\n", raw_err[:300])
        print("──────────────────────────────")

        if not raw_out:
            return f"(no stdout) stderr={raw_err}"
        return raw_out

    except Exception as e:
        return f"LLM call failed: {e}"