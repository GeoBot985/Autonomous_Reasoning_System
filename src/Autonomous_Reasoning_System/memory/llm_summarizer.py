# Autonomous_Reasoning_System/memory/llm_summarizer.py
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo


def summarize_with_local_llm(text: str) -> str:
    """
    Summarize a block of text using the locally configured LLM.
    Reads provider and model from global config each time it's called.
    Currently supports Ollama; can be extended for others.
    """
    if not text or not text.strip():
        return "(no content to summarize)"

    # --- CONFIG IMPORT (live reload each call) ---
    try:
        from Autonomous_Reasoning_System.infrastructure import config
        provider = getattr(config, "LLM_PROVIDER", "ollama")
        model = getattr(config, "DEFAULT_MODEL", "gemma3:4b")
        print(f"[DEBUG] Using provider={provider}, model={model}")
    except Exception as e:
        print(f"[WARN] Could not import infrastructure.config ({e})")
        provider, model = "ollama", "gemma3:4b"
    # ------------------------------------------------

    today = datetime.now(ZoneInfo("Africa/Johannesburg")).strftime("%d %B %Y")

    system_prompt = (
        f"You are Tyrone's reflective reasoning module. "
        f"Today's date is {today}. "
        "Summarize the following episode clearly, focusing on actions, insights, and tone. "
        "Keep it under 120 words and write naturally in first person."
    )

    full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{text}"

    if provider.lower() == "ollama":
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=full_prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
            )
            output = result.stdout.decode("utf-8", errors="replace").strip()
            if not output:
                return "(summary pending — no response from model)"
            return output
        except subprocess.TimeoutExpired:
            return "(summary pending — model timed out)"
        except FileNotFoundError:
            return "(summary pending — Ollama not found or not running)"
        except Exception as e:
            return f"(summary pending — error: {e})"

    # Future providers
    return f"(provider {provider} not yet supported)"
