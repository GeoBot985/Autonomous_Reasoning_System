# Autonomous_Reasoning_System/memory/llm_summarizer.py
from Autonomous_Reasoning_System.llm.engine import call_llm


def summarize_with_local_llm(text: str) -> str:
    """
    Summarize text using the central LLM engine (Docker-safe, no subprocess).
    """
    if not text or not text.strip():
        return "(no content)"

    system_prompt = "You are a summarizer. Summarize the following text concisely."
    try:
        return call_llm(system_prompt=system_prompt, user_prompt=text) or "(summary pending — no response from model)"
    except Exception as e:
        return f"(summary pending — error: {e})"
