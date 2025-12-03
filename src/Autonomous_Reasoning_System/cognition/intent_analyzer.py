import json
import re
from Autonomous_Reasoning_System.llm.engine import call_llm
from Autonomous_Reasoning_System.prompts import INTENT_ANALYZER_PROMPT
from Autonomous_Reasoning_System.utils.json_utils import parse_llm_json

class IntentAnalyzer:
    """
    Analyzes a text input to classify its intent and extract key entities.
    Returns structured JSON data that other modules can consume.
    """

    def __init__(self):
        self.system_prompt = INTENT_ANALYZER_PROMPT

    def analyze(self, text: str) -> dict:
        """Return structured intent and entities parsed from the LLM output."""
        raw = call_llm(system_prompt=self.system_prompt, user_prompt=text)

        # Use unified JSON parser
        result = parse_llm_json(raw)

        if result and isinstance(result, dict):
             # Ensure defaults
            result.setdefault("intent", "unknown")
            result.setdefault("family", "unknown")
            result.setdefault("subtype", "unknown")
            result.setdefault("entities", {})
            result.setdefault("reason", "(no reason provided)")
            return result
        else:
             # Fallback
            if "birthday" in text.lower():
                return {
                    "intent": "memory_store",
                    "family": "personal_facts",
                    "subtype": "birthday",
                    "entities": {},
                    "reason": "Fallback: detected birthday keyword",
                }
            else:
                return {
                    "intent": "unknown",
                    "family": "unknown",
                    "subtype": "unknown",
                    "entities": {},
                    "reason": "Fallback: invalid LLM output",
                }
