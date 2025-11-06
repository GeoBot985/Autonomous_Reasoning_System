import json
import re
from Autonomous_Reasoning_System.llm.engine import call_llm


class IntentAnalyzer:
    """
    Analyzes a text input to classify its intent and extract key entities.
    Returns structured JSON data that other modules can consume.
    """

    def __init__(self):
        self.system_prompt = (
            "You are Tyrone's Intent Analyzer. "
            "Your task is to classify the user's intent and extract any key entities. "
            "Always respond ONLY with valid JSON of the form:\n"
            '{"intent": "<one-word-intent>", "entities": {"entity1": "value", ...}, "reason": "<short reason>"}\n'
            "Do not include any text outside this JSON. "
            "Possible intents include: remind, reflect, summarize, recall, open, plan, query, greet, exit."
        )

    def analyze(self, text: str) -> dict:
        """Return structured intent and entities parsed from the LLM output."""
        raw = call_llm(system_prompt=self.system_prompt, user_prompt=text)

        # Try to extract JSON even if the model adds explanations
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = json.loads(raw)

            result.setdefault("intent", "unknown")
            result.setdefault("entities", {})
            result.setdefault("reason", "(no reason provided)")

        except Exception:
            result = {
                "intent": "unknown",
                "entities": {},
                "reason": "Fallback: invalid LLM output",
            }

        return result
