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
            '{"intent": "<one-word-intent>", "family": "<family>", "subtype": "<subtype>", "entities": {"entity1": "value", ...}, "reason": "<short reason>"}\n'
            "Do not include any text outside this JSON. "
            "Possible intents include: remind, reflect, summarize, recall, open, plan, query, greet, exit, memory_store, web_search.\n\n"
            "CRITICAL RULES:\n"
            "1. If the user asks to search google, find something online, or asks a question about current events or external facts (e.g., 'When is the next game?'), classify as 'web_search'.\n"
            "2. If the user mentions a birthday (e.g., 'X's birthday is Y', 'Remember that Z was born on...'), you MUST classify it as:\n"
            '   "intent": "memory_store", "family": "personal_facts", "subtype": "birthday"\n'
            "2. NEVER classify a birthday statement as a 'goal' or 'plan'.\n"
            "3. Extract the person's name and the date as entities if present."
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
            result.setdefault("family", "unknown")
            result.setdefault("subtype", "unknown")
            result.setdefault("entities", {})
            result.setdefault("reason", "(no reason provided)")

        except Exception:
            # Fallback heuristic for birthdays if LLM fails
            if "birthday" in text.lower():
                result = {
                    "intent": "memory_store",
                    "family": "personal_facts",
                    "subtype": "birthday",
                    "entities": {},
                    "reason": "Fallback: detected birthday keyword",
                }
            else:
                result = {
                    "intent": "unknown",
                    "family": "unknown",
                    "subtype": "unknown",
                    "entities": {},
                    "reason": "Fallback: invalid LLM output",
                }

        return result
