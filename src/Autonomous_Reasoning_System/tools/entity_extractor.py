import json
import re
from Autonomous_Reasoning_System.llm.engine import call_llm

class EntityExtractor:
    """
    Extracts key search entities from a natural language query.
    Designed to support deterministic retrieval by identifying the core subjects.
    """

    def __init__(self):
        self.system_prompt = (
            "You are a query entity extractor. "
            "Your goal is to extract the core subject and keywords from a user's question for a search engine. "
            "Ignore question words (Who, What, Where, When, How) and auxiliary verbs. "
            "Focus on proper nouns, specific nouns, and key actions. "
            "Return a JSON list of strings. "
            "Example: 'When is Cornelia's birthday?' -> [\"Cornelia\", \"birthday\"] "
            "Example: 'Show me the project plan for Alpha' -> [\"project plan\", \"Alpha\"] "
            "Respond ONLY with the JSON list."
        )

    def extract(self, text: str) -> list[str]:
        """
        Extract keywords from the text.
        """
        try:
            raw = call_llm(system_prompt=self.system_prompt, user_prompt=text)
            # clean up potential markdown blocks
            raw = re.sub(r"```json|```", "", raw).strip()

            # Attempt to parse JSON
            keywords = json.loads(raw)

            if isinstance(keywords, list):
                return [str(k) for k in keywords]
            else:
                return []
        except Exception as e:
            print(f"[EntityExtractor] Extraction failed: {e}")
            # Fallback: simple split/filter if LLM fails (basic heuristic)
            words = text.split()
            filtered = [w for w in words if w.lower() not in ["who", "what", "where", "when", "how", "is", "the", "a", "an", "in", "on", "at", "for"]]
            return filtered
