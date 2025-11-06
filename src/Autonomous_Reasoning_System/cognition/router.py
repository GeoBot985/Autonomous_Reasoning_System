import json
import re
from Autonomous_Reasoning_System.llm.engine import call_llm
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface


class Router:
    def __init__(self):
        self.memory = MemoryInterface()

        # registry of modules Tyrone can use
        self.module_registry = [
            {"name": "IntentAnalyzer", "desc": "understands what the user means"},
            {"name": "MemoryInterface", "desc": "recalls or stores experiences"},
            {"name": "RetrievalOrchestrator", "desc": "searches memory for relevant information"},
            {"name": "ContextAdapter", "desc": "builds context for reasoning"},
            {"name": "ReflectionInterpreter", "desc": "handles reflective or summary questions"},
            {"name": "Consolidator", "desc": "summarizes recent reasoning"},
            {"name": "DeterministicResponder", "desc": "handles factual or deterministic queries like time, math, or definitions"},

        ]

        self.system_prompt = (
            "You are Tyrone's cognitive router. "
            "Always respond ONLY with valid compact JSON in the exact form:\n"
            '{"intent": "<one-word-intent>", "pipeline": ["Module1","Module2"], "reason": "<short reason>"}\n'
            "Do not include any explanations or markdown outside the JSON."
        )

    def route(self, text: str, context: str = None) -> dict:
        # 1️⃣ quick deterministic check for common intent patterns
        q = text.lower()

        # recall / memory-related queries
        if re.search(r"\b(learn|remember|recall|quantization|moondream|visionassist)\b", q):
            return {
                "intent": "recall",
                "pipeline": ["IntentAnalyzer", "RetrievalOrchestrator", "ReflectionInterpreter"],
                "reason": "User asked to recall or summarize previously learned information."
            }

        # planning / scheduling
        if re.search(r"\b(plan|schedule|task|todo|reminder)\b", q):
            return {
                "intent": "plan",
                "pipeline": ["IntentAnalyzer", "PlanBuilder"],
                "reason": "Planning-related request detected."
            }

        # reflection / self-assessment
        if re.search(r"\b(reflect|progress|confidence|feeling|learned)\b", q):
            return {
                "intent": "reflect",
                "pipeline": ["IntentAnalyzer", "ReflectionInterpreter"],
                "reason": "Reflection or self-assessment requested."
            }
        
        # deterministic queries (time, math, etc.)
        if re.search(r"\b(time|date|today|now|calculate|plus|minus|divided|times)\b", q):
            return {
                "intent": "deterministic",
                "pipeline": ["DeterministicResponder"],
                "reason": "User asked for time, date, or math — handled locally."
            }

        # general knowledge queries (keep in LLM)
        if re.search(r"\b(capital|country|population|who|when|where|define|meaning of)\b", q):
            return {
                "intent": "fact_query",
                "pipeline": ["IntentAnalyzer", "ContextAdapter", "ReflectionInterpreter"],
                "reason": "User asked a general knowledge or factual question to be answered by the LLM."
            }



        # 2️⃣ otherwise use the LLM router for general reasoning
        recall = self.memory.search_similar(text)
        recall_hint = (
            f"\nRelevant memory found: {recall[0]['text']}"
            if recall else "\nNo strong prior memories found."
        )

        modules_json = json.dumps(self.module_registry, indent=2)
        user_prompt = (
            f"Input: {text}\n"
            f"Context: {context or '(none)'}\n"
            f"{recall_hint}\n\n"
            f"Available modules:\n{modules_json}\n\n"
            "Return your decision as JSON only."
        )

        raw = call_llm(system_prompt=self.system_prompt, user_prompt=user_prompt)

        # 3️⃣ parse safely
        try:
            decision = json.loads(raw)
            if not isinstance(decision.get("pipeline"), list):
                raise ValueError
            decision.setdefault("intent", "unknown")
            decision.setdefault("reason", "(no reason provided)")

            # 🧠 ensure at least one response-generating module
            if "ContextAdapter" not in decision["pipeline"]:
                decision["pipeline"].append("ContextAdapter")

        except Exception:
            # 🪞 guaranteed fallback with reflection and response
            decision = {
                "intent": "reflect",
                "pipeline": ["ContextAdapter", "ReflectionInterpreter"],
                "reason": "Fallback: invalid LLM output — defaulting to reflection.",
            }

        return decision
 