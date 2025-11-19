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
        # === 1. Deterministic fast-paths (unchanged) ===
        q = text.lower()
        lower = text.lower()
        if lower.startswith("remember") or "please remember" in lower or "just remember" in lower:
            # IMMEDIATELY store the raw user message as a sacred personal fact
            from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
            MemoryInterface().store(
                text=text.strip(),
                memory_type="personal_fact",
                importance=1.0
            )
            return {
                "intent": "fact_stored",
                "pipeline": ["ContextAdapter"],
                "reason": "Direct personal fact storage triggered"
            }

        if re.search(r"\b(learn|remember|recall|quantization|moondream|visionassist)\b", q):
            return {"intent": "recall", "pipeline": ["IntentAnalyzer", "RetrievalOrchestrator", "ReflectionInterpreter"], "reason": "Explicit recall request"}

        if re.search(r"\b(plan|schedule|task|todo|reminder)\b", q):
            return {"intent": "plan", "pipeline": ["IntentAnalyzer", "PlanBuilder"], "reason": "Planning request"}

        if re.search(r"\b(reflect|progress|confidence|feeling|learned)\b", q):
            return {"intent": "reflect", "pipeline": ["IntentAnalyzer", "ReflectionInterpreter"], "reason": "Explicit reflection"}

        if re.search(r"\b(time|date|today|now|calculate|plus|minus|divided|times)\b", q):
            return {"intent": "deterministic", "pipeline": ["DeterministicResponder"], "reason": "Time/math query"}

        if re.search(r"\b(capital|country|population|who|when|where|define|meaning of)\b", q):
            return {"intent": "fact_query", "pipeline": ["IntentAnalyzer", "ContextAdapter", "ReflectionInterpreter"], "reason": "General knowledge query"}

        if "birthday" in q or ("cornelia" in q and any(word in q for word in ["is", "="])) or "remember" in q.lower():
            return {"intent": "store_fact", "pipeline": ["MemoryInterface"], "reason": "Direct fact storage"}

        # === 2. Semantic routing with bulletproof JSON parsing ===
        recall = self.memory.search_similar(text)
        recall_hint = f"\nRelevant memory: {recall[0]['text']}" if recall else "\nNo relevant memory."

        modules_json = json.dumps(self.module_registry, indent=2)
        user_prompt = (
            f"Input: {text}\n"
            f"Context: {context or '(none)'}\n"
            f"{recall_hint}\n\n"
            f"Available modules:\n{modules_json}\n\n"
            "Respond with ONLY this exact JSON format, no markdown, no extra text:\n"
            '{"intent": "one_word", "pipeline": ["Module1", "Module2"], "reason": "short reason"}'
        )

        raw = call_llm(system_prompt=self.system_prompt, user_prompt=user_prompt)

        # ─────────────────────── NUCLEAR-LEVEL JSON EXTRACTION ───────────────────────
        try:
            # Remove common markdown wrappers
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3:]
            cleaned = cleaned.strip()

            # Find JSON object if buried in text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                cleaned = cleaned[start:end]

            decision = json.loads(cleaned)

            # Basic validation
            if not isinstance(decision.get("pipeline"), list) or len(decision.get("pipeline", [])) == 0:
                raise ValueError("Invalid or empty pipeline")

            decision.setdefault("intent", "unknown")
            decision.setdefault("reason", "Parsed successfully")

            # Force ContextAdapter if missing (critical for grounding)
            if "ContextAdapter" not in decision["pipeline"]:
                decision["pipeline"].append("ContextAdapter")

            print(f"[ROUTER] Success → {decision['intent']} | {decision['pipeline']}")
            return decision

        except Exception as e:
            print(f"[ROUTER] JSON parsing failed: {e}\nRaw output:\n{raw}\n")
            # Final desperate fallback — but now extremely rare
            return {
                "intent": "query",
                "pipeline": ["ContextAdapter"],
                "reason": "Router JSON failed — using safe single-module path",
            }
 