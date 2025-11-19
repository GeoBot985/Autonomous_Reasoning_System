import json
import re
from ..memory.singletons import get_memory_storage
from ..memory.retrieval_orchestrator import RetrievalOrchestrator
from .engine import call_llm


class ReflectionInterpreter:
    """
    Handles introspective or reflective reasoning.
    Now integrates factual context from Tyrone's memory before reflection.
    The model is instructed that retrieved memories override all other world knowledge.
    """

    def __init__(self):
        self.memory = get_memory_storage()
        self.retriever = RetrievalOrchestrator()

    # ----------------------------------------------------------------------
    def interpret(self, user_input: str, raw: bool = False) -> dict:
        """
        Reflect on user input using Tyrone's stored experiences + factual memories.
        If raw=True, return unprocessed model output directly.
        Otherwise, return structured JSON with summary, insight, and confidence_change.
        """
        # === RAW MODE ===========================================================
        if raw:
            try:
                raw_response = call_llm("", user_input)
                return raw_response if isinstance(raw_response, str) else str(raw_response)
            except Exception as e:
                print(f"[WARN] ReflectionInterpreter raw mode failed: {e}")
                return ""

        # === REFLECTIVE MODE ====================================================
        try:
            df = self.memory.get_all_memories()
            if df.empty:
                return {
                    "summary": "No reflections recorded yet.",
                    "insight": "Tyrone has not reflected before.",
                    "confidence_change": "neutral",
                }

            # 🧠 Retrieve relevant factual memories using semantic recall
            retrieved = self.retriever._semantic_retrieve(user_input, k=5)
            memory_context = (
                "\n".join([f"- {r}" for r in retrieved])
                if retrieved
                else "(no relevant factual memories found)"
            )

            # 🧩 Collect recent reflections + episodic summaries
            reflections = df[df["memory_type"].isin(["reflection", "episodic_summary"])]
            reflections = reflections.sort_values("created_at", ascending=False).head(8)
            reflection_block = "\n\n".join(reflections["text"].tolist())

            # 🧭 System prompt enforcing factual override
            system_prompt = (
                "You are Tyrone’s reflection module. "
                "The text below contains verified factual memories followed by self-reflection logs. "
                "Facts override all other world knowledge. "
                "Never introduce fictional or unrelated information. "
                "If a factual memory mentions a person, event, or detail, treat it as true. "
                "Analyze the reflections in light of the user's input and these facts. "
                "Respond only in valid JSON of the form:\n"
                '{"summary": "<short summary>", "insight": "<lesson or fact>", "confidence_change": "<positive|neutral|negative>"}'
            )

            user_prompt = (
                f"[FACTUAL CONTEXT]\n{memory_context}\n\n"
                f"[SELF REFLECTIONS]\n{reflection_block}\n\n"
                f"[USER INPUT]\n{user_input}\n\n"
                "Return only the JSON object."
            )

            print("\n[🧠 FACTUAL CONTEXT FOR REFLECTION]")
            print(memory_context[:500])
            raw_output = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)


            # --- Robust JSON extraction ---
            try:
                match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                result = json.loads(match.group()) if match else json.loads(raw_output)
            except Exception:
                return {
                    "summary": "Raw reflection",
                    "insight": raw_output.strip(),
                    "confidence_change": "neutral",
                }

            # Fill defaults safely
            result.setdefault("summary", "(no summary)")
            result.setdefault("insight", "(no insight)")
            result.setdefault("confidence_change", "neutral")

            # 🧩 Log structured reflection
            self.memory.add_memory(
                f"Reflection → {result['summary']} | Insight: {result['insight']} | Confidence: {result['confidence_change']}",
                memory_type="reflection",
            )

            print(f"[🪞 REFLECTION] {result}")
            return result

        except Exception as e:
            return {
                "summary": "ReflectionInterpreter error.",
                "insight": str(e),
                "confidence_change": "neutral",
            }
