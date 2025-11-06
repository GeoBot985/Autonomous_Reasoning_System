from ..memory.context_builder import ContextBuilder
from ..memory.storage import MemoryStorage
from ..memory.retrieval_orchestrator import RetrievalOrchestrator
from .engine import call_llm
from .consolidator import ReasoningConsolidator


class ContextAdapter:
    """
    Connects Tyrone's memory context to the reasoning engine.
    Embedding-based, entity-agnostic context integration.
    Retrieves the top semantically and deterministically relevant
    memories and ensures the LLM treats them as verified truth.
    """

    CONSOLIDATION_INTERVAL = 5  # summarize every N turns

    def __init__(self):
        self.builder = ContextBuilder()
        self.memory = MemoryStorage()
        self.retriever = RetrievalOrchestrator()
        self.consolidator = ReasoningConsolidator()
        self.turn_counter = 0

    # ------------------------------------------------------------------
    def run(self, user_input: str, system_prompt: str = None) -> str:
        """
        Builds context using semantic + deterministic recall,
        sends to the LLM, stores the reasoning episode,
        and periodically triggers consolidation.
        """
        try:
            # 1️⃣ Build standard reasoning context
            base_context = self.builder.build_context(user_input).strip()

            # 2️⃣ Retrieve factual memories via embeddings
            retrieved = self.retriever._semantic_retrieve(user_input, k=5)

            # --- relevance filtering (score > 0.6 only) ---
            relevant = []
            if retrieved:
                for r in retrieved:
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        text, score = r
                        if isinstance(score, (float, int)) and score > 0.6:
                            relevant.append((text, score))

            retrieved_texts = (
                "\n".join([f"- {r[0]}" for r in relevant])
                if relevant
                else ""
            )

            # Merge both context sources
            context_display = (
                f"{base_context}\n{retrieved_texts}".strip()
                if base_context or retrieved_texts
                else ""
            )

            # 3️⃣ Choose reasoning mode — factual vs open
            if context_display and relevant:
                # --- factual mode with hard entity constraint ---
                system_prompt = (
                    "You are Tyrone, a reasoning system that exists entirely within your own closed memory world. "
                    "You have no access to public data, the internet, or any external sources. "
                    "Everything you know comes exclusively from your verified memory store. "
                    "Each line in the context below is a factual record from your own memory. "
                    "If a name or entity appears, it exists only in this internal dataset — "
                    "you must never invent or import external information. "
                    "Respond strictly using these facts. "
                    "If information is missing, say so explicitly rather than guessing or substituting. "
                    "Do not mention being unable to discuss private individuals; all entities here are internal."
                )


                merged_prompt = (
                    f"[FACTUAL CONTEXT]\n{context_display}\n\n"
                    f"[USER MESSAGE]\n{user_input}\n\n"
                    "Respond ONLY using the factual context. "
                    "You cannot use external world knowledge."
                )


            else:
                # --- open world knowledge mode ---
                system_prompt = system_prompt or (
                    "You are Tyrone, a reasoning assistant. "
                    "No relevant factual memories were found for this question, "
                    "so you may use your general knowledge. "
                    "Provide an accurate, concise answer in plain language."
                )

                merged_prompt = (
                    f"The user asked: {user_input}\n\n"
                    "No relevant factual memories were found. "
                    "Use your general knowledge to provide an accurate answer."
                )

            # 4️⃣ Generate reply
            reply = call_llm(system_prompt, merged_prompt)

            # 5️⃣ Store this interaction as an episodic memory
            self.memory.add_memory(
                text=f"User: {user_input}\nAssistant: {reply}",
                memory_type="episode",
                importance=0.6,
                source="context_adapter",
            )

            # 6️⃣ Periodic summarization
            self.turn_counter += 1
            if self.turn_counter % self.CONSOLIDATION_INTERVAL == 0:
                print("🧩 Auto-consolidation triggered...")
                summary = self.consolidator.consolidate_recent(limit=5)
                self.memory.add_memory(
                    text=f"Reflection link: {summary}",
                    memory_type="reflection",
                    importance=0.8,
                    source="context_adapter",
                )

            return reply

        except Exception as e:
            return f"[ContextAdapter Error] {e}"
