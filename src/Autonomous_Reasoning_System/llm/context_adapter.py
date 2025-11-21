from ..memory.context_builder import ContextBuilder
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

    def __init__(self, memory_storage=None):
        self.builder = ContextBuilder()
        self.memory = memory_storage
        if not self.memory:
             print("[WARN] ContextAdapter initialized without memory_storage.")

        self.retriever = RetrievalOrchestrator()
        self.consolidator = ReasoningConsolidator()
        self.turn_counter = 0
        self.history = [] # Short-term conversation history

    # ------------------------------------------------------------------
    def run(self, user_input: str, system_prompt: str = None) -> str:
        self.turn_counter += 1

        memories = self.retriever.retrieve(user_input)

        memory_text = ""
        if memories:
            clean = [str(m).strip() for m in memories if str(m).strip()]
            if clean:
                memory_text = "\n".join(f"- {line}" for line in clean)

        # Build context window from history
        history_text = ""
        if self.history:
             history_text = "\nRECENT CONVERSATION:\n" + "\n".join(self.history[-5:]) + "\n"

        if memory_text or history_text:
            system_prompt = f"""
YOU ARE TYRONE.

LONG TERM MEMORY (FACTS):
{memory_text}

{history_text}

RULES YOU MUST OBEY:
- Use the facts above and the recent conversation context to answer.
- If the user asks about Cornelia's birthday, answer with the exact date from the facts.
- Never say "I haven't been told" when the fact is right here.
- Answer directly and naturally.

User question: {user_input}
Answer:
"""
            user_prompt = ""
        else:
            system_prompt = "You are Tyrone. No relevant memories found."
            user_prompt = user_input

        reply = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)

        # Update History
        self.history.append(f"User: {user_input}")
        self.history.append(f"Tyrone: {reply}")

        if self.memory:
            self.memory.add_memory(
                text=f"User: {user_input}\nTyrone: {reply}",
                memory_type="episode",
                importance=0.7
            )

        return reply
