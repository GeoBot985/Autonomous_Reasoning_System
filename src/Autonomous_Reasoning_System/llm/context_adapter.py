from ..memory.context_builder import ContextBuilder
from ..memory.retrieval_orchestrator import RetrievalOrchestrator
from .engine import call_llm
from .consolidator import ReasoningConsolidator
from ..tools.system_tools import get_current_time, get_current_location
import threading
import logging

logger = logging.getLogger(__name__)

class ContextAdapter:
    """
    Connects Tyrone's memory context to the reasoning engine.
    Embedding-based, entity-agnostic context integration.
    Retrieves the top semantically and deterministically relevant
    memories and ensures the LLM treats them as verified truth.
    """

    CONSOLIDATION_INTERVAL = 5  # summarize every N turns

    def __init__(self, memory_storage=None, embedding_model=None):
        self.builder = ContextBuilder()
        self.memory = memory_storage
        if not self.memory:
             logger.warning("[WARN] ContextAdapter initialized without memory_storage.")

        self.retriever = RetrievalOrchestrator(memory_storage=self.memory, embedding_model=embedding_model)
        self.consolidator = ReasoningConsolidator()
        self.turn_counter = 0
        self.history = [] # Short-term conversation history
        self.startup_context = {}  # Stores startup info like time and location
        self._context_lock = threading.Lock()

        # Load recent conversation history from persistence
        if self.memory:
            try:
                # Handle both MemoryStorage and MemoryInterface wrapper
                storage_ref = self.memory
                if hasattr(self.memory, "storage"):
                    storage_ref = self.memory.storage

                if hasattr(storage_ref, "get_all_memories"):
                    df = storage_ref.get_all_memories()
                    if not df.empty:
                        # Filter for episodes
                        episodes = df[df["memory_type"] == "episode"]
                        if not episodes.empty:
                            # Sort by created_at
                            episodes = episodes.sort_values("created_at", ascending=True)

                            # Load last 10 interactions (approx 20 lines)
                            recent = episodes.tail(10)

                            for _, row in recent.iterrows():
                                text = str(row.get("text", ""))
                                # Split into lines to match history structure (User line, Tyrone line)
                                for line in text.split("\n"):
                                    if line.strip():
                                        self.history.append(line.strip())

                            # Keep only the most recent 20 lines to avoid token bloat on restart
                            self.history = self.history[-20:]

                            logger.info(f"[ContextAdapter] Restored {len(self.history)} lines of conversation history.")
            except Exception as e:
                logger.error(f"[ContextAdapter] Error loading history: {e}")

    def set_startup_context(self, context: dict):
        """Sets the startup context (e.g. location, time)."""
        with self._context_lock:
            self.startup_context = context

    def _ensure_context(self):
        """Ensures minimal context exists if startup_context is empty."""
        # Double check locking pattern to avoid overhead if already set
        if self.startup_context:
            return

        with self._context_lock:
            if not self.startup_context:
                try:
                    # Fallback if not initialized externally
                    self.startup_context = {
                        "Current Time": get_current_time(),
                        # Skipping location to avoid API latency if not strictly needed, or we can call it.
                        # "Current Location": get_current_location()
                    }
                except Exception as e:
                    logger.warning(f"Failed to lazy-load context: {e}")

    # ------------------------------------------------------------------
    def run(self, user_input: str, system_prompt: str = None) -> str:
        self.turn_counter += 1

        self._ensure_context()

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

        # Build startup context string
        startup_info = ""
        with self._context_lock:
            if self.startup_context:
                startup_info = "\nCURRENT CONTEXT:\n"
                for key, value in self.startup_context.items():
                    startup_info += f"- {key}: {value}\n"

        if memory_text or history_text or startup_info:
            system_prompt = f"""
YOU ARE TYRONE.

{startup_info}
LONG TERM MEMORY (FACTS):
{memory_text}

{history_text}

RULES YOU MUST OBEY:
- The LONG TERM MEMORY (FACTS) are verified truth and override ALL other knowledge, including the current system date in CURRENT CONTEXT, for any personal information.
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
