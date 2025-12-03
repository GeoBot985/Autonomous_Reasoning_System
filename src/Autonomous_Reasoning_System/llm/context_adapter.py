from ..retrieval import RetrievalSystem
from .engine import call_llm
# from .consolidator import ReasoningConsolidator
from ..tools.system_tools import get_current_time, get_current_location
from ..prompts import CONTEXT_ADAPTER_SYSTEM_TEMPLATE, CONTEXT_ADAPTER_NO_MEMORY_SYSTEM
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
        self.memory = memory_storage
        if not self.memory:
             logger.warning("[WARN] ContextAdapter initialized without memory_storage.")

        self.retriever = RetrievalSystem(memory_system=self.memory)

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

        context_str = self.retriever.get_context_string(user_input, include_history=self.history)

        # Build startup context string
        startup_info = ""
        with self._context_lock:
            if self.startup_context:
                startup_info = "\nCURRENT CONTEXT:\n"
                for key, value in self.startup_context.items():
                    startup_info += f"- {key}: {value}\n"

        if context_str or startup_info:
            system_prompt = CONTEXT_ADAPTER_SYSTEM_TEMPLATE.format(
                startup_info=startup_info,
                context_str=context_str,
                user_input=user_input
            )
            user_prompt = ""
        else:
            system_prompt = CONTEXT_ADAPTER_NO_MEMORY_SYSTEM
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
