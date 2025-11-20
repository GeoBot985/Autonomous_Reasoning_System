# Autonomous_Reasoning_System/memory/memory_interface.py

from Autonomous_Reasoning_System.memory.singletons import (
    get_embedding_model,
    get_memory_storage,
    get_vector_store,
)
from Autonomous_Reasoning_System.memory.episodes import EpisodicMemory
import numpy as np
from datetime import datetime


class MemoryInterface:
    """
    Unified interface connecting symbolic, semantic, and episodic memory layers.
    Provides a clean API for Tyrone’s reasoning and self-reflection.
    """

    def __init__(self):
        self.storage = get_memory_storage()
        self.episodes = EpisodicMemory()
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()

    # ------------------------------------------------------------------
    def remember(self, text: str, metadata: dict = None):
        """
        Add a memory to the system (Unified replacement for store).
        """
        metadata = metadata or {}
        memory_type = metadata.get("type", "note")
        importance = metadata.get("importance", 0.5)
        source = metadata.get("source", "unknown")

        uid = self.storage.add_memory(text, memory_type, importance, source)
        if self.episodes.active_episode_id:
            print(f"🧠 Added memory linked to active episode {self.episodes.active_episode_id}")
        return uid

    # Legacy alias
    def store(self, text: str, memory_type="note", importance=0.5):
        return self.remember(text, {"type": memory_type, "importance": importance})

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k=5):
        """
        Retrieve top-k semantically similar memories (Unified replacement for recall).
        Combines vector search and keyword fallback if needed.
        """
        try:
            # 1. Vector Search
            if hasattr(self, "vector_store"):
                q_vec = self.embedder.embed(query)
                results = self.vector_store.search(q_vec, k)
                if results:
                    # Format for consumption
                    return [{"text": r["text"], "score": r["score"], "id": r["id"]} for r in results]

            # 2. Fallback: Keyword search
            if hasattr(self, "storage"):
                print("⚠️ Vector search yielded no results, falling back to keyword search.")
                results = self.storage.search_text(query, top_k=k)
                return [{"text": r[0], "score": r[1], "id": None} for r in results]

        except Exception as e:
            print(f"[MemoryInterface] retrieve failed: {e}")

        return []

    # Legacy alias
    def recall(self, query: str, k=5):
        results = self.retrieve(query, k)
        if not results:
            return "No relevant memories found."
        summary = "\n".join([f"- ({r['score']:.3f}) {r['text']}" for r in results])
        return summary

    # Legacy alias
    def search_similar(self, query: str, top_k: int = 3):
        return self.retrieve(query, top_k)

    # ------------------------------------------------------------------
    def update(self, uid: str, new_content: str):
        """
        Update an existing memory by ID.
        """
        return self.storage.update_memory(uid, new_content)

    # ------------------------------------------------------------------
    def summarize_and_compress(self):
        """
        Summarize recent episodes or day (Unified replacement for summarize_day/end_episode).
        """
        # For now, this delegates to summarize_day behavior but could be expanded
        # to compress older memories, etc.
        print("🗜️ Running unified memory summarization and compression...")

        # 1. Summarize the day's episodes
        def simple_summarizer(text):
            # This should ideally use an LLM
            words = len(text.split())
            return f"(summary of {words} words)\n{text[:200]}..."

        summary = self.episodes.summarize_day(simple_summarizer)

        # 2. Could add logic here to compress older memories in 'storage'
        # (e.g. delete raw logs, keep summary)

        return summary

    # Legacy alias
    def summarize_day(self):
        return self.summarize_and_compress()

    # ------------------------------------------------------------------
    def start_episode(self, description=None):
        """
        Begin a new episodic context.
        """
        eid = self.episodes.begin_episode()
        if description:
            self.remember(f"Episode started: {description}", {"type": "context", "importance": 0.4})
        return eid

    # ------------------------------------------------------------------
    def end_episode(self, summary_hint: str = None):
        """
        Close the current episode.
        """
        from Autonomous_Reasoning_System.memory.llm_summarizer import summarize_with_local_llm

        if not self.episodes.active_episode_id:
            print("⚠️ No active episode to end.")
            return None

        # Collect recent memories for this session
        df = self.storage.get_all_memories()
        combined = "\n".join(df.head(10)["text"].tolist()) if not df.empty else "(no recent memories)"

        # Merge hint + memory contents
        to_summarize = f"{summary_hint or ''}\n\n{combined}".strip()

        print("🤖 Generating episodic summary via Ollama...")
        # Note: In a real environment without Ollama this might fail or return mock
        try:
            summary = summarize_with_local_llm(to_summarize)
        except Exception as e:
            print(f"LLM Summarization failed: {e}")
            summary = "Summary generation failed."

        self.episodes.end_episode(summary)
        print("🧾 Episode summarized.")
        return summary
