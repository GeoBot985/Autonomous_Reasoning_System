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
    def store(self, text: str, memory_type="note", importance=0.5):
        """
        Add a memory to the system.
        If an episode is active, link the memory logically to that episode.
        """
        uid = self.storage.add_memory(text, memory_type, importance)
        if self.episodes.active_episode_id:
            print(f"🧠 Added memory linked to active episode {self.episodes.active_episode_id}")
        return uid

    # ------------------------------------------------------------------
    def recall(self, query: str, k=5):
        """
        Retrieve top-k semantically similar memories from the FAISS index.
        """
        q_vec = self.embedder.embed(query)
        results = self.storage.vector_store.search(q_vec, k)
        if not results:
            return "No relevant memories found."
        summary = "\n".join([f"- ({r['score']:.3f}) {r['text']}" for r in results])
        return summary

    # ------------------------------------------------------------------
    def start_episode(self, description=None):
        """
        Begin a new episodic context.
        Optionally add a short description as its first memory.
        """
        eid = self.episodes.begin_episode()
        if description:
            self.store(f"Episode started: {description}", "context", 0.4)
        return eid

    # ------------------------------------------------------------------
    def end_episode(self, summary_hint: str = None):
        """
        Close the current episode.
        Generates a natural-language summary using the local LLM via Ollama.
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
        summary = summarize_with_local_llm(to_summarize)

        self.episodes.end_episode(summary)
        print("🧾 Episode summarized (LLM).")
        return summary
    
    #----------------------------------------------------------------------
    def search_similar(self, query: str, top_k: int = 3):
        """
        Perform a semantic search for experiences similar to the query.
        Returns a list of dicts: [{'text': ..., 'score': ...}, ...]
        """
        try:
            # If vector search is available
            if hasattr(self, "vector_store"):
                q_vec = self.embedder.embed(query)
                return self.vector_store.search(q_vec, top_k=top_k)
            
            # Fallback: keyword match in storage
            if hasattr(self, "storage"):
                results = self.storage.search_text(query, top_k=top_k)
                return [{"text": r[0], "score": r[1]} for r in results]
        except Exception as e:
            print(f"[Router|MemoryInterface] search_similar failed: {e}")

        return []



    # ------------------------------------------------------------------
    def summarize_day(self):
        """
        Summarize all episodes for the current UTC day.
        """
        def simple_summarizer(text):
            words = len(text.split())
            return f"(summary of {words} words)\n{text[:200]}..."

        return self.episodes.summarize_day(simple_summarizer)
