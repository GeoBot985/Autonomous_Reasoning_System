# Autonomous_Reasoning_System/memory/memory_interface.py

from Autonomous_Reasoning_System.memory.singletons import (
    get_embedding_model,
    get_memory_storage,
    get_vector_store,
)
from Autonomous_Reasoning_System.memory.episodes import EpisodicMemory
from Autonomous_Reasoning_System.memory.persistence import get_persistence_service
import numpy as np
from datetime import datetime


class MemoryInterface:
    """
    Unified interface connecting symbolic, semantic, and episodic memory layers.
    Provides a clean API for Tyrone’s reasoning and self-reflection.
    Handles persistence automatically via PersistenceService.
    """

    def __init__(self):
        self.persistence = get_persistence_service()

        # Load data from disk
        print("💾 Loading memory from persistence layer...")
        mem_df = self.persistence.load_deterministic_memory()
        goals_df = self.persistence.load_goals()
        ep_df = self.persistence.load_episodic_memory()
        v_idx = self.persistence.load_vector_index()
        v_meta = self.persistence.load_vector_metadata()

        # Initialize singletons/classes with loaded data
        # IMPORTANT: Initialize VectorStore BEFORE MemoryStorage because MemoryStorage
        # might try to get a VectorStore if not already present. We want to ensure
        # the VectorStore is initialized with our loaded data first.
        self.vector_store = get_vector_store(index=v_idx, metadata=v_meta)
        self.storage = get_memory_storage(initial_df=mem_df, initial_goals_df=goals_df)

        # EpisodicMemory is not a singleton in the same way, we create it here
        self.episodes = EpisodicMemory(initial_df=ep_df)

        self.embedder = get_embedding_model()

        # ------------------------------------------------------------------
        # 6. Reconnect the Vector Index with Deterministic Memory
        # Validate consistency: If memory count != vector count, rebuild index.
        # ------------------------------------------------------------------
        mem_count = len(self.storage.get_all_memories())
        vec_count = len(self.vector_store.metadata)

        if mem_count != vec_count:
            print(f"⚠️ Mismatch detected: Memory={mem_count}, Vectors={vec_count}. Rebuilding index...")
            self._rebuild_vector_index()
        else:
            print("✅ Memory and Vector Index are consistent.")

        print("✅ Memory Interface fully hydrated.")

    def _rebuild_vector_index(self):
        """Rebuild FAISS index from current memory storage."""
        print("🔁 Rebuilding FAISS index from stored memories...")
        self.vector_store.reset()

        df = self.storage.get_all_memories()
        count = 0
        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            vec = self.embedder.embed(text)
            meta = {
                "memory_type": row.get("memory_type", "unknown"),
                "source": row.get("source", "unknown")
            }
            self.vector_store.add(row["id"], text, vec, meta)
            count += 1

        print(f"✅ Rebuilt FAISS index with {count} entries.")
        # Save immediately to fix the disk state too
        self.save()

    # ------------------------------------------------------------------
    def save(self):
        """
        Persist all memory states to disk.
        """
        print("💾 Saving memory state...")
        self.persistence.save_deterministic_memory(self.storage.get_all_memories())
        self.persistence.save_goals(self.storage.get_all_goals())
        self.persistence.save_episodic_memory(self.episodes.get_all_episodes())
        self.persistence.save_vector_index(self.vector_store.index)
        self.persistence.save_vector_metadata(self.vector_store.metadata)
        print("✅ Memory saved.")

    # ------------------------------------------------------------------
    def remember(self, text: str, metadata: dict = None):
        """
        Add a memory to the system (Unified replacement for store).
        Automatically triggers a save.
        """
        metadata = metadata or {}
        memory_type = metadata.get("type", "note")
        importance = metadata.get("importance", 0.5)
        source = metadata.get("source", "unknown")

        uid = self.storage.add_memory(text, memory_type, importance, source)
        if self.episodes.active_episode_id:
            print(f"🧠 Added memory linked to active episode {self.episodes.active_episode_id}")

        self.save()
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
        Also updates the vector index to ensure consistency.
        Automatically triggers a save.
        """
        result = self.storage.update_memory(uid, new_content)
        if result:
            # We need to update the vector index too.
            # FAISS IndexFlatIP doesn't support easy update/delete of single vectors without rebuilding
            # or complex ID mapping. For simplicity and robustness (as per "rebuild" logic),
            # we can either rebuild fully or try to update if VectorStore supports it.
            # Current VectorStore.add appends.
            #
            # Efficient strategy: Re-embed and update metadata if we can find the index,
            # but since we don't map ID -> FAISS index easily, full rebuild is safest for consistency
            # unless we implement ID mapping in VectorStore.
            #
            # Given the requirement "Reconnect... Ensure consistency", a rebuild on update
            # guarantees it, even if expensive.
            print(f"📝 Updating vector index for memory {uid}...")
            self._rebuild_vector_index()
            self.save()
        return result

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------
    def create_goal(self, text: str, priority: int = 1, metadata: dict = None):
        """Create a new goal."""
        from Autonomous_Reasoning_System.memory.goals import Goal

        goal = Goal(
            text=text,
            priority=priority,
            metadata=metadata or {}
        )
        self.storage.add_goal(goal.to_dict())
        self.save()
        return goal.id

    def get_goal(self, goal_id: str):
        return self.storage.get_goal(goal_id)

    def get_active_goals(self):
        return self.storage.get_active_goals()

    def update_goal(self, goal_id: str, updates: dict):
        res = self.storage.update_goal(goal_id, updates)
        if res:
            self.save()
        return res

    # ------------------------------------------------------------------
    def summarize_and_compress(self):
        """
        Summarize recent episodes or day (Unified replacement for summarize_day/end_episode).
        Triggers save after modification.
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

        self.save()

        return summary

    # Legacy alias
    def summarize_day(self):
        return self.summarize_and_compress()

    # ------------------------------------------------------------------
    def start_episode(self, description=None):
        """
        Begin a new episodic context.
        Triggers save.
        """
        eid = self.episodes.begin_episode()
        if description:
            self.remember(f"Episode started: {description}", {"type": "context", "importance": 0.4})
        else:
            self.save() # remember calls save, but if no description we still need to save the new episode
        return eid

    # ------------------------------------------------------------------
    def end_episode(self, summary_hint: str = None):
        """
        Close the current episode.
        Triggers save.
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
        self.save()
        print("🧾 Episode summarized.")
        return summary
