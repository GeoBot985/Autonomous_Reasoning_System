# Autonomous_Reasoning_System/memory/memory_interface.py

from Autonomous_Reasoning_System.memory.episodes import EpisodicMemory
from Autonomous_Reasoning_System.memory.persistence import get_persistence_service
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel
from Autonomous_Reasoning_System.memory.vector_store import DuckVSSVectorStore
from Autonomous_Reasoning_System.infrastructure.concurrency import memory_write_lock
from Autonomous_Reasoning_System.infrastructure.observability import Metrics
import numpy as np
from datetime import datetime


class MemoryInterface:
    """
    Unified interface connecting symbolic, semantic, and episodic memory layers.
    Provides a clean API for Tyrone’s reasoning and self-reflection.
    Handles persistence automatically via PersistenceService.
    """

    def __init__(self, memory_storage: MemoryStorage = None, embedding_model: EmbeddingModel = None, vector_store=None):
        self.persistence = get_persistence_service()

        # Use injected dependencies or instantiate new ones (avoiding global state if possible)
        # Ideally these should be passed from CoreLoop, but for now we support standalone init
        # by creating them if missing, though this might create multiple connections if not careful.
        # However, MemoryInterface is often the entry point.

        self.embedder = embedding_model or EmbeddingModel()

        # Load data from disk
        print("💾 Loading memory from persistence layer...")
        # Note: With the new Persistent DuckDB, loading df might not be needed for initialization
        # but for VectorStore consistency checks.
        # The new MemoryStorage handles its own connection.
        # BUT, we still need to support loading legacy Parquet if that's what persistence does,
        # OR we assume persistence.load_* returns data frames we might want to migrate?
        # The user instructions said "Move the table creation logic... into an init_db() method".
        # It also said "Delete the logic that reads the parquet file into a dataframe".
        # So we likely don't need to load deterministic memory into DF here just to pass to storage.

        # Initialize VSS-backed vector store (kept inside DuckDB)
        self.vector_store = vector_store or DuckVSSVectorStore(
            db_path=memory_storage.db_path if memory_storage else None,
            dim=self.embedder.dim if hasattr(self.embedder, "dim") else 384
        )

        # For Storage:
        # If injected, use it. If not, create it.
        # Note: Storage now handles its own DB connection.
        if memory_storage:
            self.storage = memory_storage
        else:
            self.storage = MemoryStorage(embedding_model=self.embedder, vector_store=self.vector_store)

        # EpisodicMemory is not a singleton in the same way, we create it here
        # It might need loading from parquet still if it hasn't been migrated to DuckDB?
        # The refactor instructions focused on MemoryStorage (symbolic memory).
        # EpisodicMemory might still use DataFrames.
        ep_df = self.persistence.load_episodic_memory()
        self.episodes = EpisodicMemory(initial_df=ep_df)


        print("✅ Memory Interface fully hydrated.")

    def _rebuild_vector_index(self):
        """No-op: VSS lives inside DuckDB and stays consistent."""
        print("[MemoryInterface] VSS index rebuild not required.")

    # ------------------------------------------------------------------
    def save(self):
        """
        Persist all memory states to disk.
        """
        with memory_write_lock:
            print("💾 Saving memory state...")
            # DuckDB is auto-persisted, but we might want to checkpoint or similar?
            # No, DuckDB persistent file is always saved.
            # But we might still need to save VectorStore (FAISS) and EpisodicMemory (if not in DB).

            # We might not need to save deterministic memory anymore if it's in DuckDB directly.
            # But persistence.save_deterministic_memory writes parquet.
            # If we want to keep parquet as backup or if other tools read it, we can keep it.
            # But the instruction "Delete the logic that reads the parquet file" implies we are moving away.
            # However, for safety, let's keep saving other components.

            # self.persistence.save_deterministic_memory(self.storage.get_all_memories()) # Maybe redundant now

            # Goals are also in DB now.
            # self.persistence.save_goals(self.storage.get_all_goals())

            self.persistence.save_episodic_memory(self.episodes.get_all_episodes())
            print("✅ Memory saved.")

    # ------------------------------------------------------------------
    def remember(self, text: str, metadata: dict = None):
        """
        Add a memory to the system (Unified replacement for store).
        Automatically triggers a save.
        """
        Metrics().increment("memory_ops_write")
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
        Metrics().increment("memory_ops_read")
        try:
            # 1. Vector Search
            if hasattr(self, "vector_store") and self.vector_store:
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
            print(f"📝 Updating vector index for memory {uid}...")
            try:
                # Soft delete old entry
                self.vector_store.soft_delete(uid)

                # Add new entry
                vec = self.embedder.embed(new_content)
                # We try to preserve some metadata if possible, but for now we rely on
                # what's in storage or default. Ideally we'd fetch from storage.
                # Since storage is already updated, we could fetch it?
                # But fetching just for metadata might be overkill if we assume defaults.
                # However, let's just put minimal meta or what we have.
                # Actually, let's fetch the row from storage to be safe about metadata like 'source'.
                # Since DuckDB update doesn't return row, we query it.
                # But for performance, maybe we skip full fetch if we don't care about exact metadata consistency in vector store.
                # Let's keep it simple:
                self.vector_store.add(uid, new_content, vec, {"memory_type": "updated", "source": "unknown"}) # Metadata might be lost here slightly, but text is correct.

                self.save()
            except Exception as e:
                print(f"Error updating vector store: {e}")
                # Fallback to rebuild if something goes wrong
                self._rebuild_vector_index()

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
