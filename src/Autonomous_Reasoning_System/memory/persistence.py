import os
import pickle
import pandas as pd
import faiss
from pathlib import Path
import threading

class PersistenceService:
    """
    Dedicated service for handling all disk I/O for the memory system.
    Responsible for loading and saving:
    - Deterministic memory (memory.parquet)
    - Episodic memory (episodes.parquet)
    - Vector index (vector_index.faiss)
    - Vector metadata (vector_meta.pkl)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PersistenceService, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_dir="data"):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.data_dir = Path(data_dir)
        self.memory_path = self.data_dir / "memory.parquet"
        self.episodes_path = self.data_dir / "episodes.parquet"
        self.vector_index_path = self.data_dir / "vector_index.faiss"
        self.vector_meta_path = self.data_dir / "vector_meta.pkl"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.initialized = True

    # ------------------------------------------------------------------
    # Deterministic Memory
    # ------------------------------------------------------------------
    def load_deterministic_memory(self) -> pd.DataFrame:
        """Load deterministic memory from parquet."""
        if self.memory_path.exists():
            try:
                return pd.read_parquet(self.memory_path)
            except Exception as e:
                print(f"[Persistence] Error loading deterministic memory: {e}")

        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=[
            "id", "text", "memory_type", "created_at", "last_accessed",
            "importance", "scheduled_for", "status", "source"
        ])

    def save_deterministic_memory(self, df: pd.DataFrame):
        """Save deterministic memory to parquet."""
        try:
            df.to_parquet(self.memory_path)
            print(f"[Persistence] Saved deterministic memory to {self.memory_path}")
        except Exception as e:
            print(f"[Persistence] Error saving deterministic memory: {e}")

    # ------------------------------------------------------------------
    # Episodic Memory
    # ------------------------------------------------------------------
    def load_episodic_memory(self) -> pd.DataFrame:
        """Load episodic memory from parquet."""
        if self.episodes_path.exists():
            try:
                return pd.read_parquet(self.episodes_path)
            except Exception as e:
                print(f"[Persistence] Error loading episodic memory: {e}")

        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=[
            "episode_id", "start_time", "end_time", "summary", "importance", "vector"
        ])

    def save_episodic_memory(self, df: pd.DataFrame):
        """Save episodic memory to parquet."""
        try:
            df.to_parquet(self.episodes_path)
            print(f"[Persistence] Saved episodic memory to {self.episodes_path}")
        except Exception as e:
            print(f"[Persistence] Error saving episodic memory: {e}")

    # ------------------------------------------------------------------
    # Vector Index
    # ------------------------------------------------------------------
    def load_vector_index(self):
        """Load FAISS index."""
        if self.vector_index_path.exists():
            try:
                return faiss.read_index(str(self.vector_index_path))
            except Exception as e:
                print(f"[Persistence] Error loading vector index: {e}")
        return None

    def save_vector_index(self, index):
        """Save FAISS index."""
        try:
            faiss.write_index(index, str(self.vector_index_path))
            print(f"[Persistence] Saved vector index to {self.vector_index_path}")
        except Exception as e:
            print(f"[Persistence] Error saving vector index: {e}")

    # ------------------------------------------------------------------
    # Vector Metadata
    # ------------------------------------------------------------------
    def load_vector_metadata(self) -> list:
        """Load vector metadata."""
        if self.vector_meta_path.exists():
            try:
                with open(self.vector_meta_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[Persistence] Error loading vector metadata: {e}")
        return []

    def save_vector_metadata(self, metadata: list):
        """Save vector metadata."""
        try:
            with open(self.vector_meta_path, "wb") as f:
                pickle.dump(metadata, f)
            print(f"[Persistence] Saved vector metadata to {self.vector_meta_path}")
        except Exception as e:
            print(f"[Persistence] Error saving vector metadata: {e}")

def get_persistence_service():
    return PersistenceService()
