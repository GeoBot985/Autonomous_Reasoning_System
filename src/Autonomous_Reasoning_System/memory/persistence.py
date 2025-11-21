import os
import pickle
import pandas as pd
from pathlib import Path
import threading

class PersistenceService:
    """
    Dedicated service for handling all disk I/O for the memory system.
    Responsible for loading and saving:
    - Deterministic memory (memory.parquet)
    - Episodic memory (episodes.parquet)
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
        self.goals_path = self.data_dir / "goals.parquet"
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
    # Goals
    # ------------------------------------------------------------------
    def load_goals(self) -> pd.DataFrame:
        """Load goals from parquet."""
        if self.goals_path.exists():
            try:
                return pd.read_parquet(self.goals_path)
            except Exception as e:
                print(f"[Persistence] Error loading goals: {e}")

        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=[
            "id", "text", "priority", "status", "steps", "metadata", "created_at", "updated_at"
        ])

    def save_goals(self, df: pd.DataFrame):
        """Save goals to parquet."""
        try:
            df.to_parquet(self.goals_path)
            print(f"[Persistence] Saved goals to {self.goals_path}")
        except Exception as e:
            print(f"[Persistence] Error saving goals: {e}")

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
def get_persistence_service():
    return PersistenceService()
