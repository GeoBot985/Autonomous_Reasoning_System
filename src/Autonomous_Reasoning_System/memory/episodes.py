# Autonomous_Reasoning_System/memory/episodes.py

import duckdb
import pandas as pd
from datetime import datetime
from uuid import uuid4
import threading

from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel


class EpisodicMemory:
    """
    Manages episodes: coherent clusters of related memories.
    Each episode can be summarized and semantically recalled.
    Persistence is handled by MemoryInterface via PersistenceService.
    """

    def __init__(self, initial_df: pd.DataFrame = None, embedding_model: EmbeddingModel = None):
        self.embedder = embedding_model or EmbeddingModel()
        self.active_episode_id = None

        # Initialize in-memory connection
        self.con = duckdb.connect(database=':memory:')

        if initial_df is None or initial_df.empty:
             self.con.execute("""
                CREATE TABLE episodes (
                    episode_id VARCHAR,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    summary VARCHAR,
                    importance DOUBLE,
                    vector BLOB
                )
            """)
        else:
            self.con.register('initial_episodes', initial_df)
            self.con.execute("CREATE TABLE episodes AS SELECT * FROM initial_episodes")
            self.con.unregister('initial_episodes')

            # Restore active episode if it was still open (end_time is NULL)
            # (Optional logic, depends if we want to resume sessions)
            # For now, we start fresh or let user resume manually,
            # but let's check if there's an open episode.
            try:
                res = self.con.execute("SELECT episode_id FROM episodes WHERE end_time IS NULL ORDER BY start_time DESC LIMIT 1").fetchone()
                if res:
                    self.active_episode_id = res[0]
                    print(f"[Episodic] Resuming active episode: {self.active_episode_id}")
            except Exception as e:
                print(f"[Episodic] Error restoring active episode: {e}")

    # ------------------------------------------------------------------
    def begin_episode(self):
        """Start a new active episode."""
        if self.active_episode_id:
            print(f"[Episodic] Episode already active: {self.active_episode_id}")
            return self.active_episode_id

        self.active_episode_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        self.con.execute("""
            INSERT INTO episodes (
                episode_id, start_time, end_time, summary, importance, vector
            )
            VALUES (?, ?, NULL, NULL, 0.5, NULL);
        """, (self.active_episode_id, now))

        print(f"🎬 Started new episode: {self.active_episode_id}")
        return self.active_episode_id

    # ------------------------------------------------------------------
    def end_episode(self, summary_text: str):
        """Close the active episode and store its summary + vector."""
        if not self.active_episode_id:
            print("[Episodic] No active episode to end.")
            return None

        end_time = datetime.utcnow().isoformat()
        vec = self.embedder.embed(summary_text).tobytes()

        self.con.execute("""
            UPDATE episodes
            SET end_time = ?,
                summary = ?,
                vector = ?
            WHERE episode_id = ?;
        """, (end_time, summary_text, vec, self.active_episode_id))

        print(f"🏁 Ended episode {self.active_episode_id}")
        self.active_episode_id = None

    # ------------------------------------------------------------------
    def summarize_day(self, llm_summarize_func):
        """
        Summarize all episodes for today using a provided LLM summarizer function.
        """
        today = datetime.utcnow().date()
        try:
            df = self.con.execute("""
                SELECT * FROM episodes
                WHERE start_time::DATE = ?
            """, (str(today),)).df()
        except Exception:
             return None

        if df.empty:
            print("No episodes to summarize today.")
            return None

        combined = "\n".join(df["summary"].dropna().tolist())
        if not combined:
            print("Episodes have no summaries yet.")
            return None

        final_summary = llm_summarize_func(combined)
        print("📜 Daily summary:\n", final_summary)
        return final_summary

    # ------------------------------------------------------------------
    def list_episodes(self):
        return self.con.execute("SELECT * FROM episodes ORDER BY start_time DESC").df()

    # ------------------------------------------------------------------
    def get_all_episodes(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM episodes").df()
