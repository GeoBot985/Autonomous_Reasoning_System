# Autonomous_Reasoning_System/memory/episodes.py

import duckdb
import os
from datetime import datetime
from uuid import uuid4
import numpy as np

from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel


class EpisodicMemory:
    """
    Manages episodes: coherent clusters of related memories.
    Each episode can be summarized and semantically recalled.
    """

    def __init__(self, db_path="data/episodes.parquet"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.embedder = EmbeddingModel()
        self._ensure_table()
        self.active_episode_id = None

    # ------------------------------------------------------------------
    def _ensure_table(self):
        """Create the episodes table if it doesn't exist."""
        if not os.path.exists(self.db_path):
            duckdb.sql("""
                CREATE TABLE episodes (
                    episode_id VARCHAR,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    summary VARCHAR,
                    importance DOUBLE,
                    vector BLOB
                )
            """)
            duckdb.sql(f"COPY episodes TO '{self.db_path}' (FORMAT PARQUET)")
        else:
            duckdb.sql(f"""
                CREATE OR REPLACE TABLE episodes AS
                SELECT * FROM read_parquet('{self.db_path}')
            """)

        duckdb.sql(f"COPY (SELECT * FROM episodes) TO '{self.db_path}' (FORMAT PARQUET)")

    # ------------------------------------------------------------------
    def begin_episode(self):
        """Start a new active episode."""
        if self.active_episode_id:
            print(f"[Episodic] Episode already active: {self.active_episode_id}")
            return self.active_episode_id

        self.active_episode_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        duckdb.sql(f"""
            CREATE OR REPLACE TABLE episodes_temp AS
            SELECT * FROM read_parquet('{self.db_path}');
        """)

        duckdb.sql(f"""
            INSERT INTO episodes_temp (
                episode_id, start_time, end_time, summary, importance, vector
            )
            VALUES ('{self.active_episode_id}', '{now}', NULL, NULL, 0.5, NULL);
        """)

        duckdb.sql(f"COPY episodes_temp TO '{self.db_path}' (FORMAT PARQUET, OVERWRITE TRUE);")
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

        duckdb.sql(f"""
            CREATE OR REPLACE TABLE episodes_temp AS
            SELECT * FROM read_parquet('{self.db_path}');
        """)

        duckdb.sql(f"""
            UPDATE episodes_temp
            SET end_time = '{end_time}',
                summary = '{summary_text.replace("'", "''")}',
                vector = '{vec.hex()}'
            WHERE episode_id = '{self.active_episode_id}';
        """)

        duckdb.sql(f"COPY episodes_temp TO '{self.db_path}' (FORMAT PARQUET, OVERWRITE TRUE);")
        print(f"🏁 Ended episode {self.active_episode_id}")
        self.active_episode_id = None

    # ------------------------------------------------------------------
    def summarize_day(self, llm_summarize_func):
        """
        Summarize all episodes for today using a provided LLM summarizer function.
        """
        today = datetime.utcnow().date()
        df = duckdb.sql(f"""
            SELECT * FROM read_parquet('{self.db_path}')
            WHERE start_time::DATE = '{today}'
        """).df()

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
        return duckdb.sql(f"SELECT * FROM read_parquet('{self.db_path}') ORDER BY start_time DESC").df()
