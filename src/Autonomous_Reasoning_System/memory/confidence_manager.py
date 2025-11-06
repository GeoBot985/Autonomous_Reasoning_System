import duckdb
from datetime import datetime


class ConfidenceManager:
    """
    Adjusts the 'importance' field of memories to simulate reinforcement and decay.
    """

    def __init__(self, db_path: str = "data/memory.parquet"):
        self.db_path = db_path

    def reinforce(self, mem_id: str = None, step: float = 0.05):
        """
        Increase importance slightly when memory is accessed.
        If no mem_id is provided, reinforces the most recent memory.
        """
        # Load all memories
        duckdb.sql(f"""
            CREATE OR REPLACE TABLE memory_temp AS
            SELECT * FROM read_parquet('{self.db_path}');
        """)

        # Determine which memory to update
        if mem_id is None:
            try:
                mem_id = duckdb.sql(
                    "SELECT id FROM memory_temp ORDER BY created_at DESC LIMIT 1"
                ).fetchone()[0]
            except Exception:
                mem_id = None

        if not mem_id:
            print("[⚠️ CONFIDENCE] No valid memory ID found to reinforce.")
            return

        # Apply reinforcement
        duckdb.sql(f"""
            UPDATE memory_temp
            SET importance = LEAST(1.0, COALESCE(importance, 0.0) + {step}),
                last_accessed = '{datetime.utcnow().isoformat()}'
            WHERE id = '{mem_id}';
        """)

        duckdb.sql(
            f"COPY memory_temp TO '{self.db_path}' (FORMAT PARQUET, OVERWRITE TRUE);"
        )
        print(f"[📈 CONFIDENCE] Reinforced memory {mem_id} (+{step}).")

    def decay_all(self, step: float = 0.01):
        """Decrease importance slightly across all memories over time."""
        duckdb.sql(f"""
            CREATE OR REPLACE TABLE memory_temp AS
            SELECT * FROM read_parquet('{self.db_path}');
        """)
        duckdb.sql(f"""
            UPDATE memory_temp
            SET importance = GREATEST(0.0, COALESCE(importance, 0.0) - {step})
            WHERE importance IS NOT NULL;
        """)
        duckdb.sql(
            f"COPY memory_temp TO '{self.db_path}' (FORMAT PARQUET, OVERWRITE TRUE);"
        )
        print(f"[📉 CONFIDENCE] Decayed all memories by {step}.")
