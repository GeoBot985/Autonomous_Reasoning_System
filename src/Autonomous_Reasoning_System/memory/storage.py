from email.mime import text
import duckdb
import pandas as pd
from datetime import datetime
from uuid import uuid4
import os
import numpy as np
import threading
from pathlib import Path


from Autonomous_Reasoning_System.memory.singletons import (
    get_embedding_model,
    get_vector_store,
)

# 🔒 Global DuckDB lock (shared across all MemoryStorage instances)
GLOBAL_DUCKDB_LOCK = threading.RLock()


class MemoryStorage:
    """
    Handles structured (symbolic) memory using DuckDB + Parquet backend.
    Automatically embeds each new memory and keeps FAISS index synced.
    """

    def __init__(self, db_path: str = None):
        """
        Automatically detect backend file:
        - Prefer Parquet if it exists
        - Fallback to DuckDB if found
        - Create new Parquet if nothing exists
        """
        base_dir = Path("data")
        base_dir.mkdir(exist_ok=True)

        # Determine file paths
        parquet_path = base_dir / "memory.parquet"
        duckdb_path = base_dir / "memory_store.duckdb"

        # Decide which backend to use
        if db_path:
            self.db_path = db_path
            backend = "custom"
        elif parquet_path.exists():
            self.db_path = str(parquet_path)
            backend = "parquet"
        elif duckdb_path.exists():
            self.db_path = str(duckdb_path)
            backend = "duckdb"
        else:
            self.db_path = str(parquet_path)
            backend = "new_parquet"

        print(f"🧠 Using {backend.upper()} backend → {self.db_path}")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()

        # 🔤 Initialize embedding + vector systems
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()

        # Ensure table/file structure
        self._ensure_table()

        # 🧩 Attempt to rebuild FAISS index if missing or empty
        self._rebuild_vector_index_if_needed()

    # ------------------------------------------------------------------
    def _ensure_table(self):
        """Ensure the storage file exists and matches expected schema."""
        with GLOBAL_DUCKDB_LOCK:
            # Handle Parquet vs DuckDB file types
            if self.db_path.endswith(".duckdb"):
                con = duckdb.connect(self.db_path)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS memory (
                        id VARCHAR,
                        text VARCHAR,
                        memory_type VARCHAR,
                        created_at TIMESTAMP,
                        last_accessed TIMESTAMP,
                        importance DOUBLE,
                        scheduled_for TIMESTAMP,
                        status VARCHAR,
                        source VARCHAR
                    )
                """)
                con.close()
            else:
                if not os.path.exists(self.db_path):
                    duckdb.sql("""
                        CREATE TABLE memory (
                            id VARCHAR,
                            text VARCHAR,
                            memory_type VARCHAR,
                            created_at TIMESTAMP,
                            last_accessed TIMESTAMP,
                            importance DOUBLE,
                            scheduled_for TIMESTAMP,
                            status VARCHAR,
                            source VARCHAR
                        )
                    """)
                    duckdb.sql(f"COPY memory TO '{self.db_path}' (FORMAT PARQUET)")
                else:
                    # Validate schema and backfill columns if needed
                    duckdb.sql(f"""
                        CREATE OR REPLACE TABLE memory AS
                        SELECT * FROM read_parquet('{self.db_path}')
                    """)
                    cols = duckdb.sql("PRAGMA table_info(memory)").df()
                    if "status" not in cols["name"].tolist():
                        duckdb.sql("ALTER TABLE memory ADD COLUMN status VARCHAR")
                    if "source" not in cols["name"].tolist():
                        duckdb.sql("ALTER TABLE memory ADD COLUMN source VARCHAR DEFAULT 'unknown'")
                    duckdb.sql(f"COPY (SELECT * FROM memory) TO '{self.db_path}' (FORMAT PARQUET)")

    # ------------------------------------------------------------------
    def _escape(self, text: str) -> str:
        return text.replace("'", "''") if text else text

    # ------------------------------------------------------------------
    def _rebuild_vector_index_if_needed(self, force=False):
        """
        Rebuild FAISS index from stored memories if missing, empty, or forced.
        Ensures it uses the correct data source (the Parquet file).
        """
        try:
            if force or len(self.vector_store.metadata) == 0:
                print("🔁 Rebuilding FAISS index from stored memories (forced rebuild)...")

                with GLOBAL_DUCKDB_LOCK:
                    df = duckdb.sql(f"""
                        SELECT id, text, memory_type, source
                        FROM read_parquet('{self.db_path}')
                    """).df()

                # Clear existing FAISS + metadata
                if hasattr(self.vector_store, "reset"):
                    self.vector_store.reset()
                else:
                    # brute-force delete stale FAISS + metadata
                    for f in ["data/vector_index.faiss", "data/vector_meta.pkl"]:
                        try:
                            os.remove(f)
                        except FileNotFoundError:
                            pass
                    # re-instantiate VectorStore cleanly
                    from Autonomous_Reasoning_System.memory.vector_store import VectorStore
                    self.vector_store = get_vector_store()

                added = 0
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
                    added += 1

                self.vector_store.save()
                print(f"✅ Rebuilt FAISS index with {added} entries from {self.db_path}")

        except Exception as e:
            print(f"[WARN] Could not rebuild FAISS index: {e}")


    # ------------------------------------------------------------------
    def add_memory(
        self,
        text,
        memory_type: str = "note",
        importance: float = 0.5,
        source: str = "unknown",
        scheduled_for: str | None = None,
    ):
        """Insert memory into DuckDB and automatically embed it (optionally scheduled)."""
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            new_id = str(uuid4())
            now_str = datetime.utcnow().isoformat()
            escaped_text = self._escape(text)
            escaped_type = self._escape(memory_type)
            escaped_source = self._escape(source)
            sched_str = f"'{scheduled_for}'" if scheduled_for else "NULL"

            duckdb.sql(f"""
                CREATE OR REPLACE TABLE memory_temp AS
                SELECT * FROM read_parquet('{self.db_path}');
            """)

            duckdb.sql(f"""
                INSERT INTO memory_temp (
                    id, text, memory_type, created_at, last_accessed,
                    importance, scheduled_for, status, source
                )
                VALUES (
                    '{new_id}', '{escaped_text}', '{escaped_type}',
                    '{now_str}', '{now_str}', {importance},
                    {sched_str}, NULL, '{escaped_source}'
                );
            """)
            duckdb.sql(f"COPY memory_temp TO '{self.db_path}' (FORMAT 'parquet', OVERWRITE TRUE);")

        # 2️⃣ Generate embedding + update vector store (no DB lock needed)
        try:
            vec = self.embedder.embed(text)
            self.vector_store.add(new_id, text, vec, {"memory_type": memory_type, "source": source})
            self.vector_store.save()
            print(f"🧩 Embedded + stored memory ({source}): {text[:50]}...")
        except Exception as e:
            print(f"[WARN] Could not embed text: {e}")

        return new_id

    # ------------------------------------------------------------------
    def get_all_memories(self):
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            try:
                return duckdb.sql(f"SELECT * FROM read_parquet('{self.db_path}')").df()
            except Exception as e:
                print(f"[MemoryStorage] Error reading memories: {e}")
                return pd.DataFrame()

    # ------------------------------------------------------------------
    def search_memory(self, query_text: str):
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            if not query_text or not str(query_text).strip():
                return duckdb.sql(f"SELECT * FROM read_parquet('{self.db_path}') LIMIT 0").fetchdf()
            escaped_query = self._escape(query_text)
            df = duckdb.sql(f"""
                SELECT * FROM read_parquet('{self.db_path}')
                WHERE text ILIKE '%{escaped_query}%'
            """).fetchdf()
            return df

    # ------------------------------------------------------------------
    def search_text(self, query: str, top_k: int = 3):
        """Keyword-based search fallback."""
        try:
            all_mems = self.get_all_memories()
            results = []
            for _, m in all_mems.iterrows():
                text = m["text"]
                if query.lower() in text.lower():
                    results.append((text, 1.0))
            if not results:
                for _, m in all_mems.head(top_k).iterrows():
                    results.append((m["text"], 0.1))
            return results[:top_k]
        except Exception as e:
            print(f"[MemoryStorage] search_text failed: {e}")
            return []

    # ------------------------------------------------------------------
    def delete_memory(self, phrase: str):
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            escaped = self._escape(phrase)
            duckdb.sql(f"""
                CREATE OR REPLACE TABLE memory_temp AS
                SELECT * FROM read_parquet('{self.db_path}')
                WHERE text NOT ILIKE '%{escaped}%';
            """)
            duckdb.sql(f"""
                COPY memory_temp TO '{self.db_path}' (FORMAT PARQUET, OVERWRITE TRUE);
            """)
        return True
