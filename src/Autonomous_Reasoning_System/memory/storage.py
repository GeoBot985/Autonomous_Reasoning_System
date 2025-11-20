from email.mime import text
import duckdb
import pandas as pd
from datetime import datetime
from uuid import uuid4
import threading


from Autonomous_Reasoning_System.memory.singletons import (
    get_embedding_model,
    get_vector_store,
)

# 🔒 Global DuckDB lock (shared across all MemoryStorage instances)
GLOBAL_DUCKDB_LOCK = threading.RLock()


class MemoryStorage:
    """
    Handles structured (symbolic) memory using in-memory DuckDB.
    Persistence is now handled by MemoryInterface via PersistenceService.
    """

    def __init__(self, initial_df: pd.DataFrame = None):
        """
        Initialize with an optional DataFrame.
        No longer touches disk directly on init.
        """
        self._lock = threading.Lock()

        # Initialize connection
        self.con = duckdb.connect(database=':memory:')

        if initial_df is None or initial_df.empty:
            self.con.execute("""
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
        else:
            self.con.register('initial_memories', initial_df)
            self.con.execute("CREATE TABLE memory AS SELECT * FROM initial_memories")
            self.con.unregister('initial_memories')

        # Ensure columns exist if loading from a simpler schema
        self._ensure_schema()

        # 🔤 Initialize embedding + vector systems
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()

    # ------------------------------------------------------------------
    def _ensure_schema(self):
        """Ensure the in-memory table matches expected schema."""
        cols = self.con.execute("PRAGMA table_info(memory)").df()
        existing_cols = cols["name"].tolist()

        required_cols = {
            "status": "VARCHAR",
            "source": "VARCHAR DEFAULT 'unknown'",
            "scheduled_for": "TIMESTAMP"
        }

        for col, dtype in required_cols.items():
            if col not in existing_cols:
                self.con.execute(f"ALTER TABLE memory ADD COLUMN {col} {dtype}")

    # ------------------------------------------------------------------
    def _escape(self, text: str) -> str:
        return text.replace("'", "''") if text else text

    # ------------------------------------------------------------------
    def add_memory(
        self,
        text,
        memory_type: str = "note",
        importance: float = 0.5,
        source: str = "unknown",
        scheduled_for: str | None = None,
    ):
        """Insert memory into in-memory DuckDB and embed it."""
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            new_id = str(uuid4())
            now_str = datetime.utcnow().isoformat()
            escaped_text = self._escape(text)
            escaped_type = self._escape(memory_type)
            escaped_source = self._escape(source)
            sched_str = f"'{scheduled_for}'" if scheduled_for else "NULL"

            self.con.execute(f"""
                INSERT INTO memory (
                    id, text, memory_type, created_at, last_accessed,
                    importance, scheduled_for, status, source
                )
                VALUES (
                    '{new_id}', '{escaped_text}', '{escaped_type}',
                    '{now_str}', '{now_str}', {importance},
                    {sched_str}, NULL, '{escaped_source}'
                );
            """)

        # 2️⃣ Generate embedding + update vector store
        # Note: Vector store update is still direct for now, but persistence happens at Interface level
        try:
            vec = self.embedder.embed(text)
            self.vector_store.add(new_id, text, vec, {"memory_type": memory_type, "source": source})
            print(f"🧩 Embedded memory ({source}): {text[:50]}...")
        except Exception as e:
            print(f"[WARN] Could not embed text: {e}")

        return new_id

    # ------------------------------------------------------------------
    def get_all_memories(self) -> pd.DataFrame:
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            try:
                return self.con.execute("SELECT * FROM memory").df()
            except Exception as e:
                print(f"[MemoryStorage] Error reading memories: {e}")
                return pd.DataFrame()

    # ------------------------------------------------------------------
    def search_memory(self, query_text: str):
        with GLOBAL_DUCKDB_LOCK:  # 🔒
            if not query_text or not str(query_text).strip():
                return pd.DataFrame()
            escaped_query = self._escape(query_text)
            return self.con.execute(f"""
                SELECT * FROM memory
                WHERE text ILIKE '%{escaped_query}%'
            """).df()

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
            self.con.execute(f"""
                DELETE FROM memory
                WHERE text ILIKE '%{escaped}%';
            """)
        return True

    # ------------------------------------------------------------------
    def update_memory(self, uid: str, new_text: str):
        """
        Update memory text by ID in DuckDB.
        """
        if not uid or not new_text:
            print("[MemoryStorage] Invalid update parameters.")
            return False

        with GLOBAL_DUCKDB_LOCK:
            escaped_text = self._escape(new_text)

            # Check if ID exists first
            try:
                exists = self.con.execute("SELECT count(*) FROM memory WHERE id=?", [uid]).fetchone()[0]
            except Exception as e:
                 print(f"[MemoryStorage] Error checking memory existence: {e}")
                 return False

            if exists == 0:
                print(f"[MemoryStorage] Memory ID {uid} not found.")
                return False

            self.con.execute(f"""
                UPDATE memory
                SET text = '{escaped_text}', last_accessed = '{datetime.utcnow().isoformat()}'
                WHERE id = '{uid}';
            """)

        print(f"📝 Updated memory {uid} in storage.")
        return True
