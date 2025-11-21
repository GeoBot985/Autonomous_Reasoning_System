from email.mime import text
import duckdb
import pandas as pd
from datetime import datetime
from uuid import uuid4
import os
from Autonomous_Reasoning_System.infrastructure import config

class MemoryStorage:
    """
    Handles structured (symbolic) memory using persistent DuckDB.
    """

    def __init__(self, db_path=None, embedding_model=None, vector_store=None):
        """
        Initialize with persistent connection.
        """
        self.db_path = db_path or config.MEMORY_DB_PATH

        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Initialize persistent connection
        self.con = duckdb.connect(self.db_path)

        # Initialize Schema
        self.init_db()

        # 🔤 Initialize embedding + vector systems (Injected or None)
        # If they are None here, they should be passed in or handled by MemoryInterface/Manager.
        # For backward compatibility or if not injected, we might need a way to get them,
        # but strictly following instructions: we kill singletons.
        # So we assume they are passed in.
        self.embedder = embedding_model
        self.vector_store = vector_store

        if not self.embedder:
             print("[WARN] MemoryStorage initialized without embedding_model. Vector search will fail.")
        if not self.vector_store:
             print("[WARN] MemoryStorage initialized without vector_store. Vector search will fail.")

    def init_db(self):
        """Create tables if they don't exist."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id VARCHAR PRIMARY KEY,
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

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id VARCHAR PRIMARY KEY,
                text VARCHAR,
                priority INTEGER,
                status VARCHAR,
                steps VARCHAR,
                metadata VARCHAR,
                plan_id VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

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
        """Insert memory into DuckDB and embed it."""
        new_id = str(uuid4())
        now_str = datetime.utcnow().isoformat()
        # sched_str handled via param query or manual string if using execute params

        self.con.execute("""
            INSERT INTO memory (
                id, text, memory_type, created_at, last_accessed,
                importance, scheduled_for, status, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_id, text, memory_type, now_str, now_str,
            importance, scheduled_for, None, source
        ))

        # 2️⃣ Generate embedding + update vector store
        if self.embedder and self.vector_store:
            try:
                vec = self.embedder.embed(text)
                self.vector_store.add(new_id, text, vec, {"memory_type": memory_type, "source": source})
                print(f"🧩 Embedded memory ({source}): {text[:50]}...")
            except Exception as e:
                print(f"[WARN] Could not embed text: {e}")

        return new_id

    # ------------------------------------------------------------------
    def get_all_memories(self) -> pd.DataFrame:
        try:
            return self.con.execute("SELECT * FROM memory").df()
        except Exception as e:
            print(f"[MemoryStorage] Error reading memories: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def search_memory(self, query_text: str):
        if not query_text or not str(query_text).strip():
            return pd.DataFrame()
        # Use param to prevent SQL injection, though search pattern needs concat
        escaped_query = f"%{query_text}%"
        return self.con.execute("""
            SELECT * FROM memory
            WHERE text ILIKE ?
        """, (escaped_query,)).df()

    # ------------------------------------------------------------------
    def search_text(self, query: str, top_k: int = 3):
        """Keyword-based search fallback."""
        try:
            escaped_query = f"%{query}%"
            res = self.con.execute("""
                SELECT text FROM memory
                WHERE text ILIKE ?
                LIMIT ?
            """, (escaped_query, top_k)).fetchall()

            results = [(r[0], 1.0) for r in res]
            return results
        except Exception as e:
            print(f"[MemoryStorage] search_text failed: {e}")
            return []

    # ------------------------------------------------------------------
    def delete_memory(self, phrase: str):
        escaped = f"%{phrase}%"
        self.con.execute("DELETE FROM memory WHERE text ILIKE ?", (escaped,))
        return True

    # ------------------------------------------------------------------
    def get_due_reminders(self, lookahead_minutes: int = 5) -> pd.DataFrame:
        """
        Fetch reminders that are due now or within lookahead window.
        """
        try:
             now_str = datetime.utcnow().isoformat()
             # DuckDB date arithmetic might vary, simpler to just select all pending reminders and filter in python if complex
             # But we can try simple timestamp comparison if formats are ISO.
             # However, ISO strings are comparable.

             # We assume scheduled_for is ISO string.
             # We want scheduled_for <= now + lookahead
             # Since we store as strings, string comparison works for ISO format if timezone is consistent (UTC).

             # Calculating lookahead timestamp in python
             limit_time = (datetime.utcnow() + pd.Timedelta(minutes=lookahead_minutes)).isoformat()

             # Check status too if we had one for completion?
             # The schema has 'status'. We assume 'pending' or NULL is active.
             # But add_memory sets status to None.

             return self.con.execute("""
                SELECT * FROM memory
                WHERE memory_type IN ('task', 'reminder')
                  AND scheduled_for IS NOT NULL
                  AND scheduled_for <= ?
                  AND (status IS NULL OR status != 'completed')
             """, (limit_time,)).df()
        except Exception as e:
             print(f"[MemoryStorage] Error fetching due reminders: {e}")
             return pd.DataFrame()

    # ------------------------------------------------------------------
    def update_memory(self, uid: str, new_text: str):
        """
        Update memory text by ID in DuckDB.
        """
        if not uid or not new_text:
            print("[MemoryStorage] Invalid update parameters.")
            return False

        # Check if ID exists first
        try:
            exists = self.con.execute("SELECT count(*) FROM memory WHERE id=?", (uid,)).fetchone()[0]
        except Exception as e:
                print(f"[MemoryStorage] Error checking memory existence: {e}")
                return False

        if exists == 0:
            print(f"[MemoryStorage] Memory ID {uid} not found.")
            return False

        now_str = datetime.utcnow().isoformat()
        self.con.execute("""
            UPDATE memory
            SET text = ?, last_accessed = ?
            WHERE id = ?
        """, (new_text, now_str, uid))

        print(f"📝 Updated memory {uid} in storage.")
        return True

    # ------------------------------------------------------------------
    # Goals Management
    # ------------------------------------------------------------------
    def add_goal(self, goal_data: dict):
        """Insert goal into DuckDB."""
        self.con.execute("""
            INSERT INTO goals (
                id, text, priority, status, steps, metadata, plan_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            goal_data['id'],
            goal_data.get('text', ''),
            goal_data.get('priority', 1),
            goal_data.get('status', 'pending'),
            goal_data.get('steps', '[]'),
            goal_data.get('metadata', '{}'),
            goal_data.get('plan_id', None),
            str(goal_data.get('created_at')),
            str(goal_data.get('updated_at'))
        ))
        return goal_data['id']

    def get_goal(self, goal_id: str) -> dict:
        try:
            res = self.con.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).df()
            if not res.empty:
                return res.iloc[0].to_dict()
        except Exception as e:
            print(f"[MemoryStorage] Error getting goal {goal_id}: {e}")
        return None

    def get_all_goals(self) -> pd.DataFrame:
        try:
            return self.con.execute("SELECT * FROM goals").df()
        except Exception as e:
            print(f"[MemoryStorage] Error reading goals: {e}")
            return pd.DataFrame()

    def get_active_goals(self) -> pd.DataFrame:
        try:
            return self.con.execute("SELECT * FROM goals WHERE status IN ('pending', 'active')").df()
        except Exception as e:
            print(f"[MemoryStorage] Error reading active goals: {e}")
            return pd.DataFrame()

    def update_goal(self, goal_id: str, updates: dict):
        set_clauses = []
        values = []
        for k, v in updates.items():
            set_clauses.append(f"{k} = ?")
            values.append(v)

        if not set_clauses:
            return False

        values.append(goal_id)
        set_query = ", ".join(set_clauses)
        try:
            self.con.execute(f"UPDATE goals SET {set_query} WHERE id=?", tuple(values))
            return True
        except Exception as e:
            print(f"[MemoryStorage] Error updating goal {goal_id}: {e}")
            return False

    def delete_goal(self, goal_id: str):
        try:
            self.con.execute("DELETE FROM goals WHERE id=?", (goal_id,))
            return True
        except Exception as e:
            print(f"[MemoryStorage] Error deleting goal {goal_id}: {e}")
            return False
