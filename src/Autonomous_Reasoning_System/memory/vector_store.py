# Autonomous_Reasoning_System/memory/vector_store.py
"""
DuckDB VSS-backed vector store to keep embeddings and text in the same ACID-safe DB.
"""

import duckdb
import numpy as np
import json
from typing import Optional, Dict, Any, List
from Autonomous_Reasoning_System.infrastructure import config


class DuckVSSVectorStore:
    def __init__(self, db_path: Optional[str] = None, dim: int = 384):
        self.db_path = db_path or config.MEMORY_DB_PATH
        self.dim = dim
        self.con = duckdb.connect(self.db_path)

        # Ensure VSS extension is available
        self.con.execute("INSTALL vss;")
        self.con.execute("LOAD vss;")

        # Create table + HNSW index
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vectors (
                id VARCHAR PRIMARY KEY,
                embedding FLOAT[{self.dim}],
                text VARCHAR,
                meta JSON
            )
            """
        )
        self.con.execute(
            """
            CREATE INDEX IF NOT EXISTS vector_idx
            ON vectors USING HNSW (embedding)
            WITH (metric = 'cosine');
            """
        )

    # ------------------------------------------------------------------
    def add(self, uid: str, text: str, vector: np.ndarray, meta: Dict[str, Any] | None = None):
        """Insert or update a vector entry."""
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)

        payload = (
            uid,
            vector[0].astype(np.float32).tolist(),
            text,
            json.dumps(meta or {}),
        )
        self.con.execute(
            """
            INSERT INTO vectors (id, embedding, text, meta)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE
            SET embedding = excluded.embedding,
                text = excluded.text,
                meta = excluded.meta
            """,
            payload,
        )

    def soft_delete(self, uid: str):
        """Remove an entry by id."""
        self.con.execute("DELETE FROM vectors WHERE id = ?", (uid,))
        return True

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search by embedding vector."""
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        q = query_vec[0].astype(np.float32).tolist()
        rows = self.con.execute(
            """
            SELECT id, text, meta, embedding <=> ?::FLOAT[] AS distance
            FROM vectors
            ORDER BY distance
            LIMIT ?
            """,
            (q, k),
        ).fetchall()

        results = []
        for rid, text, meta, distance in rows:
            # Convert distance to similarity score (cosine distance in [0,2])
            score = 1 - float(distance)
            try:
                meta_obj = json.loads(meta) if meta else {}
            except Exception:
                meta_obj = {}
            results.append({"id": rid, "text": text, "score": score, **meta_obj})
        return results

    def reset(self):
        """Clear all vectors."""
        self.con.execute("DELETE FROM vectors;")

    def close(self):
        self.con.close()
