import json
import logging
import threading
import duckdb
import time
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from fastembed import TextEmbedding

logger = logging.getLogger("ARS_Memory")

class MemorySystem:
    def __init__(self, db_path="data/memory.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[Memory] ⏳ Loading FastEmbed (CPU)...")
        start_t = time.time()
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.vector_dim = 384 
        print(f"[Memory] ✅ Model loaded ({time.time() - start_t:.2f}s)")

        self._lock = threading.RLock()
        print(f"[Memory] ⏳ Connecting to DuckDB...")
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()
        print(f"[Memory] ✅ Database Ready.")

    def _get_embedding(self, text: str) -> list:
        return list(self.embedder.embed([text]))[0].tolist()

    def _init_schema(self):
        with self._lock:
            try:
                self.con.execute("INSTALL vss; LOAD vss;") 
                self.con.execute("SET hnsw_enable_experimental_persistence = true;")
            except Exception: pass

            self.con.execute("CREATE TABLE IF NOT EXISTS memory (id VARCHAR PRIMARY KEY, text VARCHAR, memory_type VARCHAR, created_at TIMESTAMP, importance DOUBLE, source VARCHAR, metadata JSON)")
            self.con.execute(f"CREATE TABLE IF NOT EXISTS vectors (id VARCHAR PRIMARY KEY, embedding FLOAT[{self.vector_dim}], FOREIGN KEY (id) REFERENCES memory(id))")
            try: self.con.execute("CREATE INDEX IF NOT EXISTS idx_vec ON vectors USING HNSW (embedding) WITH (metric = 'cosine');")
            except: pass
            self.con.execute("CREATE TABLE IF NOT EXISTS triples (subject VARCHAR, relation VARCHAR, object VARCHAR, PRIMARY KEY(subject, relation, object))")
            self.con.execute("CREATE TABLE IF NOT EXISTS plans (id VARCHAR PRIMARY KEY, goal_text VARCHAR, steps JSON, status VARCHAR, created_at TIMESTAMP, updated_at TIMESTAMP)")

    # --- DEBUGGED BATCH METHOD ---
    def remember_batch(self, texts: list, memory_type: str = "episodic", importance: float = 0.5, source: str = "user", metadata_list: list = None):
        if not texts: return
        
        count = len(texts)
        print(f"\n[Memory-Debug] 🏁 Starting Batch Insert of {count} items...")
        total_start = time.time()
        
        # 1. Calculate Vectors
        t0 = time.time()
        print(f"[Memory-Debug]    🧠 Calculating {count} vectors on CPU...")
        # FastEmbed handles lists natively
        embeddings_generator = self.embedder.embed(texts)
        # Convert generator to list immediately to measure calculation time
        embeddings = list(embeddings_generator) 
        t_embed = time.time() - t0
        print(f"[Memory-Debug]    ✅ Vectors calculated in {t_embed:.2f}s ({(t_embed/count)*1000:.1f}ms per item)")

        # 2. DB Write
        t1 = time.time()
        print(f"[Memory-Debug]    💾 Writing to DuckDB transaction...")
        
        now = datetime.utcnow()
        
        with self._lock:
            self.con.execute("BEGIN TRANSACTION")
            try:
                # Prepare data for bulk insertion logic (loop in python, execute in SQL)
                for i, text in enumerate(texts):
                    mem_id = str(uuid4())
                    vector = embeddings[i].tolist()
                    meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                    meta_json = json.dumps(meta)

                    self.con.execute(
                        "INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?, ?)", 
                        (mem_id, text, memory_type, now, importance, source, meta_json)
                    )
                    self.con.execute(
                        "INSERT INTO vectors VALUES (?, ?)", 
                        (mem_id, vector)
                    )
                self.con.execute("COMMIT")
            except Exception as e:
                self.con.execute("ROLLBACK")
                print(f"[Memory-Debug] ❌ Batch DB Write Failed: {e}")
                raise e
        
        t_write = time.time() - t1
        print(f"[Memory-Debug]    ✅ DB Write finished in {t_write:.2f}s")
        print(f"[Memory-Debug] 🏁 Total Batch Time: {time.time() - total_start:.2f}s\n")

    # --- Standard Methods ---
    def remember(self, text: str, memory_type: str = "episodic", importance: float = 0.5, source: str = "user", metadata: dict = None):
        return self.remember_batch([text], memory_type, importance, source, [metadata] if metadata else None)

    def add_triple(self, subj, rel, obj):
        self.con.execute("INSERT OR IGNORE INTO triples VALUES (?, ?, ?)", (subj.lower(), rel.lower(), obj.lower()))

    def update_plan(self, plan_id, goal_text, steps, status="active"):
        now = datetime.utcnow()
        steps_json = json.dumps(steps)
        with self._lock:
            if self.con.execute("SELECT 1 FROM plans WHERE id=?", (plan_id,)).fetchone():
                self.con.execute("UPDATE plans SET steps=?, status=?, updated_at=? WHERE id=?", (steps_json, status, now, plan_id))
            else:
                self.con.execute("INSERT INTO plans VALUES (?, ?, ?, ?, ?, ?)", (plan_id, goal_text, steps_json, status, now, now))

    def search_similar(self, query: str, limit: int = 5, threshold: float = 0.4):
        query_vec = self._get_embedding(query)
        with self._lock:
            results = self.con.execute(f"""
                SELECT substr(m.text, 1, 500), m.memory_type, m.created_at, (1 - list_cosine_similarity(v.embedding, ?::FLOAT[{self.vector_dim}])) as score
                FROM vectors v
                JOIN memory m ON v.id = m.id
                ORDER BY score DESC LIMIT ?
            """, (query_vec, limit)).fetchall()
            return [{"text": r[0], "type": r[1], "date": r[2], "score": r[3]} for r in results if r[3] >= threshold]

    def search_exact(self, keyword: str, limit: int = 5):
        pattern = f"%{keyword}%"
        with self._lock:
            results = self.con.execute("SELECT substr(text, 1, 500), memory_type, created_at FROM memory WHERE text ILIKE ? ORDER BY created_at DESC LIMIT ?", (pattern, limit)).fetchall()
        return [{"text": r[0], "type": r[1], "date": r[2], "score": 1.0} for r in results]

    def get_triples(self, entity: str):
        with self._lock:
            res = self.con.execute("SELECT subject, relation, object FROM triples WHERE subject=? OR object=?", (entity.lower(), entity.lower())).fetchall()
        return res
        
    def get_active_plans(self):
        with self._lock:
            res = self.con.execute("SELECT * FROM plans WHERE status = 'active'").fetchall()
        return [{"id": r[0], "goal": r[1], "steps": json.loads(r[2]), "status": r[3]} for r in res]

    def get_recent_memories(self, limit=10):
        with self._lock:
            res = self.con.execute("SELECT text FROM memory ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in res]

def get_memory_system(db_path="data/memory.duckdb"):
    return MemorySystem(db_path)