
import re
import numpy as np
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel


class RetrievalOrchestrator:
    """
    Unified retrieval orchestrator:
    - deterministic → source/memory_type matching
    - semantic → vector similarity
    - hybrid → combines both
    - entity-aware → resolves named entities like "Cornelia" to the most relevant memory
    """

    def __init__(self, memory_storage=None, embedding_model=None):
        self.memory = memory_storage
        self.embedder = embedding_model or EmbeddingModel()  # reuse the same embedding model

    # ---------------------------------------------------
    def detect_intent(self, query: str) -> str:
        q = query.lower()
        if re.search(r"\b(show|open|read|list|display|report|document|file)\b", q):
            return "deterministic"
        if re.search(r"\b(summarize|combine|compare|mention)\b", q):
            return "hybrid"
        return "semantic"

    # ---------------------------------------------------
    def retrieve(self, query: str):
        if not self.memory:
            return []

        mode = self.detect_intent(query)
        print(f"🧭 Retrieval mode → {mode}")

        # 🧠 Entity disambiguation always runs first (adds context to any mode)
        entity_result = self._entity_resolution(query)
        if entity_result:
            print(f"[🧩 EntityResolution] Boosting '{entity_result[:40]}...'")
            # Return this prioritized entity first, followed by mode results
            base = self._mode_retrieve(query, mode)
            return [entity_result] + [r for r in base if r != entity_result]

        # fallback: normal mode
        return self._mode_retrieve(query, mode)

    # ---------------------------------------------------
    def _mode_retrieve(self, query: str, mode: str):
        if mode == "deterministic":
            return self._deterministic_retrieve(query)
        elif mode == "hybrid":
            return self._hybrid_retrieve(query)
        else:
            return self._semantic_retrieve(query)

    # ---------------------------------------------------
    def _deterministic_retrieve(self, query: str):
        df = self.memory.get_all_memories()
        q = query.lower()

        direct = df[df["source"].str.lower().str.contains("visionassist", na=False)]
        if not direct.empty:
            print(f"✅ Found {len(direct)} memories by source match.")
            return direct.sort_values("created_at")

        if "summary" in q:
            filtered = df[df["memory_type"].str.lower().eq("document_summary")]
            if not filtered.empty:
                print(f"✅ Found {len(filtered)} document summaries.")
                return filtered.sort_values("created_at")

        print("⚠️ No direct deterministic matches, falling back to semantic search.")
        return self._semantic_retrieve(query)

    # ---------------------------------------------------
    def _semantic_retrieve(self, query: str, k: int = 5):
        """Vector-based semantic retrieval with fallback to re-embedding."""
        try:
            q_vec = self.embedder.embed(query)

            # Try retrieving from DuckDB VSS if available
            if hasattr(self.memory, "vector_store"):
                print("🧠 Using DuckDB VSS for semantic retrieval...")
                results = self.memory.vector_store.search(np.array(q_vec), k)
                texts = [r.get("text") for r in results if r.get("text")]
                print(f"✅ Retrieved {len(texts)} semantic results.")
                return texts

            # --- fallback path if VSS not available ---
            print("⚠️ VSS missing, rebuilding temporary vector set...")
            mems = self.memory.get_all_memories()
            texts = mems["text"].tolist()
            vecs = np.array([self.embedder.embed(t) for t in texts])
            sims = np.dot(vecs, q_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q_vec))
            top_idx = np.argsort(sims)[-k:][::-1]
            results = [texts[i] for i in top_idx]
            print(f"✅ Retrieved {len(results)} semantic results (fallback).")
            return results

        except Exception as e:
            print(f"[ERROR] Semantic search failed: {e}")
            return []

    # ---------------------------------------------------
    def _hybrid_retrieve(self, query: str):
        df = self.memory.get_all_memories()
        subset = df[df["text"].str.contains(query.split()[0], case=False, na=False)]
        sem = self._semantic_retrieve(query)
        print(f"Hybrid: subset={len(subset)}, semantic={len(sem)}")
        return {"subset": subset, "semantic": sem}

    # ---------------------------------------------------
    def _entity_resolution(self, query: str):
        """Detects if a named entity (like 'Cornelia') exists and returns the best memory."""
        df = self.memory.get_all_memories()
        if df.empty:
            return None

        # detect capitalized name
        name_match = re.search(r"\b[A-Z][a-z]{2,}\b", query)
        if not name_match:
            return None

        name = name_match.group(0)
        subset = df[df["text"].str.contains(name, case=False, na=False)]
        if subset.empty:
            return None

        # Score each mention
        def score_row(row):
            t = str(row["text"]).lower()
            score = 0.0
            if name.lower() in t:
                score += 0.4
            if any(w in t for w in ["my ", "our ", "wife", "husband", "daughter", "son"]):
                score += 0.3
            overlap = len(set(query.lower().split()) & set(t.split()))
            score += min(0.3, overlap / 50.0)
            return score

        # Use apply but check if 'score' column exists or copy slice to avoid warning
        subset = subset.copy()
        subset["score"] = subset.apply(score_row, axis=1)
        best = subset.sort_values("score", ascending=False).iloc[0]
        print(f"[🧩 EntityResolution] '{name}' → best match score={best['score']:.3f}")
        return name
