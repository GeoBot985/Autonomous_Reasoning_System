
import re
import numpy as np
import concurrent.futures
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel
from Autonomous_Reasoning_System.tools.entity_extractor import EntityExtractor


class RetrievalOrchestrator:
    """
    Unified retrieval orchestrator:
    - deterministic → source/memory_type matching using extracted entities
    - semantic → vector similarity
    - parallel execution with prioritization
    """

    def __init__(self, memory_storage=None, embedding_model=None):
        self.memory = memory_storage
        self.embedder = embedding_model or EmbeddingModel()
        self.entity_extractor = EntityExtractor()

    # ---------------------------------------------------
    def retrieve(self, query: str):
        if not self.memory:
            return []

        print(f"🧭 Starting Parallel Hybrid Retrieval for: '{query}'")

        # 1. Extract Entities (Blocking call, fast)
        keywords = self.entity_extractor.extract(query)
        print(f"🔑 Extracted keywords: {keywords}")

        # 2. Parallel Execution of Deterministic and Semantic Search
        det_results = []
        sem_results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_det = executor.submit(self._search_deterministic, keywords)
            future_sem = executor.submit(self._search_semantic, query)

            det_results = future_det.result()
            sem_results = future_sem.result()

        # 3. Prioritization Logic
        # det_results is a list of tuples: (text, score)

        # Check for high-confidence deterministic match
        if det_results:
            best_det = det_results[0]
            # If score >= 0.9 (which we set to 1.0 in storage.py), prioritize it
            if best_det[1] >= 0.9:
                print(f"✅ High-confidence deterministic match found: '{best_det[0][:50]}...'")
                # Return just the text of the matches
                return [r[0] for r in det_results]

        print("⚠️ No high-confidence deterministic match. Falling back to hybrid ranking.")

        # 4. Fallback: Combine and Rank
        # Flatten lists
        combined_texts = set()
        final_results = []

        # Add deterministic results (lower priority than exclusive but still relevant)
        for text, score in det_results:
            if text not in combined_texts:
                combined_texts.add(text)
                final_results.append(text)

        # Add semantic results
        for text in sem_results:
            if text not in combined_texts:
                combined_texts.add(text)
                final_results.append(text)

        # Return top 5 unique results
        return final_results[:5]

    # ---------------------------------------------------
    def _search_deterministic(self, keywords: list[str]):
        """Executes the high-integrity lookup."""
        if not keywords:
            return []
        try:
            # Call the updated search_text in storage.py
            # storage.search_text returns [(text, score), ...]
            results = self.memory.search_text(keywords, top_k=3)
            print(f"🔍 Deterministic search found {len(results)} matches.")
            return results
        except Exception as e:
            print(f"[ERROR] Deterministic search failed: {e}")
            return []

    # ---------------------------------------------------
    def _search_semantic(self, query: str, k: int = 5):
        """Vector-based semantic retrieval."""
        try:
            q_vec = self.embedder.embed(query)

            # Try retrieving from DuckDB VSS if available
            if hasattr(self.memory, "vector_store") and self.memory.vector_store:
                results = self.memory.vector_store.search(np.array(q_vec), k)
                texts = [r.get("text") for r in results if r.get("text")]
                print(f"🧠 Semantic search found {len(texts)} matches.")
                return texts

            # Fallback if no vector store (shouldn't happen in prod config)
            print("⚠️ Semantic search skipped (no vector store).")
            return []

        except Exception as e:
            print(f"[ERROR] Semantic search failed: {e}")
            return []
