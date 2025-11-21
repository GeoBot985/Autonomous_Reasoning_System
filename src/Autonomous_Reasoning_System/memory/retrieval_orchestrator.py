
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

        # KG Enhancement: Expand semantic hits
        expanded_results = set()

        # Collect potential entities from semantic hits to expand
        potential_entities = set()
        # Add query keywords too
        potential_entities.update(keywords)

        for text in sem_results:
             expanded_results.add(text)
             # Extract entities from the text using our extractor
             # This fulfills "For each high-score hit, lookup its corresponding entity"
             hit_keywords = self.entity_extractor.extract(text)
             potential_entities.update(hit_keywords)

        # Limit expansion to avoid blowing up context
        # Take top 5 entities found
        target_entities = list(potential_entities)[:5]

        kg_context = self._search_kg(target_entities)
        for triple in kg_context:
             expanded_results.add(f"Fact: {triple}")

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

        # Add KG results
        for text in kg_context:
            # Format triple as text
            formatted = f"Fact: {text[0]} {text[1]} {text[2]}"
            if formatted not in combined_texts:
                combined_texts.add(formatted)
                final_results.append(formatted)

        # Return top 5 unique results
        return final_results[:5]

    # ---------------------------------------------------
    def _search_kg(self, keywords: list[str]):
        """Look up KG neighbors for keywords."""
        if not keywords:
            return []
        try:
             results = []
             # Determine if self.memory is MemoryStorage or MemoryInterface
             # MemoryStorage has _lock, MemoryInterface has storage._lock
             storage = self.memory
             if hasattr(self.memory, "storage"):
                 storage = self.memory.storage

             if not hasattr(storage, "_lock"):
                 print("[WARN] Storage object does not have expected lock structure.")
                 return []

             with storage._lock: # Use read lock
                 for kw in keywords:
                     # Find triples where keyword is subject or object
                     # We use ILIKE for loose matching
                     query = f"%{kw}%"
                     rows = storage.con.execute("""
                        SELECT subject, relation, object FROM triples
                        WHERE subject ILIKE ? OR object ILIKE ?
                        LIMIT 3
                     """, (query, query)).fetchall()
                     results.extend(rows)
             return results
        except Exception as e:
            print(f"[ERROR] KG search failed: {e}")
            return []

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
