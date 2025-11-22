
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

    def _clean_keywords(self, keywords: list[str]) -> list[str]:
        """Strips possessives and punctuation from keywords."""
        cleaned = []
        for kw in keywords:
            # Remove 's and punctuation
            clean = re.sub(r"['’]s$", "", kw, flags=re.IGNORECASE)
            clean = re.sub(r"[^\w\s]", "", clean)
            if clean.strip():
                cleaned.append(clean.strip())
        # Return unique
        return list(set(cleaned))

    # ---------------------------------------------------
    def retrieve(self, query: str):
        if not self.memory:
            return []

        print(f"🧭 Starting Parallel Hybrid Retrieval for: '{query}'")

        # 1. Extract Entities (Blocking call, fast)
        raw_keywords = self.entity_extractor.extract(query)
        keywords = self._clean_keywords(raw_keywords)
        print(f"🔑 Extracted keywords: {keywords} (raw: {raw_keywords})")

        # 1.5 Check for specific KG relations (e.g. birthdays)
        kg_direct_results = []
        if "birthday" in query.lower():
            print("🎂 Detected birthday query, checking KG specifically...")
            # Assume keywords contains the person's name
            birthday_facts = self._search_kg_relation(keywords, "has_birthday")
            if birthday_facts:
                 print(f"✅ Found birthday facts in KG: {birthday_facts}")
                 kg_direct_results = [f"Fact: {s} has_birthday {o}" for s, r, o in birthday_facts]

        # 2. Parallel Execution of Deterministic and Semantic Search
        det_results = []
        sem_results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_det = executor.submit(self._search_deterministic, keywords)
            future_sem = executor.submit(self._search_semantic, query)

            det_results = future_det.result()
            sem_results = future_sem.result()

        # 3. Prioritization Logic
        final_results = []
        combined_texts = set()

        # Priority 0: KG Direct Hits (Birthdays etc.)
        for text in kg_direct_results:
            if text not in combined_texts:
                combined_texts.add(text)
                final_results.append(text)

        if final_results:
             # If we found the specific fact, we might not need much else, but let's add context
             pass

        # Priority 1: High Confidence Deterministic
        # det_results is a list of tuples: (text, score)
        if det_results:
            best_det = det_results[0]
            if best_det[1] >= 0.9:
                print(f"✅ High-confidence deterministic match found: '{best_det[0][:50]}...'")
                for text, score in det_results:
                     if text not in combined_texts:
                         combined_texts.add(text)
                         final_results.append(text)

        print("⚠️ Checking KG context for semantic hits.")

        # KG Enhancement: Expand semantic hits
        expanded_results = set()

        # Collect potential entities from semantic hits to expand
        potential_entities = set()
        # Add query keywords too
        potential_entities.update(keywords)

        for text in sem_results:
             # Extract entities from the text using our extractor
             # This fulfills "For each high-score hit, lookup its corresponding entity"
             hit_keywords = self.entity_extractor.extract(text)
             if hit_keywords:
                 clean_hits = self._clean_keywords(hit_keywords)
                 potential_entities.update(clean_hits)

        # Limit expansion to avoid blowing up context
        # Take top 5 entities found
        target_entities = list(potential_entities)[:5]

        kg_context = self._search_kg(target_entities)
        for triple in kg_context:
             formatted = f"Fact: {triple[0]} {triple[1]} {triple[2]}"
             if formatted not in combined_texts:
                 # Insert at beginning if not already present, to prioritize KG
                 # But wait, we are building final_results list now.
                 # We want KG facts to be high up.
                 # If we haven't added them via Direct Hit, add them now.
                 combined_texts.add(formatted)
                 # Prepend or append? "Modify retrieval to check KG before vector" implies priority.
                 # We'll append here, but since we return final_results[:5], we should ensure they get in.
                 final_results.insert(len(kg_direct_results), formatted) # Insert after direct hits

        # 4. Fallback: Combine and Rank

        # Add semantic results (if not already added)
        for text in sem_results:
            if text not in combined_texts:
                combined_texts.add(text)
                final_results.append(text)

        # Also add remaining deterministic results if any were missed
        for text, score in det_results:
             if text not in combined_texts:
                combined_texts.add(text)
                final_results.append(text)

        # Return top 5 unique results
        return final_results[:5]

    # ---------------------------------------------------
    def _search_kg_relation(self, keywords: list[str], relation: str):
        """Look up specific relation in KG."""
        if not keywords: return []
        try:
             results = []
             storage = self.memory
             if hasattr(self.memory, "storage"):
                 storage = self.memory.storage

             if not hasattr(storage, "_lock"): return []

             with storage._lock:
                 for kw in keywords:
                     query = f"%{kw}%"
                     rows = storage.con.execute("""
                        SELECT subject, relation, object FROM triples
                        WHERE (subject ILIKE ? OR object ILIKE ?) AND relation = ?
                     """, (query, query, relation)).fetchall()
                     results.extend(rows)
             return results
        except Exception as e:
            print(f"[ERROR] KG relation search failed: {e}")
            return []

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
