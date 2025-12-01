import re
import datetime
import time
import logging
from typing import List, Optional, Set
from .memory import MemoryStorage

logger = logging.getLogger("ARS_Retrieval")

class RetrievalSystem:
    def __init__(self, memory_system: MemoryStorage):
        self.memory = memory_system
        
        # EXPANDED STOPWORDS (Crucial for filtering "tell me about")
        self.stop_words: Set[str] = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", 
            "if", "then", "else", "when", "where", "who", "what", "how", 
            "do", "does", "did", "can", "could", "should", "would", "to", "from",
            "of", "in", "for", "with", "about", "as", "by", "hey", "hi", "hello",
            "tyrone", "please", "thanks", "thank", "is", "are", "was", "were",
            "very", "really", "so", "much", "too", "quite", "just",
            "good", "great", "perfect", "nice", "cool", "ok", "okay", "awesome",
            "yes", "no", "sure", "right", "correct", "done", "fine",
            "stuff", "thing", "things", "something", "anything",
            "tell", "me", "you", "your", "my", "mine", "us", "we", "know", "find"
        }
        
        self.generic_terms: Set[str] = {
            "birthday", "born", "date", "time", "schedule", "plan", "detail", 
            "info", "information", "remember", "remind", "note"
        }

    def get_context_string(self, query: str, include_history: Optional[List[str]] = None) -> str:
        logger.debug(f"Building context for: '{query}'")
        start_t = time.time()
        
        context_lines = ["### SYSTEM CONTEXT ###"]
        context_lines.append(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_lines.append("Location: Cape Town, Western Cape, South Africa") 

        keywords = self._extract_keywords_fast(query)
        logger.debug(f"Keywords: {keywords}")
        
        # --- Handle Trivial Queries ---
        if not keywords:
            logger.debug("Trivial query — skipping heavy search")
            if include_history:
                context_lines.append("\n### RELEVANT CONVERSATION HISTORY ###")
                context_lines.extend(include_history[-1:]) 
            return "\n".join(context_lines)
        
        # --- Memory Retrieval (RAG) ---
        facts = self._retrieve_deterministic(keywords)
        logger.debug(f"Facts found: {len(facts)}")
        
        # --- NEW LOGIC: Strengthen Vector Query ---
        vector_query = query
        if facts:
            # Join relevant facts (avoiding overly long ones) to strengthen the query sent to the embedder.
            fact_context = " ".join([f for f in facts if len(f) < 200]) 
            vector_query = f"{query}. CONTEXT HINT: {fact_context}"
            logger.debug("Query strengthened by facts for vector search.")
        
        limit = 1 if facts else 3
        logger.debug("Requesting vectors...")
        vec_start = time.time()
        # Use the potentially strengthened query for embedding
        vectors = self.memory.search_similar(vector_query, limit=limit, threshold=0.35)
        logger.debug(f"Vectors: {len(vectors)} ({time.time() - vec_start:.2f}s)")

        if facts or vectors:
            context_lines.append("\n### RELEVANT MEMORY (FACTS) ###")
            seen_hashes = set()
            chars_added = 0
            MAX_CHARS = 2000 

            if facts:
                for f in facts:
                    clean_f = f.replace('\n', ' ')
                    h = hash(clean_f)
                    if h not in seen_hashes and chars_added < MAX_CHARS:
                        context_lines.append(f"- [FACT] {clean_f}")
                        seen_hashes.add(h)
                        chars_added += len(clean_f)
            
            if vectors:
                for v in vectors:
                    text = v['text']
                    clean_v = text.replace('\n', ' ')
                    h = hash(clean_v)
                    if h not in seen_hashes and chars_added < MAX_CHARS:
                        context_lines.append(f"- [MEMORY] {clean_v}")
                        seen_hashes.add(h)
                        chars_added += len(clean_v)
        else:
            context_lines.append("\n(No specific relevant memories found)")

        # --- Semantic History Filtering ---
        if include_history:
            # The last two entries are the immediately preceding Assistant response and User question.
            # We MUST include them for immediate follow-up context.
            guaranteed_context = include_history[-2:] if len(include_history) >= 2 else include_history

            # The rest of the history is subject to filtering
            history_to_filter = include_history[:-2]
            
            # 1. Separate content from role prefix for embedding
            history_contents = [line.split(': ', 1)[-1] for line in history_to_filter]
                
            # 2. Calculate similarities
            similarities = self.memory.calculate_similarities(query, history_contents)
            
            filtered_history = guaranteed_context[:] # Start with the two guaranteed recent turns
            THRESHOLD = 0.70 
            
            # 3. Filter and rebuild history list
            for i, sim in enumerate(similarities):
                if sim >= THRESHOLD:
                    # Append turns from the past that are still semantically relevant
                    filtered_history.append(history_to_filter[i])
            
            # 4. Limit the final history to the last 5 relevant turns (maintaining recency focus)
            if filtered_history:
                context_lines.append("\n### RELEVANT CONVERSATION HISTORY ###")
                # We still limit to 5 overall, but the two most recent are prioritized within that limit.
                context_lines.extend(filtered_history[-5:])

        logger.debug(f"Context built ({time.time() - start_t:.2f}s)")
        return "\n".join(context_lines)

    def _retrieve_deterministic(self, keywords: List[str]) -> List[str]:
        results = []
        strong_keywords = [k for k in keywords if k[0].isupper() and k.lower() not in self.generic_terms]
        
        if strong_keywords:
            search_terms = strong_keywords
            logger.debug(f"Strategy: Specific Entities Only {search_terms}")
        else:
            search_terms = keywords
            logger.debug(f"Strategy: Broad Search {search_terms}")

        for kw in search_terms:
            # 1. Get Triples
            triples = self.memory.get_triples(kw)
            for t in triples[:3]:
                # Convert ('cornelia', 'has_birthday', '22 november') -> "cornelia has_birthday 22 november"
                if isinstance(t, (tuple, list)):
                    results.append(f"{t[0]} {t[1]} {t[2]}")
                else:
                    results.append(str(t))

            # 2. Get Exact Matches
            text_matches = self.memory.search_exact(kw, limit=3)
            for tm in text_matches:
                results.append(tm['text'])

        return results

    def _extract_keywords_fast(self, text: str) -> List[str]:
        # 1. Strip possessives
        text = re.sub(r"'s\b", "", text, flags=re.IGNORECASE)
        
        # 2. Clean text BUT KEEP DOTS AND HYPHENS (Fix for kalahari.net)
        # We allow alphanumeric, spaces, dots, and hyphens
        clean_text = re.sub(r'[^\w\s\.\-]', '', text)
        
        tokens = clean_text.split()
        keywords = []
        
        for t in tokens:
            # Strip trailing dots (e.g. "sentence end.")
            t = t.strip('.')
            
            if not t: continue
            
            # 3. Filter
            if (t[0].isupper() or len(t) > 2) and t.lower() not in self.stop_words:
                keywords.append(t)
                
        return list(set(keywords))
