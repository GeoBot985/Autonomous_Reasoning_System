import re
import datetime
import time
import logging
from typing import List
from .memory import MemorySystem

logger = logging.getLogger("ARS_Retrieval")

class RetrievalSystem:
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        
        # 1. Stop Words
        self.stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", 
            "if", "then", "else", "when", "where", "who", "what", "how", 
            "do", "does", "did", "can", "could", "should", "would", "to", "from",
            "of", "in", "for", "with", "about", "as", "by", "hey", "hi", "hello",
            "tyrone", "please", "thanks", "are", "was", "were"
        }
        
        # 2. Generic Terms (Ignored for Proper Noun searches)
        self.generic_terms = {
            "birthday", "born", "date", "time", "schedule", "plan", "detail", 
            "info", "information", "remember", "remind", "note"
        }

    def get_context_string(self, query: str, include_history: List[str] = None) -> str:
        print(f"[Retrieval] 🟢 Building context for: '{query}'")
        start_t = time.time()
        
        context_lines = ["### SYSTEM CONTEXT ###"]
        context_lines.append(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_lines.append("Location: Cape Town, Western Cape, South Africa") 

        # 1. Keywords
        keywords = self._extract_keywords_fast(query)
        print(f"[Retrieval]    🔑 Keywords: {keywords}")
        
        if not keywords:
            print(f"[Retrieval]    🚀 Trivial query — skipping heavy search")
            return "\n".join(context_lines)
        
        # 2. Deterministic Search
        t_det = time.time()
        facts = self._retrieve_deterministic(keywords)
        print(f"[Retrieval]    📂 Facts found: {len(facts)} ({time.time() - t_det:.2f}s)")
        
        # DEBUG: Show what we actually found
        for i, f in enumerate(facts):
            print(f"       👉 [Fact {i}] {f}")
        
        # 3. Semantic Search
        limit = 1 if facts else 3
        print(f"[Retrieval]    🧠 Requesting vectors...")
        vec_start = time.time()
        vectors = self.memory.search_similar(query, limit=limit, threshold=0.35)
        print(f"[Retrieval]    ✅ Vectors: {len(vectors)} ({time.time() - vec_start:.2f}s)")

        # 4. Format
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

        if include_history:
            context_lines.append("\n### RECENT CONVERSATION ###")
            context_lines.extend(include_history[-5:]) 

        print(f"[Retrieval] 🏁 Context built ({time.time() - start_t:.2f}s)")
        return "\n".join(context_lines)

    def _retrieve_deterministic(self, keywords: List[str]) -> List[str]:
        results = []
        strong_keywords = [k for k in keywords if k[0].isupper() and k.lower() not in self.generic_terms]
        
        if strong_keywords:
            search_terms = strong_keywords
            print(f"[Retrieval]    🎯 Strategy: Specific Entities Only {search_terms}")
        else:
            search_terms = keywords
            print(f"[Retrieval]    🔍 Strategy: Broad Search {search_terms}")

        related_entities = set()

        # Pass 1: Direct Search
        for kw in search_terms:
            # Text Match
            text_matches = self.memory.search_exact(kw, limit=3)
            for tm in text_matches:
                results.append(tm['text'])

            # KG Lookup
            triples = self.memory.get_triples(kw)
            for t in triples:
                results.append(f"{t[0]} {t[1]} {t[2]}")
                
                # Logic: If we searched Object, Hop to Subject (and vice versa)
                s, r, o = t[0].lower(), t[1].lower(), t[2].lower()
                kw_lower = kw.lower()
                
                target = None
                # If keyword matches subject, grab object
                if kw_lower in s: target = o
                # If keyword matches object, grab subject
                elif kw_lower in o: target = s
                
                if target and target not in self.generic_terms:
                    related_entities.add(target)

        # Pass 2: Graph Expansion (Hopping)
        if related_entities:
            # Filter out entities we already searched for
            expansion_targets = [e for e in related_entities if e not in [k.lower() for k in search_terms]]
            
            if expansion_targets:
                # Truncate to avoid massive expansion loops
                targets = expansion_targets[:2] 
                print(f"[Retrieval]    🔗 Expanding graph search to: {targets}")
                
                for entity in targets:
                    # Fetch facts about the connected entity
                    triples = self.memory.get_triples(entity)
                    for t in triples:
                        results.append(f"{t[0]} {t[1]} {t[2]}")
                    
                    # Fetch text about the connected entity
                    text_matches = self.memory.search_exact(entity, limit=2)
                    for tm in text_matches:
                        results.append(tm['text'])

        return results

    def _extract_keywords_fast(self, text: str) -> List[str]:
        text = re.sub(r"'s\b", "", text, flags=re.IGNORECASE)
        clean_text = re.sub(r'[^\w\s]', '', text)
        tokens = clean_text.split()
        
        keywords = []
        for t in tokens:
            if (t[0].isupper() or len(t) > 2) and t.lower() not in self.stop_words:
                keywords.append(t)
                
        return list(set(keywords))