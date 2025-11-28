import logging
import requests
import json
import re
import datetime
import time
import threading
from typing import Optional, List

from . import config
from .memory import get_memory_system
from .tools.web_search import perform_google_search
from .retrieval import RetrievalSystem
from .reflection import get_reflector


logger = logging.getLogger("ARS_Brain")

# Config defaults (fall back to config module for host/model)
OLLAMA_BASE = getattr(config, "OLLAMA_API_BASE", "http://localhost:11434/api")
DEFAULT_MODEL = getattr(config, "LLM_MODEL", "granite4:1b")
TAGS_TIMEOUT = 5
WARMUP_TIMEOUT = 15
REQUEST_TIMEOUT = 90  # Trimmed to avoid long hangs during planning

class LLMEngine:
    def __init__(self, model: str = DEFAULT_MODEL, api_base: str = OLLAMA_BASE):
        # Configure Ollama endpoints and model once (avoid recursive init)
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.tags_url = f"{self.api_base}/tags"
        self.generate_url = f"{self.api_base}/generate"
        
        # --- NEW: Use a persistent session for better connection handling ---
        self.session = requests.Session()
        self.session.trust_env = False  # avoid proxy/env interference for local Ollama
        # ------------------------------------------------------------------
        
        self._check_model_exists()
        self._warmup()

    def ping(self, timeout: int = 2) -> bool:
        """Quick health check against the tags endpoint."""
        try:
            self.session.get(self.tags_url, timeout=timeout).raise_for_status()
            return True
        except Exception:
            return False

    def _check_model_exists(self):
        print(f"[Brain] Checking if model '{self.model}' exists locally...")
        try:
            # Use session for the request
            resp = self.session.get(self.tags_url, timeout=TAGS_TIMEOUT)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if any(self.model in m for m in models):
                    print(f"[Brain] ✅ Model '{self.model}' found.")
                else:
                    print(f"[Brain] ⚠️ WARNING: Model '{self.model}' not found in Ollama!")
        except Exception as e:
            print(f"[Brain] ⚠️ Could not list models: {e}")

    def _warmup(self):
        print("[Brain] Warming up LLM...")
        try:
            # Use session for the request
            self.session.post(self.generate_url, json={"model": self.model, "prompt": "hi", "stream": False, "keep_alive": "5m"}, timeout=WARMUP_TIMEOUT)
            print("[Brain] ✅ LLM Warmed up.")
        except Exception as e:
            print(f"[Brain] ⚠️ Warmup failed: {e}")

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, format: str = None) -> str:
        full_prompt = prompt
        if system:
            full_prompt = f"SYSTEM: {system}\n\nUSER: {prompt}"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature
        }
        if format:
            payload["format"] = format
        print(f"[LLM] Sending request to Ollama ({len(full_prompt)} chars)...")
        start_t = time.time()
        try:
            # Use session for the request (with a sane timeout to avoid UI hangs)
            resp = self.session.post(self.generate_url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            print(f"[LLM] Response received ({time.time() - start_t:.2f}s)")
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            if "resp" in locals():
                logger.error(f"Ollama status={resp.status_code}, body={resp.text[:500]}")
            return f"[Error: LLM unavailable - {e}]"

class Brain:

    def _warmup_memory(self):
        """Forces the RAG system to run a trivial vector search to load VSS indices and Embedder resources."""
        print("[Brain] 🔥 Warming up RAG/Memory System...")
        start_t = time.time()
        try:
            # Running a simple context retrieval forces the Embedder and DuckDB VSS to load cold resources.
            self.retrieval.get_context_string("quick test query for memory warmup", include_history=None)
            print(f"[Brain] ✅ RAG/Memory Warmed up ({time.time() - start_t:.2f}s).")
        except Exception as e:
            print(f"[Brain] ⚠️ RAG Warmup failed: {e}")

    def __init__(self):
        print("\n[Brain] 🟢 Initializing Brain...")
        start_t = time.time()
        self.memory = get_memory_system(db_path="data/memory.duckdb")
        self.retrieval = RetrievalSystem(self.memory)
        self.llm = LLMEngine()
        self.plugins = {}
        self.reflector = get_reflector(self.memory, self.llm)
        self._warmup_memory()
        self._start_maintenance_loop()
        self._register_basic_tools()
        print(f"[Brain] ✅ Brain Ready (Total startup: {time.time() - start_t:.2f}s)\n")

    def _run_maintenance(self):
        """Runs periodic memory decay and session consolidation."""
        while True:
            time.sleep(1800)
            print("[Brain] Running scheduled maintenance...")
            try:
                self.reflector.decay_importance()
                self.reflector.consolidate_sessions()
            except Exception as e:
                logger.error(f"Maintenance task failed: {e}")

    def _start_maintenance_loop(self):
        """Starts the maintenance thread in the background."""
        t = threading.Thread(target=self._run_maintenance, daemon=True)
        t.start()

    def think(self, user_input: str, history: List[dict] = None) -> str:
        if not user_input or not user_input.strip(): return ""
        text = user_input.strip()
        if self._is_web_search_query(text):
            return self._handle_web_search(text)
        
        plugin_response = self._check_plugins(text)
        if plugin_response: return plugin_response

        intent, metadata = self._classify_intent(text)
        print(f"[Brain] 🧭 Intent: {intent}") 
        
        if intent == "store":
            return self._handle_storage(text, metadata)
        elif intent == "plan":
            return self._handle_planning(text)
        else:
            # ✅ FIX: 'history' is now defined by the function signature
            return self._handle_chat(text, history)
        
    def _format_history(self, history: List[dict]) -> List[str]:
        """Converts the list of dicts history into a list of strings for context."""
        formatted = []
        if not history:
            return []
        
        # Exclude the very last entry, as that is the user's current message (the query itself)
        # We only want previous turns.
        previous_turns = history[:-1] 

        for turn in previous_turns:
            role = turn['role'].capitalize()
            content = turn['content']
            formatted.append(f"{role}: {content}")
        return formatted

    def _classify_intent(self, text: str):
        lower = text.lower()
        
        # 1. Explicit Storage Commands
        if any(x in lower for x in ["remember that", "don't forget", "remind me", "save this", "note that"]):
            return "store", {"source": "direct_command"}
        
        # 2. Planning Keywords
        if any(x in lower for x in ["plan a", "create a goal", "how do i", "research"]):
            return "plan", {}

        # 2.5. Treat quick web-search prompts as RAG queries.
        if lower.startswith("web search"):
            return "chat", {}

        # 3. RAG / Action Commands (The Fix!)
        # If it starts with an imperative verb, it is a request for output (Chat), not input (Store).
        rag_verbs = ["summarize", "explain", "describe", "list", "show", "find", "search", "define", "tell"]
        clean_start = lower.replace("please ", "").strip()
        if any(clean_start.startswith(v) for v in rag_verbs):
            return "chat", {}

        # 4. Question Check
        # Expanded to catch "Could you..." or "Would you..."
        is_question = lower.startswith(("when", "what", "who", "where", "how", "is ", "does", "can", "could", "would")) or "?" in lower
        
        # 5. Implicit Assertion (Store)
        # Only store if it's NOT a question AND NOT a RAG command
        if not is_question and 3 < len(lower.split()) < 20:
             return "store", {"source": "implicit_assertion"}

        return "chat", {}

    def _handle_storage(self, text: str, meta: dict) -> str:
        clean_text = re.sub(r"^(remember that|don't forget|remind me|save this|note that)\s*", "", text, flags=re.IGNORECASE)
        kg_triples = self._extract_triples_via_llm(clean_text)
        
        self.memory.remember(
            clean_text, 
            memory_type="fact", 
            importance=1.0, 
            metadata={"kg_triples": kg_triples}
        )
        
        if kg_triples:
            return f"✅ Saved fact and extracted knowledge: {kg_triples}"
        else:
            return f"✅ Saved: '{clean_text}'"

    def _is_web_search_query(self, text: str) -> bool:
        lower = text.lower().strip()
        return lower.startswith("web search") or lower.startswith("search web")

    def _handle_web_search(self, text: str) -> str:
        query = text
        lower = text.lower()

        if ":" in text:
            prefix, remainder = text.split(":", 1)
            if prefix.lower().strip() in {"web search", "search web"}:
                query = remainder.strip()
        elif lower.startswith("web search"):
            query = text[len("web search"):].strip()
        elif lower.startswith("search web"):
            query = text[len("search web"):].strip()

        query = query.strip()
        if not query:
            return "Please tell me what you'd like me to search for."

        try:
            result = perform_google_search(query)
            return f"Web search result for '{query}':\n{result}"
        except Exception as exc:
            logger.error(f"Web search failed: {exc}", exc_info=True)
            return "I tried to perform the web search but something went wrong."

    def _extract_triples_via_llm(self, text: str) -> List[tuple]:
        print(f"[Brain] 🕸️ Extracting KG Triples for: '{text}'")
        system = (
            "You are a Knowledge Graph extractor. Convert the user's text into a JSON list of triples. "
            "Format: [[\"subject\", \"relation\", \"object\"]].\n"
            "Rules:\n"
            "1. Use lower case.\n"
            "2. Convert possessives: \"Cornelia's birthday\" -> [\"cornelia\", \"has_birthday\", ...]\n"
            "3. Capture definitions: \"Password is X\" -> [\"password\", \"is\", \"x\"]\n"
            "4. Return ONLY the JSON list."
        )
        try:
            response = self.llm.generate(text, system=system, temperature=0.1)
            response = response.replace("```json", "").replace("```", "").strip()
            triples = json.loads(response)
            valid_triples = []
            if isinstance(triples, list):
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        valid_triples.append((t[0], t[1], t[2]))
            print(f"[Brain]    🕸️ Found: {valid_triples}")
            return valid_triples
        except Exception as e:
            print(f"[Brain]    ⚠️ KG Extraction failed: {e}")
            return []

    # --- brain.py patch (Full _handle_chat method) ---
    def _handle_chat(self, text: str, history: List[dict] = None) -> str:
        
        # 1. Check for specific document request
        # Regex looks for (summarize/explain/show me) followed by a file name (e.g., manual.pdf)
        doc_match = re.search(r"(summarize|explain|show me).*?(\w+\.\w+)", text, re.IGNORECASE)
        
        if doc_match:
            action = doc_match.group(1).lower()
            filename = doc_match.group(2)
            
            full_doc = self.memory.get_whole_document(filename)
            
            # Context window safeguard (15,000 chars is conservative for a 3B model)
            if full_doc and len(full_doc) < 15000: 
                context_str = f"### FULL DOCUMENT SOURCE: {filename} ###\n{full_doc}"
                # Rephrase the user query to tell the LLM what to do with the *provided content*
                text = f"{action} the provided document content." 
                print(f"[Brain] 📄 Using full document '{filename}' as context (Size: {len(full_doc)} chars).")
            else:
                # Document too long or not found, fall back to standard RAG
                formatted_history = self._format_history(history)
                context_str = self.retrieval.get_context_string(text, include_history=formatted_history)
        
        else:
            # 2. Standard RAG flow (No document explicitly requested)
            formatted_history = self._format_history(history)
            context_str = self.retrieval.get_context_string(text, include_history=formatted_history)
        
        # Detect "Soft" Intent (Summarization/Explanation) 
        is_summary = any(w in text.lower() for w in ["summarize", "list", "explain", "describe", "what is", "show"])
        
        if is_summary:
            # PERMISSIVE PROMPT: Encourages synthesis
            system_prompt = (
                "You are Tyrone. Use the provided CONTEXT to answer the user.\n"
                "Rules:\n"
                "1. Synthesize the information found in the CONTEXT facts.\n"
                "2. If the text is cut off or partial, summarize what is visible.\n"
                "3. Ignore facts that look like previous user commands (e.g. 'Please summarize...')."
            )
        else:
            # STRICT PROMPT: For specific fact retrieval
            system_prompt = (
                "You are Tyrone. Use the provided CONTEXT to answer the user.\n"
                "Rules:\n"
                "1. FACTS in the context are absolute truth.\n"
                "2. Do not guess. If the specific answer is missing, say you don't know."
            )
            
        return self.llm.generate(text, system=f"{system_prompt}\n\n{context_str}")

    def _handle_planning(self, text: str) -> str:
        try:
            print("[Brain] 🟢 Loading Planner...")
            from .plan_builder import get_planner
            print(f"[Brain] 📝 Delegating to Planner: '{text}'")
            # Pass the retrieval system to the planner factory function
            return get_planner(self.memory, self.llm, self.retrieval).process_request(text)
        except Exception as e:
            return f"⚠️ Planning error: {e}"

    def _register_basic_tools(self):
        self.plugins["time"] = lambda x: f"The current time is {datetime.datetime.now().strftime('%H:%M')}."
        self.plugins["date"] = lambda x: f"Today is {datetime.datetime.now().strftime('%A, %d %B %Y')}."
        
    def _check_plugins(self, text: str) -> Optional[str]:
        return self.plugins.get(text.lower().strip())

_brain_instance = None
def get_brain():
    global _brain_instance
    if _brain_instance is None: _brain_instance = Brain()
    return _brain_instance

if __name__ == "__main__":
    b = get_brain()
    print("🧠 Tyrone Refactored (CLI Mode)")
    while True:
        try:
            q = input("You> ")
            if q.lower() in ["exit", "quit"]: break
            print(f"Tyrone> {b.think(q)}")
        except KeyboardInterrupt:
            break
