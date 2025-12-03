import logging
import re
import datetime
import time
import threading
from typing import Optional, List, Tuple, Dict, Any

from . import config
from .memory import get_memory_system
from .control.dispatcher import Dispatcher
from .tools.web_search import perform_google_search
from .retrieval import RetrievalSystem
from .reflection import get_reflector
from .plan_builder import get_planner
from .llm.engine import LLMEngine
from .utils.json_utils import parse_llm_json
from .prompts import (
    KG_EXTRACTION_PROMPT,
    CHAT_SUMMARY_PROMPT,
    CHAT_FACTUAL_PROMPT
)

logger = logging.getLogger("ARS_Brain")

class Brain:
    def __init__(self):
        logger.info("Initializing Brain...")
        start_t = time.time()

        self.memory = get_memory_system(db_path=config.MEMORY_DB_PATH)
        self.retrieval = RetrievalSystem(self.memory)
        self.llm = LLMEngine() # Use unified engine
        self.dispatcher = Dispatcher() # Use dispatcher
        self.reflector = get_reflector(self.memory, self.llm)

        self._warmup_memory()
        self._start_maintenance_loop()
        self._register_tools()

        logger.info(f"Brain Ready (Total startup: {time.time() - start_t:.2f}s)")

    def _warmup_memory(self) -> None:
        """Forces the RAG system to run a trivial vector search to load VSS indices and Embedder resources."""
        logger.info("Warming up RAG/Memory System...")
        start_t = time.time()
        try:
            self.retrieval.get_context_string("quick test query for memory warmup", include_history=None)
            logger.info(f"RAG/Memory Warmed up ({time.time() - start_t:.2f}s).")
        except Exception as e:
            logger.warning(f"RAG Warmup failed: {e}")

    def _run_maintenance(self) -> None:
        """Runs periodic memory decay and session consolidation."""
        while True:
            time.sleep(1800)
            logger.info("Running scheduled maintenance...")
            try:
                self.reflector.decay_importance()
                self.reflector.consolidate_sessions()
            except Exception as e:
                logger.error(f"Maintenance task failed: {e}")

    def _start_maintenance_loop(self) -> None:
        """Starts the maintenance thread in the background."""
        t = threading.Thread(target=self._run_maintenance, daemon=True)
        t.start()

    def think(self, user_input: str, history: List[dict] = None) -> str:
        if not user_input or not user_input.strip():
            return ""

        text = user_input.strip()

        # Check for web search via dispatcher or direct check
        if self._is_web_search_query(text):
            return self._handle_web_search(text)
        
        # Check plugins via dispatcher (or legacy plugins dict)
        # Assuming we migrate plugins to dispatcher
        # checking legacy plugin dict for now if needed, but we should use dispatcher
        # However, for this refactor, let's keep it simple.

        intent, metadata = self._classify_intent(text)
        logger.info(f"Intent: {intent}")
        
        if intent == "store":
            return self._handle_storage(text, metadata)
        elif intent == "plan":
            return self._handle_planning(text)
        else:
            return self._handle_chat(text, history)
        
    def _format_history(self, history: List[dict]) -> List[str]:
        """Converts the list of dicts history into a list of strings for context."""
        formatted = []
        if not history:
            return []
        
        # Exclude the very last entry, as that is the user's current message
        previous_turns = history[:-1] 

        for turn in previous_turns:
            role = turn.get('role', '').capitalize()
            content = turn.get('content', '')
            formatted.append(f"{role}: {content}")
        return formatted

    def _classify_intent(self, text: str) -> Tuple[str, dict]:
        lower = text.lower()
        
        # 1. Explicit Storage Commands
        storage_keywords = ["remember that", "don't forget", "remind me", "save this", "note that"]
        if any(x in lower for x in storage_keywords):
            return "store", {"source": "direct_command"}
        
        # 2. Planning Keywords
        planning_keywords = ["plan a", "create a goal", "how do i", "research"]
        if any(x in lower for x in planning_keywords):
            return "plan", {}

        # 3. RAG / Action Commands
        # Web search handled earlier

        rag_verbs = ["summarize", "explain", "describe", "list", "show", "find", "search", "define", "tell"]
        clean_start = lower.replace("please ", "").strip()
        if any(clean_start.startswith(v) for v in rag_verbs):
            return "chat", {}

        # 4. Question Check
        question_starters = ("when", "what", "who", "where", "how", "is ", "does", "can", "could", "would")
        is_question = lower.startswith(question_starters) or "?" in lower
        
        # 5. Implicit Assertion (Store)
        if not is_question and 3 < len(lower.split()) < 20:
             return "store", {"source": "implicit_assertion"}

        return "chat", {}

    def _handle_storage(self, text: str, meta: dict) -> str:
        clean_text = re.sub(
            r"^(remember that|don't forget|remind me|save this|note that)\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
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
        # Extract query
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

        # Use Dispatcher
        response = self.dispatcher.dispatch("google_search", {"query": query})

        if response["status"] == "success":
             return response['data']
        else:
             return f"I tried to perform the web search but something went wrong: {response['errors']}"

    def _extract_triples_via_llm(self, text: str) -> List[tuple]:
        logger.debug(f"Extracting KG Triples for: '{text}'")
        try:
            response = self.llm.generate(text, system=KG_EXTRACTION_PROMPT, temperature=0.1)
            # Use unified JSON parser
            triples = parse_llm_json(response)

            valid_triples = []
            if isinstance(triples, list):
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        valid_triples.append((t[0], t[1], t[2]))
            logger.debug(f"Found: {valid_triples}")
            return valid_triples
        except Exception as e:
            logger.warning(f"KG Extraction failed: {e}")
            return []

    def _handle_chat(self, text: str, history: List[dict] = None) -> str:
        # 1. Check for specific document request
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
                logger.info(f"Using full document '{filename}' as context (Size: {len(full_doc)} chars).")
            else:
                formatted_history = self._format_history(history)
                context_str = self.retrieval.get_context_string(text, include_history=formatted_history)
        
        else:
            formatted_history = self._format_history(history)
            context_str = self.retrieval.get_context_string(text, include_history=formatted_history)
        
        is_summary = any(w in text.lower() for w in ["summarize", "list", "explain", "describe", "what is", "show"])
        
        if is_summary:
            system_prompt = CHAT_SUMMARY_PROMPT
        else:
            system_prompt = CHAT_FACTUAL_PROMPT
            
        return self.llm.generate(text, system=f"{system_prompt}\n\n{context_str}")

    def _handle_planning(self, text: str) -> str:
        try:
            logger.info("Loading Planner...")
            logger.info(f"Delegating to Planner: '{text}'")
            return get_planner(self.memory, self.llm, self.retrieval).process_request(text)
        except Exception as e:
            return f"⚠️ Planning error: {e}"

    def _register_tools(self) -> None:
        # Register Google Search with Dispatcher
        # Wrapper to handle formatting as requested
        def formatted_google_search(query: str) -> str:
            result = perform_google_search(query)
            return f"Web search result for '{query}':\n{result}"

        self.dispatcher.register_tool(
            name="google_search",
            handler=formatted_google_search,
            schema={"query": {"type": str, "required": True}}
        )
        
        # Register basic tools (legacy plugins)
        # Assuming we can invoke them via dispatcher if we wanted, or keep them as inline for now.
        # The user wanted "Unify the Tools... Ensure the new Google Search tool in the dispatcher handles the formatting logic".
        # I did that in _handle_web_search by using dispatcher.
        pass

_brain_instance = None

def get_brain() -> Brain:
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = Brain()
    return _brain_instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    b = get_brain()
    print("🧠 Tyrone Refactored (CLI Mode)")
    while True:
        try:
            q = input("You> ")
            if q.lower() in ["exit", "quit"]:
                break
            print(f"Tyrone> {b.think(q)}")
        except KeyboardInterrupt:
            break
