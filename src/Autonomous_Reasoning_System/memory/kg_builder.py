import threading
import queue
import time
import logging
from typing import List, Tuple
from Autonomous_Reasoning_System.memory.events import MemoryCreatedEvent
from Autonomous_Reasoning_System.memory.kg_validator import KGValidator
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.llm.engine import LLMEngine

logger = logging.getLogger(__name__)

class KGBuilder:
    """
    Asynchronous background module that listens for MemoryCreatedEvents,
    extracts entities/relations, validates them, and updates the KG.
    """

    def __init__(self, storage: MemoryStorage, llm_engine: LLMEngine = None):
        self.storage = storage
        self.queue = queue.Queue()
        self.validator = KGValidator()
        self.llm = llm_engine or LLMEngine()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("KGBuilder started.")

    def handle_event(self, event: MemoryCreatedEvent):
        """Callback for memory events."""
        self.queue.put(event)

    def _worker_loop(self):
        while self.running:
            try:
                event = self.queue.get(timeout=1.0)
                self._process_event(event)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in KGBuilder worker: {e}")

    def _process_event(self, event: MemoryCreatedEvent):
        # 0. Sanitize / Validate Source
        if not self.validator.is_valid_content(event.text):
            logger.info(f"KGBuilder: Skipping noise memory {event.memory_id}")
            return

        logger.info(f"Processing memory for KG: {event.memory_id}")
        try:
            # 1. Extract candidates via LLM
            candidates = self._extract_candidates(event.text)
            if not candidates:
                return

            # 2. Validate
            valid_triples = self.validator.validate_batch(candidates)

            # 3. Write to DB
            self._write_triples(valid_triples)
        except Exception as e:
            logger.error(f"Failed to process event {event.memory_id}: {e}")

    def _extract_candidates(self, text: str) -> List[Tuple[str, str, str, str, str]]:
        """
        Uses LLM to extract (Subject, Relation, Object) triples.
        """
        prompt = f"""
        Extract knowledge graph triples from the following text.
        Format each triple as: Subject | SubjectType | Relation | Object | ObjectType

        RULES:
        1. Only extract factual, stable relationships.
        2. Ignore opinions, temporary states, plan updates, or reflections.
        3. Use simple types: person, location, object, concept, event, date.
        4. IMPORTANT: If the text contains a birthday, extract it as: Person | person | has_birthday | Date | date
           Example: "Nina's birthday is 11 January" -> Nina | person | has_birthday | 11 January | date

        Text: "{text}"

        Triples:
        """
        try:
            response = self.llm.generate_response(prompt)
            lines = response.strip().split('\n')
            triples = []
            for line in lines:
                parts = line.split('|')
                if len(parts) == 5:
                    triples.append((
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip(),
                        parts[3].strip(),
                        parts[4].strip()
                    ))
                elif len(parts) == 3:
                     # Fallback for old format or hallucination
                     triples.append((
                        parts[0].strip(),
                        "unknown",
                        parts[1].strip(),
                        parts[2].strip(),
                        "unknown"
                    ))
            return triples
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    def _write_triples(self, triples: List[Tuple[str, str, str, str, str]]):
        if not triples:
            return

        with self.storage._write_lock:
            try:
                self.storage.con.begin()
                for s, s_type, r, o, o_type in triples:
                    # Insert Entities (ignore conflicts)
                    self.storage.con.execute(
                        "INSERT OR IGNORE INTO entities (entity_id, type) VALUES (?, ?)",
                        (s, s_type)
                    )
                    # Update type if it was unknown (optional, but good for refinement)

                    self.storage.con.execute(
                        "INSERT OR IGNORE INTO entities (entity_id, type) VALUES (?, ?)",
                        (o, o_type)
                    )

                    # Insert Relation
                    self.storage.con.execute(
                        "INSERT OR IGNORE INTO relations (name) VALUES (?)",
                        (r,)
                    )

                    # Insert Triple
                    self.storage.con.execute(
                        "INSERT OR IGNORE INTO triples (subject, relation, object) VALUES (?, ?, ?)",
                        (s, r, o)
                    )
                self.storage.con.commit()
                logger.info(f"KGBuilder: Wrote {len(triples)} triples.")
            except Exception as e:
                self.storage.con.rollback()
                logger.error(f"KGBuilder write failed: {e}")

    def stop(self):
        self.running = False
        self.worker_thread.join()
