# learning_manager.py
"""
LearningManager
Consumes validated experiences from the SelfValidator,
detects repeated patterns, and writes summarised "lessons" into memory.
"""
import threading

from datetime import datetime, timedelta
from collections import defaultdict
# from ..memory.singletons import get_memory_storage # Deleted


class LearningManager:
    def __init__(self, memory_storage=None):
        self.memory = memory_storage
        self.experience_buffer = []   # incoming validation results
        self.last_summary_time = datetime.utcnow()
        self.lock = threading.Lock()

    # ---------------------------------------------------------------
    # 🧠 INGESTION
    # ---------------------------------------------------------------
    def ingest(self, validation_result: dict):
        """
        Store one validation result (from SelfValidator).
        """
        if not validation_result or not isinstance(validation_result, dict):
            return False

        with self.lock:
            validation_result["timestamp"] = validation_result.get("timestamp") or datetime.utcnow().isoformat()
            self.experience_buffer.append(validation_result)
            # Keep recent 200 experiences only
            self.experience_buffer = self.experience_buffer[-200:]
        return True

    # ---------------------------------------------------------------
    # 🧩 SUMMARISATION
    # ---------------------------------------------------------------
    def summarise_recent(self, window_minutes: int = 60) -> dict:
        """
        Summarises experiences in the last N minutes into a high-level reflection.
        Returns a dict with trend summary and inserts a short "lesson" memory.
        Thread-safe to prevent DuckDB write conflicts when called by multiple threads.
        """
        with self.lock:  # 🔒 Prevent concurrent writes and reads of buffer
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=window_minutes)
            recent = [x for x in self.experience_buffer if self._ts(x["timestamp"]) >= cutoff]
            if not recent:
                return {"summary": "No recent experiences."}

            # Group by feeling and intent
            counts = defaultdict(int)
            intents = defaultdict(int)
            for r in recent:
                counts[r["feeling"]] += 1
                intents[r["intent"]] += 1

            # Handle missing keys safely
            pos = counts.get("positive", 0)
            neg = counts.get("negative", 0)
            neu = counts.get("neutral", 0)
            dominant_feeling = max(counts, key=counts.get) if counts else "neutral"
            dominant_intent = max(intents, key=intents.get) if intents else "unknown"

            lesson_text = (
                f"In the last {window_minutes} minutes, most experiences felt {dominant_feeling}. "
                f"Dominant intent: {dominant_intent}. "
                f"Summary: {pos} positive, {neu} neutral, {neg} negative results overall."
            )

            # ✅ Thread-safe write to DuckDB-backed memory (MemoryStorage handles its own locking)
            if self.memory:
                self.memory.add_memory(
                    text=lesson_text,
                    memory_type="reflection",
                    importance=0.6,
                )
            self.last_summary_time = now

            return {
                "summary": lesson_text,
                "dominant_feeling": dominant_feeling,
                "dominant_intent": dominant_intent,
                "stats": counts
            }


    # ---------------------------------------------------------------
    # 🧹 DRIFT CORRECTION
    # ---------------------------------------------------------------
    def perform_drift_correction(self):
        """
        Example placeholder for balancing memory — in future, this can downweight stale,
        repetitive, or highly negative entries.
        """
        if not self.memory:
            return "Memory storage not available."

        # ✅ Compatible call for any MemoryStorage version
        # MemoryStorage is thread-safe for reads
        if hasattr(self.memory, "get_all"):
            df = self.memory.get_all()
        elif hasattr(self.memory, "get_all_memories"):
            df = self.memory.get_all_memories()
        else:
            return "Memory interface not compatible."

        if df.empty:
            return "Memory empty, nothing to correct."

        # Simple heuristic: reduce importance of very old or negative lessons
        now = datetime.utcnow()
        updates = 0

        # Note: We are iterating over a copy of data (DataFrame), so no lock needed here.
        # Updates should use memory interface which is locked.

        for _, row in df.iterrows():
            age_days = (now - row["created_at"]).days if row["created_at"] else 0
            if "negative" in row["text"].lower() or age_days > 30:
                # new_importance = max(0.1, row["importance"] * 0.8)
                # You could later persist these via UPDATEs if required
                updates += 1
        return f"Drift correction simulated for {updates} records."

    # ---------------------------------------------------------------
    # ⏱️ HELPER
    # ---------------------------------------------------------------
    def _ts(self, t):
        if isinstance(t, datetime):
            return t
        try:
            return datetime.fromisoformat(t)
        except Exception:
            return datetime.utcnow()
