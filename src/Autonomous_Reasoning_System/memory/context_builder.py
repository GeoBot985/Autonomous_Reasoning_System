# Autonomous_Reasoning_System/memory/context_builder.py
"""
Builds Tyrone's short-term reasoning context from memory.
Combines relevant semantic memories and recent episodic summaries.
"""

from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from datetime import datetime, timedelta
import duckdb
import os


class ContextBuilder:
    """
    Generates a working-memory context for reasoning or planning.
    """

    def __init__(self, top_k: int = 5):
        self.mem = MemoryInterface()
        self.top_k = top_k

    # ------------------------------------------------------------------
    def build_context(self, query: str = None) -> str:
        """
        Returns a combined text block of:
        - top-K semantically related memories (if query given)
        - most recent episodic summaries (past 24h)
        Deduplicates repeated lines and truncates long summaries.
        """

        lines = ["### Tyrone's Working Memory Context ###"]

        # --- 1️⃣ Semantic context ---
        if query:
            recall_text = self.mem.recall(query, k=self.top_k)
            if recall_text and "No relevant memories" not in recall_text:
                # Split into individual lines and deduplicate
                seen = set()
                unique_recall = []
                for line in recall_text.splitlines():
                    normalized = line.strip()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        unique_recall.append(normalized)
                if unique_recall:
                    lines.append("\n[Recent related memories]")
                    lines.extend(unique_recall)

        # --- 2️⃣ Episodic context ---
        try:
            df = duckdb.sql(f"""
                SELECT summary, start_time
                FROM read_parquet('data/episodes.parquet')
                WHERE start_time > '{(datetime.utcnow() - timedelta(days=1)).isoformat()}'
                ORDER BY start_time DESC
                LIMIT 3;
            """).df()

            if not df.empty:
                lines.append("\n[Recent episodes]")
                for _, row in df.iterrows():
                    summary = str(row["summary"]) if row["summary"] else "(no summary)"
                    # Trim long summaries for prompt compactness
                    if len(summary) > 250:
                        summary = summary[:247] + "..."
                    lines.append(f"- ({row['start_time']}) {summary}")
        except Exception:
            lines.append("\n(No episodic data available)")

        return "\n".join(lines)

