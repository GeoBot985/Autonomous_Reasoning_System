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

    def __init__(self, memory_interface: MemoryInterface = None, top_k: int = 5):
        # Allow injection or use a new instance if needed, but ideally this should be injected
        # To avoid double init, we accept memory_interface.
        self.mem = memory_interface
        # If not provided, we assume it's not available or we'd have to create one.
        # Creating one here is risky if MemoryInterface isn't lightweight.
        # But for backward compat, we can try.
        # However, better to rely on what's passed.
        self.top_k = top_k

    # ------------------------------------------------------------------
    def build_context(self, query: str = None) -> str:
        """
        Returns a combined text block of:
        - top-K semantically related memories (if query given)
        - most recent episodic summaries (past 24h)
        Deduplicates repeated lines and truncates long summaries.
        """
        if not self.mem:
             return "### Tyrone's Working Memory Context ###\n(Memory system unavailable)"

        lines = ["### Tyrone's Working Memory Context ###"]

        # --- 1️⃣ Semantic context ---
        if query:
            # We use retrieve instead of recall (recall was legacy)
            results = self.mem.retrieve(query, k=self.top_k)
            if results:
                lines.append("\n[Recent related memories]")
                for r in results:
                    lines.append(f"- {r['text']}")

        # --- 2️⃣ Episodic context ---
        # Accessing episodes via memory interface, not direct parquet file!
        try:
            # If EpisodicMemory uses DuckDB now, we query it via self.mem.episodes
            if hasattr(self.mem, "episodes") and self.mem.episodes:
                df = self.mem.episodes.get_all_episodes()
                cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
                # Filter in pandas since we have the DF
                recent = df[df["start_time"] > cutoff].sort_values("start_time", ascending=False).head(3)

                if not recent.empty:
                    lines.append("\n[Recent episodes]")
                    for _, row in recent.iterrows():
                        summary = str(row["summary"]) if row["summary"] else "(no summary)"
                        # Trim long summaries for prompt compactness
                        if len(summary) > 250:
                            summary = summary[:247] + "..."
                        lines.append(f"- ({row['start_time']}) {summary}")
        except Exception as e:
            print(f"[ContextBuilder] Error fetching episodes: {e}")
            lines.append("\n(No episodic data available)")

        return "\n".join(lines)
