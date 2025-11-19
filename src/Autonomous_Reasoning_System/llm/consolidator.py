# Autonomous_Reasoning_System/llm/consolidator.py

from ..memory.singletons import get_memory_storage
from .engine import call_llm


class ReasoningConsolidator:
    """
    Periodically summarizes recent episodes into long-term summaries.
    """

    def __init__(self):
        self.memory = get_memory_storage()

    def consolidate_recent(self, limit: int = 5):
        """
        Fetches recent episodic memories and generates a concise summary.
        """
        try:
            df = self.memory.search_memory("Assistant:")
            if df.empty:
                return "No episodic memories to summarize."

            # Take most recent episodes
            subset = df.sort_values("created_at", ascending=False).head(limit)
            text_block = "\n\n".join(subset["text"].tolist())

            # Summarize via LLM
            prompt = (
                "Summarize the following conversation snippets into one short paragraph "
                "describing what the assistant has recently been focused on.\n\n"
                f"{text_block}"
            )
            summary = call_llm(
                "You are an episodic summarizer that writes short, coherent summaries.",
                prompt
            )

            # Store the summary as a long-term episodic memory
            self.memory.add_memory(
                text=f"Session Summary: {summary}",
                memory_type="episodic_summary",
                importance=0.9,
                source="consolidator"
            )

            return summary

        except Exception as e:
            return f"[ReasoningConsolidator Error] {e}"
