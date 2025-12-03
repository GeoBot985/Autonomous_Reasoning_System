import logging
import json
import datetime
from typing import List
from .prompts import REFLECTION_ANALYSIS_SYSTEM

# Setup simple logging
logger = logging.getLogger("ARS_Reflection")

class Reflector:
    """
    The Philosopher.
    Handles background maintenance, learning, and deep analysis.
    """

    def __init__(self, memory_system, llm_engine):
        self.memory = memory_system
        self.llm = llm_engine

    def reflect_on_recent(self, focus_topic: str = None) -> str:
        """
        Explicit reflection request. 
        Analyzes recent memories to find patterns or answer a deep question.
        """
        logger.info(f"🤔 Reflecting on: {focus_topic or 'recent events'}")
        
        # 1. Gather raw data (last 20 memories)
        recent_texts = self.memory.get_recent_memories(limit=20)
        context_str = "\n- ".join(recent_texts)
        
        # 2. Construct Prompt
        prompt = (
            f"Analyze these recent events/memories. "
            f"Focus topic: {focus_topic if focus_topic else 'General patterns'}.\n"
            f"Identify 1-3 key insights, lessons, or behavioral patterns.\n\n"
            f"MEMORIES:\n{context_str}"
        )
        
        system = REFLECTION_ANALYSIS_SYSTEM
        
        # 3. Generate Insight
        insight = self.llm.generate(prompt, system=system, temperature=0.5)
        
        # 4. Store the Insight
        self.memory.remember(
            insight, 
            memory_type="reflection", 
            importance=0.8, # Reflections are high value
            source="reflector",
            metadata={"trigger": focus_topic}
        )
        
        return insight

    def consolidate_sessions(self):
        """
        Compression Algorithm.
        Reads 'episodic' memories from today, summarizes them, 
        and saves a 'summary' memory. 
        """
        logger.info("🗜️ Consolidating memories...")
        
        # (In a real implementation, we would query by DATE. 
        # For this refactor, we take the last 50 items.)
        raw_logs = self.memory.get_recent_memories(limit=50)
        
        if not raw_logs:
            return "No memories to consolidate."
            
        block = "\n".join(raw_logs)
        
        summary = self.llm.generate(
            "Summarize the following conversation logs into a concise paragraph.",
            system=f"Logs:\n{block}"
        )
        
        # Store Summary
        self.memory.remember(
            f"Session Summary ({datetime.date.today()}): {summary}",
            memory_type="episodic_summary",
            importance=0.6,
            source="consolidator"
        )
        
        return f"Consolidated {len(raw_logs)} logs into summary."

    def decay_importance(self, decay_rate: float = 0.05):
        """
        Drift Correction.
        Reduces importance of memories that haven't been accessed recently.
        Uses a timed lock acquisition to prevent deadlocks with foreground processes.
        """
        logger.info("📉 Running memory decay...")
        
        # Attempt to acquire the MemorySystem lock with a 5-second timeout
        if self.memory._lock.acquire(timeout=5):
            try:
                # DuckDB SQL to multiply importance
                self.memory.con.execute(f"""
                    UPDATE memory 
                    SET importance = importance * {1.0 - decay_rate}
                    WHERE memory_type != 'fact' 
                    AND importance > 0.1
                """)
                return "Memory decay applied."
            except Exception as e:
                logger.error(f"Decay failed: {e}")
                return "Decay failed."
            finally:
                # Always release the lock, even if an exception occurred
                self.memory._lock.release()
        else:
            logger.warning("Skipping decay: Memory lock timeout.")
            return "Skipping decay."

# Factory pattern
def get_reflector(memory_system, llm_engine):
    return Reflector(memory_system, llm_engine)