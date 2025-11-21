from datetime import datetime
from Autonomous_Reasoning_System.memory.storage import MemoryStorage

class ConfidenceManager:
    """
    Adjusts the 'importance' field of memories to simulate reinforcement and decay.
    """

    def __init__(self, memory_storage=None):
        self.memory = memory_storage or MemoryStorage()

    def reinforce(self, mem_id: str = None, step: float = 0.05):
        """
        Increase importance slightly when memory is accessed.
        If no mem_id is provided, reinforces the most recent memory.
        """
        # Determine which memory to update
        if mem_id is None:
            try:
                res = self.memory.con.execute(
                    "SELECT id FROM memory ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                mem_id = res[0] if res else None
            except Exception as e:
                print(f"[ConfidenceManager] Error finding latest memory: {e}")
                mem_id = None

        if not mem_id:
            print("[⚠️ CONFIDENCE] No valid memory ID found to reinforce.")
            return

        # Apply reinforcement
        try:
            now = datetime.utcnow().isoformat()
            self.memory.con.execute(f"""
                UPDATE memory
                SET importance = LEAST(1.0, COALESCE(importance, 0.0) + ?),
                    last_accessed = ?
                WHERE id = ?;
            """, (step, now, mem_id))
            print(f"[📈 CONFIDENCE] Reinforced memory {mem_id} (+{step}).")
        except Exception as e:
            print(f"[ConfidenceManager] Error reinforcing memory: {e}")

    def decay_all(self, step: float = 0.01):
        """Decrease importance slightly across all memories over time."""
        try:
            self.memory.con.execute(f"""
                UPDATE memory
                SET importance = GREATEST(0.0, COALESCE(importance, 0.0) - ?)
                WHERE importance IS NOT NULL;
            """, (step,))
            print(f"[📉 CONFIDENCE] Decayed all memories by {step}.")
        except Exception as e:
            print(f"[ConfidenceManager] Error decaying memories: {e}")
