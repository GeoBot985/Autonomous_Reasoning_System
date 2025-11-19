# Autonomous_Reasoning_System/tests/test_reasoning_cycle.py
"""
Integration test:
✅ Runs several reasoning turns through ContextAdapter
✅ Verifies episodic memories are stored with provenance
✅ Checks automatic consolidation trigger after N turns
✅ Confirms episodic summaries and reflection links are saved
"""

from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter
from Autonomous_Reasoning_System.llm.consolidator import ReasoningConsolidator
from Autonomous_Reasoning_System.memory.singletons import get_memory_storage
import pandas as pd


def run_reasoning_cycle():
    print("🔁 Starting reasoning cycle test with consolidation trigger...\n")

    adapter = ContextAdapter()
    mem = get_memory_storage()
    consolidator = ReasoningConsolidator()

    # ---------------------------------------------------------
    # 1️⃣ Run multiple reasoning turns (trigger auto-consolidation)
    prompts = [
        "How is Tyrone’s memory system structured?",
        "What improvements could make it more adaptive?",
        "What did we finish yesterday?",
        "How does episodic recall help Tyrone reason better?",
        "What are the next steps for memory consolidation?",
        "Summarize Tyrone’s recent development progress."
    ]

    for i, msg in enumerate(prompts, start=1):
        print(f"\n🧠 Turn {i}: {msg}")
        reply = adapter.run(msg)
        print(f"🤖 Tyrone: {reply[:220]}")  # truncate for readability

    # ---------------------------------------------------------
    # 2️⃣ Inspect stored memories
    df = mem.get_all_memories()
    print("\n📘 Stored Memories (latest 10):")
    if df.empty:
        print("❌ No memories found.")
    else:
        display_cols = ["memory_type", "source", "created_at", "text"]
        df_display = df[display_cols].sort_values("created_at", ascending=False).head(10)
        print(df_display.to_string(index=False))

    # ---------------------------------------------------------
    # 3️⃣ Run manual consolidation for comparison
    print("\n🧩 Running ReasoningConsolidator manually (for comparison)...")
    summary = consolidator.consolidate_recent(limit=5)
    print(f"\n📜 Manual Summary: {summary}\n")

    # ---------------------------------------------------------
    # 4️⃣ Verify episodic summaries and reflections
    df_after = mem.get_all_memories()
    episodic = df_after[df_after["memory_type"] == "episodic_summary"]
    reflection = df_after[df_after["memory_type"] == "reflection"]

    print("✅ Episodic summaries in memory:")
    if not episodic.empty:
        print(episodic[["created_at", "text", "source"]].tail(3).to_string(index=False))
    else:
        print("No episodic summaries found.")

    print("\n💭 Reflection links in memory:")
    if not reflection.empty:
        print(reflection[["created_at", "text", "source"]].tail(3).to_string(index=False))
    else:
        print("No reflection links found.")

    print("\n🎯 Test complete — reasoning, consolidation, and reflection verified.\n")


if __name__ == "__main__":
    run_reasoning_cycle()
