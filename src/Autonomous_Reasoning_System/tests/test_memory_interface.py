from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface

def main():
    mem = MemoryInterface()

    # Start new episode
    mem.start_episode("Morning reasoning session")

    # Store some memories
    mem.store("Analyzed the memory system design for Tyrone.")
    mem.store("Implemented vector and episodic layers successfully.")
    mem.store("Planned integration with reasoning engine next.")

    # Query recall
    print("\n🔍 Recall test:")
    print(mem.recall("memory integration"))

    # End the episode
    print("\n🏁 Ending episode:")
    mem.end_episode("summarize key events")

    # Daily summary
    print("\n🧠 Daily summary:")
    mem.summarize_day()

if __name__ == "__main__":
    main()

