from Autonomous_Reasoning_System.memory.episodes import EpisodicMemory

def dummy_summarizer(text):
    return f"(dummy summary of {len(text.split())} words)"

def main():
    epi = EpisodicMemory()
    eid = epi.begin_episode()

    # Simulate doing stuff...
    epi.end_episode("Today I worked on the AI memory system and added episodic recall.")
    epi.begin_episode()
    epi.end_episode("Tested the new summarization system and verified re-indexing works.")

    print("\n🧾 All episodes:")
    print(epi.list_episodes())

    print("\n🧠 Summarizing today's episodes:")
    epi.summarize_day(dummy_summarizer)

if __name__ == "__main__":
    main()

