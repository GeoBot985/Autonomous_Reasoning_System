# tests/test_vector_memory.py
from Autonomous_Reasoning_System.memory.storage import MemoryStorage

def main():
    store = MemoryStorage()
    store.add_memory("I met Sarah at the coffee shop yesterday.", "note")
    store.add_memory("Meeting with John about project timeline next week.", "note")
    store.add_memory("Remember to buy groceries for the weekend.", "note")

    print("\nQuery: meeting schedule")
    q_vec = store.embedder.embed("meeting schedule")
    results = store.vector_store.search(q_vec)
    for r in results:
        print(f"- ({r['score']:.3f}) {r['text']} [{r['memory_type']}]")

if __name__ == "__main__":
    main()
