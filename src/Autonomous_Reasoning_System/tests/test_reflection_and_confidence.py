from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
from Autonomous_Reasoning_System.memory.confidence_manager import ConfidenceManager
from Autonomous_Reasoning_System.memory.storage import MemoryStorage

def run_reflection_and_confidence_test():
    print("🧠 Testing ReflectionInterpreter + ConfidenceManager...\n")

    mem = MemoryStorage()
    interpreter = ReflectionInterpreter()
    cm = ConfidenceManager()

    # 1️⃣ Ask a reflective question
    q = "What have I learned recently?"
    print(f"Q: {q}")
    print("A:", interpreter.interpret(q))

    # 2️⃣ Reinforce a random memory
    df = mem.get_all_memories()
    if not df.empty:
        mem_id = df.iloc[0]["id"]
        cm.reinforce(mem_id)
        print(f"\n🔁 Reinforced memory {mem_id}")

    # 3️⃣ Apply decay globally
    cm.decay_all()
    print("☁️ Applied importance decay to all memories.")

if __name__ == "__main__":
    run_reflection_and_confidence_test()
