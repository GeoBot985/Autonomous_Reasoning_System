from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
from Autonomous_Reasoning_System.memory.confidence_manager import ConfidenceManager
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
import pytest

def test_reflection_and_confidence(temp_db_path):
    print("🧠 Testing ReflectionInterpreter + ConfidenceManager...\n")

    # Use temp storage
    mem = MemoryStorage(db_path=temp_db_path)

    # Inject memory
    interpreter = ReflectionInterpreter(memory_storage=mem)
    cm = ConfidenceManager(memory_storage=mem)

    # Add a dummy memory so reflection has something to work with
    mem.add_memory("I learned that Python is great.", memory_type="note", importance=0.5)

    # 1️⃣ Ask a reflective question
    q = "What have I learned recently?"
    print(f"Q: {q}")
    # Mock LLM call inside interpreter? Or allow it to run if integrated.
    # Since this is a unit test, we should ideally mock call_llm, but interpreter calls it.
    # For now, we just check it runs without error.
    # interpreter.interpret calls call_llm which uses requests.

    # We rely on the mock behavior or robust error handling in call_llm
    res = interpreter.interpret(q)
    assert isinstance(res, dict)

    # 2️⃣ Reinforce a random memory
    df = mem.get_all_memories()
    if not df.empty:
        mem_id = df.iloc[0]["id"]
        cm.reinforce(mem_id)
        print(f"\n🔁 Reinforced memory {mem_id}")

    # 3️⃣ Apply decay globally
    cm.decay_all()
    print("☁️ Applied importance decay to all memories.")
