from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter


def main():
    ri = ReflectionInterpreter()
    queries = [
        "What patterns do you see in my recent work?",
        "Reflect on how I handle reminders.",
        "How could I improve my planning?",
    ]

    print("\n=== Reflection Interpreter Test ===\n")
    for q in queries:
        result = ri.interpret(q)
        print(f"🧩 Query: {q}")
        print(f"🪞 Summary: {result['summary']}")
        print(f"💡 Insight: {result['insight']}")
        print(f"📈 Confidence Change: {result['confidence_change']}\n")


if __name__ == "__main__":
    main()
