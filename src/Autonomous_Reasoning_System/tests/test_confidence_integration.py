# tests/test_confidence_integration.py
from Autonomous_Reasoning_System.control.core_loop import CoreLoop

def main():
    loop = CoreLoop()
    queries = [
        "Reflect on how confident you feel about recent progress.",
        "Reflect on a situation where performance decreased.",
        "Reflect neutrally on recent routine tasks."
    ]

    print("\n=== Confidence Integration Test ===\n")
    for q in queries:
        result = loop.run_once(q)
        print(f"💭 Reflection: {result['reflection_data']}\n")

if __name__ == "__main__":
    main()
