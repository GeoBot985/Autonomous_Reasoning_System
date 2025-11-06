# tests/test_core_loop.py
from Autonomous_Reasoning_System.control.core_loop import CoreLoop


def main():
    loop = CoreLoop()
    inputs = [
        "Remind me to test the camera tomorrow.",
        "Reflect on my recent work.",
        "Summarize today's progress.",
        "Reflect on how I handle reminders.",
    ]

    print("\n=== Core Loop (Reflection Integrated) ===\n")
    for text in inputs:
        result = loop.run_once(text)
        if result["reflection_data"]:
            print(f"💭 Reflection: {result['reflection_data']}\n")
        else:
            print(f"✅ Output summary: {result['summary']}\n")


if __name__ == "__main__":
    main()
