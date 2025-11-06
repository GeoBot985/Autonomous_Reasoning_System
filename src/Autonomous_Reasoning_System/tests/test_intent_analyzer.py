# tests/test_intent_analyzer.py
from Autonomous_Reasoning_System.cognition.intent_analyzer import IntentAnalyzer


def main():
    analyzer = IntentAnalyzer()
    tests = [
        "Remind me to test the camera tomorrow morning.",
        "What was I working on yesterday?",
        "Summarize today's work.",
        "Reflect on my last project.",
        "Open the monthly budget file.",
        "Create a plan for the next step.",
        "Hey Tyrone!",
        "Exit the session now."
    ]

    print("\n=== Intent Analyzer Test ===\n")
    for t in tests:
        result = analyzer.analyze(t)
        print(f"🧩 Input: {t}")
        print(f"🧠 Intent: {result['intent']}")
        print(f"🏷️ Entities: {result.get('entities', {})}")
        print(f"💬 Reason: {result.get('reason', '(no reason)')}\n")


if __name__ == "__main__":
    main()
