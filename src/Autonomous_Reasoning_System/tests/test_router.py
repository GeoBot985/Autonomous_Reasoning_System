# tests/test_router.py
from Autonomous_Reasoning_System.cognition.router import Router

def main():
    router = Router()

    tests = [
        "Remind me to test the camera tomorrow morning.",
        "What was I working on yesterday?",
        "Summarize everything I learned today.",
        "Open the monthly budget file.",
        "Reflect on the last conversation about memory design.",
        "Create a plan to integrate the reasoning engine.",
    ]

    print("\n=== Router Test Results ===\n")
    for t in tests:
        decision = router.route(t)
        print(f"🧩 Input: {t}")
        print(f"🧠 Intent: {decision['intent']}")
        print(f"🧭 Pipeline: {decision['pipeline']}")
        print(f"💬 Reason: {decision['reason']}\n")

if __name__ == "__main__":
    main()
