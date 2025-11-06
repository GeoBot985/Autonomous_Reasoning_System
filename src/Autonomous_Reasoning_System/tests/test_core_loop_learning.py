from Autonomous_Reasoning_System.control.core_loop import CoreLoop

def test_core_loop_learning_cycle():
    tyrone = CoreLoop()
    result = tyrone.run_once("Reflect on how confident you feel about recent progress.")
    print("\n✅ CoreLoop test completed.")
    print("Decision:", result["decision"]["intent"])
    print("Reflection:", result["reflection_data"])
    print("Summary:", result["summary"])

if __name__ == "__main__":
    test_core_loop_learning_cycle()
