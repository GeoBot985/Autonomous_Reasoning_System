from Autonomous_Reasoning_System.control.core_loop import CoreLoop
from Autonomous_Reasoning_System.infrastructure.logging_utils import setup_logging

import sys

def main():
    setup_logging()
    tyrone = CoreLoop()
    print("\n🚀 Tyrone is online and ready.\n")
    print("Type directly to interact. Type 'exit' to quit.\n")

    # If input is piped or passed as a CLI argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"You: {query}")
        result = tyrone.run_once(query)
        summary = None
        if result:
            reflection_data = result.get("reflection_data")
            if reflection_data and isinstance(reflection_data, dict):
                summary = reflection_data.get("summary")
            summary = summary or result.get("summary")

        print("\nTyrone:", summary or "I’ve noted that down.")

        return

    # Otherwise, interactive loop
    tyrone.run_interactive()


if __name__ == "__main__":
    main()
