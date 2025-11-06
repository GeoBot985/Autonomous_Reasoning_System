from Autonomous_Reasoning_System.memory.context_builder import ContextBuilder

def main():
    cb = ContextBuilder(top_k=3)
    ctx = cb.build_context("reasoning system design")
    print("\n=== Generated Working Context ===\n")
    print(ctx)

if __name__ == "__main__":
    main()
