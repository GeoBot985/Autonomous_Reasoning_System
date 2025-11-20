from Autonomous_Reasoning_System.memory.retrieval_orchestrator import RetrievalOrchestrator

def run_tests():
    r = RetrievalOrchestrator()
    queries = [
        "Show me the VisionAssist report",
        "What did I learn about quantization?",
        "Summarize all documents mentioning Moondream"
    ]

    for q in queries:
        print("\n=== QUERY:", q, "===")
        results = r.retrieve(q)
        # handle dataframe vs list/str
        if hasattr(results, "head"):
            print(results.head(2))
        elif isinstance(results, dict):
            print("Hybrid:", {k: len(v) for k,v in results.items()})
        else:
            print(str(results)[:500])

if __name__ == "__main__":
    run_tests()
