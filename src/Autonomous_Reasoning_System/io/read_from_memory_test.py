from Autonomous_Reasoning_System.memory.storage import MemoryStorage
import pandas as pd
import textwrap

def read_document_from_memory(source_name: str = None, limit: int = 20):
    memory = MemoryStorage()
    df = memory.get_all_memories()

    if not isinstance(df, pd.DataFrame):
        print("⚠️ Storage did not return a DataFrame.")
        return

    print(f"🧠 Retrieved {len(df)} records with columns: {list(df.columns)}")

    # Try to filter by source name if provided
    if source_name:
        df = df[df["source"].str.contains(source_name, case=False, na=False)]

    if df.empty:
        print(f"No matching entries for '{source_name}', showing first {limit} records instead.\n")
        df = memory.get_all_memories().head(limit)

    for i, row in df.iterrows():
        print(f"\n--- Memory {i+1} / {len(df)} ---")
        text = row.get("text", "")
        print("\n".join(textwrap.wrap(text, width=100)))
        print()

if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else None
    read_document_from_memory(source)
