import duckdb
import os

db_path = "data/memory.duckdb"

print(f"🧹 Checking database at {db_path}...")

try:
    con = duckdb.connect(db_path)
    
    # Check if table exists
    table_exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'memory'"
    ).fetchone()[0] > 0

    if not table_exists:
        print("✅ Table 'memory' does not exist yet. Database is fresh/empty.")
        print("   (You can skip this step and run the interface directly.)")
    else:
        print("🔍 Scanning for poisoned rows...")
        poison_patterns = [
            "Please summarize%",
            "Summarize my%",
            "Can you summarize%",
            "What can you tell%",
            "Describe%"
        ]

        count = 0
        for p in poison_patterns:
            n = con.execute("SELECT count(*) FROM memory WHERE text ILIKE ?", (p,)).fetchone()[0]
            if n > 0:
                print(f"   Found {n} bad rows matching '{p}'")
                con.execute("DELETE FROM memory WHERE text ILIKE ?", (p,))
                count += n

        # Clean orphaned vectors
        if count > 0:
            con.execute("DELETE FROM vectors WHERE id NOT IN (SELECT id FROM memory)")
            print(f"✅ Removed {count} bad memories and cleaned vectors.")
        else:
            print("✅ Database is clean. No poisoned commands found.")

except Exception as e:
    print(f"⚠️ Error accessing DB: {e}")