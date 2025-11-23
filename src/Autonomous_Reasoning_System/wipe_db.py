import duckdb
import os

try:
    print("💥 Connecting to DB to perform wipe...")
    con = duckdb.connect("data/memory.duckdb")
    
    # Drop tables
    print("💥 Dropping tables...")
    con.execute("DROP TABLE IF EXISTS vectors")
    con.execute("DROP TABLE IF EXISTS memory")
    con.execute("DROP TABLE IF EXISTS triples")
    con.execute("DROP TABLE IF EXISTS plans")
    
    # Shrink file
    print("🧹 Vacuuming...")
    con.execute("VACUUM")
    print("✅ Database wiped clean.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("If this fails, close ALL python terminals and delete 'data/memory.duckdb' manually.")