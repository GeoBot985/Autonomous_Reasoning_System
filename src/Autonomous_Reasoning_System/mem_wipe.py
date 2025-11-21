import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

# =========================================================================
# HARDCODED PATH FIX: 
# This ensures Python can find the 'Autonomous_Reasoning_System' package 
# even when run as a script from within the package directory.
# =========================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now the imports will work
from Autonomous_Reasoning_System.infrastructure import config 

# --- Configuration ---
# =========================================================================
# HARDCODED DATA DIRECTORY: 
# Based on your confirmed structure:
# D:\Projects\Autonomous_Reasoning_System\src\data
# We use the absolute path relative to the script location (D:\...src\...)
# =========================================================================
# This resolves to D:\Projects\Autonomous_Reasoning_System\src\data
DATA_DIR = Path(os.path.dirname(__file__)).parent / "data"

DB_PATH = DATA_DIR / "memory.duckdb"

PARQUET_FILES = [
    DATA_DIR / "memory.parquet",
    DATA_DIR / "goals.parquet",
    DATA_DIR / "episodes.parquet"
]

DUCKDB_TABLES = [
    "memory",
    "goals",
    "vectors" # Assuming the vectors table exists in memory.duckdb
]

def wipe_all_memory():
    """Wipes all data from DuckDB tables and Parquet files."""
    print("⚠️  [WIPE] Starting FULL memory wipe...")

    # 1. Truncate DuckDB Tables
    if DB_PATH.exists():
        try:
            con = duckdb.connect(str(DB_PATH))
            print(f"✅ Connected to DuckDB at: {DB_PATH}")

            for table in DUCKDB_TABLES:
                try:
                    con.execute(f"TRUNCATE TABLE {table};")
                    print(f"   -> TRUNCATED table: {table}")
                except duckdb.CatalogException:
                    print(f"   -> Skipping table: {table} (Does not exist)")
                except Exception as e:
                    print(f"   ❌ Error truncating {table}: {e}")

            con.close()
            print("✅ DuckDB memory cleared and connection closed.")
        except Exception as e:
            print(f"❌ Failed to connect or truncate DuckDB: {e}")
            sys.exit(1)
    else:
        print(f"⚠️ DuckDB file not found at {DB_PATH}. Skipping DB wipe.")


    # 2. Recreate Empty Parquet Files
    for p_path in PARQUET_FILES:
        try:
            if p_path.exists():
                # Read columns and save an empty DataFrame to zero-out the file
                # Use pyarrow engine for reliability if pandas default fails
                df_schema = pd.read_parquet(p_path).head(0)
                df_schema.to_parquet(p_path)
                print(f"✅ ZEROED out Parquet file: {p_path.name}")
            else:
                print(f"⚠️ Parquet file not found at {p_path}. Skipping file.")
        except Exception as e:
            print(f"❌ Failed to zero-out Parquet file {p_path.name}: {e}")


    print("\n🎉 [WIPE] Memory wipe complete! Relaunch the agent to hydrate the clean memory.")


if __name__ == "__main__":
    
    # --- CONFIRMATION CHECK ---
    print("\n⚠️  WARNING: DESTRUCTIVE OPERATION ⚠️")
    print("This action will **PERMANENTLY DELETE ALL STORED DATA** (facts, goals, episodes).")
    confirm = input("Type 'WIPE' to proceed with memory deletion: ")
    if confirm.upper() == "WIPE":
        wipe_all_memory()
    else:
        print("Operation aborted.")