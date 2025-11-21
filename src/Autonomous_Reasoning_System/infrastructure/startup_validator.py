import os
import sys
import logging
import duckdb
import pandas as pd
import faiss
import pickle
from pathlib import Path
from Autonomous_Reasoning_System.infrastructure import config

logger = logging.getLogger(__name__)

def validate_startup():
    """
    Performs a fail-safe boot check.
    If any critical data file is missing or corrupted, it HALTS execution.
    It does NOT auto-rebuild.
    """
    print("[Startup Validator] Verifying system integrity...")

    # Determine Data Directory
    # We assume MEMORY_DB_PATH is like "data/memory.duckdb"
    db_path = Path(config.MEMORY_DB_PATH)
    data_dir = db_path.parent

    if not data_dir.exists():
        print(f"CRITICAL ERROR: Data directory not found at {data_dir}")
        sys.exit(1)

    # 1. Check DuckDB
    if not db_path.exists():
        print(f"CRITICAL ERROR: DuckDB file missing at {db_path}")
        sys.exit(1)

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        # Check for required tables
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        required_tables = ["memory", "goals"]
        for rt in required_tables:
            if rt not in table_names:
                print(f"CRITICAL ERROR: DuckDB missing table '{rt}'")
                sys.exit(1)
        con.close()
    except Exception as e:
        print(f"CRITICAL ERROR: DuckDB corrupted or unreadable: {e}")
        sys.exit(1)

    # 2. Check Parquet Files
    parquet_files = ["memory.parquet", "goals.parquet", "episodes.parquet"]
    for p_file in parquet_files:
        p_path = data_dir / p_file
        if not p_path.exists():
            print(f"CRITICAL ERROR: Parquet file missing: {p_path}")
            sys.exit(1)
        try:
            pd.read_parquet(p_path)
        except Exception as e:
            print(f"CRITICAL ERROR: Parquet file corrupted: {p_path} ({e})")
            sys.exit(1)

    # 3. Check FAISS Index
    faiss_path = data_dir / "vector_index.faiss"
    if not faiss_path.exists():
        print(f"CRITICAL ERROR: FAISS index missing at {faiss_path}")
        sys.exit(1)
    try:
        faiss.read_index(str(faiss_path))
    except Exception as e:
        print(f"CRITICAL ERROR: FAISS index corrupted: {e}")
        sys.exit(1)

    # 4. Check Metadata Pickle
    meta_path = data_dir / "vector_meta.pkl"
    if not meta_path.exists():
        print(f"CRITICAL ERROR: Metadata pickle missing at {meta_path}")
        sys.exit(1)
    try:
        with open(meta_path, "rb") as f:
            pickle.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Metadata pickle corrupted: {e}")
        sys.exit(1)

    print("[Startup Validator] System integrity verified. Proceeding to boot.")
