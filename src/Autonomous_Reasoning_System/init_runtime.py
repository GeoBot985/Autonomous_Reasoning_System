import os
import sys
import shutil
import argparse
import logging
import duckdb
import pandas as pd
import faiss
import pickle
from pathlib import Path
from Autonomous_Reasoning_System.infrastructure import config

# Setup basic logging for CLI tool
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def bootstrap_runtime():
    """
    Creates a fresh, empty memory store.
    Safe to call if data directory is missing.
    """
    db_path = Path(config.MEMORY_DB_PATH)
    data_dir = db_path.parent

    # Create data directory
    if not data_dir.exists():
        logger.info(f"Creating data directory at {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create DuckDB
    # We can rely on MemoryStorage's init_db logic, but here we do it explicitly to ensure clean slate without side effects of other classes.
    if db_path.exists():
        logger.warning(f"DuckDB already exists at {db_path}. Skipping creation.")
    else:
        try:
            con = duckdb.connect(str(db_path))
            con.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id VARCHAR PRIMARY KEY,
                    text VARCHAR,
                    memory_type VARCHAR,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    importance DOUBLE,
                    scheduled_for TIMESTAMP,
                    status VARCHAR,
                    source VARCHAR
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id VARCHAR PRIMARY KEY,
                    text VARCHAR,
                    priority INTEGER,
                    status VARCHAR,
                    steps VARCHAR,
                    metadata VARCHAR,
                    plan_id VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            con.close()
            logger.info("Initialized fresh DuckDB.")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            sys.exit(1)

    # 2. Create Empty Parquet Files
    parquet_schemas = {
        "memory.parquet": ["id", "text", "memory_type", "created_at", "last_accessed", "importance", "scheduled_for", "status", "source"],
        "goals.parquet": ["id", "text", "priority", "status", "steps", "metadata", "created_at", "updated_at"],
        "episodes.parquet": ["episode_id", "start_time", "end_time", "summary", "importance", "vector"]
    }

    for filename, cols in parquet_schemas.items():
        p_path = data_dir / filename
        if not p_path.exists():
            df = pd.DataFrame(columns=cols)
            df.to_parquet(p_path)
            logger.info(f"Created empty {filename}.")

    # 3. Create Empty FAISS Index
    faiss_path = data_dir / "vector_index.faiss"
    if not faiss_path.exists():
        # Dimension 384 is standard for all-MiniLM-L6-v2 which seems to be implied (default in embeddings.py usually)
        # But let's verify if we can find the dimension. vector_store.py defaults to 384.
        index = faiss.IndexFlatIP(384)
        faiss.write_index(index, str(faiss_path))
        logger.info("Created empty FAISS index.")

    # 4. Create Empty Metadata Pickle
    meta_path = data_dir / "vector_meta.pkl"
    if not meta_path.exists():
        with open(meta_path, "wb") as f:
            pickle.dump([], f)
        logger.info("Created empty metadata pickle.")

    logger.info("Initialized fresh memory store (first launch).")


def rebuild_runtime():
    """
    Completely wipes the data directory and rebuilds it.
    Requires explicit operator confirmation.
    """
    db_path = Path(config.MEMORY_DB_PATH)
    data_dir = db_path.parent

    print("\n⚠️  WARNING: DESTRUCTIVE OPERATION ⚠️")
    print(f"You are about to DELETE ALL MEMORY in {data_dir}.")
    print("This action CANNOT be undone.")

    confirm = input("Type 'DELETE' to confirm: ")
    if confirm != "DELETE":
        print("Operation aborted.")
        return

    confirm2 = input("Are you absolutely sure? (y/n): ")
    if confirm2.lower() != "y":
        print("Operation aborted.")
        return

    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            logger.info(f"Deleted directory: {data_dir}")
        except Exception as e:
            logger.error(f"Failed to delete {data_dir}: {e}")
            sys.exit(1)

    bootstrap_runtime()
    logger.info("Operator-triggered rebuild — developer not responsible.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize or rebuild ARS runtime environment.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the memory store.")

    args = parser.parse_args()

    if args.rebuild:
        rebuild_runtime()
    else:
        # If run directly without args, maybe we want to just bootstrap?
        # But the tool description says "Add CLI command... init_runtime --rebuild".
        # I'll make it so running it checks/bootstraps safely.
        bootstrap_runtime()
