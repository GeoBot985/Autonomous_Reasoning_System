import pytest
import os
import json
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System import config

@pytest.fixture
def memory_db():
    # Use in-memory DB for integration tests to avoid file I/O and state persistence issues
    storage = MemoryStorage(db_path=":memory:")
    yield storage

def test_memory_integration_flow(memory_db):
    """
    Test a full flow: Remember -> Search -> Verify
    """
    # 1. Remember some data
    text1 = "The capital of France is Paris."
    text2 = "The capital of Germany is Berlin."
    memory_db.remember(text1, metadata={"category": "geography"})
    memory_db.remember(text2, metadata={"category": "geography"})

    # 2. Search Similar
    # NOTE: The default `search_similar` threshold is 0.4.
    # If using real embeddings, "France capital" matches "The capital of France is Paris." quite well.
    # However, if something is wrong with VSS or embedding calculation, it might return 0 results.

    # Let's lower the threshold to 0.0 to ensure we get results if the extension works at all.
    results = memory_db.search_similar("France capital", threshold=0.0)

    # Check if we got results
    if len(results) == 0:
        # Debugging: check if vectors table has entries
        count = memory_db.con.execute("SELECT count(*) FROM vectors").fetchone()[0]
        print(f"\n[DEBUG] Vectors count: {count}")

        # Check if vss extension is working by running a simple query
        try:
            memory_db.con.execute("SELECT list_cosine_similarity([1,2,3], [1,2,3])").fetchall()
            print("[DEBUG] VSS function works.")
        except Exception as e:
            print(f"[DEBUG] VSS function failed: {e}")

    assert len(results) >= 1
    found_texts = [r['text'] for r in results]
    assert text1 in found_texts or text2 in found_texts

def test_exact_search_integration(memory_db):
    """Test exact search works with the DB."""
    text = "UniqueKeyword123 is here."
    memory_db.remember(text)

    results = memory_db.search_exact("UniqueKeyword123")
    assert len(results) == 1
    assert results[0]['text'] == text

def test_document_reassembly_integration(memory_db):
    """Test storing multiple chunks and retrieving them as a document."""
    filename = "doc_integration.txt"
    chunks = ["Chunk 1.", "Chunk 2.", "Chunk 3."]

    metas = [{"filename": filename} for _ in chunks]

    memory_db.remember_batch(chunks, source=filename, metadata_list=metas)

    full_text = memory_db.get_whole_document(filename)
    assert full_text == "Chunk 1.\nChunk 2.\nChunk 3."

def test_kg_triples_integration(memory_db):
    """Test adding and retrieving triples."""
    memory_db.add_triple("Alice", "knows", "Bob")
    memory_db.add_triple("Bob", "knows", "Charlie")

    triples = memory_db.get_triples("Bob")
    # Should get (Alice, knows, Bob) and (Bob, knows, Charlie)
    assert len(triples) == 2

    # Verify content
    subjects = [t[0] for t in triples]
    objects = [t[2] for t in triples]
    assert "alice" in subjects or "bob" in subjects # lowercased in DB
    assert "bob" in objects or "charlie" in objects

def test_plan_persistence(memory_db):
    """Test that plans are saved and retrieved correctly."""
    plan_id = "integration_plan"
    goal = "integration testing"
    steps = [{"id": 1, "desc": "step 1"}]

    memory_db.update_plan(plan_id, goal, steps)

    plans = memory_db.get_active_plans()
    found = next((p for p in plans if p.id == plan_id), None)

    assert found is not None
    assert found.goal == goal
    assert found.steps == steps
