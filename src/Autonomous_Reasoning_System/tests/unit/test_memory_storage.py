import pytest
from unittest.mock import MagicMock, patch, ANY
import json
from Autonomous_Reasoning_System.memory.storage import MemoryStorage

@pytest.fixture
def mock_duckdb():
    with patch('duckdb.connect') as mock_connect:
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        yield mock_con

@pytest.fixture
def memory_storage(mock_duckdb, mock_embedding_model):
    with patch('Autonomous_Reasoning_System.memory.storage.TextEmbedding', return_value=mock_embedding_model):
        storage = MemoryStorage(db_path=":memory:")
        return storage

def test_init_schema(memory_storage, mock_duckdb):
    """Test that schema initialization commands are executed."""
    # We expect several execute calls for schema creation
    # checking for 'CREATE TABLE IF NOT EXISTS memory'
    calls = [args[0] for args, _ in mock_duckdb.execute.call_args_list]
    assert any("CREATE TABLE IF NOT EXISTS memory" in call for call in calls)
    assert any("CREATE TABLE IF NOT EXISTS vectors" in call for call in calls)
    assert any("CREATE TABLE IF NOT EXISTS triples" in call for call in calls)
    assert any("CREATE TABLE IF NOT EXISTS plans" in call for call in calls)

def test_remember_batch(memory_storage, mock_duckdb):
    """Test remember_batch inserts data into memory and vectors tables."""
    texts = ["hello world", "another memory"]
    metadata_list = [{"meta": 1}, {"meta": 2}]

    memory_storage.remember_batch(texts, metadata_list=metadata_list)

    # Verify transaction calls
    mock_duckdb.execute.assert_any_call("BEGIN TRANSACTION")
    mock_duckdb.execute.assert_any_call("COMMIT")

    # Verify insert calls
    # We can inspect the arguments passed to execute
    # INSERT INTO memory ...
    insert_memory_calls = [
        call for call in mock_duckdb.execute.call_args_list
        if "INSERT INTO memory VALUES" in call[0][0]
    ]
    assert len(insert_memory_calls) == 2

    # INSERT INTO vectors ...
    insert_vector_calls = [
        call for call in mock_duckdb.execute.call_args_list
        if "INSERT INTO vectors VALUES" in call[0][0]
    ]
    assert len(insert_vector_calls) == 2

def test_get_whole_document(memory_storage, mock_duckdb):
    """Test retrieving and reassembling a document."""
    filename = "test_doc.txt"

    # Mock return value of fetchall
    mock_duckdb.execute.return_value.fetchall.return_value = [
        ("Part 1 content",),
        ("Part 2 content",)
    ]

    result = memory_storage.get_whole_document(filename)

    assert result == "Part 1 content\nPart 2 content"

    # Verify query
    args, _ = mock_duckdb.execute.call_args
    query = args[0]
    assert "SELECT text" in query
    assert "WHERE source = ?" in query
    assert "ORDER BY created_at ASC" in query

def test_add_triple(memory_storage, mock_duckdb):
    """Test adding a triple."""
    memory_storage.add_triple("Subj", "Rel", "Obj")
    mock_duckdb.execute.assert_called_with(
        "INSERT OR IGNORE INTO triples VALUES (?, ?, ?)",
        ("subj", "rel", "obj")
    )

def test_update_plan_new(memory_storage, mock_duckdb):
    """Test creating a new plan."""
    # Mock select to return None (plan doesn't exist)
    mock_duckdb.execute.return_value.fetchone.return_value = None

    plan_id = "plan1"
    goal = "do something"
    steps = ["step1", "step2"]

    memory_storage.update_plan(plan_id, goal, steps)

    # Verify insert
    args_list = mock_duckdb.execute.call_args_list
    insert_call = [call for call in args_list if "INSERT INTO plans VALUES" in call[0][0]]
    assert len(insert_call) == 1

def test_update_plan_existing(memory_storage, mock_duckdb):
    """Test updating an existing plan."""
    # Mock select to return True (plan exists)
    mock_duckdb.execute.return_value.fetchone.return_value = (1,)

    plan_id = "plan1"
    goal = "do something"
    steps = ["step1", "step2"]

    memory_storage.update_plan(plan_id, goal, steps)

    # Verify update
    args_list = mock_duckdb.execute.call_args_list
    update_call = [call for call in args_list if "UPDATE plans SET" in call[0][0]]
    assert len(update_call) == 1

def test_search_similar(memory_storage, mock_duckdb):
    """Test similarity search."""
    # Mock fetchall result
    mock_duckdb.execute.return_value.fetchall.return_value = [
        ("Result text", "episodic", "2023-01-01", 0.9)
    ]

    results = memory_storage.search_similar("query")

    assert len(results) == 1
    assert results[0]["text"] == "Result text"
    assert results[0]["score"] == 0.9

def test_search_exact(memory_storage, mock_duckdb):
    """Test exact keyword search."""
    mock_duckdb.execute.return_value.fetchall.return_value = [
        ("Exact match", "episodic", "2023-01-01")
    ]

    results = memory_storage.search_exact("keyword")

    assert len(results) == 1
    assert results[0]["text"] == "Exact match"
    assert results[0]["score"] == 1.0
