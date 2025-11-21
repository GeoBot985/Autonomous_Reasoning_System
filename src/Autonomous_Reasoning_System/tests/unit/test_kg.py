import pytest
from unittest.mock import MagicMock, patch
import time
from Autonomous_Reasoning_System.memory.kg_builder import KGBuilder
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.events import MemoryCreatedEvent
from Autonomous_Reasoning_System.memory.kg_validator import KGValidator

@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=MemoryStorage)
    storage.con = MagicMock()
    storage._write_lock = MagicMock()
    storage._write_lock.__enter__ = MagicMock()
    storage._write_lock.__exit__ = MagicMock()
    return storage

@pytest.fixture
def kg_builder(mock_storage):
    with patch('Autonomous_Reasoning_System.memory.kg_builder.LLMEngine') as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.generate_response.return_value = "" # Default return
        builder = KGBuilder(mock_storage)
        builder.llm = mock_instance # Replace with mock instance
        yield builder
        builder.stop()

def test_kg_validator():
    validator = KGValidator()

    # Valid triple
    assert validator.validate_triple("User", "owns", "Laptop") == True

    # Invalid relation (opinion)
    assert validator.validate_triple("User", "likes", "Pizza") == False

    # Invalid entity (ephemeral)
    assert validator.validate_triple("User", "met", "today") == False

    # Type constraint
    assert validator.validate_triple("User", "controls", "Device", "person", "device") == True
    assert validator.validate_triple("User", "controls", "Apple", "person", "fruit") == False # Invalid control

    # Canonicalization
    assert validator.canonicalize("  Apple  ") == "apple"

def test_kg_builder_process(kg_builder):
    # Mock LLM response
    kg_builder.llm.generate_response.return_value = "Alice | person | knows | Bob | person\nCharlie | person | owns | Car | object"

    event = MemoryCreatedEvent(
        text="Alice knows Bob and Charlie owns a Car.",
        timestamp="2023-01-01",
        source="test",
        memory_id="123",
        metadata={}
    )

    kg_builder.handle_event(event)

    # Allow thread to process
    time.sleep(0.5)

    # Verify calls to DB
    calls = kg_builder.storage.con.execute.call_args_list
    assert len(calls) > 0

    insert_triples = [c for c in calls if "INSERT OR IGNORE INTO triples" in str(c)]
    assert len(insert_triples) >= 2

    args0 = insert_triples[0][0][1]
    assert args0 == ('alice', 'knows', 'bob') or args0 == ('charlie', 'owns', 'car')

    # Verify types inserted
    insert_entities = [c for c in calls if "INSERT OR IGNORE INTO entities" in str(c)]
    assert len(insert_entities) >= 4
    # Check if one of arguments was 'person'
    types = [c[0][1][1] for c in insert_entities]
    assert 'person' in types
