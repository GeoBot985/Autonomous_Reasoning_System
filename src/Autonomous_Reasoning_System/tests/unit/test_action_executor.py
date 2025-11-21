
import pytest
from unittest.mock import MagicMock
from Autonomous_Reasoning_System.tools.action_executor import ActionExecutor

class MockWorkspace:
    def __init__(self):
        self.data = {}
    def get(self, key, default=None):
        return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value

@pytest.fixture
def mock_memory_storage():
    return MagicMock()

@pytest.fixture
def action_executor(mock_memory_storage):
    return ActionExecutor(memory_storage=mock_memory_storage)

def test_action_executor_init(mock_memory_storage):
    executor = ActionExecutor(memory_storage=mock_memory_storage)
    assert executor.memory == mock_memory_storage

def test_action_executor_ocr_mock(action_executor):
    import sys
    # Configure the global mock to return the expected string
    sys.modules["Autonomous_Reasoning_System.tools.ocr"].run.return_value = "MOCK OCR TEXT"

    workspace = MockWorkspace()
    step_description = "Extract text using OCR from the image"

    result = action_executor.execute_step(step_description, workspace)

    assert result["success"] is True
    assert "OCR extracted text" in result["result"]
    assert workspace.get("extracted_text") == "MOCK OCR TEXT"
    action_executor.memory.add_memory.assert_called()

def test_action_executor_load_image(action_executor):
    workspace = MockWorkspace()
    step_description = "Load image from data/sample.jpg"

    result = action_executor.execute_step(step_description, workspace)

    assert result["success"] is True
    assert "Loaded sample image" in result["result"]
    assert workspace.get("image_path") == "data/sample_image.jpg"
    action_executor.memory.add_memory.assert_called()

def test_action_executor_store_text(action_executor):
    workspace = MockWorkspace()
    workspace.set("extracted_text", "some extracted text")
    step_description = "Store extracted text to memory"

    result = action_executor.execute_step(step_description, workspace)

    assert result["success"] is True
    assert "Stored OCR text" in result["result"]
    action_executor.memory.add_memory.assert_called()

def test_action_executor_unknown_action(action_executor):
    workspace = MockWorkspace()
    step_description = "Do some unknown magic"

    result = action_executor.execute_step(step_description, workspace)

    assert result["success"] is False
    assert "No matching tool found" in result["result"]
    action_executor.memory.add_memory.assert_called()

def test_action_executor_exception(action_executor):
    workspace = MockWorkspace()
    # Force an exception by passing a workspace that raises error on get
    bad_workspace = MagicMock()
    bad_workspace.get.side_effect = Exception("Workspace error")

    result = action_executor.execute_step("ocr", bad_workspace)

    assert result["success"] is False
    assert "Error executing step" in result["result"]
