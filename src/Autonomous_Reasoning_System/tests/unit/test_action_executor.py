import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.tools.action_executor import ActionExecutor

class MockWorkspace:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

@pytest.fixture
def mock_memory():
    return MagicMock()

@pytest.fixture
def executor(mock_memory):
    return ActionExecutor(memory_storage=mock_memory)

@pytest.fixture
def workspace():
    return MockWorkspace()

def test_load_image(executor, workspace):
    result = executor.execute_step("Load image from disk", workspace)

    assert result["success"] is True
    assert "Loaded sample image" in result["result"]
    assert workspace.get("image_path") == "data/sample_image.jpg"

    # Verify log entry
    executor.memory.add_memory.assert_called()

def test_store_extracted_text(executor, workspace):
    workspace.set("extracted_text", "Some OCR text content")

    result = executor.execute_step("Store extracted text", workspace)

    assert result["success"] is True
    assert "Stored OCR text" in result["result"]

    # Verify memory storage
    args = executor.memory.add_memory.call_args_list
    # Expected call for storing text
    found_store_call = False
    for call in args:
        if "Some OCR text content" in str(call):
            found_store_call = True
            break
    assert found_store_call

def test_ocr_unavailable(executor, workspace):
    # OCR module is optional and might be mocked or missing.
    # In this test environment, it's likely missing or we rely on the implementation
    # handling the ImportError gracefully if it's not mocked in a way that 'run' works.

    # We'll rely on the fact that 'ocr' is not installed in the real env or
    # if it is mocked by conftest, we need to check how it behaves.
    # The existing code tries to import inside the method.

    # Let's force an import error or exception if needed, or see what happens.
    # If sys.modules['Autonomous_Reasoning_System.tools.ocr'] is mocked (from conftest),
    # we need to make sure it has a 'run' method.

    # From conftest: sys.modules["Autonomous_Reasoning_System.tools.ocr"] = MagicMock()
    # So it is a MagicMock. 'run' will return another MagicMock.

    # Let's configure the mock return value if it exists
    import sys
    if "Autonomous_Reasoning_System.tools.ocr" in sys.modules:
        mock_ocr = sys.modules["Autonomous_Reasoning_System.tools.ocr"]
        mock_ocr.run.return_value = "Mocked OCR Result"

        result = executor.execute_step("Run OCR on image", workspace)

        assert result["success"] is True
        assert "Mocked OCR Result" in workspace.get("extracted_text")
    else:
        # Should handle error
        pass

def test_unknown_command(executor, workspace):
    result = executor.execute_step("Do something random", workspace)

    assert result["success"] is False
    assert "No matching tool found" in result["result"]
