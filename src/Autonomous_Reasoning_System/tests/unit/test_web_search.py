import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.tools.web_search import perform_google_search

@pytest.fixture
def mock_playwright():
    with patch('Autonomous_Reasoning_System.tools.web_search.sync_playwright') as mock:
        yield mock

def test_perform_google_search_success(mock_playwright):
    """Test successful google search with results."""
    # Setup mock chain
    mock_p = mock_playwright.return_value.__enter__.return_value
    mock_browser = mock_p.chromium.launch.return_value
    mock_page = mock_browser.new_context.return_value.new_page.return_value

    # Mock evaluate return values
    # First call is main results, Second call is quick answer
    # But wait, evaluate is called twice in the code.
    # 1. results extraction
    # 2. quick answer extraction

    expected_results = [
        "Title: Test Title\nLink: http://example.com\nSnippet: Test Snippet"
    ]

    def evaluate_side_effect(script):
        if "document.querySelectorAll('.g')" in script:
            return expected_results
        if "document.querySelector('.Iz6qV')" in script:
            return "Quick Answer: 42"
        return None

    mock_page.evaluate.side_effect = evaluate_side_effect

    result = perform_google_search("test query")

    # Verify browser interaction
    mock_page.goto.assert_called()
    assert "google.com/search?q=test+query" in mock_page.goto.call_args[0][0]

    # Verify results
    assert "Quick Answer: 42" in result
    assert "Title: Test Title" in result
    assert "Snippet: Test Snippet" in result

def test_perform_google_search_no_results(mock_playwright):
    """Test search with no results found."""
    mock_p = mock_playwright.return_value.__enter__.return_value
    mock_page = mock_p.chromium.launch.return_value.new_context.return_value.new_page.return_value

    mock_page.evaluate.return_value = [] # No results for both calls

    result = perform_google_search("weird query")

    assert result == "No search results found."

def test_perform_google_search_error(mock_playwright):
    """Test search when exception occurs."""
    mock_p = mock_playwright.return_value.__enter__.return_value
    mock_browser = mock_p.chromium.launch.return_value
    # Make launching raise an error
    mock_browser.new_context.side_effect = Exception("Browser crashed")

    result = perform_google_search("crash me")

    assert "Error performing search" in result
    assert "Browser crashed" in result
