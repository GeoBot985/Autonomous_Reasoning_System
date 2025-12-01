
import pytest
from unittest.mock import MagicMock
from Autonomous_Reasoning_System.control.router import Router, IntentFamily

def test_router_web_search_resolution():
    """
    Verifies that web search related queries are correctly routed to the WEB_SEARCH family
    and the 'google_search' pipeline.
    """
    mock_dispatcher = MagicMock()

    # Setup mock dispatcher to return specific intent for testing
    # In a real scenario, analyze_intent would return 'google', 'web_search', etc.
    def mock_dispatch(tool_name, arguments=None):
        if tool_name == "analyze_intent":
            text = arguments.get("text", "").lower()
            if "google" in text:
                return {"status": "success", "data": {"intent": "google", "family": "web_search"}}
            if "search" in text:
                return {"status": "success", "data": {"intent": "web_search", "family": "web_search"}}
            return {"status": "success", "data": {"intent": "unknown"}}
        return {"status": "success", "data": {}}

    mock_dispatcher.dispatch.side_effect = mock_dispatch

    router = Router(dispatcher=mock_dispatcher)

    # Test cases
    test_queries = [
        ("google python 3.12 release date", "google"),
        ("search web for autonomous agents", "web_search"),
    ]

    for query, expected_intent in test_queries:
        result = router.resolve(query)

        assert result["intent"] == expected_intent
        assert result["family"] == IntentFamily.WEB_SEARCH
        assert result["pipeline"] == ["google_search"]

def test_router_web_search_fallback():
    """
    Verifies that if the intent analyzer misses the family but catches the intent,
    the Router still maps it correctly using its internal map.
    """
    mock_dispatcher = MagicMock()

    # Analyzer returns correct intent but NO family
    def mock_dispatch(tool_name, arguments=None):
        if tool_name == "analyze_intent":
            return {"status": "success", "data": {"intent": "google", "family": "unknown"}}
        return {"status": "success", "data": {}}

    mock_dispatcher.dispatch.side_effect = mock_dispatch

    router = Router(dispatcher=mock_dispatcher)

    result = router.resolve("google something")

    assert result["intent"] == "google"
    # Router should use its internal map to find the family for 'google'
    assert result["family"] == IntentFamily.WEB_SEARCH
    assert result["pipeline"] == ["google_search"]
