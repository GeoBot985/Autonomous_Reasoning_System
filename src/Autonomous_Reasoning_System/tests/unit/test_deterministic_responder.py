import pytest
from unittest.mock import patch, MagicMock
from Autonomous_Reasoning_System.tools.deterministic_responder import DeterministicResponder
import datetime

@pytest.fixture
def responder():
    return DeterministicResponder()

def test_responder_time(responder):
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime.datetime(2023, 10, 27, 10, 0, 0)
        mock_datetime.now.return_value.strftime.return_value = "Friday, 27 October 2023 10:00:00"

        result = responder.run("what time is it?")
        assert "10:00:00" in result

        result = responder.run("what is the date today?")
        assert "October" in result

def test_responder_math_simple(responder):
    assert responder.run("2 + 2") == "4"
    assert responder.run("10 * 5") == "50"
    assert responder.run("100 / 2") == "50.0"

def test_responder_math_invalid(responder):
    # Not a math expression that is safe or recognized
    assert "I'm not sure" in responder.run("what is 2 plus 2")

def test_responder_wikipedia(responder):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"extract": "Python is a programming language."}
        mock_get.return_value = mock_response

        # The code uses the whole query as the wikipedia page title
        # q.replace(' ', '_')
        result = responder.run("Python_(programming_language)")
        assert result == "Python is a programming language."

def test_responder_wikipedia_fail(responder):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.ok = False
        mock_get.return_value = mock_response

        # Ensure the query string doesn't accidentally contain time keywords like 'now'
        result = responder.run("ArbitraryQueryString")
        assert "I'm not sure" in result

def test_responder_unknown(responder):
    result = responder.run("what is the meaning of life?")
    assert "I'm not sure" in result
