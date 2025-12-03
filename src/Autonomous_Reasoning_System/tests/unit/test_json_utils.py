import pytest
from Autonomous_Reasoning_System.utils.json_utils import parse_llm_json

def test_parse_simple_json():
    text = '{"key": "value", "num": 1}'
    result = parse_llm_json(text)
    assert result == {"key": "value", "num": 1}

def test_parse_markdown_json():
    text = '```json\n{"key": "value"}\n```'
    result = parse_llm_json(text)
    assert result == {"key": "value"}

def test_parse_json_with_text_around():
    text = 'Here is the json: {"key": "value"} thank you.'
    result = parse_llm_json(text)
    assert result == {"key": "value"}

def test_parse_list_json():
    text = '[1, 2, 3]'
    result = parse_llm_json(text)
    assert result == [1, 2, 3]

def test_parse_invalid_json():
    text = 'Not a json'
    result = parse_llm_json(text)
    assert result is None

def test_parse_nested_structure():
    text = 'Some text {"a": [1, 2], "b": {"c": 3}} end.'
    result = parse_llm_json(text)
    assert result == {"a": [1, 2], "b": {"c": 3}}
