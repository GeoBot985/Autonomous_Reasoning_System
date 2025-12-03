import json
import logging
import re
from typing import Any, Union, List, Dict

logger = logging.getLogger(__name__)

def parse_llm_json(text: str) -> Union[Dict[str, Any], List[Any], None]:
    """
    Parses a JSON string from an LLM response.
    Handles markdown code blocks (```json ... ```) and common formatting issues.

    Args:
        text: The raw string response from the LLM.

    Returns:
        The parsed JSON object (dict or list) or None if parsing fails.
    """
    if not text:
        return None

    # Remove markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()

    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding the first JSON object or array using regex
    try:
        # Match from the first { to the last } or [ to ]
        match_dict = re.search(r"\{.*\}", text, re.DOTALL)
        match_list = re.search(r"\[.*\]", text, re.DOTALL)

        candidate = None
        if match_dict and match_list:
            # If both found, take the one that starts earlier
            if match_dict.start() < match_list.start():
                candidate = match_dict.group()
            else:
                candidate = match_list.group()
        elif match_dict:
            candidate = match_dict.group()
        elif match_list:
            candidate = match_list.group()

        if candidate:
            return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from LLM text via regex: {e}")

    logger.warning(f"Could not parse JSON from text: {text[:100]}...")
    return None
