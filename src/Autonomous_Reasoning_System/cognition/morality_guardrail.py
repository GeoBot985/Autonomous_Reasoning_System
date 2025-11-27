import logging

logger = logging.getLogger(__name__)

def check_safety(text: str) -> dict:
    """
    Checks input text for potential safety violations.
    Returns a dictionary with 'safe' (bool) and 'reason' (str).
    """
    unsafe_keywords = [
        "destroy humans", "kill all", "harm humans", "delete all files", "rm -rf /"
    ]

    normalized_text = text.lower()

    for keyword in unsafe_keywords:
        if keyword in normalized_text:
            logger.warning(f"Safety violation detected: {keyword}")
            return {
                "safe": False,
                "reason": f"Contains unsafe keyword: {keyword}"
            }

    return {"safe": True, "reason": "No violations found"}
