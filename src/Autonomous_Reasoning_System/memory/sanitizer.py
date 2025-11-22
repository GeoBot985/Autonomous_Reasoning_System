import re

class MemorySanitizer:
    """
    Sanitizes memory text before storage.
    Removes noise, plan metadata, and wrappers.
    """

    NOISE_PATTERNS = [
        r"^plan update",
        r"^introspection",
        r"^reflecting on",
        r"^user stated:",
        r"^summary of",
    ]

    SKIP_PATTERNS = [
        r"^plan update",
        r"^introspection",
        r"^reflecting on",
        r"^summary of",
    ]

    @staticmethod
    def sanitize(text: str) -> str:
        """
        Cleans the text. Returns None if the text should be skipped entirely.
        """
        if not text:
            return None

        # Check for skip patterns
        for pattern in MemorySanitizer.SKIP_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return None

        # Remove wrappers
        cleaned = text
        if re.match(r"^user stated:\s*", cleaned, re.IGNORECASE):
            cleaned = re.sub(r"^user stated:\s*", "", cleaned, flags=re.IGNORECASE)

        # Specific check for plan metadata (heuristic)
        if "priority:" in cleaned.lower() and "status:" in cleaned.lower():
            # Likely a plan object dump
            return None

        return cleaned.strip()

    @staticmethod
    def is_valid_for_kg(text: str) -> bool:
        """
        Checks if text is valid for KG extraction.
        """
        if not text: return False
        for pattern in MemorySanitizer.NOISE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True
