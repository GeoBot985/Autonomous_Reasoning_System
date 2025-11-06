# planning/workspace.py
"""
Transient shared workspace for inter-step data exchange.
Used within a single Plan to pass results between steps.
"""

class Workspace:
    def __init__(self):
        self.data = {}

    def set(self, key: str, value):
        """Store a key–value pair in working memory."""
        self.data[key] = value

    def get(self, key: str, default=None):
        """Retrieve a value from working memory."""
        return self.data.get(key, default)

    def clear(self):
        """Wipe all temporary data."""
        self.data.clear()

    def snapshot(self) -> dict:
        """Return a shallow copy of the workspace contents."""
        return dict(self.data)
