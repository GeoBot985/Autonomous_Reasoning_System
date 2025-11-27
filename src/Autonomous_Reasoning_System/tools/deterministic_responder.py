import datetime
import math
import requests


class DeterministicResponder:
    """
    Handles factual, numeric, and system-level queries without invoking the LLM.
    Uses local logic or small public lookups.
    """

    def run(self, text: str) -> str:
        q = text.lower().strip()

        # --- date/time ---
        if any(k in q for k in ["time", "date", "today", "now"]):
            return datetime.datetime.now().strftime("%A, %d %B %Y %H:%M:%S")

        # --- math ---
        try:
            if any(op in q for op in ["+", "-", "*", "/"]) and all(
                c.isdigit() or c.isspace() or c in "+-*/." for c in q
            ):
                return str(eval(q))
        except Exception:
            pass

        # --- factual lookup (Wikipedia REST API) ---
        try:
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ', '_')}",
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                if "extract" in data:
                    return data["extract"]
        except Exception:
            pass

        return "I'm not sure, but I can look it up later."
