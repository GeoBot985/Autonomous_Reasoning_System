# self_validator.py
"""
Self-Validator Module
Evaluates each reasoning or action cycle outcome and returns a confidence-based “feeling”.
"""

from datetime import datetime

class SelfValidator:
    def __init__(self):
        self.history = []  # stores last few validation results for trend analysis

    def evaluate(self, input_text: str, output_text: str, meta: dict | None = None) -> dict:
        """
        Evaluates the success and emotional outcome of a reasoning/action cycle.

        Args:
            input_text (str): the original user or system prompt
            output_text (str): the result Tyrone produced
            meta (dict, optional): may include {'intent': ..., 'confidence': float, 'error': ...}

        Returns:
            dict: {
                "success": bool,
                "feeling": str,  # "positive" | "neutral" | "negative"
                "reason": str,
                "timestamp": datetime
            }
        """
        meta = meta or {}
        conf = meta.get("confidence", 0.5)
        intent = meta.get("intent", "unknown")
        error = meta.get("error")

        # ---- Primary heuristic rules ----
        if error:
            feeling = "negative"
            reason = f"Encountered error: {error}"
            success = False
        elif "sorry" in output_text.lower() or "error" in output_text.lower():
            feeling = "negative"
            reason = "Response indicates apology or failure."
            success = False
        elif conf >= 0.8:
            feeling = "positive"
            reason = f"High confidence ({conf:.2f}) on intent '{intent}'."
            success = True
        elif 0.5 <= conf < 0.8:
            feeling = "neutral"
            reason = f"Moderate confidence ({conf:.2f}); acceptable but uncertain."
            success = True
        else:
            feeling = "negative"
            reason = f"Low confidence ({conf:.2f}); uncertain about result."
            success = False

        # ---- Save short history (rolling window of 20) ----
        record = {
            "success": success,
            "feeling": feeling,
            "reason": reason,
            "intent": intent,
            "confidence": conf,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.history.append(record)
        self.history = self.history[-20:]

        return record

    def summary(self) -> dict:
        """
        Returns aggregate metrics over recent history.
        """
        if not self.history:
            return {"avg_conf": None, "success_rate": None, "trend": "n/a"}

        avg_conf = sum(r["confidence"] for r in self.history) / len(self.history)
        success_rate = sum(1 for r in self.history if r["success"]) / len(self.history)
        trend = "up" if success_rate > 0.7 else "flat" if success_rate > 0.4 else "down"

        return {
            "avg_conf": round(avg_conf, 3),
            "success_rate": round(success_rate, 3),
            "trend": trend
        }
