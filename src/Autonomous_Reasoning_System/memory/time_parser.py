# Autonomous_Reasoning_System/memory/time_parser.py

import re
from datetime import datetime, timedelta
from dateparser import parse
from dateparser.search import search_dates

# Settings tuned for South Africa / DMY order
SETTINGS = {
    "PREFER_DATES_FROM": "future",
    "DATE_ORDER": "DMY",
    "PREFER_LOCALE_DATE_ORDER": True,
    "RETURN_AS_TIMEZONE_AWARE": False,
}

# Simple relative expressions: "tomorrow 9am"
REL_DAY_TIME = re.compile(
    r"\b(?P<day>today|tomorrow)\s*(?:at\s*)?(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm)?\b",
    re.IGNORECASE
)

def extract_datetime(text: str):
    """
    Extract a datetime from natural language text.
    Supports phrases like 'tomorrow 9am', '25 December 2025 at 8:00',
    or 'next Tuesday at 14:30'.
    Returns naive datetime or None.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip().lower()
    now = datetime.now()

    # 1️⃣ Deterministic relative parsing
    m = REL_DAY_TIME.search(s)
    if m:
        day = m.group("day")
        hour = int(m.group("h"))
        minute = int(m.group("m")) if m.group("m") else 0
        ampm = m.group("ampm")

        if ampm:
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0

        base = now
        if day == "tomorrow":
            base += timedelta(days=1)

        return base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 2️⃣ Fuzzy parser
    dt = parse(text, settings=SETTINGS, languages=["en"])
    if dt:
        return dt

    # 3️⃣ Fuzzy in-text scan
    hits = search_dates(text, settings=SETTINGS, languages=["en"])
    if hits:
        best = max(hits, key=lambda t: len(t[0]))
        return best[1]

    return None
