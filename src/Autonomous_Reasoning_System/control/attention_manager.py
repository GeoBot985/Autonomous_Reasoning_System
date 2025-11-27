import threading
import time
import sys
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_last_user_activity = 0
PAUSE_DURATION = 90
_silent = False     # 🔇 new flag to suppress mid-input prints


def set_silent(value: bool):
    """Enable/disable console prints from attention manager."""
    global _silent
    _silent = value


def acquire():
    """Called when user starts typing → pause background tasks."""
    global _last_user_activity
    _last_user_activity = time.time()
    _lock.acquire()
    if not _silent:
        logger.info("[🧭 ATTENTION] User input detected — pausing autonomous tasks.")


def release():
    """Release after input handled → resume background tasks."""
    if _lock.locked():
        _lock.release()
        if not _silent:
            logger.info("[🧭 ATTENTION] User input handled — resuming autonomous tasks.")


def user_activity_detected():
    """Called by heartbeat when it senses user activity."""
    global _last_user_activity
    _last_user_activity = time.time()
    if not _silent:
        logger.info("[🧭 ATTENTION] User or recent activity detected — pausing background tasks for 90 s.")


def should_pause_autonomous() -> bool:
    """Return True if autonomous threads should stay paused."""
    return (time.time() - _last_user_activity) < PAUSE_DURATION

attention = sys.modules[__name__]
