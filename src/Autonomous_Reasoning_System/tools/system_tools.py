import datetime
import requests
import logging

logger = logging.getLogger(__name__)

def get_current_time() -> str:
    """Returns the current date and time as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_location() -> str:
    """Returns the current location based on IP address."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                city = data.get("city", "Unknown City")
                country = data.get("country", "Unknown Country")
                region_name = data.get("regionName", "")
                return f"{city}, {region_name}, {country}"
    except Exception as e:
        logger.warning(f"Failed to retrieve location: {e}")

    return "Location unavailable"
