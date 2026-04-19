"""
Produce clean (noiseless) event sequences from routines.yaml.

Loads config/routines.yaml and config/devices.yaml at module level.
Generates on/off event pairs for each routine step that matches a given date.
"""

import os
import datetime
import yaml

from utils.time_utils import parse_time_string, combine_date_time, is_weekday, day_name

# ---------------------------------------------------------------------------
# Load configs at module level
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEVICES_PATH = os.path.join(_BASE_DIR, "config", "devices.yaml")
_ROUTINES_PATH = os.path.join(_BASE_DIR, "config", "routines.yaml")

with open(_DEVICES_PATH, "r") as f:
    DEVICES_CONFIG = yaml.safe_load(f)

with open(_ROUTINES_PATH, "r") as f:
    ROUTINES_CONFIG = yaml.safe_load(f)

DEVICES = DEVICES_CONFIG["devices"]
ROUTINES = ROUTINES_CONFIG["routines"]

# Build a lookup: device_id -> device dict
DEVICE_MAP = {dev["id"]: dev for dev in DEVICES}


def _matches_day(routine_days, date: datetime.date) -> bool:
    """Check if a routine's 'days' field matches the given date.
    
    Supported formats:
      - "all"           -> matches every day
      - "weekday"       -> matches Monday–Friday
      - "weekend"       -> matches Saturday–Sunday
      - ["saturday"]    -> matches explicit day names (list)
      - "monday"        -> matches a single explicit day name (string)
    """
    dn = day_name(date).lower()  # e.g. "monday"
    
    if isinstance(routine_days, str):
        routine_days_lower = routine_days.lower()
        if routine_days_lower == "all":
            return True
        if routine_days_lower == "weekday":
            return is_weekday(date)
        if routine_days_lower == "weekend":
            return not is_weekday(date)
        # Single explicit day name
        return routine_days_lower == dn
    
    if isinstance(routine_days, list):
        return dn in [d.lower() for d in routine_days]
    
    return False


def generate_clean_day(date: datetime.date) -> list[dict]:
    """Generate clean (noiseless) events for a single day.
    
    For each routine whose 'days' field matches the given date, produces
    two events per step: an 'on' event at the specified time and an 'off'
    event at time + duration_minutes.
    
    Args:
        date: The calendar date to generate events for.
        
    Returns:
        A list of event dicts sorted by timestamp, each with schema:
        {
            "timestamp": "2024-01-08T06:30:00",
            "device_id": "bathroom_light",
            "action": "on",
            "source": "routine",
            "routine_name": "morning_routine"
        }
    """
    events = []
    
    for routine in ROUTINES:
        if not _matches_day(routine["days"], date):
            continue
        
        routine_name = routine["name"]
        
        for step in routine["steps"]:
            device_id = step["device"]
            time_on = parse_time_string(step["time"])
            duration = step["duration_minutes"]
            
            # ON event
            dt_on = combine_date_time(date, time_on)
            events.append({
                "timestamp": dt_on.strftime("%Y-%m-%dT%H:%M:%S"),
                "device_id": device_id,
                "action": "on",
                "source": "routine",
                "routine_name": routine_name,
            })
            
            # OFF event
            dt_off = dt_on + datetime.timedelta(minutes=duration)
            events.append({
                "timestamp": dt_off.strftime("%Y-%m-%dT%H:%M:%S"),
                "device_id": device_id,
                "action": "off",
                "source": "routine",
                "routine_name": routine_name,
            })
    
    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp"])
    return events
