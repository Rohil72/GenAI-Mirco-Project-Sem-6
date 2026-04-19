"""
Apply realistic noise to clean event lists.

Implements five noise functions that progressively distort clean events
to create training data that mimics real-world smart-home log variability.
"""

import copy
import random
import datetime

from utils.logging_utils import get_logger

logger = get_logger("noise")


def add_time_jitter(events: list[dict], max_jitter_minutes: int = 12) -> list[dict]:
    """Randomly shift each event's timestamp by ±0 to max_jitter_minutes.
    
    Uses a Gaussian distribution with sigma = max_jitter_minutes / 3.
    Re-sorts events by timestamp after jittering.
    """
    sigma = max_jitter_minutes / 3.0
    jittered = []
    
    for event in events:
        ev = copy.deepcopy(event)
        dt = datetime.datetime.strptime(ev["timestamp"], "%Y-%m-%dT%H:%M:%S")
        jitter = random.gauss(0, sigma)
        # Clamp to max range
        jitter = max(-max_jitter_minutes, min(max_jitter_minutes, jitter))
        dt += datetime.timedelta(minutes=jitter)
        ev["timestamp"] = dt.strftime("%Y-%m-%dT%H:%M:%S")
        jittered.append(ev)
    
    jittered.sort(key=lambda e: e["timestamp"])
    return jittered


def skip_events(events: list[dict], skip_probability: float = 0.08) -> list[dict]:
    """Drop events with a given probability. Skips on/off pairs atomically.
    
    When an 'on' event is dropped, its corresponding 'off' event (matched by
    device_id and being the next 'off' for that device) is also dropped to
    avoid orphaned events.
    """
    # Build set of indices to skip
    skip_indices = set()
    
    for i, event in enumerate(events):
        if i in skip_indices:
            continue
        if event["action"] == "on" and random.random() < skip_probability:
            skip_indices.add(i)
            # Find the matching off event for this device
            for j in range(i + 1, len(events)):
                if (events[j]["device_id"] == event["device_id"] and 
                        events[j]["action"] == "off" and
                        j not in skip_indices):
                    skip_indices.add(j)
                    break
    
    return [ev for i, ev in enumerate(events) if i not in skip_indices]


def add_irregular_events(events: list[dict], devices: list[dict], 
                          num_irregular: int = 3) -> list[dict]:
    """Add completely random device events at random times during the day.
    
    Each irregular event has a matching 'off' event 5–60 minutes later.
    Marked with source='irregular' and routine_name=None.
    """
    if not events:
        return events
    
    result = copy.deepcopy(events)
    
    # Determine the day from the first event
    first_dt = datetime.datetime.strptime(events[0]["timestamp"], "%Y-%m-%dT%H:%M:%S")
    day_date = first_dt.date()
    
    for _ in range(num_irregular):
        device = random.choice(devices)
        device_id = device["id"]
        
        # Random time during waking hours (06:00 – 23:00)
        random_hour = random.randint(6, 22)
        random_minute = random.randint(0, 59)
        dt_on = datetime.datetime(day_date.year, day_date.month, day_date.day,
                                   random_hour, random_minute)
        
        # Duration: 5–60 minutes
        duration = random.randint(5, 60)
        dt_off = dt_on + datetime.timedelta(minutes=duration)
        
        result.append({
            "timestamp": dt_on.strftime("%Y-%m-%dT%H:%M:%S"),
            "device_id": device_id,
            "action": "on",
            "source": "irregular",
            "routine_name": None,
        })
        result.append({
            "timestamp": dt_off.strftime("%Y-%m-%dT%H:%M:%S"),
            "device_id": device_id,
            "action": "off",
            "source": "irregular",
            "routine_name": None,
        })
    
    result.sort(key=lambda e: e["timestamp"])
    return result


def add_duration_noise(events: list[dict], 
                        max_duration_delta_minutes: int = 15) -> list[dict]:
    """For each 'off' event, randomly extend or shorten its time.
    
    Shifts the off-event timestamp by a random amount in the range
    [-max_duration_delta_minutes, +max_duration_delta_minutes].
    """
    result = []
    for event in events:
        ev = copy.deepcopy(event)
        if ev["action"] == "off":
            dt = datetime.datetime.strptime(ev["timestamp"], "%Y-%m-%dT%H:%M:%S")
            delta = random.uniform(-max_duration_delta_minutes, 
                                    max_duration_delta_minutes)
            dt += datetime.timedelta(minutes=delta)
            ev["timestamp"] = dt.strftime("%Y-%m-%dT%H:%M:%S")
        result.append(ev)
    
    result.sort(key=lambda e: e["timestamp"])
    return result


def apply_all_noise(events: list[dict], devices: list[dict], 
                     config: dict = None) -> list[dict]:
    """Master function: applies all noise transformations in sequence.
    
    Args:
        events: Clean event list from patterns.generate_clean_day().
        devices: List of device dicts from devices.yaml.
        config: Optional dict with noise parameters. Supported keys:
            - max_jitter_minutes (default 12)
            - skip_probability (default 0.08)
            - num_irregular (default 3)
            - max_duration_delta_minutes (default 15)
    
    Returns:
        Noisy event list sorted by timestamp.
    """
    if config is None:
        config = {}
    
    result = add_time_jitter(
        events, 
        max_jitter_minutes=config.get("max_jitter_minutes", 12)
    )
    result = skip_events(
        result, 
        skip_probability=config.get("skip_probability", 0.08)
    )
    result = add_irregular_events(
        result, 
        devices, 
        num_irregular=config.get("num_irregular", 3)
    )
    result = add_duration_noise(
        result, 
        max_duration_delta_minutes=config.get("max_duration_delta_minutes", 15)
    )
    
    return result
