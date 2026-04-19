"""
Manage simulation state: device on/off states, time progression, event queue.

Reads a JSONL log file and replays events in accelerated time.
"""

import json
import datetime


class Simulator:
    """Replay smart-home events from a log file with time acceleration.
    
    Maintains device states (on/off) and advances simulation time based on
    real elapsed time multiplied by a speed factor.
    """
    
    def __init__(self, log_path: str):
        """Load events from a .jsonl log file.
        
        Args:
            log_path: Path to the event log (.jsonl format).
        """
        self.events = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    event = json.loads(line)
                    event["_dt"] = datetime.datetime.strptime(
                        event["timestamp"], "%Y-%m-%dT%H:%M:%S"
                    )
                    self.events.append(event)
        
        # Sort by timestamp
        self.events.sort(key=lambda e: e["_dt"])
        
        # Initialize state
        self.device_states: dict[str, bool] = {}
        self._init_device_states()
        
        # Time tracking
        if self.events:
            self.current_time = self.events[0]["_dt"]
        else:
            self.current_time = datetime.datetime(2024, 1, 1)
        
        self._event_index = 0  # next event to fire
        self._finished = False
    
    def _init_device_states(self):
        """Set all unique devices to off."""
        device_ids = set()
        for event in self.events:
            device_ids.add(event["device_id"])
        self.device_states = {did: False for did in device_ids}
    
    def tick(self, delta_real_seconds: float, speed_multiplier: float = 300.0):
        """Advance simulation time and fire pending events.
        
        Args:
            delta_real_seconds: Real wall-clock time elapsed since last tick.
            speed_multiplier: How many sim-seconds per real-second.
                Default 300 = 5 minutes of sim time per 1 real second.
        """
        if self._finished:
            return
        
        sim_delta = datetime.timedelta(seconds=delta_real_seconds * speed_multiplier)
        self.current_time += sim_delta
        
        # Fire all events up to current_time
        while (self._event_index < len(self.events) and 
               self.events[self._event_index]["_dt"] <= self.current_time):
            event = self.events[self._event_index]
            device_id = event["device_id"]
            action = event["action"]
            
            self.device_states[device_id] = (action == "on")
            self._event_index += 1
        
        if self._event_index >= len(self.events):
            self._finished = True
    
    def get_state(self) -> dict:
        """Return current simulation state.
        
        Returns:
            Dict with 'current_time' (datetime) and 'device_states' (dict[str, bool]).
        """
        return {
            "current_time": self.current_time,
            "device_states": dict(self.device_states),
        }
    
    def is_finished(self) -> bool:
        """Return True if all events have been fired."""
        return self._finished
    
    def get_progress(self) -> float:
        """Return progress as a fraction (0.0 to 1.0)."""
        if not self.events:
            return 1.0
        return self._event_index / len(self.events)
    
    def get_time_range(self) -> tuple[datetime.datetime, datetime.datetime]:
        """Return the (start, end) datetime of the log."""
        if not self.events:
            now = datetime.datetime(2024, 1, 1)
            return (now, now)
        return (self.events[0]["_dt"], self.events[-1]["_dt"])
