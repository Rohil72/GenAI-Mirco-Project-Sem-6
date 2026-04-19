"""
Parse the model's text output into structured Python objects.

Uses regex + simple string parsing (no second LLM call) to extract
routine information from the model's natural language response.
"""

import re
from utils.logging_utils import get_logger

logger = get_logger("routine_extractor")


class RoutineExtractor:
    """Extract structured routine data from model text output.
    
    Parses text like:
        Routine detected: morning_routine
        - bathroom_light: ON ~06:30, duration ~20 min
        - coffee_maker: ON ~06:35, duration ~15 min
    
    Into structured dicts.
    """
    
    # Regex patterns
    _ROUTINE_HEADER = re.compile(
        r"Routine detected:\s*(\w+)", re.IGNORECASE
    )
    _DEVICE_LINE = re.compile(
        r"-\s*(\w+):\s*ON\s*~?(\d{1,2}:\d{2}),?\s*duration\s*~?(\d+)\s*min",
        re.IGNORECASE,
    )
    _IRREGULAR_LINE = re.compile(
        r"Irregular event detected:\s*(\w+)\s+at\s+(\d{1,2}:\d{2})",
        re.IGNORECASE,
    )
    
    def extract(self, model_output: str) -> list[dict]:
        """Parse model output into structured routine dicts.
        
        Args:
            model_output: Raw text output from the model.
            
        Returns:
            List of routine dicts, e.g.:
            [
                {
                    "routine_name": "morning_routine",
                    "devices": [
                        {"device_id": "bathroom_light", "action": "on",
                         "typical_time": "06:30", "duration_min": 20},
                    ]
                }
            ]
            Returns empty list on parse failure.
        """
        try:
            return self._parse(model_output)
        except Exception as e:
            logger.warning(f"Failed to parse model output: {e}")
            return []
    
    def _parse(self, text: str) -> list[dict]:
        """Internal parsing logic."""
        routines = []
        current_routine = None
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Check for routine header
            header_match = self._ROUTINE_HEADER.search(line)
            if header_match:
                # Save previous routine if exists
                if current_routine is not None:
                    routines.append(current_routine)
                current_routine = {
                    "routine_name": header_match.group(1),
                    "devices": [],
                }
                continue
            
            # Check for device line (only if inside a routine)
            if current_routine is not None:
                device_match = self._DEVICE_LINE.search(line)
                if device_match:
                    current_routine["devices"].append({
                        "device_id": device_match.group(1),
                        "action": "on",
                        "typical_time": device_match.group(2),
                        "duration_min": int(device_match.group(3)),
                    })
                    continue
            
            # Check for irregular event (standalone, not part of a routine)
            irregular_match = self._IRREGULAR_LINE.search(line)
            if irregular_match:
                # Store irregulars as a special "routine" with no name
                routines.append({
                    "routine_name": None,
                    "devices": [{
                        "device_id": irregular_match.group(1),
                        "action": "on",
                        "typical_time": irregular_match.group(2),
                        "duration_min": None,
                    }],
                })
        
        # Don't forget the last routine
        if current_routine is not None:
            routines.append(current_routine)
        
        return routines
    
    def to_summary(self, routines: list[dict]) -> str:
        """Convert structured routines back to a human-readable summary."""
        if not routines:
            return "No routines detected."
        
        lines = []
        for r in routines:
            if r["routine_name"]:
                lines.append(f"Routine: {r['routine_name']}")
                for d in r["devices"]:
                    dur = f", duration ~{d['duration_min']} min" if d.get("duration_min") else ""
                    lines.append(f"  - {d['device_id']}: ON ~{d['typical_time']}{dur}")
            else:
                for d in r["devices"]:
                    lines.append(f"Irregular: {d['device_id']} at {d['typical_time']}")
        
        return "\n".join(lines)
