"""
Convert raw .jsonl logs into instruction-tuning format, then split into train/val.

Groups events by calendar day and creates prompt/completion pairs for
instruction-following fine-tuning.

Usage:
    python -m training.format_data

Can also be called programmatically via main() function.
"""

import os
import sys
import json
import random
import argparse
import datetime
from collections import defaultdict

import yaml

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from utils.logging_utils import get_logger
from utils.time_utils import day_name

logger = get_logger("format_data")

# Instruction template (constant)
INSTRUCTION_TEXT = (
    "You are a smart home routine analyzer. Given the following device event log "
    "for a single day, identify any recurring routines present. List each routine "
    "with its name, the devices involved, and the typical times."
)


def _load_events(raw_path: str) -> list[dict]:
    """Load events from a JSONL file, skipping blank lines."""
    events = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _group_by_day(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by calendar date string (YYYY-MM-DD)."""
    days = defaultdict(list)
    for event in events:
        date_str = event["timestamp"][:10]  # "2024-01-08"
        days[date_str].append(event)
    return dict(days)


def _format_input_section(date_str: str, events: list[dict]) -> str:
    """Format a day's events into the ### Input section.
    
    Example output:
        Day: Monday 2024-01-08
        Events:
        - 06:31 bathroom_light ON
        - 06:50 bathroom_light OFF
    """
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    dn = day_name(dt.date())
    
    lines = [f"Day: {dn} {date_str}", "Events:"]
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    
    for event in sorted_events:
        time_str = event["timestamp"][11:16]  # "HH:MM"
        device_id = event["device_id"]
        action = event["action"].upper()
        lines.append(f"- {time_str} {device_id} {action}")
    
    return "\n".join(lines)


def _format_response_section(events: list[dict]) -> str:
    """Build the ### Response section from ground-truth routine_name tags.
    
    Groups events by routine_name, reconstructs approximate on-times and durations.
    Irregular events are listed separately.
    """
    # Separate routine vs irregular events
    routine_events = defaultdict(list)  # routine_name -> list of events
    irregular_events = []
    
    for event in events:
        if event.get("source") == "irregular" or event.get("routine_name") is None:
            irregular_events.append(event)
        else:
            routine_events[event["routine_name"]].append(event)
    
    lines = []
    
    # Format each routine
    for routine_name, revents in sorted(routine_events.items()):
        lines.append(f"Routine detected: {routine_name}")
        
        # Group by device_id to find on/off pairs
        device_events = defaultdict(list)
        for ev in revents:
            device_events[ev["device_id"]].append(ev)
        
        for device_id, devs in sorted(device_events.items()):
            on_events = [e for e in devs if e["action"] == "on"]
            off_events = [e for e in devs if e["action"] == "off"]
            
            if on_events:
                on_time = on_events[0]["timestamp"][11:16]
                duration = "?"
                if off_events:
                    on_dt = datetime.datetime.strptime(
                        on_events[0]["timestamp"], "%Y-%m-%dT%H:%M:%S")
                    off_dt = datetime.datetime.strptime(
                        off_events[0]["timestamp"], "%Y-%m-%dT%H:%M:%S")
                    duration = str(int((off_dt - on_dt).total_seconds() / 60))
                lines.append(
                    f"- {device_id}: ON ~{on_time}, duration ~{duration} min")
    
    # Format irregular events
    for ev in irregular_events:
        if ev["action"] == "on":
            time_str = ev["timestamp"][11:16]
            lines.append(
                f"Irregular event detected: {ev['device_id']} at {time_str}")
    
    # If no routines and no irregular, still provide something
    if not lines:
        lines.append("No routines detected for this day.")
    
    return "\n".join(lines)


def format_sample(date_str: str, events: list[dict]) -> str:
    """Create a full instruction-tuning sample for one day.
    
    Returns a single string with ### Instruction, ### Input, and ### Response sections.
    """
    input_section = _format_input_section(date_str, events)
    response_section = _format_response_section(events)
    
    sample = (
        f"### Instruction:\n{INSTRUCTION_TEXT}\n\n"
        f"### Input:\n{input_section}\n\n"
        f"### Response:\n{response_section}"
    )
    return sample


def format_and_split(raw_path: str, train_path: str, val_path: str,
                     val_split: float = 0.1, seed: int = 42) -> tuple[int, int]:
    """Convert raw logs to instruction-tuning format, shuffle, and split.
    
    Args:
        raw_path: Path to raw synthetic_logs.jsonl.
        train_path: Output path for train.jsonl.
        val_path: Output path for val.jsonl.
        val_split: Fraction of samples for validation.
        seed: Random seed for shuffling.
    
    Returns:
        Tuple of (train_count, val_count).
    """
    events = _load_events(raw_path)
    logger.info(f"Loaded {len(events)} events from {raw_path}")
    
    # Group by day
    days = _group_by_day(events)
    logger.info(f"Found {len(days)} unique days")
    
    # Format each day into a sample
    samples = []
    for date_str in sorted(days.keys()):
        day_events = days[date_str]
        text = format_sample(date_str, day_events)
        samples.append({"text": text})
    
    # Shuffle
    random.seed(seed)
    random.shuffle(samples)
    
    # Split
    val_count = max(1, int(len(samples) * val_split))
    train_count = len(samples) - val_count
    
    train_samples = samples[:train_count]
    val_samples = samples[train_count:]
    
    # Write output
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Train samples: {train_count} -> {train_path}")
    logger.info(f"Val samples:   {val_count} -> {val_path}")
    
    return train_count, val_count


def main(args=None):
    """CLI entry point for data formatting."""
    parser = argparse.ArgumentParser(description="Format raw logs into training data")
    parser.add_argument("--raw", type=str, default=None, help="Path to raw .jsonl")
    parser.add_argument("--train", type=str, default=None, help="Output train.jsonl path")
    parser.add_argument("--val", type=str, default=None, help="Output val.jsonl path")
    parser.add_argument("--val-split", type=float, default=None, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parsed = parser.parse_args(args)
    
    # Load defaults from training.yaml
    config_path = os.path.join(_BASE_DIR, "config", "training.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    raw_path = parsed.raw or os.path.join(_BASE_DIR, config["data"]["raw_path"])
    train_path = parsed.train or os.path.join(_BASE_DIR, config["data"]["train_path"])
    val_path = parsed.val or os.path.join(_BASE_DIR, config["data"]["val_path"])
    val_split = parsed.val_split if parsed.val_split is not None else config["data"]["val_split"]
    
    format_and_split(raw_path, train_path, val_path, val_split, parsed.seed)


if __name__ == "__main__":
    main()
