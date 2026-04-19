"""
Orchestrate generation of a full multi-day synthetic log file.

Usage:
    python -m data.generator.generate_data --days 90 --output data/raw/synthetic_logs.jsonl --seed 42

Can also be called programmatically via main() function.
"""

import os
import sys
import json
import random
import argparse
import datetime

import numpy as np
import yaml

# Ensure the project root is on sys.path for imports
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from data.generator.patterns import generate_clean_day, DEVICES
from data.generator.noise import apply_all_noise
from utils.logging_utils import get_logger

logger = get_logger("generate_data")


def generate(num_days: int, output_path: str, seed: int = 42, 
             noise_config: dict = None) -> None:
    """Generate synthetic smart-home logs for multiple consecutive days.
    
    Args:
        num_days: Number of days to generate (starting from 2024-01-01).
        output_path: Path to write the .jsonl output file.
        seed: Random seed for reproducibility.
        noise_config: Optional noise parameter overrides.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if noise_config is None:
        noise_config = {}
    
    all_events = []
    start_date = datetime.date(2024, 1, 1)
    
    for day_offset in range(num_days):
        current_date = start_date + datetime.timedelta(days=day_offset)
        
        # 1. Generate clean events for this day
        clean_events = generate_clean_day(current_date)
        
        # 2. Apply noise
        noisy_events = apply_all_noise(clean_events, DEVICES, noise_config)
        
        all_events.extend(noisy_events)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write as newline-delimited JSON
    with open(output_path, "w", encoding="utf-8") as f:
        for event in all_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    # Print summary
    total = len(all_events)
    routine_count = sum(1 for e in all_events if e.get("source") == "routine")
    irregular_count = sum(1 for e in all_events if e.get("source") == "irregular")
    
    logger.info(f"Generation complete!")
    logger.info(f"  Days generated: {num_days}")
    logger.info(f"  Total events:   {total}")
    logger.info(f"  Routine events: {routine_count}")
    logger.info(f"  Irregular events: {irregular_count}")
    logger.info(f"  Output: {output_path}")


def main(args=None):
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic smart-home logs")
    parser.add_argument("--days", type=int, default=None,
                        help="Number of days to generate (default: from training.yaml)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .jsonl file path (default: from training.yaml)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    parsed = parser.parse_args(args)
    
    # Load defaults from training.yaml
    config_path = os.path.join(_BASE_DIR, "config", "training.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    num_days = parsed.days or config["data"]["num_days"]
    output_path = parsed.output or os.path.join(_BASE_DIR, config["data"]["raw_path"])
    
    generate(num_days=num_days, output_path=output_path, seed=parsed.seed)


if __name__ == "__main__":
    main()
