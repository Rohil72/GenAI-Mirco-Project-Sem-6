"""
Utility functions for time parsing and manipulation.
Used across the data generator, training pipeline, and simulation.
"""

import datetime


def parse_time_string(time_str: str) -> datetime.time:
    """Parse '06:30' into a datetime.time object."""
    return datetime.datetime.strptime(time_str, "%H:%M").time()


def combine_date_time(date: datetime.date, time: datetime.time) -> datetime.datetime:
    """Combine a date and time into a naive datetime."""
    return datetime.datetime.combine(date, time)


def is_weekday(date: datetime.date) -> bool:
    """Return True if date is Monday–Friday."""
    return date.weekday() < 5  # 0=Monday, 4=Friday


def day_name(date: datetime.date) -> str:
    """Return full day name e.g. 'Monday'."""
    return date.strftime("%A")


def minutes_between(dt1: datetime.datetime, dt2: datetime.datetime) -> float:
    """Return signed minutes between two datetimes (dt2 - dt1)."""
    delta = dt2 - dt1
    return delta.total_seconds() / 60.0
