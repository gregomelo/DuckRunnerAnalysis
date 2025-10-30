"""
Utility functions for converting units.

This module provides helpers to:
- Convert pace values between ``min:sec`` and decimal minutes.
- Convert speed in meters per second (m/s) to pace in minutes per kilometer (min/km).
- Format decimal pace values into time strings (``M:SS``).

These conversions are frequently used for normalizing data from running devices
(e.g., Garmin, Apple Watch) before analysis or visualization.

Functions
---------
pace_minsec_to_float
speed_meters_per_second_to_pace
pace_float_to_time
"""


def pace_minsec_to_float(mins: int, secs: int) -> float:
    """
    Convert a pace value from minutes and seconds into a float representation.

    Parameters
    ----------
    mins : int
        Minutes component of the pace value.
    secs : int
        Seconds component of the pace value.

    Returns
    -------
    float
        Pace represented in decimal minutes (``min/km``), rounded to two
        decimal places (e.g., ``4 min 30 s → 4.5``).

    Examples
    --------
    >>> pace_minsec_to_float(5, 30)
    5.5
    >>> pace_minsec_to_float(4, 45)
    4.75
    """
    return round(mins + secs / 60, 2)


def speed_meters_per_second_to_pace(meter_per_second: float) -> float:
    """
    Convert speed from meters per second (m/s) to pace in minutes per kilometer.

    The conversion uses the formula::

        pace = 1000 / (speed * 60)

    where ``speed`` is in meters per second and the result represents
    minutes per kilometer. The value is rounded to two decimal places.

    Parameters
    ----------
    meter_per_second : float
        Speed value in meters per second (m/s).

    Returns
    -------
    float
        Equivalent pace in minutes per kilometer (``min/km``), rounded to two
        decimal places.

    Examples
    --------
    >>> speed_meters_per_second_to_pace(3.33)
    5.0
    >>> speed_meters_per_second_to_pace(2.78)
    6.0
    """
    return round((1000 / meter_per_second / 60), 2)


def pace_float_to_time(mins_per_kilometer: float) -> str:
    """
    Convert a decimal pace value into a time string in the format ``M:SS``.

    This function transforms a pace value expressed in decimal minutes
    (e.g., ``4.75``) into a formatted string showing minutes and seconds
    (e.g., ``"4:45"``). The seconds are always displayed with two digits.

    Parameters
    ----------
    mins_per_kilometer : float
        Pace in decimal minutes per kilometer (``min/km``).

    Returns
    -------
    str
        Pace formatted as a string in the format ``M:SS``.

    Examples
    --------
    >>> pace_float_to_time(4.75)
    '4:45'
    >>> pace_float_to_time(5.5)
    '5:30'
    >>> pace_float_to_time(6.02)
    '6:01'
    """
    integer_part = int(mins_per_kilometer)
    secs = round((abs(mins_per_kilometer) - abs(integer_part)) * 60)

    # Adjust if rounding leads to 60 seconds
    if secs == 60:
        integer_part += 1
        secs = 0

    return f"{integer_part}:{secs:02d}"
