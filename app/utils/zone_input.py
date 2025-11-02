"""
Streamlit helpers for configuring and displaying training pace zones.

This module defines visual and logical helpers used in the DuckRunnerAnalysis
Streamlit app for handling training zones (Z1–Z5). It provides a reference
dictionary for default pace thresholds and color mapping, as well as utility
functions for value conversion and dynamic UI elements that store their state in
``st.session_state``.

Notes
-----
- ``ZONE_REF`` stores each zone’s start/end limits (minutes, seconds) and color.
- ``zone_input_st`` renders dynamic numeric inputs in the Streamlit sidebar,
  maintaining user modifications across interactions through ``st.session_state``.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, TypedDict, cast

import streamlit as st


class PaceBoundary(TypedDict, total=False):
    """Minute/second boundary for a zone."""

    min: int
    s: int


class ZoneConfig(TypedDict):
    """Configuration for a training zone."""

    start: Optional[PaceBoundary]
    end: Optional[PaceBoundary]
    color: str


ZONE_REF: Dict[str, ZoneConfig] = {
    "Z1": {"start": None, "end": {"min": 6, "s": 6}, "color": "#4169E1"},
    "Z2": {"start": None, "end": {"min": 5, "s": 20}, "color": "#008000"},
    "Z3": {
        "start": {"min": 4, "s": 59},
        "end": {"min": 4, "s": 38},
        "color": "#FFD700",
    },
    "Z4": {"start": {"min": 4, "s": 33}, "end": None, "color": "#FF4500"},
    "Z5": {"start": {"min": 4, "s": 2}, "end": None, "color": "#800080"},
}


def zone_input_st(
    zone_name: str, information: Literal["start", "end"], unit: Literal["min", "s"]
) -> None:
    """
    Render a Streamlit number input for a pace-zone boundary and store its state.

    The function dynamically builds a labeled numeric input (minutes or seconds)
    for a given training zone, using the color defined in ``ZONE_REF``. The input
    is displayed with a small colored dot and descriptive text ("Início" or
    "Término"). Updated values are saved in
    ``st.session_state["zones_current"]`` to persist across app reruns.

    Parameters
    ----------
    zone_name : str
        Zone identifier (e.g., ``"Z1"``–``"Z5"``).
    information : str
        Either ``"start"`` or ``"end"``, indicating which bound is being edited.
    unit : str
        Either ``"min"`` or ``"s"``, specifying the unit for the number input.

    Returns
    -------
    None
        The function updates ``st.session_state`` in place without returning a
        value.

    Examples
    --------
    >>> zone_input_st("Z2", "end", "min")
    # Renders a labeled number input for Z2 end-minute threshold in Streamlit.
    """
    # Accessing session memory
    zone_ref_session = cast(Dict[str, ZoneConfig], st.session_state["zones_current"])

    color = ZONE_REF[zone_name]["color"]
    number_label = (
        "<div style='margin-bottom:-6px;font-size:14px;'>"
        + f"<span style='color:{color};font-size:14px;'>⬤</span>"
        + "<span>"
        + f" {'Início' if information == 'start' else 'Término'} {zone_name} ({unit})"
        + "</span></div>"
    )

    st.markdown(number_label, unsafe_allow_html=True)

    default_boundary = ZONE_REF[zone_name][information]
    default_value = 0 if default_boundary is None else default_boundary[unit]

    number_input = st.number_input(
        label=f"{zone_name}_{information}_{unit}",
        value=default_value,
        min_value=0,
        max_value=59,
        label_visibility="collapsed",
    )

    # Asseting that zone exists
    if zone_ref_session.get(zone_name) is None:
        zone_ref_session[zone_name] = {"start": None, "end": None, "color": color}

    if zone_ref_session[zone_name][information] is None:
        zone_ref_session[zone_name][information] = {}  # type: ignore[assignment]

    boundary = cast(PaceBoundary, zone_ref_session[zone_name][information])
    boundary[unit] = number_input
