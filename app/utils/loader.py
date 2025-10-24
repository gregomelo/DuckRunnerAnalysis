"""
Load and transform FIT data into analysis-ready pandas DataFrames.

This module exposes a single public function, :func:`fit_to_df`, which decodes a
Garmin FIT binary and returns five pandas DataFrames representing sessions,
events, laps, per-record samples, and per-kilometer splits. The transformations
standardize timestamps, compute elapsed time, derive pace in min/km, attach lap
labels to records, and summarize metrics per lap and per split.

Notes
-----
- Requires ``garmin_fit_sdk`` for decoding and ``pandas`` for tabular handling.
- Pace (``speed_pace``) is computed as ``1000 / speed / 60`` (min per km).
- Latitude/longitude are scaled from 32-bit integer representation by 11,930,465.
- Lap numbers are 1-indexed and inferred from order after sorting by start time.
"""

import pandas as pd
from garmin_fit_sdk import Decoder, Stream


def fit_to_df(fit_file):
    """
    Decode a FIT file and return session, event, lap, record, and split tables.

    The input is a bytes-like object (e.g., Streamlit ``UploadedFile`` or
    ``io.BytesIO``). The function decodes the FIT messages, builds DataFrames,
    computes helper columns (e.g., pace, elapsed time), joins lap information
    into records, and produces aggregated summaries per lap and per kilometer
    split.

    Parameters
    ----------
    fit_file : BinaryIO or bytes-like
        Bytes buffer for a single FIT activity. Typically the result of
        ``st.file_uploader`` or a file-like object opened in binary mode.

    Returns
    -------
    df_session : pandas.DataFrame
        Session-level messages as returned by the FIT decoder.
    df_event : pandas.DataFrame
        Event-level messages as returned by the FIT decoder.
    df_lap : pandas.DataFrame
        Lap summary table with timing, cumulative elapsed time, distance, and
        aggregated heart-rate/pace metrics. Columns include (not exhaustive):
        ``lap``, ``start_time``, ``end_time``, ``total_elapsed_time``,
        ``cum_elapsed_time``, ``total_distance``, ``heart_rate_*``,
        ``speed_pace_min/max/mean/avg``.
    df_record : pandas.DataFrame
        Per-record samples (time series) with at least:
        ``timestamp``, ``elapsed_time`` (s), ``distance`` (m), ``heart_rate``,
        ``enhanced_altitude`` (m), ``speed_pace`` (min/km), and ``lap``.
    df_split : pandas.DataFrame
        Per-kilometer split summary with timing and aggregated metrics
        (``heart_rate_*`` and ``speed_pace_*``), plus ``total_elapsed_time`` and
        ``speed_pace_avg``.

    Raises
    ------
    ValueError
        If the FIT stream cannot be decoded or required message groups are
        missing expected fields for the transformations.
    KeyError
        If expected keys are not present in the decoder output.

    See Also
    --------
    garmin_fit_sdk.Decoder : FIT decoding utility.
    garmin_fit_sdk.Stream : Helper for building a decode stream from bytes.

    Examples
    --------
    Decode a file uploaded via Streamlit::

        uploaded = st.file_uploader("Upload FIT", type="fit")
        if uploaded is not None:
            dfs = fit_to_df(uploaded)
            df_session, df_event, df_lap, df_record, df_split = dfs

    """
    stream = Stream.from_bytes_io(fit_file)
    decoder = Decoder(stream)
    messages, errors = decoder.read()

    df_session = pd.DataFrame(messages["session_mesgs"])

    df_event = df_session = pd.DataFrame(messages["event_mesgs"])

    df_lap = pd.DataFrame(messages["lap_mesgs"])

    df_lap["lap"] = df_lap.index + 1

    df_lap = df_lap.sort_values("start_time")

    df_lap = df_lap.rename(columns={"timestamp": "end_time"})

    df_lap["cum_elapsed_time"] = df_lap["total_elapsed_time"].cumsum()

    df_record = pd.DataFrame(messages["record_mesgs"])

    # Coordenates are store using 32-bit integer
    # Dividing by 11930465 will give a decimal value
    df_record["position_lat"] = df_record["position_lat"] / 11930465
    df_record["position_long"] = df_record["position_long"] / 11930465

    df_record = df_record.groupby(["timestamp"], as_index=False).mean()

    # To add a limit do speed_pace, add the code .mask(lambda s: s >=10)
    df_record["speed_pace"] = (1000 / df_record["speed"] / 60).round(2)

    df_record["elapsed_time"] = df_record["timestamp"] - df_record["timestamp"].min()

    df_record["elapsed_time"] = df_record["elapsed_time"].dt.seconds

    df_record = df_record.sort_values("timestamp")

    df_record = pd.merge_asof(
        df_record,
        df_lap[["start_time", "lap"]],
        left_on="timestamp",
        right_on="start_time",
        direction="backward",
    )

    df_record = df_record.drop(columns=["start_time"])

    df_record["enhanced_altitude"] = df_record["enhanced_altitude"].round(2)

    df_summary_lap = df_record.groupby("lap", as_index=False).agg(
        heart_rate_min=("heart_rate", lambda x: int(x.min())),
        heart_rate_max=("heart_rate", lambda x: int(x.max())),
        heart_rate_mean=("heart_rate", lambda x: int(round(x.mean()))),
        speed_pace_min=("speed_pace", "min"),
        speed_pace_max=("speed_pace", "max"),
        speed_pace_mean=("speed_pace", lambda x: round(x.mean(), 2)),
    )

    df_lap = df_lap.join(df_summary_lap, on="lap", rsuffix="_agg")

    df_lap["speed_pace_avg"] = (
        1000 / (df_lap["total_distance"] / df_lap["total_elapsed_time"]) / 60
    ).round(2)

    df_lap = df_lap[
        [
            "lap",
            "start_time",
            "end_time",
            "total_elapsed_time",
            "cum_elapsed_time",
            "total_distance",
            "heart_rate_min",
            "heart_rate_max",
            "heart_rate_mean",
            "speed_pace_min",
            "speed_pace_max",
            "speed_pace_mean",
            "speed_pace_avg",
        ]
    ]

    df_record["split"] = 1 + df_record["distance"] // 1000

    df_record["split"] = df_record["split"].ffill().astype("int")

    df_summary_split = df_record.groupby("split", as_index=False).agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        end_time_seconds=("elapsed_time", "max"),
        heart_rate_min=("heart_rate", lambda x: int(x.min())),
        heart_rate_max=("heart_rate", lambda x: int(x.max())),
        heart_rate_mean=("heart_rate", lambda x: int(round(x.mean()))),
        speed_pace_min=("speed_pace", "min"),
        speed_pace_max=("speed_pace", "max"),
        speed_pace_mean=("speed_pace", lambda x: round(x.mean(), 2)),
    )

    df_summary_split["start_elapsed_time"] = df_summary_split["end_time_seconds"].shift(
        periods=1, fill_value=0
    )

    df_summary_split["total_elapsed_time"] = (
        df_summary_split["end_time_seconds"] - df_summary_split["start_elapsed_time"]
    )

    df_summary_split["speed_pace_avg"] = (
        df_summary_split["total_elapsed_time"] / 60
    ).round(2)

    df_summary_split = df_summary_split[
        [
            "split",
            "start_time",
            "end_time",
            "total_elapsed_time",
            "heart_rate_min",
            "heart_rate_max",
            "heart_rate_mean",
            "speed_pace_min",
            "speed_pace_max",
            "speed_pace_mean",
            "speed_pace_avg",
        ]
    ]

    return df_session, df_event, df_lap, df_record, df_summary_split
