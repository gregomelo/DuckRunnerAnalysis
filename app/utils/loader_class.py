"""
Running FIT loader and lap summarization utilities.

This module provides the `DataLoaderFromFIT` class to decode Garmin FIT files,
transform their messages into structured pandas DataFrames, and compute
per-record and per-lap metrics suitable for visualization (e.g., Streamlit/Plotly)
or downstream analysis.

The processing pipeline is:

1. Decode FIT → in-memory message dict (sessions, events, laps, records).
2. Build raw DataFrames: `raw_session`, `raw_event`, `raw_lap`, `raw_record`.
3. Pre-process records at 1 Hz (resampling, interpolation, flags for missing data,
   elapsed time, coordinate conversion from semicircles to degrees).
4. Smooth speed with an adaptive moving-window method that shortens the window
   on rapid changes and lengthens it in steady segments, followed by a centered
   rolling average.
5. Generate processed series (speed components and pace), reconstruct cumulative
   distance from processed speed, and scale to match the device distance.
6. Map records to laps via time-aware merge-as-of and compute per-lap summaries
   (heart rate, altitude, min/max/avg speeds, per-lap distance).
7. Produce a tidy ``process_lap`` table with speeds and their pace equivalents.

Notes
-----
- Coordinates are converted from *semicircles* to degrees using:
  ``degrees = semicircles * 180 / 2**31``.
- Record timestamps are resampled to 1 second; missing values are imputed with
  time-based interpolation (limits differ for "fast" and "slow" signals) and
  forward/backward fill as a safety net.
- Speed smoothing uses an **adaptive moving window**. The window size is adjusted
  by comparing the current speed to a short recent average:
  larger windows in steady segments and shorter windows during rapid changes.
  A centered rolling average is applied afterward to stabilize local noise.
- Distance reconstruction integrates the processed speed (1 Hz) and scales the
  cumulative total to match the device-reported maximum distance.
- Laps are associated to records via a backward merge-as-of on
  ``timestamp`` (record) and ``start_time`` (lap). Ensure both sides are sorted.

API Overview
------------
DataLoaderFromFIT
    Orchestrates the end-to-end pipeline from FIT decoding to tabular outputs.
"""

from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Self

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream

from .unit_converter import pace_float_to_time, speed_meters_per_second_to_pace


class DataLoaderFromFIT:
    """
    Decode, transform, and summarize running data from FIT files.

    This class orchestrates a complete pipeline to convert a FIT file into
    structured pandas DataFrames representing sessions, laps, and records.
    It applies adaptive smoothing to speed signals, derives pace metrics,
    reconstructs distance at 1 Hz, and aggregates per-lap statistics.

    The end-to-end processing is triggered by calling :meth:`start`, which
    executes the steps below in order:

    1. :meth:`extract_message` — decode the FIT stream into message dicts.
    2. :meth:`load_raw_data` — convert message lists into raw DataFrames.
    3. :meth:`pre_process_record_data` — resample to 1 Hz, impute, augment.
    4. :meth:`process_record_data` — adaptive speed smoothing, pace, distance,
       split markers, and lap mapping (:func:`pandas.merge_asof`).
    5. :meth:`process_lap_data` — per-lap aggregation and pace conversions.

    Parameters
    ----------
    fit_file : str, pathlib.Path, or BinaryIO
        Path or readable binary stream of the FIT file.

    Attributes
    ----------
    message : dict
        Parsed FIT messages grouped by type (e.g., ``session_mesgs``,
        ``lap_mesgs``, ``record_mesgs``).
    error : list
        List of decoding issues reported by the FIT SDK (if any).
    raw_session : pandas.DataFrame
        Untouched DataFrame built from ``session_mesgs``.
    raw_event : pandas.DataFrame
        Untouched DataFrame built from ``event_mesgs``.
    raw_lap : pandas.DataFrame
        Untouched DataFrame built from ``lap_mesgs``.
    raw_record : pandas.DataFrame
        Untouched DataFrame built from ``record_mesgs``.
    pre_process_record : pandas.DataFrame
        Record-level table on a 1 Hz timeline with missing-data flags,
        elapsed time, and coordinates converted from semicircles to degrees.
    process_record : pandas.DataFrame
        Smoothed record-level data with pace fields, reconstructed distance
        aligned to the device total, split markers, and mapped lap ids.
    process_lap : pandas.DataFrame
        One row per lap with heart-rate/altitude stats, min/max/avg speeds and
        their pace representations, cumulative elapsed time, and distances.

    Raises
    ------
    IOError
        If the FIT stream cannot be created from the provided input.
    TypeError
        If ``fit_file`` is neither a path-like object nor a readable binary.
    RuntimeError
        If FIT decoding fails unexpectedly.
    ValueError
        If the FIT decoder reports structured errors.

    Notes
    -----
    - **Coordinates:** semicircles are converted to degrees using
      ``degrees = semicircles * 180 / 2**31``.
    - **Resampling & imputation:** records are resampled to 1 s, then
      interpolated (time-based) with different limits for fast/slow signals,
      with forward/backward fill as safety net.
    - **Adaptive smoothing:** speed is smoothed with a moving window that
      shrinks under rapid changes and widens in steady segments, followed by
      a centered rolling average for local stabilization. See
      :meth:`_speed_smoothing` for details.
    - **Distance reconstruction:** processed speed (1 Hz) is cumulatively
      integrated and scaled to match the device-reported maximum distance.
    - **Lap mapping:** records are associated to laps via a backward
      merge-as-of on record ``timestamp`` and lap ``start_time``; both inputs
      must be sorted.

    Examples
    --------
    >>> from pathlib import Path
    >>> loader = DataLoaderFromFIT(Path("run.fit")).start()
    >>> loader.process_record.head()
    >>> loader.process_lap[["lap", "pace_avg_time", "heart_rate_mean"]]
    """

    _fit_stream: Stream
    _fit_file: str
    message: Dict[str, Any]
    error: List[Any]
    raw_session: pd.DataFrame
    raw_event: pd.DataFrame
    raw_lap: pd.DataFrame
    raw_record: pd.DataFrame
    pre_process_record: pd.DataFrame
    process_record: pd.DataFrame
    process_lap: pd.DataFrame

    def __init__(self, fit_file: str | Path | BinaryIO):
        """
        Initialize the loader with a FIT file path or binary stream.

        Parameters
        ----------
        fit_file : str, Path, or BinaryIO
            Source of the FIT data. If a path is provided, the file is opened
            for reading; if a binary stream is provided, bytes are consumed
            directly.

        Raises
        ------
        IOError
            If the underlying FIT stream cannot be created from the provided
            path or binary object.
        TypeError
            If `fit_file` is neither a path-like object nor a readable binary.
        """
        self._fit_file = (
            str(fit_file) if isinstance(fit_file, (str, Path)) else "<binary stream>"
        )
        try:
            if isinstance(fit_file, str):
                self._fit_stream = Stream.from_file(str(fit_file))
            elif isinstance(fit_file, Path):
                self._fit_stream = Stream.from_file(str(fit_file))
            elif hasattr(fit_file, "read"):
                self._fit_stream = Stream.from_bytes_io(fit_file)
            else:
                raise TypeError(
                    "The argument 'fit_file' must be a path (str/Path) or a binary "
                    "object (BinaryIO). "
                    "O argumento 'fit_file' deve ser um caminho (str/Path) ou um "
                    "objeto binário (BinaryIO)."
                )
        except Exception as e:
            raise IOError(
                f"Failed to read FIT file: {self._fit_file}. "
                f"Não foi possível ler o arquivo FIT: {self._fit_file}."
            ) from e

    def start(self) -> Self:
        """
        Run the full processing pipeline and return `self`.

        This method executes, in order:
        1) :meth:`extract_message`
        2) :meth:`load_raw_data`
        3) :meth:`pre_process_record_data`
        4) :meth:`process_record_data`
        5) :meth:`process_lap_data`

        Returns
        -------
        Self
            The instance with all output DataFrames populated.

        Raises
        ------
        RuntimeError
            If FIT decoding fails unexpectedly.
        ValueError
            If the FIT decoder reports structured errors.
        """
        self.extract_message()
        self.load_raw_data()
        self.pre_process_record_data()
        self.process_record_data()
        self.process_lap_data()

        return self

    def extract_message(self) -> None:
        """
        Decode the FIT stream into in-memory message dictionaries.

        Populates `message` and `error` attributes with the parsed output
        from `garmin_fit_sdk.Decoder`.

        Raises
        ------
        RuntimeError
            If the decoder raises an unexpected exception during parsing.
        ValueError
            If the FIT decoder reports errors in the `error` field.
        """
        try:
            decoder = Decoder(self._fit_stream)
            message, error = decoder.read()
            self.message = message
            self.error = error
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode FIT file: {self._fit_file}. "
                f"Não foi possível decodificar o arquivo FIT: {self._fit_file}. "
                f"Details/Detalhes: {e}"
            ) from e

        if self.error:
            raise ValueError(
                f"Decoder reported an error for FIT file {self._fit_file}: "
                + f"{self.error}. O decodificador reportou um erro para o arquivo "
                + f"{self._fit_file}: {self.error}."
            )

    def load_raw_data(self):
        """
        Build raw DataFrames from decoded FIT messages.

        The following attributes are populated from `self.message`:
        - `raw_session` from ``session_mesgs``
        - `raw_event` from ``event_mesgs``
        - `raw_lap` from ``lap_mesgs``
        - `raw_record` from ``record_mesgs``

        Notes
        -----
        This method does not perform any cleaning or type conversions.
        """
        self.raw_session = pd.DataFrame(self.message["session_mesgs"])
        self.raw_event = pd.DataFrame(self.message["event_mesgs"])
        self.raw_lap = pd.DataFrame(self.message["lap_mesgs"])
        self.raw_record = pd.DataFrame(self.message["record_mesgs"])

    def pre_process_record_data(self):
        """
        Resample, clean, and augment record data to a uniform 1 Hz timeline.

        This method performs all preliminary transformations required to create
        a consistent record-level time series before smoothing and distance
        reconstruction. It converts coordinates, removes implausible speeds,
        imputes missing data, and computes auxiliary columns such as flags and
        elapsed time.

        Steps
        -----
        1. Convert latitude/longitude from semicircles to degrees.
        2. Remove duplicate timestamps by averaging rows with the same timestamp.
        3. Sort by timestamp and resample at a fixed 1 Hz frequency.
        4. Apply physical clamps to speed (remove values < 0.5 m/s or > 8.0 m/s).
        5. Flag missing values for key signals (speed, power, heart rate,
        altitude).
        6. Compute ``elapsed_time`` as seconds since the first record.
        7. Interpolate missing data timewise with different limits:
        - *Fast* signals: ``speed``, ``power``, ``position_lat``,
            ``position_long`` (limit = 1 s)
        - *Slow* signals: ``heart_rate``, ``enhanced_altitude``,
            ``enhanced_speed`` (limit = 5 s)
        8. Forward/backward fill as a final safety net and round numeric values.

        Populates
        ---------
        pre_process_record : pandas.DataFrame
            Record-level DataFrame on a 1 Hz timeline, containing:
            - Interpolated numeric values
            - Flags for missing data
            - Elapsed time in seconds
            - Latitude/longitude in degrees

        Notes
        -----
        - FIT coordinates are stored as 32-bit integers; conversion uses
        ``degrees = semicircles * 180 / 2**31``.
        - The resampling step assumes timestamps are consistent and monotonic;
        only relative timing is required for subsequent computations.
        - Interpolation uses a time-based method; the `limit` parameter prevents
        long gaps from being filled unrealistically.
        """
        pre_process_record = self.raw_record.copy()

        # Coordenates are store using 32-bit integer
        # Dividing by 11930465 will give a decimal value
        SEMICIRCLE_TO_DEG = 180.0 / (1 << 31)
        pre_process_record["position_lat"] = (
            pre_process_record["position_lat"] * SEMICIRCLE_TO_DEG
        )
        pre_process_record["position_long"] = (
            pre_process_record["position_long"] * SEMICIRCLE_TO_DEG
        )

        # Removing duplicated timestamps
        pre_process_record = pre_process_record.groupby(
            ["timestamp"], as_index=False
        ).mean()

        pre_process_record = pre_process_record.sort_values("timestamp")

        # Imputing missing timestamps
        pre_process_record = (
            pre_process_record.set_index("timestamp")
            .resample("1s")
            .asfreq()
            .reset_index()
        )

        # Removing fisic clamps
        pre_process_record.loc[
            (pre_process_record["speed"] < 0.5) | (pre_process_record["speed"] > 8.0),
            "speed",
        ] = np.nan

        # Marking missing points
        pre_process_record["speed_missing"] = pre_process_record["speed"].isna()
        pre_process_record["power_missing"] = pre_process_record["power"].isna()
        pre_process_record["hr_missing"] = pre_process_record["heart_rate"].isna()
        pre_process_record["ea_missing"] = pre_process_record[
            "enhanced_altitude"
        ].isna()

        # Adding elapsed_time
        pre_process_record["elapsed_time"] = (
            pre_process_record["timestamp"] - pre_process_record["timestamp"].min()
        )
        pre_process_record["elapsed_time"] = (
            pre_process_record["elapsed_time"].dt.total_seconds().astype(int)
        )

        # Immputing missing data
        num_cols_fast = ["speed", "power", "position_lat", "position_long"]
        num_cols_slow = ["heart_rate", "enhanced_altitude", "enhanced_speed"]

        pre_process_record = pre_process_record.set_index("timestamp")

        pre_process_record[num_cols_fast] = pre_process_record[
            num_cols_fast
        ].interpolate("time", limit=1)

        pre_process_record[num_cols_slow] = pre_process_record[
            num_cols_slow
        ].interpolate("time", limit=5)

        pre_process_record = pre_process_record.reset_index()

        pre_process_record[num_cols_fast + num_cols_slow] = (
            pre_process_record[num_cols_fast + num_cols_slow].ffill().bfill()
        )

        pre_process_record["speed"] = pre_process_record["speed"].round(3)
        pre_process_record["enhanced_speed"] = pre_process_record[
            "enhanced_speed"
        ].round(3)
        pre_process_record["enhanced_altitude"] = pre_process_record[
            "enhanced_altitude"
        ].round(1)
        pre_process_record["heart_rate"] = pre_process_record["heart_rate"].astype(
            "int"
        )
        pre_process_record["power"] = pre_process_record["power"].astype("int")

        self.pre_process_record = pre_process_record

    def _speed_smoothing(
        self,
        speed_series,
        base_window=15,
        change_threshold=0.3,
        window_rolling=7,
    ):
        """
        Apply adaptive moving-window smoothing to a speed time series (1 Hz).

        The algorithm adjusts the local averaging window based on recent speed
        variation: it shortens the window during rapid changes and lengthens it
        under steady conditions. After the adaptive pass, a centered rolling
        mean is applied for local stabilization.

        The *change ratio* is computed against the mean of the last 10 samples.
        If the ratio exceeds ``change_threshold``, a shorter window is used;
        otherwise a longer window is used. Finally, a centered rolling window
        of size ``window_rolling`` is applied to the adaptively smoothed series.

        Parameters
        ----------
        speed_series : pandas.Series
            Speed values in m/s on a 1 Hz timeline (index used to preserve time).
            It is expected to be free of large gaps and NaNs (see
            :meth:`pre_process_record_data`).
        base_window : int, default=15
            Base window length (samples) around which the adaptive window is
            reduced or expanded.
        change_threshold : float, default=0.3
            Relative change threshold (unitless) used to detect rapid variation,
            computed as ``abs(curr - recent_avg) / recent_avg``.
        window_rolling : int, default=7
            Size (samples) of the final centered rolling mean applied after the
            adaptive pass.

        Returns
        -------
        pandas.Series
            Smoothed speed series (m/s), indexed as ``speed_series`` and rounded
            to three decimals.

        Notes
        -----
        - **Adaptation window:** recent average uses up to the last 10 samples;
        adaptation begins after the first 10 points.
        - **Window bounds:** during rapid changes the window is reduced toward
        ``max(7, base_window - 8)``; in steady segments it expands toward
        ``min(25, base_window + 8)``.
        - **Division by zero:** if the recent average is (near) zero, the change
        ratio becomes ill-defined; in prática, use desta função assume séries
        pré-processadas com velocidades > 0 em movimento (paradas prolongadas
        devem ter sido tratadas no pré-processamento).
        - **Complexidade:** O(n · w̄), onde w̄ é a janela média resultante da
        adaptação; adequada para séries de treino típicas (escala de minutos
        a horas em 1 Hz).
        - **Pré-requisito:** Recomenda-se chamar
        :meth:`pre_process_record_data` antes deste método, garantindo série
        contínua, sem duplicatas e com limites físicos aplicados.
        """
        speeds = speed_series.values
        smoothed = np.zeros_like(speeds)
        current_window = base_window

        for i in range(len(speeds)):
            # Detect significant speed changes
            if i > 10:
                recent_speeds = speeds[max(0, i - 10) : i]
                recent_avg = np.mean(recent_speeds)
                current_speed = speeds[i]
                change_ratio = abs(current_speed - recent_avg) / recent_avg

                if change_ratio > change_threshold:
                    # Speed changing rapidly - use shorter window
                    current_window = max(7, base_window - 8)
                else:
                    # Steady state - use longer window
                    current_window = min(25, base_window + 8)

            # Apply smoothing with current window
            start_idx = max(0, i - current_window // 2)
            end_idx = min(len(speeds), i + current_window // 2 + 1)

            if end_idx - start_idx > 0:
                smoothed[i] = np.mean(speeds[start_idx:end_idx])
            else:
                smoothed[i] = speeds[i]

        smoothed_serie = pd.Series(smoothed, index=speed_series.index)

        smoothed_serie_rolling = (
            smoothed_serie.rolling(window=window_rolling, center=True, min_periods=1)
            .mean()
            .round(3)
        )

        return smoothed_serie_rolling

    def _transform_speed_to_pace(
        self, dataframe: pd.DataFrame, cols_to_transform: List[str]
    ) -> pd.DataFrame:
        """
        Derive pace (float and formatted time) from speed columns.

        For each column in `cols_to_transform`, this method:
        1) Rounds the speed to three decimals,
        2) Computes pace as minutes per kilometer (float),
        3) Adds a formatted `M:SS` pace string with suffix ``_time``.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Input table containing the speed columns to transform.
        cols_to_transform : list of str
            Column names of speed-like series (m/s) to convert (e.g.,
            ``["speed_processed", "speed_rolling"]``).

        Returns
        -------
        pandas.DataFrame
            A copy of `dataframe` with additional pace columns
            (``pace_*`` and ``pace_*_time``).

        Notes
        -----
        The conversion assumes speed is in meters per second and pace is
        expressed as minutes per kilometer.
        """
        expanded_df = dataframe.copy()

        for col in cols_to_transform:
            expanded_df[col] = expanded_df[col].round(3)

            # Adding pace float column
            pace_column = col.replace("speed", "pace")
            expanded_df[pace_column] = speed_meters_per_second_to_pace(expanded_df[col])

            # Adding pace time column
            expanded_df[pace_column + "_time"] = expanded_df[pace_column].apply(
                pace_float_to_time
            )

        return expanded_df

    def process_record_data(self) -> None:
        """
        Pipeline to process record data from FIT files.

        Apply adaptive speed smoothing, derive pace, reconstruct distance,
        generate splits, and map records to laps.

        Steps
        -----
        1. Smooth speed with :meth:`_speed_smoothing` → ``speed_processed``.
        2. Derive pace (float and ``M:SS``) for ``["speed", "speed_processed"]``.
        3. Reconstruct cumulative ``distance_processed`` at 1 Hz by cumulative
        sum of ``speed_processed`` and **scale** to match the device distance
        (``raw_record["distance"].max()``).
        4. Compute integer ``split`` (1 km buckets) from ``distance_processed``.
        5. Map records to laps using a backward ``merge_asof`` of record
        ``timestamp`` to lap ``start_time`` (sorted on both sides), attaching
        the lap index (``lap``).

        Populates
        ---------
        process_record : pandas.DataFrame
            Record-level table with the following (in adição às colunas de entrada):
            - ``speed_processed`` : float
            - ``pace`` / ``pace_time`` for each speed col (e.g. ``pace_time`` for
            ``speed`` and ``speed_processed``)
            - ``distance_processed`` : float (meters, scaled to device total)
            - ``split`` : int (1, 2, 3, …) per every 1000 m
            - ``lap`` : int (1-based), from merge-as-of with lap ``start_time``

        Notes
        -----
        - **Units:** speed in m/s; pace in min/km; distance in meters.
        - **Scaling:** the scale factor is
        ``device_max = distance.max(); scale = device_max / distance_processed.max()``.
        This assumes that ``distance`` exists and that
        ``distance_processed.max() > 0``.
        - **Sorting:** both records and laps are sorted by their time columns prior
        to ``merge_asof``; the merge uses ``direction="backward"``.
        - **Splits:** computed as
        ``1 + floor(distance_processed / 1000)`` with dtype ``int``.

        Raises
        ------
        KeyError
            If required columns are missing in ``pre_process_record`` (e.g.,
            ``"timestamp"``, ``"speed"``, ``"distance"``) or in ``raw_lap``
            (e.g., ``"start_time"``).
        ValueError
            If ``distance_processed.max()`` equals zero, scaling would be
            undefined; ensure the speed series is non-empty and positive after
            preprocessing.
        """
        process_record = self.pre_process_record.copy()
        raw_lap = self.raw_lap.copy()

        # Smoothing speed
        process_record["speed_processed"] = self._speed_smoothing(
            process_record["speed"]
        )

        # Creting pace columns based on speed
        speed_cols = ["speed", "speed_processed"]
        process_record = self._transform_speed_to_pace(process_record, speed_cols)

        # Distance from processed speed (scale to match device max distance)
        process_record["distance_processed"] = process_record[
            "speed_processed"
        ].cumsum()
        scale = (
            process_record["distance"].max()
            / process_record["distance_processed"].max()
        )
        process_record["distance_processed"] = (
            process_record["distance_processed"] * scale
        ).round(2)

        # Adding split
        process_record["split"] = 1 + process_record["distance_processed"] // 1000
        process_record["split"] = process_record["split"].astype("int")

        # Adding lap
        raw_lap = raw_lap.sort_values("start_time")
        raw_lap["lap"] = raw_lap.index + 1

        process_record = process_record.sort_values("timestamp")
        process_record = pd.merge_asof(
            process_record,
            raw_lap[["start_time", "lap"]],
            left_on="timestamp",
            right_on="start_time",
            direction="backward",
        ).drop(columns=["start_time"])

        self.process_record = process_record

    def _get_summary_record(
        self, process_record: pd.DataFrame, column_reference: str
    ) -> pd.DataFrame:
        """
        Aggregate record-level metrics by a reference column (e.g., lap or split).

        The aggregation computes:
        - Heart-rate min/max/mean (integers),
        - Speed min/max (from `speed_processed`),
        - Altitude min/max/avg (rounded),
        - Per-group distance as the groupwise max of `distance_processed`
        followed by a first-difference to obtain the segment distance.

        Parameters
        ----------
        process_record : pandas.DataFrame
            Record-level table produced by :meth:`process_record_data`.
        column_reference : str
            Column name to group by (e.g., `"lap"` or `"split"`).

        Returns
        -------
        pandas.DataFrame
            One row per group (`column_reference`) with aggregated metrics.

        Notes
        -----
        Expects `distance_processed` to be monotonically nondecreasing by time.
        """
        df_summary = process_record.groupby(column_reference, as_index=False).agg(
            heart_rate_min=("heart_rate", lambda x: int(x.min())),
            heart_rate_max=("heart_rate", lambda x: int(x.max())),
            heart_rate_mean=("heart_rate", lambda x: int(round(x.mean()))),
            speed_min=("speed_processed", "min"),
            speed_max=("speed_processed", "max"),
            enhanced_altitude_min=("enhanced_altitude", lambda x: round(x.min(), 1)),
            enhanced_altitude_max=("enhanced_altitude", lambda x: round(x.max(), 1)),
            enhanced_altitude_avg=("enhanced_altitude", lambda x: round(x.mean(), 1)),
            distance_processed=("distance_processed", "max"),
        )

        # Total distance seg
        df_summary["distance_processed"] = (
            df_summary["distance_processed"]
            .diff()
            .fillna(df_summary["distance_processed"].iloc[0])
        )

        return df_summary

    def process_lap_data(self) -> None:
        """
        Build a per-lap summary table and derive speed/pace metrics.

        Steps performed:
        1. Prepare lap index (`lap`), order by `start_time`, and compute
        cumulative elapsed time (`cum_elapsed_time`).
        2. Merge per-lap aggregates from :meth:`_get_summary_record` (by `"lap"`).
        3. Compute `speed_avg` as per-lap distance divided by per-lap elapsed time.
        4. Convert min/max/avg speed to pace (float and `M:SS`).
        5. Reorder columns to a tidy, analysis-friendly schema.

        Populates
        ---------
        process_lap : pandas.DataFrame
            One row per lap with heart-rate/altitude stats, min/max/avg speeds
            and pace representations, cumulative elapsed time, and per-lap
            distance.

        Notes
        -----
        - `distance_processed` is expected to be present in the joined summary.
        - If `total_distance` is not available from the device, remove it from
        the final column ordering to avoid `KeyError`.
        """
        process_lap = self.raw_lap.copy()
        process_record_to_lap = self.process_record.copy()

        process_lap["lap"] = process_lap.index + 1
        process_lap = process_lap.sort_values("start_time")

        process_lap = process_lap.rename(columns={"timestamp": "end_time"})
        process_lap["cum_elapsed_time"] = process_lap["total_elapsed_time"].cumsum()

        process_record_to_lap = self._get_summary_record(process_record_to_lap, "lap")

        # O problema está aqui nesse jpin
        process_lap = process_lap.merge(
            process_record_to_lap, on="lap", how="left", suffixes=("", "_agg")
        )

        process_lap["speed_avg"] = (
            process_lap["distance_processed"] / process_lap["total_elapsed_time"]
        )

        speed_cols = ["speed_min", "speed_max", "speed_avg"]

        process_lap = self._transform_speed_to_pace(process_lap, speed_cols)

        ordered_cols = [
            "lap",
            "start_time",
            "end_time",
            "total_elapsed_time",
            "cum_elapsed_time",
            "total_distance",
            "distance_processed",
            "heart_rate_min",
            "heart_rate_max",
            "heart_rate_mean",
            "speed_min",
            "speed_max",
            "speed_avg",
            "pace_min",
            "pace_max",
            "pace_avg",
            "pace_min_time",
            "pace_max_time",
            "pace_avg_time",
            "enhanced_altitude_min",
            "enhanced_altitude_max",
            "enhanced_altitude_avg",
        ]

        process_lap = process_lap[ordered_cols]

        self.process_lap = process_lap
