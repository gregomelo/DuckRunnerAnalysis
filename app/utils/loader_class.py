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
4. Optimize smoothing for speed (Savitzky–Golay + centered rolling) via a small
   grid-search that minimizes a composite loss (fidelity vs. smoothness).
5. Generate processed series (speed components and pace), reconstruct cumulative
   distance from processed speed, and scale to match the device distance.
6. Map records to laps via time-aware merge-as-of and compute per-lap summaries
   (heart rate, altitude, min/max/avg speeds, per-lap distance).
7. Produce a tidy `process_lap` table with speeds and their pace equivalents.

Notes
-----
- Coordinates are converted from *semicircles* to degrees using:
  ``degrees = semicircles * 180 / 2**31``.
- Record timestamps are resampled to 1 second; missing values are imputed with
  time-based interpolation (limits differ for "fast" and "slow" signals) and
  forward/backward fill as a safety net.
- Speed smoothing parameters are selected from a small grid to balance fidelity
  to the raw signal and second-order smoothness; the chosen params are exposed
  in ``best_smoothing_params``.
- Distance reconstruction integrates the processed speed (1 Hz) and scales the
  cumulative total to match the device-reported maximum distance.
- Laps are associated to records via a backward merge-as-of on
  ``timestamp`` (record) and ``start_time`` (lap). Ensure both sides are sorted.

Main Class
----------
DataLoaderFromFIT
    Orchestrates the end-to-end pipeline from FIT decoding to tabular outputs.
"""

from itertools import product
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Self

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream
from scipy.signal import savgol_filter

from .unit_converter import pace_float_to_time, speed_meters_per_second_to_pace


class DataLoaderFromFIT:
    """
    Decode, transform, and summarize running data from Garmin FIT files.

    This class orchestrates a complete pipeline for converting a FIT file into
    structured pandas DataFrames representing sessions, laps, and records. It
    supports smoothing of speed signals, pace conversion, reconstruction of
    distance, and computation of per-lap summaries.

    The processing sequence is executed by calling :meth:`start`, which performs
    the following steps in order:

    1. :meth:`extract_message` — decodes the FIT file into message dictionaries.
    2. :meth:`load_raw_data` — converts message lists into raw DataFrames.
    3. :meth:`pre_process_record_data` — resamples, imputes, and augments record data.
    4. :meth:`process_record_data` — optimizes and applies speed smoothing, converts
       to pace, reconstructs distance, and maps records to laps.
    5. :meth:`process_lap_data` — aggregates per-lap statistics and finalizes
       the `process_lap` table.

    The resulting DataFrames are suitable for analytical workflows or visual
    applications such as Streamlit dashboards.

    Parameters
    ----------
    fit_file : str, Path, or BinaryIO
        Path or binary stream of the FIT file to be processed.

    Attributes
    ----------
    message : dict
        Parsed FIT messages grouped by type (e.g., ``session_mesgs``,
        ``lap_mesgs``, ``record_mesgs``).
    error : list
        List of decoding errors reported by the FIT SDK.
    raw_session, raw_event, raw_lap, raw_record : pandas.DataFrame
        Untouched DataFrames created from the raw message data.
    pre_process_record : pandas.DataFrame
        1 Hz record-level table with interpolated values, flags for missing
        data, elapsed time, and converted coordinates.
    process_record : pandas.DataFrame
        Smoothed record-level data with pace columns, reconstructed distance,
        and mapped lap identifiers.
    process_lap : pandas.DataFrame
        Per-lap summary table with heart-rate, altitude, and pace statistics.
    best_smoothing_params : dict or None
        Optimal Savitzky–Golay and rolling parameters selected during
        smoothing (keys: ``wl``, ``po``, ``wr``, ``loss``).

    Raises
    ------
    IOError
        If the FIT file cannot be opened or read.
    RuntimeError
        If the decoding process fails unexpectedly.
    ValueError
        If the FIT decoder reports an error or if no valid smoothing
        configuration is found.

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
    best_smoothing_params: Dict[str, float | int] | None = None
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
        Resample, impute, and augment record data to a 1 Hz timeline.

        Operations performed:
        1. Convert `position_lat/long` from semicircles to degrees.
        2. Remove duplicate timestamps by averaging rows with the same timestamp.
        3. Sort by timestamp and resample to 1-second frequency.
        4. Flag missing values for selected signals.
        5. Compute `elapsed_time` (seconds from the first record).
        6. Interpolate missing values (time-based) with different limits for
        "fast" (`speed`, `power`, coordinates) and "slow" signals
        (`heart_rate`, `enhanced_altitude`, `enhanced_speed`).
        7. Forward/backward fill as a final safety net and round/cast types.

        Populates
        ---------
        pre_process_record : pandas.DataFrame
            Cleaned, 1 Hz record table with flags and augmented fields.

        Notes
        -----
        - Assumes the FIT timestamps are timezone-aware or consistent; only
        relative timing is used downstream.
        - Columns listed for interpolation must exist in the input.
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

    def _optimize_speed_params(
        self,
        speed: pd.Series,
        alpha: float = 0.7,
        objective: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> tuple[dict, np.ndarray]:
        """
        Grid-search Savitzky–Golay + rolling parameters to smooth speed.

        The default objective balances fidelity and smoothness:
        ``loss = alpha * ||processed - raw||^2 + (1 - alpha) * ||Δ²(processed)||^2``.

        Parameters
        ----------
        speed : pandas.Series
            Raw speed sampled at 1 Hz (uniform).
        alpha : float, default 0.7
            Weight for fidelity versus smoothness in the composite loss.
        objective : callable(processed, raw) -> float, optional
            Custom loss function. If provided, it overrides the default.

        Returns
        -------
        (params, processed) : (dict, numpy.ndarray)
            Best hyperparameters (``{'wl','po','wr','loss'}``) and the resulting
            processed speed series as a NumPy array.

        Raises
        ------
        ValueError
            If no valid (window_length, polyorder, rolling_window) combination is
            found (e.g., series too short for the chosen grids).

        Notes
        -----
        - Candidate grids are small by design for performance.
        - Assumes uniform 1 Hz sampling (resampled upstream).
        """
        wl_grid = (5, 7, 9, 11, 13)
        po_grid = (2, 3)
        roll_grid = (5, 7)

        raw = speed.to_numpy(dtype=float, copy=False)
        n = raw.size

        def valid(wl: int, po: int, wr: int) -> bool:
            # SavGol constraints
            if wl % 2 == 0 or wl <= po or wl > n:
                return False
            return wr >= 1

        best_loss = np.inf
        best_processed = None
        best_params = {}

        for wl, po, wr in product(wl_grid, po_grid, roll_grid):
            if not valid(wl, po, wr):
                continue

            # Savitzky–Golay; assumes uniform sampling (you already resampled to 1s)
            sg = savgol_filter(raw, window_length=wl, polyorder=po, mode="interp")

            # Centered rolling mean
            roll = (
                pd.Series(raw)
                .rolling(window=wr, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

            processed = 0.7 * sg + 0.3 * roll

            if objective is not None:
                loss = objective(processed, raw)
            else:
                resid = processed - raw
                fidelity = float(np.dot(resid, resid))
                rough = np.diff(processed, n=2)
                smoothness = float(np.dot(rough, rough))
                loss = alpha * fidelity + (1.0 - alpha) * smoothness

            if loss < best_loss:
                best_loss = loss
                best_processed = processed
                best_params = {"wl": wl, "po": po, "wr": wr, "loss": loss}

        if best_processed is None:
            raise ValueError(
                "No valid (wl, po, wr) combination found for given series."
            )

        return best_params, best_processed

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
        Apply smoothing to speed, convert to pace, and reconstruct distance.

        Steps performed:
        1. Select optimal smoothing parameters via :meth:`_optimize_speed_params`.
        2. Compute `speed_sav_gol`, `speed_rolling`, and `speed_processed`.
        3. Convert speed columns to pace (float and `M:SS`).
        4. Reconstruct `distance_processed` by cumulative sum of processed speed
        (1 Hz) and scale to match the device maximum distance.
        5. Compute `split` (integer) from `distance_processed` (every 1 km).
        6. Associate records with laps via a backward merge-as-of between record
        `timestamp` and lap `start_time`.

        Populates
        ---------
        process_record : pandas.DataFrame
            Record-level table with smoothed speeds, pace fields, processed
            distance, split markers, and lap identifiers.

        Notes
        -----
        - Requires `pre_process_record` and `raw_lap` to be available.
        - Both records and laps are sorted before the merge-as-of step.
        """
        process_record = self.pre_process_record.copy()
        raw_lap = self.raw_lap.copy()

        # Applying smooth tunning
        params, processed = self._optimize_speed_params(process_record["speed"])
        self.best_smoothing_params = params

        # Recompute individual components for transparency using best params
        wl, po, wr = params["wl"], params["po"], params["wr"]

        process_record["speed_sav_gol"] = savgol_filter(
            process_record["speed"].to_numpy(dtype=float, copy=False),
            window_length=wl,
            polyorder=po,
            mode="interp",
        )
        process_record["speed_rolling"] = (
            process_record["speed"]
            .rolling(window=wr, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        process_record["speed_processed"] = processed

        speed_cols = ["speed_sav_gol", "speed_rolling", "speed_processed"]

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
