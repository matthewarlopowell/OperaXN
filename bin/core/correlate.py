"""
Electrochemistry parsing and scan/echem time-correlation for the core pipeline.

The canonical .nxs file always stores absolute timestamps. RELATIVE mode,
selected at upload, matches scans to echem by relative offsets (for
unsynchronised instrument clocks); relative display time is a GUI-side view
transform (`generate` runs this processor with the display rewrite off).
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ECHEM_LOG_MIN_POINTS, ECHEM_TIME_TOLERANCE
from .model import DataSourceType, Scan, TimeMethod
from .classify import LOGBOOK_TIME_FORMAT, NeutronFileGrouper, NeutronMetadataParser

logger = logging.getLogger(__name__)


def _nan_to_none(value: Any) -> Any:
    """Convert NaN/NaT to None for safe downstream use."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


# ============================================================================
# Echem Parser
# ============================================================================

class EchemParser:
    """Parses tab-delimited electrochemistry files into timestamp/voltage/current DataFrames."""

    COLUMN_PATTERNS = {
        "time": ["time", "date"],
        "voltage": ["voltage", "v/", "ecell", "ewe"],
        "current": ["current", "i/"]
    }

    HEADER_KEYWORDS = ["time", "date", "ecell", "ewe", "voltage", "current", "i/", "v/"]

    def parse(self, path: str) -> Optional[pd.DataFrame]:
        """Parse one echem file into a timestamp/echem_data/current DataFrame;
        None when nothing parses. Dates are read day-first (UK convention)."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                return None

            has_header = any(h in lines[0].lower() for h in self.HEADER_KEYWORDS)

            if has_header:
                data_rows = [ln for ln in lines[1:] if ln.strip()]
                columns = self._detect_columns(
                    lines[0],
                    data_rows[0] if data_rows else None,
                    data_rows[-1] if data_rows else None)
                current_scale = self._current_unit_scale(lines[0], columns)
                data_lines = lines[1:]
            else:
                columns = {"time": 0, "voltage": 1, "current": 2}
                current_scale = 1.0
                data_lines = lines

            data = self._parse_data_lines(data_lines, columns, current_scale)

            if data:
                return pd.DataFrame(data)

            return None

        except Exception as e:
            logger.error(f"Failed to parse echem file: {e}")
            return None

    def _detect_columns(self, header_line: str,
                        sample_line: Optional[str] = None,
                        last_sample_line: Optional[str] = None) -> Dict[str, int]:
        """Map column names to indices via keyword matching; the first and
        last data rows validate that a date-named column really carries
        per-row timestamps."""
        header_parts = [part.strip().lower() for part in header_line.strip().split("\t")]

        detected: Dict[str, int] = {}

        column_mapping: Dict[str, str] = {}
        for col_type, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                column_mapping[pattern] = col_type

        time_matches = []
        matched_indices = set()
        for i, part in enumerate(header_parts):
            clean_part = part.replace("(", "").replace(")", "").replace("/", "").replace(" ", "")

            if clean_part in column_mapping:
                detected[column_mapping[clean_part]] = i
                matched_indices.add(i)
                if column_mapping[clean_part] == "time":
                    time_matches.append((i, part))
                continue

            for key, value in column_mapping.items():
                if key in part:
                    detected[value] = i
                    matched_indices.add(i)
                    if value == "time":
                        time_matches.append((i, part))
                    break

        # When several columns match time keywords, prefer a date-named one:
        # cycler exports (e.g. Arbin) pair Date_Time with an elapsed-seconds
        # Test_Time(s) column that must not win the timestamp slot. Among
        # date candidates, only one whose data carries a time of day (':')
        # AND varies across the file qualifies: a bare date would collapse
        # rows to midnight, and a constant metadata datetime (Start_Date)
        # would silently break correlation.
        date_matches = [i for i, part in time_matches if "date" in part]
        if date_matches:
            first_parts = (sample_line.strip().split("\t")
                           if sample_line else [])
            last_parts = (last_sample_line.strip().split("\t")
                          if last_sample_line else [])
            for i in date_matches:
                if not first_parts:
                    detected["time"] = i
                    break
                first_cell = first_parts[i] if i < len(first_parts) else ""
                last_cell = last_parts[i] if i < len(last_parts) else ""
                has_clock = ":" in first_cell or ":" in last_cell
                varies = (not last_parts or last_sample_line is sample_line
                          or first_cell != last_cell)
                if has_clock and varies:
                    detected["time"] = i
                    break
            else:
                # No date candidate qualified: if a bare/constant date column
                # holds the slot via last-match-wins, hand it to a non-date
                # candidate so the file fails safe instead of collapsing every
                # row to midnight
                non_date = [i for i, part in time_matches if "date" not in part]
                if detected.get("time") in date_matches and non_date:
                    detected["time"] = non_date[-1]

        # Default positional indices for undetected columns
        columns = {"time": 0, "voltage": 1, "current": 2}
        columns.update(detected)

        # Resolve index collisions between defaults and detected columns.
        # Every keyword-matched index counts as occupied, so a column vacated
        # by the date promotion can never be re-bound by a positional default
        used_indices = set(detected.values()) | matched_indices
        for col_name in columns:
            if col_name not in detected and columns[col_name] in used_indices:
                logger.warning(
                    f"Default column '{col_name}' at index {columns[col_name]} "
                    f"collides with detected column. Disabling."
                )
                columns[col_name] = -1

        return columns

    @staticmethod
    def _current_unit_scale(header_line: str, columns: Dict[str, int]) -> float:
        """1000.0 when a current-named column is unit-labelled in amps, e.g.
        Arbin's 'Current(A)'; stored current values are milliamps."""
        idx = columns.get("current", -1)
        parts = [p.strip().lower() for p in header_line.strip().split("\t")]
        if 0 <= idx < len(parts):
            part = parts[idx].replace(" ", "")
            if "(a)" in part and ("current" in part or part.startswith("i(")
                                  or part.startswith("i/")):
                return 1000.0
        return 1.0

    @staticmethod
    def _parse_data_lines(lines: List[str], columns: Dict[str, int],
                          current_scale: float = 1.0) -> List[Dict[str, Any]]:
        """Rows as dicts, skipping unparseable lines and placeholder rows;
        ambiguous dates are read day-first (UK convention, by design)."""
        # A disabled time/voltage column (-1) would silently index parts[-1]
        if columns["time"] < 0 or columns["voltage"] < 0:
            logger.warning("Echem time/voltage column could not be resolved; skipping file")
            return []

        data = []
        max_idx = max(columns.values())

        with warnings.catch_warnings():
            # pandas warns when a value can only be month-first (e.g. 05/13);
            # day-first stays the intended reading, so the warning is noise
            warnings.filterwarnings("ignore", message=".*dayfirst.*")

            for line in lines:
                parts = line.strip().split("\t")

                if len(parts) <= max_idx:
                    continue

                # Skip epoch-zero placeholder rows
                ts_str = parts[columns["time"]]
                if ts_str.startswith("1970/01/01"):
                    continue

                try:
                    if ts_str[:4].isdigit() and ts_str[4:5] in ("-", "/"):
                        # Year-first strings, e.g. converted xlsx dates:
                        # dayfirst=True silently swaps month and day on these
                        timestamp = pd.to_datetime(ts_str.replace("/", "-"),
                                                   format="ISO8601")
                    else:
                        timestamp = pd.to_datetime(ts_str, dayfirst=True)
                except (ValueError, TypeError):
                    continue

                # Blank cells parse to NaT without raising; a NaT row would
                # win the nearest-timestamp match and void every correlation
                if pd.isna(timestamp):
                    continue

                try:
                    voltage = float(parts[columns["voltage"]])
                except (ValueError, IndexError):
                    continue

                current = None
                if 0 <= columns["current"] < len(parts):
                    try:
                        current = float(parts[columns["current"]]) * current_scale
                    except (ValueError, IndexError):
                        pass

                data.append({
                    "timestamp": timestamp,
                    "echem_data": voltage,
                    "current": current
                })

        return data


# ============================================================================
# Scan Processor
# ============================================================================

class ScanProcessor:
    """Builds the ordered scan list and correlates scans with echem data."""

    def __init__(self, time_method: TimeMethod = TimeMethod.ABSOLUTE,
                 data_source: DataSourceType = DataSourceType.INHOUSE,
                 echem_time_tolerance: float = ECHEM_TIME_TOLERANCE,
                 apply_relative_display: bool = True):
        self.time_method = time_method
        self.data_source = data_source
        self.echem_time_tolerance = echem_time_tolerance
        # RELATIVE affects two things: how scans are *matched* to echem
        # (relative offsets, for unsynchronised clocks) and how timestamps are
        # *displayed*. The canonical pipeline keeps the matching but disables
        # the display rewrite so files always store absolute time.
        self.apply_relative_display = apply_relative_display
        self.xrd_reference_time: Optional[pd.Timestamp] = None
        self.echem_reference_time: Optional[pd.Timestamp] = None
        self.neutron_reference_time: Optional[pd.Timestamp] = None
        self.echem_parser = EchemParser()

    def process_scans(self, df: pd.DataFrame) -> Tuple[List[Scan], pd.DataFrame]:
        """Main pipeline: parse echem -> build scans -> adjust timestamps -> correlate."""
        combined_echem_df = self._process_echem_data(df)

        neutron_metadata_df = None
        if self.data_source == DataSourceType.NEUTRON:
            neutron_metadata_df = self._process_neutron_metadata(df)
            if neutron_metadata_df is not None:
                logger.info(f"Found {len(neutron_metadata_df)} neutron scans")

        scan_list = self._create_scan_list(df, neutron_metadata_df)

        # Compute midpoint-adjusted correlation timestamps
        self._adjust_for_exposure_time(scan_list)

        if self.time_method == TimeMethod.RELATIVE:
            self._set_reference_times(combined_echem_df, scan_list)

        # Correlate scans with echem using absolute timestamps (before formatting)
        if not combined_echem_df.empty:
            self._correlate_with_echem(scan_list, combined_echem_df)
            self._annotate_echem_window(scan_list, combined_echem_df)

        # Convert display timestamps to relative HH:MM:SS strings
        if self.time_method == TimeMethod.RELATIVE and self.apply_relative_display:
            self._apply_relative_time(scan_list, combined_echem_df)

        return scan_list, combined_echem_df

    def _process_neutron_metadata(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Parse and concatenate every logbook file; None when none parse."""
        neutron_meta_paths = df[df["neutron_meta"].notna()]["neutron_meta"].tolist()

        logger.info(f"Found {len(neutron_meta_paths)} neutron metadata files")

        if not neutron_meta_paths:
            return None

        neutron_dfs = []
        for path in neutron_meta_paths:
            meta_df = NeutronMetadataParser.parse(path)
            if meta_df is not None:
                meta_df["source_file"] = path
                neutron_dfs.append(meta_df)

        if neutron_dfs:
            combined_df = pd.concat(neutron_dfs, ignore_index=True)
            logger.info(f"Combined neutron metadata: {len(combined_df)} total entries")
            return combined_df

        return None

    def _process_echem_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse all echem files, concatenate, and sort by timestamp."""
        echem_rows = df[df["echem"].notna()]
        echem_paths = echem_rows["echem"].tolist()
        # source_file records the user's file, not a converted temp copy
        if "original_path" in df.columns:
            originals = echem_rows["original_path"].tolist()
        else:
            originals = echem_paths
        echem_dfs = []

        for e_path, orig in zip(echem_paths, originals):
            e_df = self.echem_parser.parse(e_path)
            if e_df is not None:
                e_df["source_file"] = orig if pd.notna(orig) else e_path
                echem_dfs.append(e_df)

        if echem_dfs:
            combined_df = (pd.concat(echem_dfs, ignore_index=True)
                           .sort_values("timestamp").reset_index(drop=True))
            logger.info(f"Echem data: {len(combined_df)} rows")
            return combined_df

        return pd.DataFrame(columns=["timestamp", "echem_data", "current", "source_file"])

    def _set_reference_times(self, echem_df: pd.DataFrame,
                             scan_list: Optional[List[Scan]] = None) -> None:
        """Set t=0 reference for each data stream in relative time mode."""

        # XRD/synchrotron reference: earliest scan midpoint
        if self.data_source != DataSourceType.NEUTRON and scan_list:
            mids = [
                scan.timestamp_for_correlation
                for scan in scan_list
                if scan.timestamp_for_correlation is not None and (scan.oned or scan.twod)
            ]
            self.xrd_reference_time = min(mids) if mids else None
        else:
            # Neutron dataframes carry no oned/twod rows, and an empty scan
            # list implies none either -- there is nothing to reference.
            self.xrd_reference_time = None

        # Echem reference: earliest echem point
        if not echem_df.empty:
            try:
                self.echem_reference_time = pd.to_datetime(echem_df["timestamp"]).min()
            except (ValueError, TypeError):
                self.echem_reference_time = None

        # Neutron reference: earliest neutron midpoint
        if self.data_source == DataSourceType.NEUTRON and scan_list:
            mids = [
                scan.timestamp_for_correlation
                for scan in scan_list
                if scan.timestamp_for_correlation is not None
            ]
            self.neutron_reference_time = min(mids) if mids else None

    @staticmethod
    def _create_neutron_scan_list(df: pd.DataFrame,
                                  neutron_metadata_df: pd.DataFrame) -> List[Scan]:
        """Match neutron .dat file groups to logbook entries, skip unmatched scans."""
        scan_list: List[Scan] = []

        neutron_files = [
            row["path"] for _, row in df.iterrows()
            if pd.notna(row["path"]) and row["path"].endswith('.dat')
        ]
        logger.info(f"Found {len(neutron_files)} .dat files")

        grouper = NeutronFileGrouper()
        neutron_groups = grouper.group_neutron_files(neutron_files)

        scans_without_data = 0

        for _, meta_row in neutron_metadata_df.iterrows():
            scan_id = str(meta_row["scan_id"])

            neutron_data_files = neutron_groups.get(scan_id, {})

            if not neutron_data_files:
                scans_without_data += 1
                continue

            start_time = meta_row["start_time"]
            end_time = meta_row["end_time"]

            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time, format=LOGBOOK_TIME_FORMAT)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time, format=LOGBOOK_TIME_FORMAT)

            midpoint = start_time + (end_time - start_time) / 2

            logbook = {
                key: meta_row[key] for key in
                ("run_title", "users", "proposal", "full_line")
                if key in meta_row and pd.notna(meta_row.get(key))
            }

            scan = Scan(
                scan_num=0,
                neutron_meta=meta_row.get("source_file"),
                neutron_files=neutron_data_files,
                neutron_start=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                neutron_end=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                timestamp=midpoint.strftime('%Y-%m-%d %H:%M:%S'),
                original_timestamp=midpoint.strftime('%Y-%m-%d %H:%M:%S'),
                timestamp_for_correlation=midpoint,
                logbook=logbook or None
            )
            scan_list.append(scan)

        logger.info(f"Created {len(scan_list)} neutron scans with data")
        if scans_without_data:
            logger.warning(
                f"{scans_without_data} logbook entries had no matching data files and were skipped")

        return scan_list

    def _create_scan_list(self, df: pd.DataFrame,
                          neutron_metadata_df: Optional[pd.DataFrame] = None) -> List[Scan]:
        """Build scan list from classified DataFrame, sorted by timestamp."""
        scan_list: List[Scan] = []

        if self.data_source == DataSourceType.NEUTRON and neutron_metadata_df is not None:
            scan_list = self._create_neutron_scan_list(df, neutron_metadata_df)
        elif self.data_source == DataSourceType.SYNCHROTRON:
            synchrotron_df = df[(df["oned"].notna()) | (df["twod"].notna())]
            for _, row in synchrotron_df.iterrows():
                scan = Scan(
                    scan_num=0,
                    oned=_nan_to_none(row["oned"]),
                    twod=_nan_to_none(row["twod"]),
                    timestamp=_nan_to_none(row["timestamp"]),
                    original_timestamp=_nan_to_none(row["timestamp"]),
                    exposure_time=_nan_to_none(row.get("exposure_time")),
                    oned_exposure=_nan_to_none(row.get("exposure_time")),
                    twod_exposure=_nan_to_none(row.get("exposure_time")),
                    source_nxs=_nan_to_none(row.get("path"))
                )
                scan_list.append(scan)
        else:
            # Laboratory: pair 1D and 2D by matching timestamps
            oned_df = df[df["oned"].notna()]
            twod_df = df[df["twod"].notna()]

            for _, row in oned_df.iterrows():
                scan = Scan(
                    scan_num=0,
                    oned=_nan_to_none(row["oned"]),
                    timestamp=_nan_to_none(row.get("timestamp")),
                    original_timestamp=_nan_to_none(row.get("timestamp")),
                    oned_exposure=_nan_to_none(row.get("exposure_time"))
                )
                scan_list.append(scan)

            for _, row in twod_df.iterrows():
                row_ts = _nan_to_none(row.get("timestamp"))
                existing = None
                for existing_scan in scan_list:
                    if (pd.notna(row_ts) and pd.notna(existing_scan.timestamp)
                            and existing_scan.timestamp == row_ts):
                        existing = existing_scan
                        break

                if existing:
                    existing.twod = _nan_to_none(row["twod"])
                    existing.twod_exposure = _nan_to_none(row.get("exposure_time"))
                else:
                    scan = Scan(
                        scan_num=0,
                        twod=_nan_to_none(row["twod"]),
                        timestamp=_nan_to_none(row.get("timestamp")),
                        original_timestamp=_nan_to_none(row.get("timestamp")),
                        twod_exposure=_nan_to_none(row.get("exposure_time"))
                    )
                    scan_list.append(scan)

        # Sort by timestamp; safe sentinel avoids pandas datetime resolution mismatches
        _sort_sentinel = pd.Timestamp("1900-01-01")
        scan_list.sort(key=lambda s: pd.to_datetime(s.timestamp) if s.timestamp else _sort_sentinel)

        # Assign sequential scan numbers
        for i, scan in enumerate(scan_list, start=1):
            scan.scan_num = i

        return scan_list

    def _adjust_for_exposure_time(self, scan_list: List[Scan]) -> None:
        """Set timestamp_for_correlation to the exposure midpoint for each scan."""
        for scan in scan_list:
            exposure_time = self._determine_exposure_time(scan)
            scan.exposure_time = exposure_time

            if self.data_source == DataSourceType.NEUTRON:
                # Neutron midpoint already computed from start/end
                scan.timestamp_for_correlation = pd.to_datetime(scan.timestamp) if scan.timestamp else None
            else:
                if exposure_time and scan.timestamp:
                    original_ts = pd.to_datetime(scan.timestamp)
                    adjusted_ts = original_ts + pd.Timedelta(seconds=exposure_time / 2)
                    scan.timestamp_for_correlation = adjusted_ts
                else:
                    scan.timestamp_for_correlation = pd.to_datetime(scan.timestamp) if scan.timestamp else None

    def _determine_exposure_time(self, scan: Scan) -> Optional[float]:
        """Return best available exposure time; average 1D/2D if both present."""
        if self.data_source == DataSourceType.NEUTRON and scan.neutron_start and scan.neutron_end:
            start = pd.to_datetime(scan.neutron_start)
            end = pd.to_datetime(scan.neutron_end)
            return (end - start).total_seconds()

        if scan.oned and scan.twod:
            if scan.oned_exposure and scan.twod_exposure:
                return (scan.oned_exposure + scan.twod_exposure) / 2
        elif scan.oned and scan.oned_exposure:
            return scan.oned_exposure
        elif scan.twod and scan.twod_exposure:
            return scan.twod_exposure
        return None

    def _correlate_with_echem(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Dispatch to relative or absolute correlation per time_method."""
        if self.time_method == TimeMethod.RELATIVE:
            self._correlate_relative_time(scan_list, echem_df)
        else:
            self._correlate_absolute_time(scan_list, echem_df)

    def _correlate_relative_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Match scans to nearest echem point by relative offset from respective t=0."""
        if self.data_source == DataSourceType.NEUTRON:
            reference_time = self.neutron_reference_time
        else:
            reference_time = self.xrd_reference_time

        if not reference_time or not self.echem_reference_time:
            logger.warning("No reference times available for relative correlation")
            return

        echem_timestamps = pd.to_datetime(echem_df["timestamp"])
        echem_relative_seconds = (echem_timestamps - self.echem_reference_time).dt.total_seconds()

        for scan in scan_list:
            if not scan.timestamp_for_correlation:
                scan.echem_timestamp = None
                continue

            scan_relative_seconds = (scan.timestamp_for_correlation - reference_time).total_seconds()

            time_diffs = np.abs(echem_relative_seconds.values - scan_relative_seconds)
            nearest_idx = int(np.argmin(time_diffs))
            min_diff_seconds = time_diffs[nearest_idx]

            if min_diff_seconds < self.echem_time_tolerance:
                scan.echem = float(echem_df.iloc[nearest_idx]["echem_data"])
                current_val = echem_df.iloc[nearest_idx]["current"]
                scan.current = float(current_val) if pd.notna(current_val) else None
                scan.echem_timestamp = str(echem_df.iloc[nearest_idx]["timestamp"])
            else:
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None

    @staticmethod
    def _parse_echem_timestamps(echem_df: pd.DataFrame) -> Optional[pd.Series]:
        """Parse the echem timestamp column, trying the known day-first formats."""
        for kwargs in ({}, {"format": "%d/%m/%Y %H:%M:%S.%f"},
                       {"format": "%d/%m/%Y %H:%M:%S"}):
            try:
                return pd.to_datetime(echem_df["timestamp"], **kwargs)
            except (ValueError, TypeError):
                continue
        logger.error("Could not parse echem timestamps")
        return None

    def _correlate_absolute_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Match scans to nearest echem point by absolute timestamp proximity."""
        echem_timestamps = self._parse_echem_timestamps(echem_df)
        if echem_timestamps is None:
            return

        echem_start = echem_timestamps.min()
        echem_end = echem_timestamps.max()
        tolerance = pd.Timedelta(seconds=self.echem_time_tolerance)

        for scan in scan_list:
            scan_time = scan.timestamp_for_correlation
            if not scan_time:
                scan.echem_timestamp = None
                continue

            if isinstance(scan_time, str):
                scan_time = pd.to_datetime(scan_time)

            # Skip scans outside echem window + tolerance
            if scan_time < echem_start - tolerance or scan_time > echem_end + tolerance:
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None
                continue

            time_diffs = abs(echem_timestamps - scan_time)
            nearest_idx = int(np.argmin(time_diffs.values))
            min_diff = time_diffs.iloc[nearest_idx]

            if min_diff.total_seconds() < self.echem_time_tolerance:
                scan.echem = float(echem_df.iloc[nearest_idx]["echem_data"])
                current_val = echem_df.iloc[nearest_idx]["current"]
                scan.current = float(current_val) if pd.notna(current_val) else None
                scan.echem_timestamp = str(echem_df.iloc[nearest_idx]["timestamp"])
            else:
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None

    def _acquisition_window(self, scan: Scan) -> Tuple[Optional[pd.Timestamp],
                                                       Optional[pd.Timestamp]]:
        """Return (start, end) of a scan's acquisition as absolute timestamps."""
        if self.data_source == DataSourceType.NEUTRON:
            if not scan.neutron_start:
                return None, None
            start = pd.to_datetime(scan.neutron_start)
            end = pd.to_datetime(scan.neutron_end) if scan.neutron_end else None
            return start, end

        ts = scan.original_timestamp or scan.timestamp
        if not ts:
            return None, None
        try:
            start = pd.to_datetime(ts)
        except (ValueError, TypeError):
            return None, None
        if scan.exposure_time:
            return start, start + pd.Timedelta(seconds=scan.exposure_time)
        return start, None

    def _annotate_echem_window(self, scan_list: List[Scan],
                               echem_df: pd.DataFrame) -> None:
        """Fill per-scan echem window summaries: voltage/current min/max,
        0-based indices into the sorted operando arrays, and (neutron) NXlog
        segments. Runs before the relative-display rewrite of echem_df."""
        echem_timestamps = self._parse_echem_timestamps(echem_df)
        if echem_timestamps is None:
            return

        # In relative mode the streams have unsynchronised clocks: shift the
        # scan window into the echem time base with the same references the
        # nearest-neighbour matching uses.
        offset = pd.Timedelta(0)
        if self.time_method == TimeMethod.RELATIVE:
            reference_time = (self.neutron_reference_time
                              if self.data_source == DataSourceType.NEUTRON
                              else self.xrd_reference_time)
            if reference_time is None or self.echem_reference_time is None:
                return
            offset = self.echem_reference_time - reference_time

        ts_values = echem_timestamps.values
        voltage = pd.to_numeric(echem_df["echem_data"], errors="coerce").to_numpy(dtype=float)
        current = None
        if "current" in echem_df.columns:
            current = pd.to_numeric(echem_df["current"], errors="coerce").to_numpy(dtype=float)

        for scan in scan_list:
            start, end = self._acquisition_window(scan)
            if start is None:
                continue
            start = start + offset
            end = (end + offset) if end is not None else start

            i0 = int(np.searchsorted(ts_values, start.to_datetime64(), side="left"))
            i1 = int(np.searchsorted(ts_values, end.to_datetime64(), side="right"))
            if i1 <= i0:
                continue

            scan.echem_index_start = i0
            scan.echem_index_end = i1 - 1
            window_v = voltage[i0:i1]
            scan.voltage_min = float(np.nanmin(window_v))
            scan.voltage_max = float(np.nanmax(window_v))
            window_i = current[i0:i1] if current is not None else None
            has_current = window_i is not None and not np.all(np.isnan(window_i))
            if has_current:
                scan.current_min = float(np.nanmin(window_i))
                scan.current_max = float(np.nanmax(window_i))

            if (self.data_source == DataSourceType.NEUTRON
                    and i1 - i0 >= ECHEM_LOG_MIN_POINTS):
                seg_ts = echem_timestamps.iloc[i0:i1]
                segment = {
                    "start": seg_ts.iloc[0].isoformat(),
                    "time_s": (seg_ts - seg_ts.iloc[0]).dt.total_seconds().to_numpy(),
                    "voltage": window_v.copy(),
                }
                if has_current:
                    segment["current"] = window_i.copy()
                scan.echem_segment = segment

    def _apply_relative_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Replace absolute timestamps with HH:MM:SS relative strings for display."""
        if self.data_source == DataSourceType.NEUTRON:
            reference_time = self.neutron_reference_time
        else:
            reference_time = self.xrd_reference_time

        if reference_time:
            for scan in scan_list:
                base_time = None

                if scan.timestamp_for_correlation is not None:
                    base_time = scan.timestamp_for_correlation
                elif scan.timestamp:
                    try:
                        base_time = pd.to_datetime(scan.timestamp)
                    except (ValueError, TypeError):
                        base_time = None

                if base_time is not None:
                    scan.original_timestamp = scan.timestamp
                    relative_seconds = (base_time - reference_time).total_seconds()
                    scan.timestamp = format_relative_time(relative_seconds)

        if not echem_df.empty and self.echem_reference_time:
            original_timestamps = pd.to_datetime(echem_df["timestamp"])
            relative_seconds = (original_timestamps - self.echem_reference_time).dt.total_seconds()

            echem_df["original_timestamp"] = echem_df["timestamp"].copy()
            echem_df["timestamp"] = [format_relative_time(sec) for sec in relative_seconds]


def format_relative_time(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_relative_time(value: str) -> Optional[float]:
    """Parse an HH:MM:SS relative string back to seconds; None if not one."""
    if not isinstance(value, str) or "-" in value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
