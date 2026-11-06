"""
Electrochemistry parsing and scan/echem time-correlation for the core pipeline.

The canonical .nxs file always stores absolute timestamps. RELATIVE mode,
selected at upload, matches scans to echem by relative offsets (for
unsynchronised instrument clocks); relative display time is a GUI-side view
transform.
"""

import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ECHEM_TIME_TOLERANCE
from .model import DataSourceType, Scan, TimeMethod
from .classify import NeutronFileGrouper, NeutronMetadataParser, name_list, strip_utc_offset

logger = logging.getLogger(__name__)


def _is_number(text: str) -> bool:
    """True when text reads as a plain number ('12.5', '1e3')."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _unit(label: str) -> str:
    """Return a column label's unit, the text after its last '/', '(' or '[' ('Ewe/V' -> 'v')."""
    return re.split(r"[/(\[]", label.lower().replace(" ", ""))[-1].rstrip(")]")


def _voltage_scale(label: str) -> Optional[float]:
    """Return the factor taking a voltage label's unit to volts (V 1.0, mV 0.001), else None."""
    scale = {"v": 1.0, "mv": 0.001}.get(_unit(label))
    if scale is None:
        # A V or mV unit followed by more text, e.g. 'Voltage(V) avg' or 'Ewe/V vs. Li'
        match = re.search(r"[/(\[]\s*(mv|v)(?![a-z])", label.lower())
        if match:
            scale = 1.0 if match.group(1) == "v" else 0.001
    return scale


def _current_scale(label: str) -> Optional[float]:
    """Return the factor taking a current label's unit to milliamps, None when it names none."""
    part = label.lower().replace(" ", "")
    scale = None
    if "current" in part or re.match(r"<?i>?[/(\[]", part):
        scale = {"a": 1000.0, "ma": 1.0, "ua": 0.001, "µa": 0.001, "μa": 0.001,
                 "na": 1e-6, "pa": 1e-9}.get(_unit(label))
        if scale is None and "(a)" in part:
            scale = 1000.0  # an amp unit followed by more text, e.g. 'Current(A) avg'
    return scale


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
    """Parser for tab-delimited electrochemistry files (timestamp, voltage, current)."""

    COLUMN_PATTERNS = {
        "time": ["time", "date"],
        "voltage": ["voltage", "v/", "ecell", "ewe"],
        "current": ["current", "i/"]
    }

    HEADER_KEYWORDS = ["time", "date", "ecell", "ewe", "voltage", "current", "i/", "v/"]

    def parse(self, path: str, name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Parse an echem file into a timestamp/echem_data/current DataFrame, or None.
        Ambiguous dates are read day-first (UK convention); messages call the file `name`."""
        name = name or os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                return None

            has_header = any(h in lines[0].lower() for h in self.HEADER_KEYWORDS)

            if has_header:
                # An ANSI header is read as cp1252: dropping its bytes would turn "I/µA" into
                # amps. It is decoded on its own, so a stray byte in a data row cannot do the same
                with open(path, "rb") as f:
                    header = f.readline().splitlines()[0]
                try:
                    header.decode("utf-8")
                except UnicodeDecodeError:
                    lines[0] = header.decode("cp1252", errors="replace") + "\n"
                data_rows = [ln for ln in lines[1:] if ln.strip()]
                columns = self._detect_columns(
                    lines[0],
                    data_rows[0] if data_rows else None,
                    data_rows[-1] if data_rows else None,
                    name=name)
                if not columns:
                    return None
                voltage_scale = self._voltage_unit_scale(lines[0], columns, name)
                current_scale = self._current_unit_scale(lines[0], columns, name)
                data_lines = lines[1:]
            else:
                columns = {"time": 0, "voltage": 1, "current": 2}
                voltage_scale = current_scale = 1.0
                data_lines = lines

            data = self._parse_data_lines(data_lines, columns, voltage_scale, current_scale,
                                          name=name)

            if data:
                return pd.DataFrame(data)

            return None

        except Exception as e:
            logger.error(f"Failed to parse echem file {name}: {e}")
            return None

    def _detect_columns(self, header_line: str,
                        sample_line: Optional[str] = None,
                        last_sample_line: Optional[str] = None,
                        name: str = "") -> Dict[str, int]:
        """Map time/voltage/current to header indices by keyword (-1 disabled; {} skips the file).
        The first and last data rows confirm a date-named column carries per-row timestamps."""
        header_parts = [part.strip().lower() for part in header_line.strip().split("\t")]

        detected: Dict[str, int] = {}

        column_mapping: Dict[str, str] = {}
        for col_type, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                column_mapping[pattern] = col_type

        # Match strength ranks, so derived or auxiliary columns can never
        # steal a slot from the real one regardless of column order:
        # exact clean name (4) > keyword plus at most a unit letter, e.g.
        # Voltage(V) -> 'voltagev' (3) > keyword inside a longer name, e.g.
        # Aux_Voltage or Test_Time (2) > slashed shorthand 'v/', 'i/', as in
        # dV/dt(V/s) (1); equal strengths keep last-match-wins
        time_matches = []
        matched_indices = set()
        strengths: Dict[str, int] = {}

        def _claim(col_type, i, part, strength):
            matched_indices.add(i)
            if col_type == "time":
                time_matches.append((i, part))
            if strength >= strengths.get(col_type, 0):
                detected[col_type] = i
                strengths[col_type] = strength

        for i, part in enumerate(header_parts):
            clean_part = part.replace("(", "").replace(")", "").replace("/", "").replace(" ", "")

            if clean_part in column_mapping:
                _claim(column_mapping[clean_part], i, part, 4)
                continue

            for key, value in column_mapping.items():
                if key in part:
                    if "/" in key:
                        strength = 1
                    elif (clean_part.startswith(key) and
                          len(clean_part) <= len(key) + 2):
                        strength = 3
                    else:
                        strength = 2
                    _claim(value, i, part, strength)
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
                    f"{name}: default column '{col_name}' at index {columns[col_name]} "
                    f"collides with a detected column; it is not read"
                )
                columns[col_name] = -1

        # A voltage or current taken by position must carry its unit (V or mV; a current unit
        # as _current_scale reads it), so a column of another log is never read as echem
        if columns["time"] >= 0 and columns["voltage"] >= 0:
            cells = header_line.strip().split("\t")
            for col_name in ("voltage", "current"):
                idx = columns[col_name]
                if col_name in detected or idx < 0:
                    continue
                label = cells[idx].strip().strip("\"'").strip() if idx < len(cells) else ""
                has_unit = (_voltage_scale(label) if col_name == "voltage"
                            else _current_scale(label)) is not None
                if not has_unit:
                    logger.warning(f"Skipping {name}: column '{label}' is not a {col_name} column"
                                   if label else f"Skipping {name}: no {col_name} column")
                    return {}

        return columns

    @staticmethod
    def _voltage_unit_scale(header_line: str, columns: Dict[str, int], name: str = "") -> float:
        """Return the factor taking the voltage column's unit to volts (1.0 for V or no unit).
        The unit follows the last '/', '(' or '[' of the label; any but V or mV is read as V."""
        idx = columns.get("voltage", -1)
        cells = header_line.strip().split("\t")
        if not 0 <= idx < len(cells):
            return 1.0
        label = cells[idx].strip().strip("\"'").strip()
        column = f"Echem voltage column '{label}'" + (f" in {name}" if name else "")
        scale = _voltage_scale(label)
        if scale is None:
            if re.search(r"[/(\[]", label):
                logger.warning(f"{column} is not in V; values read as V")
            return 1.0
        if scale != 1.0:
            logger.info(f"{column} converted to V (x{scale:g})")
        return scale

    @staticmethod
    def _current_unit_scale(header_line: str, columns: Dict[str, int], name: str = "") -> float:
        """Return the factor taking the current column's unit to milliamps (1.0 when unlabelled).
        The unit follows the last '/', '(' or '[' of a current label: A, mA, uA, µA, nA or pA."""
        idx = columns.get("current", -1)
        cells = header_line.strip().split("\t")
        if not 0 <= idx < len(cells):
            return 1.0
        label = cells[idx].strip().strip("\"'").strip()
        column = f"Echem current column '{label}'" + (f" in {name}" if name else "")
        scale = _current_scale(label)
        if scale is None:
            logger.warning(f"{column} has no recognised unit (A, mA, uA, nA, pA); values read as mA")
            return 1.0
        if scale != 1.0:
            logger.info(f"{column} converted to mA (x{scale:g})")
        return scale

    @staticmethod
    def _parse_data_lines(lines: List[str], columns: Dict[str, int], voltage_scale: float,
                          current_scale: float, name: str) -> List[Dict[str, Any]]:
        """Parse data lines into row dicts, skipping unparseable and placeholder rows.
        Ambiguous dates are read day-first (UK convention, by design)."""
        # A disabled time/voltage column (-1) would silently index parts[-1]
        if columns["time"] < 0 or columns["voltage"] < 0:
            logger.warning(f"Skipping {name}: echem time/voltage column could not be resolved")
            return []

        data = []
        offsets = set()
        # A row needs its time and voltage; a blank or missing current is read as None
        rows = [parts for parts in (line.strip().split("\t") for line in lines)
                if len(parts) > max(columns["time"], columns["voltage"])]
        # Voltage-only files (no row reaches the current column) are not read
        if not any(len(parts) > columns["current"] for parts in rows):
            return []

        # Elapsed-time exports hold plain numbers, which dateutil reads as years or clocks
        cells = [parts[columns["time"]] for parts in rows if parts[columns["time"]].strip()]
        if cells and all(_is_number(cell) for cell in cells):
            logger.warning(f"Skipping {name}: echem time column holds plain numbers "
                           "(elapsed seconds?), not dates")
            return []

        with warnings.catch_warnings():
            # pandas warns when a value can only be month-first (e.g. 05/13);
            # day-first stays the intended reading, so the warning is noise
            warnings.filterwarnings("ignore", message=".*dayfirst.*")

            for parts in rows:
                # Skip epoch-zero placeholder rows
                ts_str = parts[columns["time"]]
                if ts_str.startswith("1970/01/01"):
                    continue

                try:
                    if ts_str[:4].isdigit() and ts_str[4:5] in ("-", "/"):
                        # Year-first strings, e.g. converted xlsx dates:
                        # dayfirst=True silently swaps month and day on these
                        iso = ts_str.replace("/", "-")
                        try:
                            timestamp = pd.to_datetime(iso, format="ISO8601")
                        except ValueError:  # 'UTC' or 'GMT' after the clock
                            timestamp = pd.to_datetime(strip_utc_offset(iso), format="ISO8601")
                    else:
                        timestamp = pd.to_datetime(ts_str, dayfirst=True)
                except (ValueError, TypeError):
                    continue

                # Blank cells parse to NaT without raising; a NaT row would
                # win the nearest-timestamp match and void every correlation
                if pd.isna(timestamp):
                    continue
                if timestamp.tzinfo is not None:
                    offsets.add(timestamp.strftime("%z"))
                timestamp = strip_utc_offset(timestamp)

                try:
                    voltage = float(parts[columns["voltage"]]) * voltage_scale
                except ValueError:
                    continue

                current = None
                if 0 <= columns["current"] < len(parts):
                    try:
                        current = float(parts[columns["current"]]) * current_scale
                    except ValueError:
                        pass

                data.append({
                    "timestamp": timestamp,
                    "echem_data": voltage,
                    "current": current
                })

        if len(offsets) > 1:
            logger.warning(f"{name}: echem times change UTC offset ({', '.join(sorted(offsets))}); "
                           "matched by wall clock, so the hour around a clock change may pair "
                           "with the wrong pass")
        early = sum(1 for row in data if row["timestamp"].year < 1900)
        if early:
            logger.warning(f"{name}: {early} echem row{'s' if early != 1 else ''} dated before 1900 "
                           f"(placeholders?) {'were' if early != 1 else 'was'} skipped")
            data = [row for row in data if row["timestamp"].year >= 1900]
        return data


# ============================================================================
# Scan Processor
# ============================================================================

class ScanProcessor:
    """Processor building the ordered scan list and correlating it with echem data."""

    def __init__(self, time_method: TimeMethod = TimeMethod.ABSOLUTE,
                 data_source: DataSourceType = DataSourceType.INHOUSE,
                 echem_time_tolerance: float = ECHEM_TIME_TOLERANCE):
        self.time_method = time_method
        self.data_source = data_source
        self.echem_time_tolerance = echem_time_tolerance
        self.xrd_reference_time: Optional[pd.Timestamp] = None
        self.echem_reference_time: Optional[pd.Timestamp] = None
        self.neutron_reference_time: Optional[pd.Timestamp] = None
        self.echem_parser = EchemParser()

    def process_scans(self, df: pd.DataFrame) -> Tuple[List[Scan], pd.DataFrame]:
        """Run the pipeline: parse echem, build scans, set exposure midpoints, correlate."""
        combined_echem_df = self._process_echem_data(df)

        neutron_metadata_df = None
        if self.data_source == DataSourceType.NEUTRON:
            neutron_metadata_df = self._process_neutron_metadata(df)

        scan_list = self._create_scan_list(df, neutron_metadata_df)

        # Compute midpoint-adjusted correlation timestamps
        self._adjust_for_exposure_time(scan_list)

        if self.time_method == TimeMethod.RELATIVE:
            self._set_reference_times(combined_echem_df, scan_list)

        # Correlate scans with echem by time (absolute or relative)
        if not combined_echem_df.empty:
            self._correlate_with_echem(scan_list, combined_echem_df)
            matched = sum(scan.echem is not None for scan in scan_list)
            if 0 < matched < len(scan_list):
                logger.info(f"Echem matched {matched} of {len(scan_list)} scans within "
                            f"{self.echem_time_tolerance:g} s")
            self._annotate_echem_window(scan_list, combined_echem_df)

        return scan_list, combined_echem_df

    def _process_neutron_metadata(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Parse and concatenate every logbook file; None when none parse."""
        neutron_meta_paths = df[df["neutron_meta"].notna()]["neutron_meta"].tolist()

        logger.info(f"Found {len(neutron_meta_paths)} neutron metadata "
                    f"file{'s' if len(neutron_meta_paths) != 1 else ''}")

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
            # A run listed in two logbooks (e.g. cumulative exports) is one scan
            duplicates = int(combined_df["scan_id"].duplicated().sum())
            if duplicates:
                logger.warning(f"{duplicates} logbook entries repeat a run already listed "
                               "and were skipped" if duplicates != 1 else
                               "1 logbook entry repeats a run already listed and was skipped")
                combined_df = combined_df.drop_duplicates(subset="scan_id").reset_index(drop=True)
            logger.info(f"Combined neutron metadata: {len(combined_df)} total "
                        f"{'entries' if len(combined_df) != 1 else 'entry'}")
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
            e_df = self.echem_parser.parse(
                e_path, os.path.basename(orig) if pd.notna(orig) else None)
            if e_df is not None:
                e_df["source_file"] = orig if pd.notna(orig) else e_path
                echem_dfs.append(e_df)

        if len(echem_dfs) > 1:
            names = sorted(os.path.basename(str(e_df["source_file"].iloc[0])) for e_df in echem_dfs)
            logger.info(f"Echem data from {len(names)} files: {name_list(names)}")
        if echem_dfs:
            combined_df = (pd.concat(echem_dfs, ignore_index=True)
                           .sort_values("timestamp").reset_index(drop=True))
            logger.info(f"Echem data: {len(combined_df)} row{'s' if len(combined_df) != 1 else ''}")
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
            # list implies none either: there is nothing to reference.
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
        """Match neutron .dat file groups to logbook entries, skipping entries without data."""
        scan_list: List[Scan] = []

        neutron_files = [
            row["path"] for _, row in df.iterrows()
            if pd.notna(row["path"]) and row["path"].lower().endswith('.dat')
        ]
        logger.info(f"Found {len(neutron_files)} .dat file{'s' if len(neutron_files) != 1 else ''}")

        grouper = NeutronFileGrouper()
        neutron_groups = grouper.group_neutron_files(neutron_files)

        unmatched: List[str] = []  # logbook runs with no data files

        for _, meta_row in neutron_metadata_df.iterrows():
            scan_id = str(meta_row["scan_id"])

            neutron_data_files = neutron_groups.get(scan_id, {})

            if not neutron_data_files:
                unmatched.append(scan_id)
                continue

            start_time = meta_row["start_time"]
            end_time = meta_row["end_time"]
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
                # Exact midpoint: odd-second runs land on .5 s. The display
                # layer rounds labels to the second; correlation must not.
                timestamp=str(midpoint),
                original_timestamp=str(midpoint),
                timestamp_for_correlation=midpoint,
                logbook=logbook or None
            )
            scan_list.append(scan)

        logger.info(f"Created {len(scan_list)} neutron "
                    f"scan{'s' if len(scan_list) != 1 else ''} with data")
        if unmatched:
            logger.warning(
                f"{len(unmatched)} logbook entries had no matching data files and were skipped: "
                f"{name_list(sorted(unmatched, key=int))}" if len(unmatched) != 1 else
                f"1 logbook entry had no matching data files and was skipped: {unmatched[0]}")
        # Every bank file counts as used, so a run the logbook does not list is named here
        logged = {str(scan_id) for scan_id in neutron_metadata_df["scan_id"]}
        unlogged = sorted((run for run in neutron_groups if run not in logged), key=int)
        if unlogged:
            logger.warning(
                f"{len(unlogged)} runs have bank files but no logbook entry and were skipped: "
                f"{name_list(unlogged)}" if len(unlogged) != 1 else
                f"1 run has bank files but no logbook entry and was skipped: {unlogged[0]}")

        return scan_list

    def _create_scan_list(self, df: pd.DataFrame,
                          neutron_metadata_df: Optional[pd.DataFrame] = None) -> List[Scan]:
        """Build the scan list from the classified DataFrame, sorted by timestamp."""
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

        # A scan without a start time cannot be placed, correlated or written validly
        if self.data_source != DataSourceType.NEUTRON:
            untimed = [s for s in scan_list if not s.timestamp]
            if untimed:
                names = [os.path.basename(str(s.source_nxs or s.oned or s.twod)) for s in untimed]
                logger.warning(f"{len(untimed)} scans have no start time and were skipped: "
                               f"{name_list(names)}" if len(untimed) != 1 else
                               f"1 scan has no start time and was skipped: {names[0]}")
                scan_list = [s for s in scan_list if s.timestamp]

        scan_list.sort(key=lambda s: pd.to_datetime(s.timestamp))

        for i, scan in enumerate(scan_list, start=1):
            scan.scan_num = i

        return scan_list

    def _adjust_for_exposure_time(self, scan_list: List[Scan]) -> None:
        """Set each scan's exposure_time and its exposure-midpoint timestamp_for_correlation."""
        for scan in scan_list:
            exposure_time = self._determine_exposure_time(scan)
            scan.exposure_time = exposure_time

            if self.data_source == DataSourceType.NEUTRON:
                # The exact midpoint was set from the logbook start/end; fall
                # back to the display timestamp only when it is missing
                if scan.timestamp_for_correlation is None and scan.timestamp:
                    scan.timestamp_for_correlation = pd.to_datetime(scan.timestamp)
            else:
                if exposure_time and scan.timestamp:
                    original_ts = pd.to_datetime(scan.timestamp)
                    adjusted_ts = original_ts + pd.Timedelta(seconds=exposure_time / 2)
                    scan.timestamp_for_correlation = adjusted_ts
                else:
                    scan.timestamp_for_correlation = pd.to_datetime(scan.timestamp) if scan.timestamp else None

    def _determine_exposure_time(self, scan: Scan) -> Optional[float]:
        """Return the exposure time in seconds (neutron: the logbook run length), or None.
        With both 1D and 2D files it is their mean, None unless both exposures are known."""
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
        """Return the echem timestamps as datetime64; None, logged, when no one unit holds them all."""
        try:
            return pd.to_datetime(echem_df["timestamp"])
        except (ValueError, TypeError):
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
        scan_times = []

        for scan in scan_list:
            scan_time = scan.timestamp_for_correlation
            if not scan_time:
                scan.echem_timestamp = None
                continue
            scan_times.append(scan_time)

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

        # Not one match is usually a day/month swap in the echem dates or the wrong time method
        if scan_times and all(scan.echem is None for scan in scan_list):
            n = len(scan_times)
            logger.warning(
                f"No scan matched the echem data: {n} scan{'s' if n != 1 else ''} from "
                f"{min(scan_times).isoformat(' ', 'minutes')} to "
                f"{max(scan_times).isoformat(' ', 'minutes')}, echem from "
                f"{echem_start.isoformat(' ', 'minutes')} to "
                f"{echem_end.isoformat(' ', 'minutes')}, "
                f"tolerance {self.echem_time_tolerance:g} s; check the echem date format and "
                "the time method")

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
        """Fill per-scan voltage/current extrema over the acquisition window, bounds inclusive.
        Clock-aligned like the nearest-point match (the Figure 4 window summary)."""
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
            if start is None or end is None:
                # No window without both bounds (XRD with unknown exposure)
                continue
            start, end = start + offset, end + offset

            # Every point with start <= t <= end, both bounds inclusive
            i0 = int(np.searchsorted(ts_values, start.to_datetime64(), side="left"))
            i1 = int(np.searchsorted(ts_values, end.to_datetime64(), side="right"))
            if i1 <= i0:
                continue

            window_v = voltage[i0:i1]
            if not np.all(np.isnan(window_v)):
                scan.voltage_min = float(np.nanmin(window_v))
                scan.voltage_max = float(np.nanmax(window_v))
            if current is not None:
                window_i = current[i0:i1]
                if not np.all(np.isnan(window_i)):
                    scan.current_min = float(np.nanmin(window_i))
                    scan.current_max = float(np.nanmax(window_i))


def format_relative_time(seconds: float) -> str:
    """Format elapsed seconds as an HH:MM:SS string."""
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
