"""
File classification, grouping, and metadata extraction for the core pipeline.

Classifiers and the NeXus extractor are deliberately uncached (cheap header reads), so
results track a file's current contents. One neutron scan-ID rule (SCAN_ID_MIN/MAX_DIGITS)
serves the logbook parser, the file grouper and display (`NeutronFileGrouper.extract_scan_id`).
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd

from .config import MAX_EXPOSURE_TIME, SCAN_ID_MAX_DIGITS, SCAN_ID_MIN_DIGITS
from .model import DataSourceType, DataType, FileType, SUPPORTED_EXTENSIONS
from .readers import FABIO_AVAILABLE, fabio

logger = logging.getLogger(__name__)

WEEKDAYS = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
# Logbook user field: comma-separated surnames ("Smith,Jones")
USER_TOKEN_RE = re.compile(r"[A-Za-z ,.\-']+")
# A lone surname (no comma marks it as the user field): one capitalised
# word, hyphens and apostrophes allowed ("Perez", "O'Neil", "Smith-Jones")
SURNAME_RE = re.compile(r"[A-Z][A-Za-z.\-']*")
LOGBOOK_TIME_FORMAT = '%a %b %d %H:%M:%S %Y'


def _valid_scan_id(scan_id: str) -> bool:
    """True when scan_id is SCAN_ID_MIN_DIGITS to SCAN_ID_MAX_DIGITS digits (the scan-ID rule)."""
    return scan_id.isdigit() and SCAN_ID_MIN_DIGITS <= len(scan_id) <= SCAN_ID_MAX_DIGITS


# ============================================================================
# Per-file metadata harvesting (schema v4 NXmonopd/NXtofnpd fields)
# ============================================================================

def parse_mantid_header(path: str) -> Dict[str, Any]:
    """Harvest instrument, axis units and spectrum number from a Mantid .dat header.
    Reads at most 10 lines, stopping at the first data line."""
    info: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(10):
                line = f.readline()
                if not line.startswith('#') and 'XYDATA' not in line:
                    if line.strip() and not line.lstrip().startswith('#'):
                        break
                    continue

                m = re.search(r'Instrument\s+(\S+)', line)
                if m:
                    info['instrument'] = m.group(1)
                m = re.search(r'X-axis unit is:\s*([^,]+)', line)
                if m:
                    info['x_unit'] = m.group(1).strip()
                m = re.search(r'Y-axis unit is:\s*(.+)$', line)
                if m:
                    info['y_unit'] = m.group(1).strip()
                m = re.search(r'#\s*Spectrum\s+(\d+)', line)
                if m:
                    info['spectrum'] = int(m.group(1))
    except Exception as e:
        logger.debug(f"Could not parse Mantid header of {path}: {e}")
    return info


# EDF header counter that feeds the per-scan monitor integral: the beam
# monitor reading of one exposure (zero on instruments without a monitor).
# The Pilatus counters (pilct*, pilai*) are the detector's own totals and the
# Intensity/TransmittedFlux entries SPEC-derived values; none is recorded.
EDF_MONITOR_KEYS = ['Monitor']
# EDF header fields that describe the instrument, harvested from the first
# image for the entry-level NXinstrument
EDF_INSTRUMENT_KEYS = {
    # header key -> nexus-ish name
    'WaveLength': 'wavelength_m',
    'Wavelength': 'wavelength_m',
    'SampleDistance': 'distance_m',
    'DetectorModel': 'detector_model',
    'PSize_1': 'x_pixel_size_m',
    'PSize_2': 'y_pixel_size_m',
    'Center_1': 'beam_center_x',
    'Center_2': 'beam_center_y',
    'ExposureTime': 'count_time',
    'Comment': 'comment',
}


def extract_edf_scan_fields(edf_path: str) -> Dict[str, Any]:
    """Harvest an EDF header as {'monitor': {...}, 'instrument': {...}}; {} when unreadable."""
    if not FABIO_AVAILABLE:
        return {}
    try:
        header = fabio.open(edf_path).header
    except Exception as e:
        logger.debug(f"Could not read EDF header of {edf_path}: {e}")
        return {}

    monitor: Dict[str, Any] = {}
    for key in EDF_MONITOR_KEYS:
        if key in header:
            try:
                monitor[key] = float(header[key])
            except (ValueError, TypeError):
                pass

    instrument: Dict[str, Any] = {}
    for key, name in EDF_INSTRUMENT_KEYS.items():
        if key in header and name not in instrument:
            raw = header[key]
            try:
                instrument[name] = float(raw)
            except (ValueError, TypeError):
                if str(raw).strip():
                    instrument[name] = str(raw).strip()

    return {'monitor': monitor, 'instrument': instrument}


# ============================================================================
# Classifiers
# ============================================================================

class FileClassifierBase(ABC):
    """Abstract classifier mapping a file to (data_type, timestamp, exposure_time)."""

    @abstractmethod
    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Return (data_type, timestamp, exposure_time), Nones when unrecognised."""


class DATClassifier(FileClassifierBase):
    """Classifier for 1D .dat files (timestamp and exposure from the header)."""

    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Classify as 1D when the first 30 lines hold a '# Date' line; exposure is optional."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(30)]

            timestamp = None
            exposure_time = None

            for line in lines:
                if not line:
                    break

                line_lower = line.lstrip().lower()

                if line_lower.startswith("# date"):
                    timestamp = line.split()[-1].replace("T", " ")

                elif "exposuretime" in line_lower.replace(" ", ""):
                    try:
                        exposure_time = float(line.split()[-1])
                    except (ValueError, IndexError):
                        pass

            if timestamp:
                return DataType.ONED.value, timestamp, exposure_time

        except Exception as e:
            logger.error(f"Error classifying DAT file {path}: {e}")

        return None, None, None


class EDFClassifier(FileClassifierBase):
    """Classifier for 2D .edf images (timestamp and exposure from the header)."""

    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Read Date and exposure fields from the EDF header via fabio."""
        if not FABIO_AVAILABLE:
            return None, None, None

        try:
            image = fabio.open(path)

            raw_date = image.header.get("Date")
            timestamp = raw_date.replace("T", " ") if raw_date else None

            exposure_time = None
            for key in ["ExposureTime", "Exposure_Time", "ExpTime", "Exposure"]:
                if key in image.header:
                    try:
                        exposure_time = float(image.header[key])
                        break
                    except (ValueError, TypeError):
                        pass

            if timestamp:
                return DataType.TWOD.value, timestamp, exposure_time

        except Exception as e:
            logger.error(f"Error classifying EDF file {path}: {e}")

        return None, None, None


class TXTClassifier(FileClassifierBase):
    """Classifier telling electrochemistry .txt tables from neutron logbooks."""

    ECHEM_KEYWORDS = ["time", "absolute", "ecell", "voltage", "current", "i/", "ewe", "v/"]

    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Distinguish echem tables from neutron logbooks by header content."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)

            if not lines:
                return None, None, None

            first_line_lower = lines[0].lower()
            if any(keyword in first_line_lower for keyword in self.ECHEM_KEYWORDS):
                return DataType.ECHEM.value, None, None

            # Neutron logbook: numeric scan ID + weekday timestamps
            for line in lines[:5]:
                parts = line.strip().split('\t')

                if (len(parts) >= 8 and _valid_scan_id(parts[0])
                        and any(weekday in line for weekday in WEEKDAYS)):
                    logger.debug(f"Identified {path} as neutron metadata")
                    return DataType.NEUTRON_META.value, None, None

            return None, None, None

        except Exception as e:
            logger.error(f"Error classifying TXT file {path}: {e}")
            return None, None, None


class FileClassificationManager:
    """Dispatcher running the per-extension classifiers over a DataFrame of files."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE):
        self.data_source = data_source
        self.classifiers: Dict[FileType, FileClassifierBase] = {
            FileType.DAT: DATClassifier(),
            FileType.EDF: EDFClassifier(),
            FileType.TXT: TXTClassifier()
        }

    def classify_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify unclassified rows into the type/timestamp/exposure columns of a copy of df.
        Neutron mode classifies .txt files only."""
        df = df.copy()

        # Ensure all classification columns exist
        for col in ["oned", "twod", "echem", "neutron_meta", "source_nxs"]:
            if col not in df.columns:
                df[col] = pd.array([None] * len(df), dtype="object")
            else:
                df[col] = df[col].astype("object")
        for col in ["exposure_time", "timestamp"]:
            if col not in df.columns:
                df[col] = pd.array([None] * len(df), dtype="object")

        for idx, row in df.iterrows():
            # Skip already-classified rows
            if pd.notna(row["oned"]) or pd.notna(row["twod"]) or pd.notna(row["neutron_meta"]):
                continue

            file_path = row["path"]
            ext = os.path.splitext(file_path)[1].lower()

            # Neutron mode classifies .txt only (echem or logbook)
            if self.data_source == DataSourceType.NEUTRON and ext != '.txt':
                continue

            file_type = FileType(ext) if ext in SUPPORTED_EXTENSIONS else None
            if not (file_type and file_type in self.classifiers):
                continue

            classifier = self.classifiers[file_type]
            data_type, timestamp, exposure_time = classifier.classify(file_path)

            if data_type:
                df.at[idx, data_type] = file_path
                if timestamp:
                    df.at[idx, "timestamp"] = timestamp
                if exposure_time is not None:
                    df.at[idx, "exposure_time"] = exposure_time

        return df


# ============================================================================
# Neutron logbook + file grouping
# ============================================================================

class NeutronMetadataParser:
    """Parser for tab-delimited neutron logbooks (scan ID, start/end time, logbook extras)."""

    @staticmethod
    def parse(path: str) -> Optional[pd.DataFrame]:
        """Parse a logbook into a scans DataFrame; None when no valid entries."""
        try:
            logger.debug(f"Parsing neutron metadata file: {path}")

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                logger.debug("No lines found in file")
                return None

            data = []

            for line_num, line in enumerate(lines):
                parts = line.strip().split('\t')

                if len(parts) >= 8:
                    try:
                        scan_id = parts[0].strip()

                        if not _valid_scan_id(scan_id):
                            continue

                        # Start and end times: the first two fields naming a weekday
                        start_time = None
                        end_time = None

                        for part in parts:
                            if any(day in part for day in WEEKDAYS):
                                if not start_time:
                                    start_time = part.strip()
                                else:
                                    end_time = part.strip()
                                    break

                        if start_time and end_time:
                            start_dt = pd.to_datetime(start_time, format=LOGBOOK_TIME_FORMAT)
                            end_dt = pd.to_datetime(end_time, format=LOGBOOK_TIME_FORMAT)

                            data.append({
                                "scan_id": scan_id,
                                "start_time": start_dt,
                                "end_time": end_dt,
                                "full_line": line.strip(),
                                **NeutronMetadataParser._harvest_extras(parts, scan_id),
                            })

                    except (ValueError, IndexError) as e:
                        if line_num < 3:
                            logger.debug(f"Error parsing line {line_num}: {e}")
                        continue

            if data:
                logger.info(f"Successfully parsed {len(data)} logbook entries")
                return pd.DataFrame(data)

            logger.debug("No valid entries found")
            return None

        except Exception as e:
            logger.error(f"Error parsing neutron metadata: {e}")
            return None

    @staticmethod
    def _harvest_extras(parts: List[str], scan_id: str) -> Dict[str, Any]:
        """Harvest the optional run title, users and proposal number of a logbook line.
        Best effort: the column layout varies between logbooks."""
        extras: Dict[str, Any] = {}

        # Run title: second column, when it is free text
        if len(parts) > 1:
            title = parts[1].strip().strip('"')
            if title and not title.isdigit() and not any(d in title for d in WEEKDAYS):
                extras["run_title"] = title

        proposal_idx = None
        for idx, part in enumerate(parts[2:], start=2):
            token = part.strip().strip('"')
            if not token or any(d in token for d in WEEKDAYS):
                continue
            # Users: comma-separated surnames (e.g. "Smith,Jones")
            if ("users" not in extras and "," in token
                    and USER_TOKEN_RE.fullmatch(token)):
                extras["users"] = token
            # Proposal / experiment number: long numeric field that is not the run
            elif ("proposal" not in extras and token.isdigit()
                  and len(token) >= 6 and token != scan_id):
                extras["proposal"] = token
                proposal_idx = idx

        # A single surname carries no comma; in every logbook layout seen so
        # far the user field is the one just before the proposal number.
        # Only a lone capitalised word is accepted there, so free text in
        # that column ("no beam") is left unassigned rather than stored as
        # the user
        if "users" not in extras and proposal_idx is not None and proposal_idx >= 3:
            token = parts[proposal_idx - 1].strip().strip('"')
            if SURNAME_RE.fullmatch(token):
                extras["users"] = token

        return extras


class NeutronFileGrouper:
    """Grouper of neutron .dat files by scan ID, measurement number and tof/d type."""

    # Lookbehind stops an 8+ digit run number matching by its last digits
    _ID_MEAS_PATTERN = re.compile(
        r'(?<!\d)(\d{%d,%d})-(\d)' % (SCAN_ID_MIN_DIGITS, SCAN_ID_MAX_DIGITS))
    _POL_PATTERN = re.compile(r'POL(\d+)-b_(\d)')

    def group_neutron_files(self, file_list: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Group .dat paths as {scan_id: {measurement: {'tof'|'d': path}}}."""
        groups: Dict[str, Dict[str, Dict[str, str]]] = {}

        logger.debug(f"Processing {len(file_list)} neutron files")

        for filepath in file_list:
            basename = os.path.basename(filepath)

            if not basename.endswith('.dat'):
                continue

            scan_info = self._extract_neutron_file_info(basename)

            if scan_info:
                scan_id = scan_info['scan_id']
                measurement_num = scan_info['measurement']
                data_type = scan_info['type']

                groups.setdefault(scan_id, {}).setdefault(measurement_num, {})[data_type] = filepath

        logger.info(f"Grouped {len(groups)} scans")
        return groups

    @classmethod
    def extract_scan_id(cls, filename: str) -> Optional[str]:
        """Return the scan ID for a neutron filename, or None.
        Display code must use this, never its own regex, so shown IDs match grouping/correlation."""
        info = cls._extract_neutron_file_info(filename)
        return info['scan_id'] if info else None

    @classmethod
    def _extract_neutron_file_info(cls, filename: str) -> Optional[Dict[str, str]]:
        """Parse scan ID, measurement number (1-5) and tof/d type from a neutron filename."""
        name_no_ext = filename[:-4] if filename.endswith('.dat') else filename
        is_dspacing = ('-d-' in name_no_ext or '_d_' in name_no_ext
                       or name_no_ext.endswith('_d') or name_no_ext.endswith('-d'))
        kind = 'd' if is_dspacing else 'tof'

        try:
            # Pattern: POL123456-b_1-d.dat
            match = cls._POL_PATTERN.search(name_no_ext.replace('-d', '').replace('_d', ''))
            if match and _valid_scan_id(match.group(1)) and 1 <= int(match.group(2)) <= 5:
                return {'scan_id': match.group(1), 'measurement': match.group(2), 'type': kind}

            # Pattern: 12345-1-d.dat
            match = cls._ID_MEAS_PATTERN.search(name_no_ext)
            if match and 1 <= int(match.group(2)) <= 5:
                return {'scan_id': match.group(1), 'measurement': match.group(2), 'type': kind}

            # Fallback: split on hyphen
            parts = name_no_ext.split('-')
            if (len(parts) >= 2 and _valid_scan_id(parts[0]) and parts[1][:1].isdigit()
                    and 1 <= int(parts[1][0]) <= 5):
                return {'scan_id': parts[0], 'measurement': parts[1][0], 'type': kind}

            return None

        except (ValueError, IndexError):
            return None


# ============================================================================
# Synchrotron grouping + NeXus metadata
# ============================================================================

class SynchrotronFileGrouper:
    """Grouper of related synchrotron .nxs, .hdf and .xy files by scan ID."""

    def group_files(self, file_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Group paths as {scan_id: {'nxs'|'hdf'|'xy': path}}; a .nxs anchors each group."""
        groups: Dict[str, Dict[str, str]] = {}
        nxs_ids: Set[str] = set()

        # First pass: collect the scan IDs of all .nxs files
        for extracted_path in file_dict:
            basename = os.path.basename(extracted_path)
            ext = os.path.splitext(basename)[1].lower()

            if ext == '.nxs':
                scan_id = self._extract_scan_id(os.path.splitext(basename)[0])
                if scan_id:
                    nxs_ids.add(scan_id)

        # Second pass: assign each file to its group
        for extracted_path in file_dict:
            basename = os.path.basename(extracted_path)
            ext = os.path.splitext(basename)[1].lower()

            if ext in ['.hdf', '.nxs', '.xy']:
                group_id = self._determine_group_id(basename, ext, nxs_ids)

                if group_id:
                    groups.setdefault(group_id, {})[ext.lstrip('.')] = extracted_path

        return groups

    @classmethod
    def extract_scan_id(cls, filename: str) -> Optional[str]:
        """Return the display scan ID by the grouping rule, .xy integration suffix included."""
        base = os.path.splitext(filename)[0]
        if '_integration_' in base:
            base = base.split('_integration_')[0]
        return cls._extract_scan_id(base)

    @staticmethod
    def _extract_scan_id(stem: str) -> Optional[str]:
        """Return the last digit run of an extension-free stem ('_integration' dropped), or None.
        Callers strip the extension once, so a dot inside the stem (e.g. '0.5C') is kept."""
        match = re.search(r'(\d+)(?!.*\d)', stem.replace('_integration', ''))
        return match.group(1) if match else None

    def _determine_group_id(self, basename: str, ext: str, nxs_ids: Set[str]) -> Optional[str]:
        """Return the scan ID of the .nxs group a file belongs to, or None."""
        base = os.path.splitext(basename)[0]
        if ext == '.nxs':
            return self._extract_scan_id(base)

        if ext == '.xy' and '_integration_' in base:
            file_id = self._extract_scan_id(base.split('_integration_')[0])
        else:
            file_id = self._extract_scan_id(base)

        return file_id if file_id in nxs_ids else None


class NexusMetadataExtractor:
    """Extractor of timestamps and exposure times from synchrotron .nxs files."""

    TIMESTAMP_PATHS = [
        '/entry1/start_time',
        '/entry/start_time',
        '/entry1/instrument/detector/start_time',
        '/entry1/end_time',
    ]

    TIME_PATH_PAIRS = [
        ('/entry1/start_time', '/entry1/end_time'),
        ('/entry/start_time', '/entry/end_time'),
    ]

    EXPOSURE_PATHS = [
        '/entry1/instrument/detector/exposure_time',
        '/entry/instrument/detector/exposure_time',
        '/entry1/instrument/detector/count_time',
        '/entry1/instrument/detector/preset',
    ]

    def extract(self, nxs_path: str) -> Optional[Dict[str, Any]]:
        """Return the timestamp/exposure metadata of a scan .nxs; None if absent or unreadable."""
        try:
            with h5py.File(nxs_path, 'r') as f:
                metadata: Dict[str, Any] = {}

                timestamp = self._extract_timestamp(f)
                if timestamp:
                    metadata['timestamp'] = timestamp

                exposure_time = self._extract_exposure_time(f)
                if exposure_time:
                    metadata['exposure_time'] = exposure_time

                return metadata or None

        except Exception as e:
            logger.error(f"Failed to extract NXS metadata from {nxs_path}: {e}")
            return None

    def _extract_timestamp(self, h5file: h5py.File) -> Optional[str]:
        """Return the normalised timestamp at the first TIMESTAMP_PATHS entry present."""
        for ts_path in self.TIMESTAMP_PATHS:
            if ts_path in h5file:
                timestamp_str = self._decode_value(h5file[ts_path][()])
                return self._parse_nexus_timestamp(timestamp_str)
        return None

    def _extract_exposure_time(self, h5file: h5py.File) -> Optional[float]:
        """Return exposure seconds from an exposure dataset, else from the start/end times."""
        for exp_path in self.EXPOSURE_PATHS:
            if exp_path in h5file:
                try:
                    exp_time = float(h5file[exp_path][()])
                    if exp_time > 0:
                        return exp_time
                except (ValueError, TypeError):
                    pass

        for start_path, end_path in self.TIME_PATH_PAIRS:
            if start_path in h5file and end_path in h5file:
                start_str = self._decode_value(h5file[start_path][()])
                end_str = self._decode_value(h5file[end_path][()])

                start_time = self._parse_nexus_timestamp(start_str)
                end_time = self._parse_nexus_timestamp(end_str)

                if start_time and end_time:
                    start_dt = pd.to_datetime(start_time)
                    end_dt = pd.to_datetime(end_time)
                    exposure_seconds = (end_dt - start_dt).total_seconds()

                    if 0 < exposure_seconds < MAX_EXPOSURE_TIME:
                        return exposure_seconds

        return None

    @staticmethod
    def _decode_value(value: Any) -> str:
        """Return an h5py scalar as str, decoding bytes."""
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    @staticmethod
    def _parse_nexus_timestamp(timestamp_str: str) -> str:
        """Normalise an ISO/NeXus timestamp to 'YYYY-MM-DD HH:MM:SS'; others pass through."""
        if 'T' in timestamp_str:
            base_time = timestamp_str.split('+')[0].split('Z')[0].split('.')[0]
            return base_time.replace('T', ' ')
        return timestamp_str


# ============================================================================
# Instrument profile derivation (Mantid-processed files)
# ============================================================================
# Derives the geometry block of a config.INSTRUMENT_PROFILES entry from a
# Mantid SaveNexusProcessed file: one focussed bank per mantid_workspace_N
# and L1 from the embedded instrument definition. `operaxn --profile FILE`
# prints the result ready to paste into config.py.

_WORKSPACE_RE = re.compile(r"mantid_workspace_(\d+)$")
# Below this ratio of mean transverse unit vector to mean transverse extent
# a bank surrounds the beam (a ring) and has no meaningful azimuth
_AZIMUTH_COHERENCE_MIN = 0.5


def _mantid_text(node: Any) -> Optional[str]:
    """Decode a Mantid string dataset (bytes, shape (1,) or scalar); None if missing or empty."""
    if node is None:
        return None
    try:
        value = node[()]
    except Exception:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


def _idf_facts(idf: str) -> Dict[str, Any]:
    """Return the identity and source-to-sample distance (m) of a Mantid IDF (XML text)."""
    facts: Dict[str, Any] = {"idf_name": None, "idf_valid_from": None,
                             "source_type_name": None,
                             "pre_sample_flightpath_m": None}
    head = re.search(r"<instrument\b[^>]*>", idf)
    if head:
        for key, attr in (("idf_name", "name"), ("idf_valid_from", "valid-from")):
            m = re.search(r'\s%s="([^"]*)"' % attr, head.group(0))
            if m:
                facts[key] = m.group(1).strip() or None

    # z (m, along the beam) of the component whose type declares is="<role>":
    # 0.0 for a placed component without a z, None if absent
    def component_z(role: str) -> Optional[float]:
        for tag in re.findall(r"<type\b[^>]*>", idf):
            if not re.search(r'\bis="%s"' % role, tag, re.I):
                continue
            name = re.search(r'\bname="([^"]+)"', tag)
            if not name:
                continue
            if role == "Source":
                facts["source_type_name"] = name.group(1)
            comp = re.search(r'<component\s+type="%s"[^>]*>\s*<location\b([^>]*)>'
                             % re.escape(name.group(1)), idf)
            if not comp:
                return None
            z = re.search(r'\bz="\s*(-?[0-9.eE+-]+)\s*"', comp.group(1))
            return float(z.group(1)) if z else 0.0
        return None

    z_source = component_z("Source")
    if z_source is not None:
        z_sample = component_z("SamplePos") or 0.0
        facts["pre_sample_flightpath_m"] = round(abs(z_sample - z_source), 3)
    return facts


def _bank_from_positions(positions: Any) -> Dict[str, Any]:
    """Return nominal bank geometry from Mantid detector_positions rows (distance, 2theta, phi).
    {} when the array has no usable rows."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] < 3:
        return {}
    r, two_theta, phi = pos[:, 0], pos[:, 1], pos[:, 2]
    # Azimuth only for a bank localised in azimuth: the mean transverse unit
    # vector is as long as the mean transverse extent for a single panel and
    # vanishes for a ring around the beam
    sin_t = np.sin(np.radians(two_theta))
    x, y = np.cos(np.radians(phi)) * sin_t, np.sin(np.radians(phi)) * sin_t
    extent = float(sin_t.mean())
    coherence = float(np.hypot(x.mean(), y.mean()) / extent) if extent > 0 else 0.0
    azimuth = (round(float(np.degrees(np.arctan2(y.mean(), x.mean()))), 2)
               if coherence >= _AZIMUTH_COHERENCE_MIN else None)
    return {
        "distance_m": round(float(r.mean()), 3),
        "polar_angle_deg": round(float(two_theta.mean()), 2),
        "azimuthal_angle_deg": azimuth,
        "info": {
            "elements": int(pos.shape[0]),
            "distance_range_m": (round(float(r.min()), 3), round(float(r.max()), 3)),
            "polar_angle_range_deg": (round(float(two_theta.min()), 2),
                                      round(float(two_theta.max()), 2)),
            "azimuthal_coherence": round(coherence, 3),
        },
    }


def _monitor_table(monitors: h5py.Group) -> Dict[int, Dict[str, Any]]:
    """Return a Mantid physical_monitors group as the profile monitors table, by detector number.
    Angles are None when missing or mis-sized; {} without matching distance and ID arrays."""
    distance = monitors.get("distance")
    numbers = monitors.get("detector_number")
    if not isinstance(distance, h5py.Dataset) or not isinstance(numbers, h5py.Dataset):
        return {}
    distance = np.asarray(distance[()], dtype=float).ravel()
    numbers = np.asarray(numbers[()]).ravel()
    if distance.size != numbers.size:
        return {}

    def column(name: str) -> Optional[np.ndarray]:
        ds = monitors.get(name)
        if not isinstance(ds, h5py.Dataset):
            return None
        values = np.asarray(ds[()], dtype=float).ravel()
        return values if values.size == distance.size else None

    polar, azimuth = column("polar_angle"), column("azimuthal_angle")
    table: Dict[int, Dict[str, Any]] = {}
    for i, number in enumerate(numbers):
        table[int(number)] = {
            "distance_m": round(float(distance[i]), 3),
            "polar_angle_deg": round(float(polar[i]), 2) if polar is not None else None,
            "azimuthal_angle_deg": (round(float(azimuth[i]), 2)
                                    if azimuth is not None else None),
        }
    return table


def profile_from_mantid_nexus(path: str) -> Dict[str, Any]:
    """Derive an instrument profile's geometry block from a Mantid SaveNexusProcessed file.
    Bank N is mantid_workspace_N and L1 comes from the embedded IDF; unknown values are None
    (monitor_mode always) and 'info' is never written. ValueError without mantid_workspace_N."""
    profile: Dict[str, Any] = {
        "instrument_name": None,
        "pre_sample_flightpath_m": None,
        "detector_banks": {},
        "monitor_mode": None,
        "monitors": {},
        "provenance": None,
        "info": {"banks": {}, "source_file": os.path.basename(path)},
    }
    idf_text = None
    mantid_version = None
    with h5py.File(path, "r") as f:
        workspaces = []
        for key in f:
            m = _WORKSPACE_RE.match(key)
            if m and isinstance(f[key], h5py.Group):
                workspaces.append((int(m.group(1)), key))
        if not workspaces:
            raise ValueError(f"{path}: no mantid_workspace_N entries "
                             "(not a Mantid processed file)")
        for number, key in sorted(workspaces):
            ws = f[key]
            inst = ws.get("instrument")
            if not isinstance(inst, h5py.Group):
                continue
            if profile["instrument_name"] is None:
                profile["instrument_name"] = _mantid_text(inst.get("name"))
            positions = inst.get("detector/detector_positions")
            if isinstance(positions, h5py.Dataset):
                bank = _bank_from_positions(positions[()])
                if bank:
                    profile["info"]["banks"][number] = bank.pop("info")
                    profile["detector_banks"][number] = bank
            if idf_text is None:
                idf_text = _mantid_text(inst.get("instrument_xml/data"))
            monitors = inst.get("physical_monitors")
            if not profile["monitors"] and isinstance(monitors, h5py.Group):
                profile["monitors"] = _monitor_table(monitors)
            if mantid_version is None:
                env = _mantid_text(ws.get("process/MantidEnvironment/data"))
                m = re.search(r"Framework Version:\s*(\S+)", env or "")
                if m:
                    mantid_version = m.group(1)

    facts = _idf_facts(idf_text) if idf_text else {}
    profile["pre_sample_flightpath_m"] = facts.get("pre_sample_flightpath_m")
    for key in ("idf_name", "idf_valid_from", "source_type_name"):
        profile["info"][key] = facts.get(key)
    profile["info"]["mantid_version"] = mantid_version

    parts: List[str] = []
    if facts.get("idf_name"):
        idf = f"Mantid IDF {facts['idf_name']}"
        if facts.get("idf_valid_from"):
            idf += f" (valid-from {facts['idf_valid_from']})"
        parts.append(idf)
    origin = f"read from {profile['info']['source_file']}"
    if mantid_version:
        origin += f" (Mantid {mantid_version})"
    parts.append(origin)
    profile["provenance"] = ", ".join(parts)
    return profile


def format_profile_block(profile: Dict[str, Any]) -> str:
    """Format a profile as a paste-ready config.INSTRUMENT_PROFILES entry, info as comments."""
    def geometry_row(key, spec):
        return (f'        {key}: {{"distance_m": {spec.get("distance_m")!r}, '
                f'"polar_angle_deg": {spec.get("polar_angle_deg")!r}, '
                f'"azimuthal_angle_deg": {spec.get("azimuthal_angle_deg")!r}}},')

    name = profile.get("instrument_name")
    info = profile.get("info") or {}
    banks = profile.get("detector_banks") or {}
    lines = [f'"{(name or "instrument").lower()}": {{']
    lines.append(f'    "instrument_name": {name!r},')
    lines.append(f'    # {profile.get("provenance") or "provenance unknown"}')
    lines.append(f'    "pre_sample_flightpath_m": {profile.get("pre_sample_flightpath_m")!r},')
    for bank in sorted(banks):
        b = (info.get("banks") or {}).get(bank, {})
        lo_r, hi_r = b.get("distance_range_m", (None, None))
        lo_t, hi_t = b.get("polar_angle_range_deg", (None, None))
        lines.append(f'    # bank {bank}: {b.get("elements")} elements, '
                     f'L2 {lo_r}-{hi_r} m, 2theta {lo_t}-{hi_t} deg, '
                     f'azimuthal coherence {b.get("azimuthal_coherence")}')
    lines.append('    "detector_banks": {')
    for bank in sorted(banks):
        lines.append(geometry_row(bank, banks[bank]))
    lines.append('    },')
    lines.append('    # Counting mode is not in the file: "timer" (runs end on the clock)')
    lines.append('    # or "monitor" (runs end on monitor counts), from the run records')
    lines.append(f'    "monitor_mode": {profile.get("monitor_mode")!r},')
    monitors = profile.get("monitors") or {}
    if monitors:
        lines.append('    # Every beam-monitor element of the IDF (physical_monitors):')
        lines.append('    # 2theta 180 is upstream of the sample, 0 downstream')
        lines.append('    "monitors": {')
        for number in sorted(monitors):
            lines.append(geometry_row(number, monitors[number]))
        lines.append('    },')
    if info.get("source_type_name"):
        lines.append(f'    # IDF source type: {info["source_type_name"]}')
    lines.append('},')
    return "\n".join(lines)
