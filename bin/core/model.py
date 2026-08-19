"""
Shared data model for the OperaXN core pipeline.

`Scan` describes source files on disk (paths, timestamps, correlation
results); `ScanData` carries the loaded data arrays. `source_nxs` on a Scan
feeds the NeXus writer's global metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DataSourceType(Enum):
    """Instrument family an experiment was collected on."""
    INHOUSE = "inhouse"
    SYNCHROTRON = "synchrotron"
    NEUTRON = "neutron"


class FileType(Enum):
    """Supported input file extensions."""
    DAT = ".dat"
    EDF = ".edf"
    TXT = ".txt"
    HDF = ".hdf"
    NXS = ".nxs"
    XY = ".xy"
    ZIP = ".zip"
    XLSX = ".xlsx"
    CSV = ".csv"


SUPPORTED_EXTENSIONS = {ft.value for ft in FileType}


class DataType(Enum):
    """What a classified file contains (also the FileRecord column it fills)."""
    ONED = "oned"
    TWOD = "twod"
    ECHEM = "echem"
    NEUTRON_META = "neutron_meta"


class TimeMethod(Enum):
    """Display-time handling. The canonical .nxs always stores absolute
    timestamps; RELATIVE is a view transform applied by the GUI."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass
class FileRecord:
    """Single collected file with its classification and extracted metadata."""
    path: str
    original_path: str
    oned: Optional[str] = None
    twod: Optional[str] = None
    echem: Optional[str] = None
    neutron_meta: Optional[str] = None
    neutron_files: Optional[Dict[str, Dict[str, str]]] = None
    timestamp: Optional[str] = None
    exposure_time: Optional[float] = None
    source_nxs: Optional[str] = None


@dataclass
class Scan:
    """A complete scan entry with all correlated data across sources."""
    scan_num: int
    oned: Optional[str] = None
    twod: Optional[str] = None
    echem: Optional[float] = None
    current: Optional[float] = None
    echem_timestamp: Optional[str] = None
    neutron_meta: Optional[str] = None
    neutron_files: Optional[Dict[str, Dict[str, str]]] = None
    timestamp: Optional[str] = None
    original_timestamp: Optional[str] = None
    exposure_time: Optional[float] = None
    oned_exposure: Optional[float] = None
    twod_exposure: Optional[float] = None
    neutron_start: Optional[str] = None
    neutron_end: Optional[str] = None
    source_nxs: Optional[str] = None
    timestamp_for_correlation: Optional[pd.Timestamp] = None
    # Harvested logbook extras (run_title, users, proposal, full_line)
    logbook: Optional[Dict[str, Any]] = None
    # Echem summary over the acquisition window [start, end]
    voltage_min: Optional[float] = None
    voltage_max: Optional[float] = None
    current_min: Optional[float] = None
    current_max: Optional[float] = None
    # 0-based positional window into the operando echem arrays (inclusive end)
    echem_index_start: Optional[int] = None
    echem_index_end: Optional[int] = None
    # Echem stream segment spanning the window (long acquisitions):
    # {"start": iso_str, "time_s": ndarray, "voltage": ndarray,
    #  "current": Optional[ndarray]}
    echem_segment: Optional[Dict[str, Any]] = None
    # Capacity (mAh); reserved — written when set, the core does not
    # compute it yet
    capacity: Optional[float] = None


@dataclass
class ScanData:
    """A single scan as read back from a .nxs file.

    Unlike `Scan` (which references raw files on disk), this holds the data
    itself: 1D arrays, an embedded 2D image (or an external source reference),
    and neutron bank arrays.
    """
    scan_num: int
    timestamp: Optional[str] = None
    midpoint_timestamp: Optional[str] = None
    exposure_time: Optional[float] = None
    echem: Optional[float] = None            # voltage (V)
    current: Optional[float] = None          # current (mA)
    echem_timestamp: Optional[str] = None
    # Echem summary over the acquisition window, when present in the file
    voltage_min: Optional[float] = None
    voltage_max: Optional[float] = None
    current_min: Optional[float] = None
    current_max: Optional[float] = None
    echem_index_start: Optional[int] = None
    echem_index_end: Optional[int] = None
    capacity: Optional[float] = None         # capacity (mAh), when stored
    # 1D XRD: {"x": ndarray, "y": ndarray, "e": ndarray?, "source": str} or None
    oned: Optional[Dict[str, Any]] = None
    # 2D XRD: embedded image array, or None
    twod: Optional[Any] = None
    twod_source: Optional[str] = None        # original filename reference
    twod_embedded: bool = False
    # Neutron banks: {bank_num: {"tof": {"x","y","e"?}, "d": {"x","y","e"?}}}
    neutron: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    neutron_start: Optional[str] = None
    neutron_end: Optional[str] = None
    # Beam monitor record (mode/preset/integral + raw counters), if present
    monitor: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentModel:
    """Everything the GUI needs, loaded from a canonical .nxs file."""
    scans: List[ScanData] = field(default_factory=list)
    echem_df: Optional[pd.DataFrame] = None          # operando echem
    standard_echem: List[Dict[str, Any]] = field(default_factory=list)
    global_metadata: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    source_path: Optional[str] = None                # the .nxs this came from
