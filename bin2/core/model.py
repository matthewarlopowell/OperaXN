"""
Shared data model for the OperaXN/NexusGen core pipeline.

Unifies the previously duplicated dataclasses: `Scan` carries both
`source_nxs` (needed by the NeXus writer for global metadata) and `to_dict()`
(consumed by the visualiser GUI).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DataSourceType(Enum):
    INHOUSE = "inhouse"
    SYNCHROTRON = "synchrotron"
    NEUTRON = "neutron"


class FileType(Enum):
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan to plain dictionary (consumed by the visualiser GUI)."""
        return {
            "scan_num": self.scan_num,
            "oned": self.oned,
            "twod": self.twod,
            "echem": self.echem,
            "current": self.current,
            "echem_timestamp": self.echem_timestamp,
            "neutron_meta": self.neutron_meta,
            "neutron_files": self.neutron_files,
            "timestamp": self.timestamp,
            "original_timestamp": self.original_timestamp,
            "exposure_time": self.exposure_time,
            "oned_exposure": self.oned_exposure,
            "twod_exposure": self.twod_exposure,
            "neutron_start": self.neutron_start,
            "neutron_end": self.neutron_end,
            "source_nxs": self.source_nxs,
            "timestamp_for_correlation": self.timestamp_for_correlation,
        }


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
