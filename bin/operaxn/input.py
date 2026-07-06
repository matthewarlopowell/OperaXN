"""
Input Module for Data Processing
"""

import logging
import os
import re
import sys
import tempfile
import threading
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Callable, List, Optional, Tuple, Union, Any, Dict

import h5py
import numpy as np
import pandas as pd

from .config import (
    BATCH_SIZE,
    CACHE_ENABLED,
    DataSourceType,
    ECHEM_TIME_TOLERANCE,
    FILE_READ_RETRIES,
    FILE_READ_RETRY_DELAY,
    LRU_CACHE_MAXSIZE,
    MAX_CACHE_SIZE_MB,
    MAX_DATASET_ELEMENTS,
    MAX_EXPOSURE_TIME,
    MAX_WORKERS,
    OPERAXNTheme,
    PARALLEL_PROCESSING,
    PARALLEL_PROCESSING_THRESHOLD,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
    TARGET_DISPLAY_PIXELS,
    WINDOW_SIZES,
)

try:
    import fabio

    FABIO_AVAILABLE = True
except ImportError:
    FABIO_AVAILABLE = False


# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

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
# Data Models and Enums
# ============================================================================

class FileType(Enum):
    """Supported file type extensions."""
    DAT = ".dat"
    EDF = ".edf"
    TXT = ".txt"
    HDF = ".hdf"
    NXS = ".nxs"
    XY = ".xy"
    ZIP = ".zip"


class DataType(Enum):
    """Scientific data type identifiers."""
    ONED = "oned"
    TWOD = "twod"
    ECHEM = "echem"
    NEUTRON_META = "neutron_meta"
    NEUTRON_TOF = "neutron_tof"
    NEUTRON_D = "neutron_d"


class TimeMethod(Enum):
    """Absolute vs. relative time correlation mode."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass
class FileRecord:
    """Single processed file with path, classification, and metadata."""
    path: str
    original_path: str
    oned: Optional[str] = None
    twod: Optional[str] = None
    echem: Optional[str] = None
    neutron_meta: Optional[str] = None
    neutron_files: Optional[Dict[str, Dict[str, str]]] = None
    timestamp: Optional[str] = None
    exposure_time: Optional[float] = None


@dataclass
class Scan:
    """Single scan with timestamps, file paths, and correlated echem values."""
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
    timestamp_for_correlation: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan to plain dictionary."""
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
            "timestamp_for_correlation": self.timestamp_for_correlation
        }


# ============================================================================
# Cache Management
# ============================================================================

class FileCache:
    """Thread-safe LRU file data cache with size limits."""

    def __init__(self, max_size_mb: int = MAX_CACHE_SIZE_MB) -> None:
        self.cache: Dict[str, Any] = {}
        self.max_size_bytes: int = max_size_mb * 1024 * 1024
        self.current_size: int = 0
        self.lock: threading.Lock = threading.Lock()
        self.access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Any]:
        """Return cached item or None."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
        return None

    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> None:
        """Store item, evicting LRU entries if over size limit."""
        with self.lock:
            if size_bytes is None:
                size_bytes = sys.getsizeof(value)

            while self.current_size + size_bytes > self.max_size_bytes and self.cache:
                self._evict_lru()

            self.cache[key] = value
            self.current_size += size_bytes
            self.access_count[key] = 1

    def _evict_lru(self) -> None:
        """Evict least-accessed item."""
        if not self.cache:
            return

        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count.get(k, 0))

        if lru_key in self.cache:
            value = self.cache.pop(lru_key)
            self.current_size -= sys.getsizeof(value)
            self.access_count.pop(lru_key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.current_size = 0


# Global cache instance
_file_cache = FileCache()


# ============================================================================
# Data Readers
# ============================================================================

class DataReader(ABC):
    """Abstract base for file readers with optional caching."""

    @abstractmethod
    def _read_impl(self, path: str) -> np.ndarray:
        """Read raw data from file."""
        pass

    def read(self, path: str, use_cache: bool = True) -> np.ndarray:
        """Read data, returning cached result when available."""
        if not use_cache or not CACHE_ENABLED:
            return self._read_impl(path)

        cache_key = self._get_cache_key(path)
        cached_data = _file_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        data = self._read_impl(path)
        _file_cache.put(cache_key, data)
        return data

    def _get_cache_key(self, path: str) -> str:
        """Generate cache key from path, size, and mtime."""
        stat = os.stat(path)
        return f"{path}_{stat.st_size}_{stat.st_mtime}"


class DATReader(DataReader):
    """Reader for DAT files (XRD and neutron)."""

    def __init__(self, data_type: str = "xrd") -> None:
        super().__init__()
        self.data_type = data_type

    def _read_impl(self, path: str) -> np.ndarray:
        """Read DAT file as 2-column array."""
        try:
            # Try fast numpy loading first
            try:
                data = np.loadtxt(path, comments='#')
                if data.ndim == 1:
                    data = data.reshape(-1, 2)
                return data
            except (ValueError, IOError, OSError) as e:
                logger.debug(f"NumPy loadtxt failed for {path}: {e}")

            # Fallback to manual parsing
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Find data start (skip comments)
            data_start = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith("#") and line.strip():
                    data_start = i
                    break

            # Parse data
            data = []
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        data.append([x_val, y_val])
                    except ValueError:
                        continue

            if not data:
                file_type = "neutron" if self.data_type == "neutron" else "DAT"
                raise ValueError(f"No valid data found in {file_type} file: {path}")

            return np.array(data, dtype=float)

        except Exception as e:
            file_type = "neutron DAT" if self.data_type == "neutron" else "DAT"
            raise IOError(f"Error reading {file_type} file {path}: {e}")


class EDFReader(DataReader):
    """Reader for EDF 2D detector images via fabio."""

    def _read_impl(self, path: str) -> np.ndarray:
        """Read EDF file as 2D array."""
        if not FABIO_AVAILABLE:
            raise ImportError("fabio is required to read EDF files")

        try:
            arr = np.asarray(fabio.open(path).data.astype(float))
            pos = arr > 0
            floor = float(arr[pos].min()) if pos.any() else 0.0
            return np.clip(arr, floor, None)
        except Exception as e:
            raise IOError(f"Error reading EDF file {path}: {e}")


class XYReader(DataReader):
    """Reader for XY synchrotron integrated data."""

    def _read_impl(self, path: str) -> np.ndarray:
        """Read XY file as 2-column array."""
        try:
            data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            return data
        except Exception as e:
            raise IOError(f"Error reading XY file {path}: {e}")


class HDFReader(DataReader):
    """Reader for HDF5 2D detector data with display downsampling."""

    COMMON_DATA_PATHS = [
        '/entry/instrument/detector/data',
        '/entry1/instrument/detector/data',
        '/entry/data/data',
        '/entry1/data/data',
        '/entry/data',
        '/entry1/data',
        '/data',
    ]

    DETECTOR_SIZES = [
        (2880, 2881),  # Pixium detector
        (2048, 2048),
        (1024, 1024),
        (512, 512),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.max_display_size: int = SYNCHROTRON_MAX_DISPLAY_SIZE
        self.original_shape: Optional[Tuple[int, ...]] = None

    def _read_impl(self, path: str) -> np.ndarray:
        """Read HDF file as 2D array."""
        try:
            with h5py.File(path, 'r', swmr=True) as f:
                data = self._find_data_array(f)

                if data is None:
                    raise ValueError(f"No suitable data found in HDF file")

                data = self._process_data_shape(data)
                data = self._apply_floor_clipping(data)
                self.original_shape = data.shape
                data = self._downsample_if_needed(data)

                return data
        except Exception as e:
            raise IOError(f"Error reading HDF file {path}: {e}")

    def _find_data_array(self, h5file: h5py.File) -> Optional[np.ndarray]:
        """Locate the primary data array in an HDF5 file."""
        for path in self.COMMON_DATA_PATHS:
            if path in h5file:
                dataset = h5file[path]
                if dataset.size > MAX_DATASET_ELEMENTS:
                    return self._sample_large_dataset(dataset)
                else:
                    return np.array(dataset)

        # Search for largest dataset
        largest_dataset = None
        largest_size = 0

        def find_largest(name: str, obj: Any) -> None:
            nonlocal largest_dataset, largest_size
            if isinstance(obj, h5py.Dataset):
                if obj.size > largest_size and obj.ndim in [2, 3]:
                    largest_size = obj.size
                    largest_dataset = name

        h5file.visititems(find_largest)

        if largest_dataset:
            dataset = h5file[largest_dataset]
            if dataset.size > MAX_DATASET_ELEMENTS:
                return self._sample_large_dataset(dataset)
            else:
                return np.array(dataset)

        return None

    @staticmethod
    def _sample_large_dataset(dataset: h5py.Dataset) -> np.ndarray:
        """Sub-sample a large dataset to fit in memory."""
        shape = dataset.shape

        if dataset.ndim == 3:
            data = dataset[0]
        else:
            data = dataset

        if data.size > MAX_DATASET_ELEMENTS:
            height, width = data.shape[-2:]
            step = max(1, int(np.sqrt(data.size / TARGET_DISPLAY_PIXELS)))
            if dataset.ndim == 2:
                return dataset[::step, ::step]
            else:
                return dataset[0, ::step, ::step]

        return np.array(data)

    def _process_data_shape(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is a 2D array."""
        if data.ndim == 3:
            return data[0]
        elif data.ndim == 1:
            return self._reshape_1d_data(data)
        elif data.ndim == 2:
            return data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

    def _reshape_1d_data(self, data: np.ndarray) -> np.ndarray:
        """Reshape 1D array to 2D using known detector dimensions."""
        for height, width in self.DETECTOR_SIZES:
            if data.size == height * width:
                return data.reshape((height, width))

        sqrt_size = int(np.sqrt(data.size))
        if sqrt_size * sqrt_size == data.size:
            return data.reshape((sqrt_size, sqrt_size))

        raise ValueError(f"Cannot determine shape for 1D data of size {data.size}")

    @staticmethod
    def _apply_floor_clipping(data: np.ndarray) -> np.ndarray:
        """Clip values below the positive-pixel floor."""
        pos = data > 0
        if pos.any():
            floor = float(data[pos].min())
            return np.clip(data, floor, None)
        return data

    def _downsample_if_needed(self, data: np.ndarray) -> np.ndarray:
        """Downsample if either dimension exceeds display limit; 0 = disabled."""
        if self.max_display_size <= 0:
            return data

        height, width = data.shape

        if height > self.max_display_size or width > self.max_display_size:
            return self._downsample_for_display(data)

        return data

    def _downsample_for_display(self, data: np.ndarray) -> np.ndarray:
        """Stride-downsample data to fit max display size."""
        height, width = data.shape

        scale_factor = max(
            height / self.max_display_size,
            width / self.max_display_size
        )

        if scale_factor <= 1:
            return data

        step = int(np.ceil(scale_factor))
        downsampled = data[::step, ::step]

        return downsampled


# ============================================================================
# File Processing
# ============================================================================

class FileProcessor:
    """Collects, extracts, and groups files from paths and ZIPs."""

    def __init__(self, progress_callback: Optional[Callable] = None,
                 data_source: DataSourceType = DataSourceType.INHOUSE) -> None:
        self.progress_callback = progress_callback
        self.data_source = data_source
        self.tempdir: Optional[str] = None
        self.processed_files: set = set()
        self.synchrotron_grouper = SynchrotronFileGrouper()
        self.nexus_extractor = NexusMetadataExtractor()

    def __enter__(self) -> "FileProcessor":
        self.tempdir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.tempdir and os.path.exists(self.tempdir):
            import shutil
            shutil.rmtree(self.tempdir, ignore_errors=True)

    def update_progress(self, message: str) -> None:
        """Emit progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def process_paths(self, selected_paths: List[str]) -> List[FileRecord]:
        """Collect and classify all files from selected paths."""
        all_files = self._collect_all_files(selected_paths)

        if not all_files:
            return []

        file_dict = {extracted: original for extracted, original in all_files}

        if self.data_source == DataSourceType.NEUTRON:
            records = self._process_neutron_files(all_files)
        elif self.data_source == DataSourceType.SYNCHROTRON:
            synchrotron_groups = self.synchrotron_grouper.group_files(file_dict)
            records = self._process_synchrotron_groups(synchrotron_groups)
        else:
            records = []

        # Process remaining files
        remaining_files = []
        for extracted_path, original_path in all_files:
            if extracted_path not in self.processed_files:
                remaining_files.append((extracted_path, original_path))

        if remaining_files:
            if PARALLEL_PROCESSING and len(remaining_files) > PARALLEL_PROCESSING_THRESHOLD:
                remaining_records = self._process_remaining_files_parallel(remaining_files)
            else:
                remaining_records = self._process_remaining_files_sequential(remaining_files)
            records.extend(remaining_records)

        return records

    def _process_neutron_files(self, all_files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Create FileRecords for neutron data and metadata files."""
        records = []

        for extracted_path, original_path in all_files:
            basename = os.path.basename(extracted_path)
            ext = os.path.splitext(basename)[1].lower()

            self.processed_files.add(extracted_path)

            record = FileRecord(
                path=extracted_path,
                original_path=original_path
            )
            records.append(record)

            if ext == '.txt':
                logger.debug(f"Added potential metadata file: {basename}")
            elif ext == '.dat':
                logger.debug(f"Added neutron data file: {basename}")

        logger.info(f"Processed {len(records)} neutron-related files")
        return records

    def _collect_all_files(self, selected_paths: List[str]) -> List[Tuple[str, str]]:
        """Gather files from paths, extracting ZIPs as needed."""
        all_files = []

        for path in selected_paths:
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext == FileType.ZIP.value:
                    extracted = self._extract_zip_files(path)
                    all_files.extend(extracted)
                else:
                    all_files.append((path, path))
            elif os.path.isdir(path):
                dir_files = self._collect_directory_files(path)
                all_files.extend(dir_files)

        logger.info(f"Collected {len(all_files)} total files")
        return all_files

    def _extract_zip_files(self, zip_path: str) -> List[Tuple[str, str]]:
        """Extract supported files from a ZIP archive."""
        extracted = []

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                members = archive.namelist()

                for member in members:
                    if (member.endswith("/") or
                            member.startswith("__MACOSX") or
                            member.startswith(".")):
                        continue

                    member_ext = os.path.splitext(member)[1].lower()
                    supported_exts = [ft.value for ft in FileType if ft != FileType.ZIP]

                    if member_ext in supported_exts:
                        # Validate path to prevent ZIP Slip (path traversal)
                        target = os.path.realpath(os.path.join(self.tempdir, member))
                        if not target.startswith(os.path.realpath(self.tempdir) + os.sep):
                            logger.warning(f"Skipping ZIP entry with invalid path: {member}")
                            continue
                        extracted_path = archive.extract(member, self.tempdir)
                        original_path = os.path.join(zip_path, member)
                        extracted.append((extracted_path, original_path))

        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")

        return extracted

    def _collect_directory_files(self, directory: str) -> List[Tuple[str, str]]:
        """Recursively collect supported files from a directory."""
        files = []

        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                file_path = os.path.join(root, filename)
                ext = os.path.splitext(file_path)[1].lower()

                if ext == FileType.ZIP.value:
                    extracted = self._extract_zip_files(file_path)
                    files.extend(extracted)
                elif ext in [ft.value for ft in FileType]:
                    files.append((file_path, file_path))

        return files

    def _process_synchrotron_groups(self, synchrotron_groups: Dict[str, Dict[str, str]]) -> List[FileRecord]:
        """Create FileRecords from grouped synchrotron file sets."""
        records = []

        for base_id, file_group in synchrotron_groups.items():
            if 'nxs' in file_group:
                record = self._create_synchrotron_record(base_id, file_group)
                if record:
                    records.append(record)
                    for file_path in file_group.values():
                        self.processed_files.add(file_path)

        return records

    def _process_remaining_files_parallel(self, files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Process unclassified files using a thread pool."""
        chunk_size = max(BATCH_SIZE, len(files) // MAX_WORKERS)
        chunks = [files[i:i + chunk_size]
                  for i in range(0, len(files), chunk_size)]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunk_results = list(executor.map(self._process_remaining_chunk, chunks))

        records = []
        for chunk_records in chunk_results:
            records.extend(chunk_records)

        return records

    def _process_remaining_files_sequential(self, files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Process unclassified files in a single thread."""
        return self._process_remaining_chunk(files)

    def _process_remaining_chunk(self, files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Convert a batch of files into FileRecords."""
        records = []
        for extracted_path, original_path in files:
            ext = os.path.splitext(extracted_path)[1].lower()
            if ext in [FileType.DAT.value, FileType.EDF.value, FileType.TXT.value]:
                self.processed_files.add(extracted_path)
                records.append(FileRecord(
                    path=extracted_path,
                    original_path=original_path
                ))
        return records

    def _create_synchrotron_record(self, base_id: str, file_group: Dict[str, str]) -> Optional[FileRecord]:
        """Build a FileRecord from a synchrotron NXS group."""
        nxs_path = file_group.get('nxs')
        if not nxs_path:
            return None

        metadata = self.nexus_extractor.extract(nxs_path)
        if not metadata:
            return FileRecord(
                path=nxs_path,
                original_path=nxs_path,
                oned=file_group.get('xy'),
                twod=file_group.get('hdf'),
                timestamp=None,
                exposure_time=None
            )

        return FileRecord(
            path=nxs_path,
            original_path=nxs_path,
            oned=file_group.get('xy'),
            twod=file_group.get('hdf'),
            timestamp=metadata.get('timestamp'),
            exposure_time=metadata.get('exposure_time')
        )


# ============================================================================
# File Classification
# ============================================================================

class FileClassifierBase(ABC):
    """Abstract base for file type classifiers."""

    @abstractmethod
    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Return (data_type, timestamp, exposure_time) for a file."""
        pass


class DATClassifier(FileClassifierBase):
    """Extract timestamp and exposure from DAT headers."""

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Classify DAT file as 1D data and extract header metadata."""
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
                    raw = line.strip().split()[-1]
                    timestamp = raw.replace("T", " ")

                elif "exposuretime" in line_lower.replace(" ", ""):
                    try:
                        exposure_str = line.strip().split()[-1]
                        exposure_time = float(exposure_str)
                    except (ValueError, IndexError):
                        pass

            if timestamp:
                return DataType.ONED.value, timestamp, exposure_time

        except Exception as e:
            logger.error(f"Error classifying DAT file {path}: {e}")

        return None, None, None


class EDFClassifier(FileClassifierBase):
    """Extract timestamp and exposure from EDF headers."""

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Classify EDF file as 2D data and extract header metadata."""
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
    """Distinguish echem data from neutron metadata in TXT files."""

    ECHEM_KEYWORDS = ["time", "absolute", "ecell", "voltage", "current", "i/", "ewe", "v/"]

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Classify TXT as echem data or neutron logbook."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i in range(10):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)

            if not lines:
                return None, None, None

            first_line_lower = lines[0].lower()
            if any(keyword in first_line_lower for keyword in self.ECHEM_KEYWORDS):
                return DataType.ECHEM.value, None, None

            # Check for neutron metadata format
            weekdays = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
            is_neutron_logbook = False

            for line in lines[:5]:
                parts = line.strip().split('\t')

                if len(parts) >= 8:
                    if parts[0].isdigit() and 5 <= len(parts[0]) <= 7:
                        line_text = ' '.join(parts)
                        if any(weekday in line_text for weekday in weekdays):
                            is_neutron_logbook = True
                            break

            if is_neutron_logbook:
                logger.debug(f"Identified {path} as neutron metadata")
                return DataType.NEUTRON_META.value, None, None

            return None, None, None

        except Exception as e:
            logger.error(f"Error classifying TXT file {path}: {e}")
            return None, None, None


class FileClassificationManager:
    """Dispatch classifier by file type and populate DataFrame columns."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE) -> None:
        self.data_source = data_source
        self.classifiers = {
            FileType.DAT: DATClassifier(),
            FileType.EDF: EDFClassifier(),
            FileType.TXT: TXTClassifier()
        }

    def classify_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each row's file and populate metadata columns."""
        df = df.copy()

        for col in ["oned", "twod", "echem", "neutron_meta"]:
            if col not in df.columns:
                df[col] = pd.array([None] * len(df), dtype="object")
            else:
                df[col] = df[col].astype("object")
        if "exposure_time" not in df.columns:
            df["exposure_time"] = pd.array([None] * len(df), dtype="object")
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.array([None] * len(df), dtype="object")

        for idx, row in df.iterrows():
            if pd.notna(row["oned"]) or pd.notna(row["twod"]) or pd.notna(row["neutron_meta"]):
                continue

            path = row["path"]
            ext = os.path.splitext(path)[1].lower()

            if self.data_source == DataSourceType.NEUTRON:
                if ext == '.txt':
                    file_type = FileType(ext) if ext in [ft.value for ft in FileType] else None
                    if file_type and file_type in self.classifiers:
                        classifier = self.classifiers[file_type]
                        data_type, timestamp, exposure_time = classifier.classify(path)

                        if data_type == DataType.NEUTRON_META.value:
                            df.at[idx, "neutron_meta"] = path
                        elif data_type == DataType.ECHEM.value:
                            df.at[idx, "echem"] = path

                        if timestamp:
                            df.at[idx, "timestamp"] = timestamp
                        if exposure_time is not None:
                            df.at[idx, "exposure_time"] = exposure_time

            else:
                file_type = FileType(ext) if ext in [ft.value for ft in FileType] else None
                if file_type and file_type in self.classifiers:
                    classifier = self.classifiers[file_type]
                    data_type, timestamp, exposure_time = classifier.classify(path)

                    if data_type:
                        df.at[idx, data_type] = path
                        if timestamp:
                            df.at[idx, "timestamp"] = timestamp
                        if exposure_time is not None:
                            df.at[idx, "exposure_time"] = exposure_time

        return df


# ============================================================================
# Neutron Data Processing
# ============================================================================

class NeutronMetadataParser:
    """Parse neutron logbook TXT files into scan DataFrames."""

    @staticmethod
    def parse(path: str) -> Optional[pd.DataFrame]:
        """Parse neutron logbook and return scan entries as DataFrame."""
        try:
            logger.debug(f"Parsing neutron metadata file: {path}")

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                logger.debug("No lines found in file")
                return None

            data = []
            weekdays = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

            for line_num, line in enumerate(lines):
                parts = line.strip().split('\t')

                if len(parts) >= 8:
                    try:
                        scan_id = parts[0].strip()

                        if not (scan_id.isdigit() and 5 <= len(scan_id) <= 7):
                            continue

                        start_time = None
                        end_time = None

                        for i in range(len(parts)):
                            if any(day in parts[i] for day in weekdays):
                                if not start_time:
                                    start_time = parts[i].strip()
                                elif not end_time:
                                    end_time = parts[i].strip()
                                    break

                        if start_time and end_time:
                            start_dt = pd.to_datetime(start_time, format='%a %b %d %H:%M:%S %Y')
                            end_dt = pd.to_datetime(end_time, format='%a %b %d %H:%M:%S %Y')

                            data.append({
                                "scan_id": scan_id,
                                "start_time": start_dt,
                                "end_time": end_dt,
                                "full_line": line.strip()
                            })

                            if len(data) <= 3:
                                logger.debug(f"Parsed scan {scan_id}: {start_dt} -> {end_dt}")

                    except (ValueError, IndexError) as e:
                        if line_num < 3:
                            logger.debug(f"Error parsing line {line_num}: {e}")
                        continue

            if data:
                logger.info(f"Successfully parsed {len(data)} entries")
                return pd.DataFrame(data)
            else:
                logger.debug("No valid entries found")

            return None

        except Exception as e:
            logger.error(f"Error parsing neutron metadata: {e}")
            return None


class NeutronFileGrouper:
    """Group neutron DAT files by scan ID and measurement number."""

    def group_neutron_files(self, file_list: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Return {scan_id: {measurement: {type: path}}} mapping."""
        groups = {}

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

                if scan_id not in groups:
                    groups[scan_id] = {}

                if measurement_num not in groups[scan_id]:
                    groups[scan_id][measurement_num] = {}

                groups[scan_id][measurement_num][data_type] = filepath

        logger.info(f"Grouped {len(groups)} scans")
        return groups

    @staticmethod
    def extract_scan_id(filename: str) -> Optional[str]:
        """Return the 5-7 digit scan ID for a neutron filename, or None."""
        info = NeutronFileGrouper._extract_neutron_file_info(filename)
        return info['scan_id'] if info else None

    @staticmethod
    def _extract_neutron_file_info(filename: str) -> Optional[Dict[str, str]]:
        """Parse scan ID, measurement number, and type from filename."""
        name_no_ext = filename[:-4] if filename.endswith('.dat') else filename
        is_dspacing = ('-d-' in name_no_ext or '_d_' in name_no_ext
                       or name_no_ext.endswith('_d') or name_no_ext.endswith('-d'))

        try:
            # Pattern: POL123456-b_1-d.dat
            pol_pattern = r'POL(\d+)-b_(\d)'
            pol_match = re.search(pol_pattern, name_no_ext.replace('-d', '').replace('_d', ''))

            if pol_match:
                scan_id = pol_match.group(1)
                measurement_num = pol_match.group(2)

                # Same 5-7 digit rule as the logbook parser, or the group
                # could never match a logbook entry
                if 5 <= len(scan_id) <= 7 and 1 <= int(measurement_num) <= 5:
                    return {
                        'scan_id': scan_id,
                        'measurement': measurement_num,
                        'type': 'd' if is_dspacing else 'tof'
                    }

            # Pattern: 12345-1-d.dat or 1234567-1-d.dat (5-7 digit scan IDs).
            # Lookbehind stops an 8+ digit run number matching by its last 7 digits.
            pattern = r'(?<!\d)(\d{5,7})-(\d)'
            match = re.search(pattern, name_no_ext)

            if match:
                scan_id = match.group(1)
                measurement_num = match.group(2)

                if 1 <= int(measurement_num) <= 5:
                    return {
                        'scan_id': scan_id,
                        'measurement': measurement_num,
                        'type': 'd' if is_dspacing else 'tof'
                    }

            # Fallback: split on hyphen
            if '-' in name_no_ext:
                parts = name_no_ext.split('-')
                if len(parts) >= 2:
                    if parts[0].isdigit() and 5 <= len(parts[0]) <= 7:
                        scan_id = parts[0]
                        if parts[1] and parts[1][0].isdigit():
                            measurement_num = parts[1][0]
                            if 1 <= int(measurement_num) <= 5:
                                return {
                                    'scan_id': scan_id,
                                    'measurement': measurement_num,
                                    'type': 'd' if is_dspacing else 'tof'
                                }

            return None

        except (ValueError, IndexError):
            return None


# ============================================================================
# Specialised Processors
# ============================================================================

class EchemParser:
    """Parse tab-delimited electrochemistry TXT files."""

    COLUMN_PATTERNS = {
        "time": ["time", "date"],
        "voltage": ["voltage", "v/", "ecell", "ewe"],
        "current": ["current", "i/"]
    }

    def parse(self, path: str) -> Optional[pd.DataFrame]:
        """Read echem TXT and return timestamp/voltage/current DataFrame."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                return None

            has_header = any(h in lines[0].lower() for h in
                             ["time", "date", "ecell", "ewe", "voltage", "current", "i/", "v/"])

            if has_header:
                columns = self._detect_columns(lines[0])
                data_lines = lines[1:]
            else:
                columns = {"time": 0, "voltage": 1, "current": 2}
                data_lines = lines

            data = self._parse_data_lines(data_lines, columns)

            if data:
                return pd.DataFrame(data)

            return None

        except Exception as e:
            logger.error("Failed to parse echem file: %s", e)
            return None

    def _detect_columns(self, header_line: str) -> Dict[str, int]:
        """Map column names to indices via keyword matching."""
        header_parts = [part.strip().lower() for part in header_line.strip().split("\t")]

        detected: Dict[str, int] = {}

        column_mapping: Dict[str, str] = {}
        for col_type, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                column_mapping[pattern] = col_type

        for i, part in enumerate(header_parts):
            clean_part = part.replace("(", "").replace(")", "").replace("/", "").replace(" ", "")

            if clean_part in column_mapping:
                detected[column_mapping[clean_part]] = i
                continue

            for key, value in column_mapping.items():
                if key in part:
                    detected[value] = i
                    break

        # Default positional indices for undetected columns
        columns = {"time": 0, "voltage": 1, "current": 2}
        columns.update(detected)

        # Resolve index collisions between defaults and detected columns
        used_indices = set(detected.values())
        for col_name in columns:
            if col_name not in detected and columns[col_name] in used_indices:
                logger.warning(
                    "Default column '%s' at index %d collides with detected column. Disabling.",
                    col_name, columns[col_name]
                )
                columns[col_name] = -1

        return columns

    def _parse_data_lines(self, lines: List[str], columns: Dict[str, int]) -> List[Dict[str, Any]]:
        """Convert tab-delimited lines to dicts of timestamp, voltage, current."""
        # A disabled time/voltage column (-1) would silently index parts[-1]
        if columns["time"] < 0 or columns["voltage"] < 0:
            logger.warning("Echem time/voltage column could not be resolved; skipping file")
            return []

        data = []
        max_idx = max(columns.values())

        for line in lines:
            parts = line.strip().split("\t")

            if len(parts) <= max_idx:
                continue

            ts_str = parts[columns["time"]]
            if ts_str.startswith("1970/01/01"):
                continue

            try:
                timestamp = pd.to_datetime(ts_str, dayfirst=True)
            except (ValueError, TypeError):
                continue

            try:
                voltage = float(parts[columns["voltage"]])
            except (ValueError, IndexError):
                continue

            current = None
            if 0 <= columns["current"] < len(parts):
                try:
                    current = float(parts[columns["current"]])
                except (ValueError, IndexError):
                    pass

            data.append({
                "timestamp": timestamp,
                "echem_data": voltage,
                "current": current
            })

        return data


class SynchrotronFileGrouper:
    """Group NXS, HDF, and XY files by scan ID."""

    def group_files(self, file_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Return {scan_id: {ext_key: path}} grouping."""
        groups = {}
        nxs_to_id = {}

        # First pass: identify NXS files and their IDs
        for extracted_path, original_path in file_dict.items():
            basename = os.path.basename(extracted_path)
            ext = os.path.splitext(basename)[1].lower()

            if ext == '.nxs':
                scan_id = self._extract_scan_id(basename)
                if scan_id:
                    nxs_to_id[extracted_path] = scan_id

        # Second pass: group all files
        for extracted_path, original_path in file_dict.items():
            basename = os.path.basename(extracted_path)
            ext = os.path.splitext(basename)[1].lower()

            if ext in ['.hdf', '.nxs', '.xy']:
                group_id = self._determine_group_id(basename, ext, nxs_to_id)

                if group_id:
                    if group_id not in groups:
                        groups[group_id] = {}

                    if ext == '.hdf':
                        groups[group_id]['hdf'] = extracted_path
                    elif ext == '.nxs':
                        groups[group_id]['nxs'] = extracted_path
                    elif ext == '.xy':
                        groups[group_id]['xy'] = extracted_path

        return groups

    def _extract_scan_id(self, filename: str) -> Optional[str]:
        """Extract trailing numeric scan ID from filename."""
        base_name = os.path.splitext(filename)[0]
        base_name = base_name.replace('_integration', '')

        match = re.search(r'(\d+)(?!.*\d)', base_name)
        return match.group(1) if match else None

    def _determine_group_id(self, basename: str, ext: str, nxs_to_id: Dict[str, str]) -> Optional[str]:
        """Match a file to its NXS-based group ID."""
        file_id = self._extract_scan_id(basename)

        if ext == '.hdf':
            if file_id:
                for nxs_path, nxs_id in nxs_to_id.items():
                    if nxs_id == file_id:
                        return nxs_id

        elif ext == '.nxs':
            return file_id

        elif ext == '.xy':
            base_parts = os.path.splitext(basename)[0]
            if '_integration_' in base_parts:
                base_without_integration = base_parts.split('_integration_')[0]
                xy_id = self._extract_scan_id(base_without_integration)
                if xy_id:
                    for nxs_path, nxs_id in nxs_to_id.items():
                        if nxs_id == xy_id:
                            return nxs_id
            else:
                if file_id:
                    for nxs_path, nxs_id in nxs_to_id.items():
                        if nxs_id == file_id:
                            return nxs_id

        return None


class NexusMetadataExtractor:
    """Extract timestamps and exposure times from NeXus files."""

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

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def extract(self, nxs_path: str) -> Optional[Dict[str, Any]]:
        """Return dict with timestamp and exposure_time from NXS."""
        try:
            with h5py.File(nxs_path, 'r') as f:
                metadata = {}

                timestamp = self._extract_timestamp(f)
                if timestamp:
                    metadata['timestamp'] = timestamp

                exposure_time = self._extract_exposure_time(f)
                if exposure_time:
                    metadata['exposure_time'] = exposure_time

                midpoint = self._calculate_midpoint_timestamp(f)
                if midpoint:
                    metadata['midpoint_timestamp'] = midpoint

                return metadata if metadata else None

        except Exception as e:
            logger.error("Failed to extract NXS metadata from %s: %s", nxs_path, e)
            return None

    def _extract_timestamp(self, h5file: h5py.File) -> Optional[str]:
        """Read start_time from known HDF5 paths."""
        for path in self.TIMESTAMP_PATHS:
            if path in h5file:
                timestamp_str = self._decode_value(h5file[path][()])
                return self._parse_nexus_timestamp(timestamp_str)
        return None

    def _extract_exposure_time(self, h5file: h5py.File) -> Optional[float]:
        """Read or calculate exposure time in seconds."""
        for path in self.EXPOSURE_PATHS:
            if path in h5file:
                try:
                    exp_time = float(h5file[path][()])
                    if exp_time > 0:
                        return exp_time
                except (ValueError, TypeError):
                    pass

        # Calculate from start/end times
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

    def _calculate_midpoint_timestamp(self, h5file: h5py.File) -> Optional[str]:
        """Compute midpoint between start_time and end_time."""
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

                    if exposure_seconds > 0:
                        midpoint = start_dt + pd.Timedelta(seconds=exposure_seconds / 2)
                        return midpoint.strftime('%Y-%m-%d %H:%M:%S')

        return None

    @staticmethod
    def _decode_value(value: Any) -> str:
        """Decode bytes to str for HDF5 values."""
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    @staticmethod
    def _parse_nexus_timestamp(timestamp_str: str) -> str:
        """Normalise ISO/NeXus timestamp to 'YYYY-MM-DD HH:MM:SS'."""
        if 'T' in timestamp_str:
            base_time = timestamp_str.split('+')[0].split('Z')[0]
            if '.' in base_time:
                base_time = base_time.split('.')[0]
            return base_time.replace('T', ' ')
        return timestamp_str


# ============================================================================
# Scan Processor
# ============================================================================

class ScanProcessor:
    """Build scan lists and correlate with echem timestamps."""

    def __init__(self, time_method: TimeMethod = TimeMethod.ABSOLUTE,
                 data_source: DataSourceType = DataSourceType.INHOUSE) -> None:
        self.time_method = time_method
        self.data_source = data_source
        self.xrd_reference_time = None
        self.echem_reference_time = None
        self.neutron_reference_time = None
        self.echem_parser = EchemParser()
        self.neutron_parser = NeutronMetadataParser()

    def process_scans(self, df: pd.DataFrame) -> Tuple[List[Scan], pd.DataFrame]:
        """Build scans from DataFrame and correlate with echem."""
        combined_echem_df = self._process_echem_data(df)

        neutron_metadata_df = None
        if self.data_source == DataSourceType.NEUTRON:
            neutron_metadata_df = self._process_neutron_metadata(df)
            logger.info(f"Processed neutron metadata: {neutron_metadata_df is not None}")
            if neutron_metadata_df is not None:
                logger.info(f"Found {len(neutron_metadata_df)} neutron scans")

        scan_list = self._create_scan_list(df, neutron_metadata_df)

        # Compute midpoint-adjusted correlation timestamps
        self._adjust_for_exposure_time(scan_list)

        if self.time_method == TimeMethod.RELATIVE:
            self._set_reference_times(df, combined_echem_df, neutron_metadata_df, scan_list)

        # Correlate scans with echem using absolute timestamps (before formatting)
        if not combined_echem_df.empty:
            self._correlate_with_echem(scan_list, combined_echem_df)

        # Convert display timestamps to relative HH:MM:SS strings
        if self.time_method == TimeMethod.RELATIVE:
            self._apply_relative_time(scan_list, combined_echem_df)

        return scan_list, combined_echem_df

    def _process_neutron_metadata(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Parse and combine all neutron metadata files."""
        neutron_meta_paths = df[df["neutron_meta"].notna()]["neutron_meta"].tolist()

        logger.info(f"Found {len(neutron_meta_paths)} neutron metadata files")

        if not neutron_meta_paths:
            return None

        neutron_dfs = []
        for path in neutron_meta_paths:
            logger.debug(f"Parsing metadata file: {path}")
            meta_df = self.neutron_parser.parse(path)
            if meta_df is not None:
                meta_df["source_file"] = path
                neutron_dfs.append(meta_df)
                logger.debug(f"Successfully parsed {len(meta_df)} entries")
            else:
                logger.debug("Failed to parse metadata")

        if neutron_dfs:
            combined_df = pd.concat(neutron_dfs, ignore_index=True)
            logger.info(f"Combined neutron metadata: {len(combined_df)} total entries")
            return combined_df

        return None

    def _process_echem_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and concatenate all echem files."""
        echem_paths = df[df["echem"].notna()]["echem"].tolist()
        echem_dfs = []

        for e_path in echem_paths:
            e_df = self.echem_parser.parse(e_path)
            if e_df is not None:
                e_df["source_file"] = e_path
                echem_dfs.append(e_df)

        if echem_dfs:
            combined_df = pd.concat(echem_dfs, ignore_index=True).sort_values("timestamp")

            logger.info(f"Echem data: {len(combined_df)} rows")
            if len(combined_df) > 0:
                logger.debug(f"First echem timestamp: {combined_df.iloc[0]['timestamp']}")
                logger.debug(f"Last echem timestamp: {combined_df.iloc[-1]['timestamp']}")

            return combined_df

        return pd.DataFrame(columns=["timestamp", "echem_data", "current", "source_file"])

    def _set_reference_times(self, df: pd.DataFrame, echem_df: pd.DataFrame,
                             neutron_df: Optional[pd.DataFrame] = None,
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
            xrd_timestamps = []
            for _, row in df.iterrows():
                if pd.notna(row.get("timestamp")) and (pd.notna(row.get("oned")) or pd.notna(row.get("twod"))):
                    try:
                        xrd_timestamps.append(pd.to_datetime(row["timestamp"]))
                    except (ValueError, TypeError):
                        pass
            self.xrd_reference_time = min(xrd_timestamps) if xrd_timestamps else None

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
        elif neutron_df is not None and not neutron_df.empty:
            neutron_timestamps = []
            for _, row in neutron_df.iterrows():
                if pd.notna(row.get("start_time")):
                    try:
                        neutron_timestamps.append(pd.to_datetime(row["start_time"]))
                    except (ValueError, TypeError):
                        pass
            self.neutron_reference_time = min(neutron_timestamps) if neutron_timestamps else None

    @staticmethod
    def _create_neutron_scan_list(df: pd.DataFrame,
                                  neutron_metadata_df: pd.DataFrame) -> List[Scan]:
        """Build Scan objects from neutron metadata and grouped files."""
        scan_list = []

        neutron_files = []
        logger.debug(f"DataFrame has {len(df)} rows")

        for idx, row in df.iterrows():
            if pd.notna(row["path"]) and row["path"].endswith('.dat'):
                neutron_files.append(row["path"])
                if len(neutron_files) <= 5:
                    logger.debug(f"Found neutron data file: {os.path.basename(row['path'])}")

        logger.info(f"Found {len(neutron_files)} .dat files")

        grouper = NeutronFileGrouper()
        neutron_groups = grouper.group_neutron_files(neutron_files)
        logger.info(f"Created {len(neutron_groups)} neutron groups")

        scans_with_data = 0
        scans_without_data = 0

        for idx, meta_row in neutron_metadata_df.iterrows():
            scan_id = str(meta_row["scan_id"])

            neutron_data_files = neutron_groups.get(scan_id, {})

            if not neutron_data_files:
                scans_without_data += 1
                if scans_without_data <= 5:
                    logger.debug(f"Info: No data files for scan {scan_id} - skipping")
                continue

            scans_with_data += 1

            start_time = meta_row["start_time"]
            end_time = meta_row["end_time"]

            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time, format='%a %b %d %H:%M:%S %Y')
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time, format='%a %b %d %H:%M:%S %Y')

            midpoint = start_time + (end_time - start_time) / 2

            if scans_with_data <= 3:
                logger.debug(f"Scan {scan_id}: Start: {start_time}, End: {end_time}, Midpoint: {midpoint}")

            scan = Scan(
                scan_num=0,
                neutron_meta=meta_row.get("source_file"),
                neutron_files=neutron_data_files,
                neutron_start=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                neutron_end=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                timestamp=midpoint.strftime('%Y-%m-%d %H:%M:%S'),
                original_timestamp=midpoint.strftime('%Y-%m-%d %H:%M:%S'),
                timestamp_for_correlation=midpoint
            )
            scan_list.append(scan)

        logger.info(f"Created {len(scan_list)} neutron scans with data")
        if scans_without_data > 5:
            logger.info(f"({scans_without_data} scans in metadata had no data files and were skipped)")

        return scan_list

    def _create_scan_list(self, df: pd.DataFrame,
                          neutron_metadata_df: Optional[pd.DataFrame] = None) -> List[Scan]:
        """Create sorted, numbered Scan list from classified DataFrame."""
        scan_list = []

        if self.data_source == DataSourceType.NEUTRON and neutron_metadata_df is not None:
            scan_list = self._create_neutron_scan_list(df, neutron_metadata_df)
        elif self.data_source == DataSourceType.SYNCHROTRON:
            synchrotron_df = df[(df["oned"].notna()) | (df["twod"].notna())]
            for idx, row in synchrotron_df.iterrows():
                scan = Scan(
                    scan_num=0,
                    oned=_nan_to_none(row["oned"]),
                    twod=_nan_to_none(row["twod"]),
                    timestamp=_nan_to_none(row["timestamp"]),
                    original_timestamp=_nan_to_none(row["timestamp"]),
                    exposure_time=_nan_to_none(row.get("exposure_time")),
                    oned_exposure=_nan_to_none(row.get("exposure_time")),
                    twod_exposure=_nan_to_none(row.get("exposure_time"))
                )
                scan_list.append(scan)
        else:
            oned_df = df[df["oned"].notna()]
            twod_df = df[df["twod"].notna()]

            for idx, row in oned_df.iterrows():
                scan = Scan(
                    scan_num=0,
                    oned=_nan_to_none(row["oned"]),
                    timestamp=_nan_to_none(row.get("timestamp")),
                    original_timestamp=_nan_to_none(row.get("timestamp")),
                    oned_exposure=_nan_to_none(row.get("exposure_time"))
                )
                scan_list.append(scan)

            for idx, row in twod_df.iterrows():
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

        for num, scan in enumerate(scan_list, start=1):
            scan.scan_num = num

        return scan_list

    def _adjust_for_exposure_time(self, scan_list: List[Scan]) -> None:
        """Shift timestamps to exposure midpoint for echem correlation."""
        logger.debug("Adjusting timestamps to midpoint")
        for i, scan in enumerate(scan_list):
            exposure_time = self._determine_exposure_time(scan)
            scan.exposure_time = exposure_time

            if self.data_source == DataSourceType.NEUTRON:
                scan.timestamp_for_correlation = pd.to_datetime(scan.timestamp) if scan.timestamp else None
            else:
                if exposure_time and scan.timestamp:
                    original_ts = pd.to_datetime(scan.timestamp)
                    adjusted_ts = original_ts + pd.Timedelta(seconds=exposure_time / 2)
                    scan.timestamp_for_correlation = adjusted_ts
                else:
                    scan.timestamp_for_correlation = pd.to_datetime(scan.timestamp) if scan.timestamp else None

    def _determine_exposure_time(self, scan: Scan) -> Optional[float]:
        """Return best available exposure time in seconds."""
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
        """Dispatch to absolute or relative echem correlation."""
        if self.time_method == TimeMethod.RELATIVE:
            self._correlate_relative_time(scan_list, echem_df)
        else:
            self._correlate_absolute_time(scan_list, echem_df)

    def _correlate_relative_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Match scans to echem by relative elapsed seconds."""
        reference_time = None
        if self.data_source == DataSourceType.NEUTRON:
            reference_time = self.neutron_reference_time
        else:
            reference_time = self.xrd_reference_time

        if not reference_time or not self.echem_reference_time:
            logger.warning("No reference times available for relative correlation")
            return

        echem_timestamps = pd.to_datetime(echem_df["timestamp"])
        echem_relative_seconds = (echem_timestamps - self.echem_reference_time).dt.total_seconds()

        logger.debug(
            f"Relative Time Correlation - XRD/Neutron ref: {reference_time}, Echem ref: {self.echem_reference_time}")

        for i, scan in enumerate(scan_list):
            if not scan.timestamp_for_correlation:
                scan.echem_timestamp = None
                continue

            scan_relative_seconds = (scan.timestamp_for_correlation - reference_time).total_seconds()

            time_diffs = np.abs(echem_relative_seconds.values - scan_relative_seconds)
            nearest_idx = int(np.argmin(time_diffs))
            min_diff_seconds = time_diffs[nearest_idx]

            if min_diff_seconds < ECHEM_TIME_TOLERANCE:
                scan.echem = float(echem_df.iloc[nearest_idx]["echem_data"])
                current_val = echem_df.iloc[nearest_idx]["current"]
                scan.current = float(current_val) if pd.notna(current_val) else None
                scan.echem_timestamp = str(echem_df.iloc[nearest_idx]["timestamp"])
            else:
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None

    @staticmethod
    def _correlate_absolute_time(scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Match scans to nearest echem point by absolute timestamp."""
        echem_timestamps = None
        try:
            echem_timestamps = pd.to_datetime(echem_df["timestamp"])
        except (ValueError, TypeError):
            try:
                echem_timestamps = pd.to_datetime(echem_df["timestamp"], format='%d/%m/%Y %H:%M:%S.%f')
            except (ValueError, TypeError):
                try:
                    echem_timestamps = pd.to_datetime(echem_df["timestamp"], format='%d/%m/%Y %H:%M:%S')
                except (ValueError, TypeError):
                    logger.error("Could not parse echem timestamps")
                    return

        echem_start = echem_timestamps.min()
        echem_end = echem_timestamps.max()

        logger.debug(f"Absolute Time Correlation - Echem range: {echem_start} to {echem_end}")

        for i, scan in enumerate(scan_list):
            scan_time = scan.timestamp_for_correlation
            if not scan_time:
                scan.echem_timestamp = None
                continue

            if isinstance(scan_time, str):
                scan_time = pd.to_datetime(scan_time)

            if (scan_time < echem_start - pd.Timedelta(seconds=ECHEM_TIME_TOLERANCE) or
                    scan_time > echem_end + pd.Timedelta(seconds=ECHEM_TIME_TOLERANCE)):
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None
                continue

            time_diffs = abs(echem_timestamps - scan_time)
            nearest_idx = int(np.argmin(time_diffs.values))
            min_diff = time_diffs.iloc[nearest_idx]

            if min_diff.total_seconds() < ECHEM_TIME_TOLERANCE:
                scan.echem = float(echem_df.iloc[nearest_idx]["echem_data"])
                current_val = echem_df.iloc[nearest_idx]["current"]
                scan.current = float(current_val) if pd.notna(current_val) else None
                scan.echem_timestamp = str(echem_df.iloc[nearest_idx]["timestamp"])
            else:
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None

    def _apply_relative_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Convert scan and echem timestamps to HH:MM:SS offsets."""
        reference_time = None
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
                    scan.timestamp = self._format_relative_time(relative_seconds)

        if not echem_df.empty and self.echem_reference_time:
            original_timestamps = pd.to_datetime(echem_df["timestamp"])
            relative_seconds = (original_timestamps - self.echem_reference_time).dt.total_seconds()

            echem_df["original_timestamp"] = echem_df["timestamp"].copy()
            echem_df["timestamp"] = [self._format_relative_time(s) for s in relative_seconds]

    @staticmethod
    def _format_relative_time(seconds: float) -> str:
        """Format elapsed seconds as HH:MM:SS string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# Data Reader Factory
# ============================================================================

class DataReaderFactory:
    """Dispatch DataReader by file extension."""

    READERS = {
        FileType.EDF: EDFReader(),
        FileType.DAT: DATReader("xrd"),
        FileType.HDF: HDFReader(),
        FileType.XY: XYReader()
    }

    NEUTRON_READER = DATReader("neutron")

    @classmethod
    def get_reader(cls, file_path: str, is_neutron: bool = False) -> DataReader:
        """Return the DataReader for a given file path."""
        if is_neutron and file_path.endswith('.dat'):
            return cls.NEUTRON_READER

        ext = os.path.splitext(file_path)[1].lower()
        file_type = FileType(ext) if ext in [ft.value for ft in FileType] else None

        if file_type and file_type in cls.READERS:
            return cls.READERS[file_type]

        raise ValueError(f"No reader available for file type: {ext}")

    @classmethod
    def read_file(cls, file_path: str, use_cache: bool = True, is_neutron: bool = False) -> np.ndarray:
        """Read file with the appropriate reader and optional caching."""
        reader = cls.get_reader(file_path, is_neutron)
        return reader.read(file_path, use_cache)


# ============================================================================
# Main API Functions
# ============================================================================

def process_paths(selected_paths: List[str],
                  progress_callback: Optional[Callable] = None,
                  time_method: Optional[TimeMethod] = None,
                  data_source: DataSourceType = DataSourceType.INHOUSE) -> Tuple[
                  List[Dict[str, Any]], pd.DataFrame, str]:
    """Process paths into scan dicts, echem DataFrame, and time method."""
    with FileProcessor(progress_callback, data_source) as processor:
        records = processor.process_paths(selected_paths)

    if not records:
        return [], pd.DataFrame(), TimeMethod.ABSOLUTE.value

    records_df = pd.DataFrame([r.__dict__ for r in records])

    if progress_callback:
        progress_callback("Classifying files...")

    classifier_manager = FileClassificationManager(data_source)
    sorted_df = classifier_manager.classify_files(records_df)

    if time_method is None:
        time_method = TimeSortingDialog.ask_method()

    if progress_callback:
        progress_callback("Processing scans and correlating with echem...")

    scan_processor = ScanProcessor(time_method, data_source)
    scans, echem_df = scan_processor.process_scans(sorted_df)

    scan_dicts = [scan.to_dict() for scan in scans]

    return scan_dicts, echem_df, time_method.value


def make_neutron_arrays(scans: List[Dict[str, Any]], state: Any = None) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Build {scan_num: {measurement: {type: {x,y}}}} from neutron scans."""
    plot_data = {}
    reader = DATReader("neutron")

    for scan in scans:
        if not scan.get("neutron_files"):
            continue

        scan_data = {}

        for measurement_num, measurement_files in scan["neutron_files"].items():
            measurement_data = {}

            for key in ("tof", "d"):
                if key not in measurement_files:
                    continue
                path = measurement_files[key]

                def _load(p=path):
                    if state and hasattr(state, 'get_cached_data'):
                        return state.get_cached_data(p, lambda: reader.read(p))
                    return reader.read(p)

                label = "TOF" if key == "tof" else "d-spacing"
                data = _read_with_retry(_load, path, scan["scan_num"], label)
                if data is not None:
                    measurement_data[key] = {"x": data[:, 0], "y": data[:, 1]}

            if measurement_data:
                scan_data[measurement_num] = measurement_data

        if scan_data:
            plot_data[scan["scan_num"]] = scan_data

    return plot_data


def _read_with_retry(reader_func: Callable, path: str, scan_num: int,
                     data_label: str) -> Optional[np.ndarray]:
    """Retry reader_func up to FILE_READ_RETRIES on failure."""
    import time
    last_error = None
    for attempt in range(FILE_READ_RETRIES + 1):
        try:
            return reader_func()
        except Exception as e:
            last_error = e
            if attempt < FILE_READ_RETRIES:
                logger.debug(f"Retry {attempt + 1}/{FILE_READ_RETRIES} for "
                             f"{data_label} scan {scan_num} ({path}): {e}")
                time.sleep(FILE_READ_RETRY_DELAY)
    logger.error(f"Failed to read {data_label} for scan {scan_num} "
                 f"after {FILE_READ_RETRIES + 1} attempts: {last_error}")
    return None


def make_oned_arrays(scans: List[Dict[str, Any]], state: Any = None) -> Dict[int, Dict[str, Any]]:
    """Build {scan_num: {x, y, timestamp, echem, current}} from 1D scans."""
    plot_data = {}
    reader_factory = DataReaderFactory()

    for scan in scans:
        if not scan.get("oned"):
            continue

        def _load(path=scan["oned"]):
            if state and hasattr(state, 'get_cached_data'):
                return state.get_cached_data(
                    path, lambda: reader_factory.read_file(path))
            return reader_factory.read_file(path)

        data = _read_with_retry(_load, scan["oned"], scan["scan_num"], "1D data")

        if data is not None:
            plot_data[scan["scan_num"]] = {
                "x": data[:, 0],
                "y": data[:, 1],
                "timestamp": scan["timestamp"],
                "echem": scan.get("echem"),
                "current": scan.get("current"),
            }
        else:
            plot_data[scan["scan_num"]] = {"error": True}

    return plot_data


def make_twod_arrays(scans: List[Dict[str, Any]], state: Any = None) -> Dict[int, Dict[str, Any]]:
    """Build {scan_num: {image, timestamp, echem, current}} from 2D scans."""
    plot_data = {}
    reader_factory = DataReaderFactory()

    # Configure HDFReader with user-selected display size
    hdf_reader = reader_factory.READERS.get(FileType.HDF)
    if hdf_reader and state and hasattr(state, 'synchrotron_max_size'):
        hdf_reader.max_display_size = state.synchrotron_max_size

    for scan in scans:
        if not scan.get("twod"):
            continue

        def _load(path=scan["twod"]):
            if state and hasattr(state, 'get_cached_data'):
                return state.get_cached_data(
                    path, lambda: reader_factory.read_file(path))
            return reader_factory.read_file(path)

        image = _read_with_retry(_load, scan["twod"], scan["scan_num"], "2D data")

        if image is not None:
            plot_data[scan["scan_num"]] = {
                "image": image,
                "timestamp": scan["timestamp"],
                "echem": scan.get("echem"),
                "current": scan.get("current"),
            }
        else:
            plot_data[scan["scan_num"]] = {"error": True}

    return plot_data


def make_echem_arrays(echem_df: Optional[pd.DataFrame],
                      time_method: str = "absolute") -> Dict[str, Union[np.ndarray, List]]:
    """Convert echem DataFrame to {x, y, current, timestamps} arrays."""
    if echem_df is None or echem_df.empty:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "current": np.array([]),
            "timestamps": []
        }

    if time_method == TimeMethod.RELATIVE.value or time_method == "relative":
        time_seconds = []
        for ts in echem_df["timestamp"]:
            if isinstance(ts, str) and ":" in ts and "-" not in ts:
                parts = ts.split(":")
                if len(parts) == 3:
                    try:
                        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        time_seconds.append(seconds)
                    except ValueError:
                        time_seconds.append(0)
                else:
                    time_seconds.append(0)
            else:
                time_seconds.append(0)
        time_seconds = np.array(time_seconds)
    else:
        try:
            timestamps = pd.to_datetime(echem_df["timestamp"])
            time_seconds = (timestamps - timestamps.min()).dt.total_seconds().values
        except Exception as e:
            logger.error(f"Error processing timestamps: {e}")
            time_seconds = np.arange(len(echem_df))

    echem_values = echem_df["echem_data"].values
    current_values = (echem_df["current"].values if "current" in echem_df.columns
                      else np.full(len(echem_df), np.nan))

    return {
        "x": time_seconds,
        "y": echem_values,
        "current": current_values,
        "timestamps": echem_df["timestamp"].tolist()
    }


def get_correlated_data(scans: List[Dict[str, Any]],
                        echem_df: Optional[pd.DataFrame],
                        scan_num: int,
                        state: Any = None) -> Optional[Dict[str, Any]]:
    """Load all data (1D, 2D, neutron, echem) for a single scan."""
    scan = next((s for s in scans if s["scan_num"] == scan_num), None)
    if not scan:
        return None

    reader_factory = DataReaderFactory()

    result = {
        "scan_num": scan_num,
        "timestamp": scan["timestamp"],
        "echem_value": scan.get("echem"),
        "current_value": scan.get("current")
    }

    # Standard XRD data
    if scan.get("oned"):
        def _load_oned(path=scan["oned"]):
            if state and hasattr(state, 'get_cached_data'):
                return state.get_cached_data(path, lambda: reader_factory.read_file(path))
            return reader_factory.read_file(path)

        data = _read_with_retry(_load_oned, scan["oned"], scan_num, "1D data")
        result["oned"] = {"x": data[:, 0], "y": data[:, 1]} if data is not None else None
    else:
        result["oned"] = None

    if scan.get("twod"):
        def _load_twod(path=scan["twod"]):
            if state and hasattr(state, 'get_cached_data'):
                return state.get_cached_data(path, lambda: reader_factory.read_file(path))
            return reader_factory.read_file(path)

        image = _read_with_retry(_load_twod, scan["twod"], scan_num, "2D data")
        result["twod"] = image
    else:
        result["twod"] = None

    # Neutron data
    if scan.get("neutron_files"):
        neutron_data = {}
        reader = DATReader("neutron")

        for measurement_num, measurement_files in scan["neutron_files"].items():
            measurement_data = {}

            for key in ("tof", "d"):
                if key not in measurement_files:
                    continue
                path = measurement_files[key]

                def _load_neutron(p=path):
                    if state and hasattr(state, 'get_cached_data'):
                        return state.get_cached_data(p, lambda: reader.read(p))
                    return reader.read(p)

                label = "TOF" if key == "tof" else "d-spacing"
                data = _read_with_retry(_load_neutron, path, scan_num, label)
                if data is not None:
                    measurement_data[key] = {"x": data[:, 0], "y": data[:, 1]}

            if measurement_data:
                neutron_data[measurement_num] = measurement_data

        if neutron_data:
            result["neutron"] = neutron_data

    return result


# ============================================================================
# UI Components
# ============================================================================

class TimeSortingDialog:
    """Tkinter dialog for choosing absolute vs. relative time."""

    @staticmethod
    def ask_method() -> TimeMethod:
        """Show modal dialog and return chosen TimeMethod."""
        import tkinter as tk

        dialog = tk.Toplevel()
        dialog.title("Time Sorting Method")
        dialog.geometry(WINDOW_SIZES['time'])
        dialog.transient()
        dialog.grab_set()

        dialog.configure(bg=OPERAXNTheme.COLORS['bg_primary'])

        dialog.protocol("WM_DELETE_WINDOW", lambda: [result.update({"method": None}), dialog.destroy()])

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        result = {"method": TimeMethod.ABSOLUTE}

        tk.Label(
            dialog,
            text="How should timestamps be handled?",
            font=OPERAXNTheme.FONTS['heading'],
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary']
        ).pack(pady=20)

        var = tk.StringVar(value=TimeMethod.ABSOLUTE.value)

        options = [
            (TimeMethod.ABSOLUTE.value, "Absolute time (use actual timestamps)"),
            (TimeMethod.RELATIVE.value, "Relative time (XRD and echem each start at 00:00:00)")
        ]

        for value, text in options:
            tk.Radiobutton(
                dialog,
                text=text,
                variable=var,
                value=value,
                font=OPERAXNTheme.FONTS['body'],
                bg=OPERAXNTheme.COLORS['bg_primary'],
                fg=OPERAXNTheme.COLORS['text_primary'],
                activebackground=OPERAXNTheme.COLORS['bg_primary'],
                activeforeground=OPERAXNTheme.COLORS['accent_primary'],
                selectcolor=OPERAXNTheme.COLORS['bg_tertiary']
            ).pack(anchor="w", padx=30, pady=5)

        def confirm():
            result["method"] = TimeMethod(var.get())
            dialog.destroy()

        button_frame = tk.Frame(dialog, bg=OPERAXNTheme.COLORS['bg_primary'])
        button_frame.pack(pady=10)

        confirm_btn = tk.Button(
            button_frame,
            text="OK",
            command=confirm,
            width=12,
            bg=OPERAXNTheme.COLORS['accent_primary'],
            fg=OPERAXNTheme.COLORS['bg_primary'],
            font=OPERAXNTheme.FONTS['button'],
            relief=tk.FLAT,
            cursor='hand2'
        )
        confirm_btn.pack()

        confirm_btn.bind("<Enter>", lambda e: confirm_btn.config(bg=OPERAXNTheme.COLORS['accent_hover']))
        confirm_btn.bind("<Leave>", lambda e: confirm_btn.config(bg=OPERAXNTheme.COLORS['accent_primary']))

        dialog.wait_window()
        return result["method"]


# ============================================================================
# Performance Utility Functions
# ============================================================================

def clear_global_cache() -> None:
    """Clear the global file cache."""
    global _file_cache
    _file_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Return cache size, item count, and limit."""
    global _file_cache
    return {
        "size_bytes": _file_cache.current_size,
        "size_mb": _file_cache.current_size / (1024 * 1024),
        "num_items": len(_file_cache.cache),
        "max_size_mb": MAX_CACHE_SIZE_MB
    }
