"""Nexus Generator."""

import logging
import os
import re
import shutil
import tempfile
import threading
import tkinter as tk
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import fabio
import h5py
import numpy as np
import pandas as pd

# --- Logging ---

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Constants ---

APP_NAME = "nexusgen"
APP_VERSION = "1.0.0"

# --- Configuration ---

ECHEM_TIME_TOLERANCE = 300  # seconds
MAX_WORKERS = 8
PARALLEL_PROCESSING = True
BATCH_SIZE = 20
SYNCHROTRON_MAX_DISPLAY_SIZE = 4096
TARGET_DISPLAY_PIXELS = 2048 * 2048
MAX_DATASET_ELEMENTS = 100_000_000

# Header fields excluded from global metadata extraction
EDF_EXCLUDE_FIELDS = {
    'Date', 'ExposureTime', 'Image', 'Monitor', 'Intensity1', 'title',
    'SumForIntensity1', 'TransmittedFlux', 'Saturation',
    'pilai0', 'pilai1', 'pilct0', 'pilct1', 'pilroi0', 'pilroi1', 'Pil_Roi0'
}
NXS_EXCLUDE_FIELDS = {'start_time', 'end_time', 'count_time', 'scan_identifier'}


# --- Utility Functions ---

def convert_xlsx_to_txt(xlsx_path: str) -> str:
    """Convert .xlsx to tab-delimited .txt; returns original path on failure."""
    if not xlsx_path.lower().endswith('.xlsx'):
        return xlsx_path

    try:
        df = pd.read_excel(xlsx_path)
        base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
        fd, txt_path = tempfile.mkstemp(prefix=f"{base_name}_", suffix=".txt")
        os.close(fd)
        df.to_csv(txt_path, sep='\t', index=False)
        logger.info(f"Converted {os.path.basename(xlsx_path)} to txt format")
        return txt_path
    except Exception as e:
        logger.error(f"Error converting xlsx file {xlsx_path}: {e}")
        return xlsx_path


def convert_csv_to_txt(csv_path: str) -> str:
    """Convert .csv to tab-delimited .txt; returns original path on failure."""
    if not csv_path.lower().endswith('.csv'):
        return csv_path

    try:
        df = pd.read_csv(csv_path)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        fd, txt_path = tempfile.mkstemp(prefix=f"{base_name}_", suffix=".txt")
        os.close(fd)
        df.to_csv(txt_path, sep='\t', index=False)
        logger.info(f"Converted {os.path.basename(csv_path)} to txt format")
        return txt_path
    except Exception as e:
        logger.error(f"Error converting csv file {csv_path}: {e}")
        return csv_path


def _nan_to_none(value: Any) -> Any:
    """Convert NaN/NaT to None for safe HDF5 storage."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


# --- Data Models ---

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
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass
class NXSConfig:
    echem_time_tolerance: int = ECHEM_TIME_TOLERANCE
    max_workers: int = MAX_WORKERS
    parallel_processing: bool = PARALLEL_PROCESSING
    synchrotron_max_display_size: int = 0
    include_2d_images: bool = False


config = NXSConfig()


@dataclass
class FileRecord:
    """Single file with its classification and extracted metadata."""
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


# --- Data Readers ---

class DataReader(ABC):
    """Base class for all file format readers."""

    @abstractmethod
    def _read_impl(self, path: str) -> np.ndarray:
        pass

    def read(self, path: str) -> np.ndarray:
        return self._read_impl(path)


class DATReader(DataReader):
    """Reads .dat files (XRD or neutron) as two-column arrays."""

    def __init__(self, data_type: str = "xrd"):
        super().__init__()
        self.data_type = data_type

    def _read_impl(self, path: str) -> np.ndarray:
        try:
            # Fast path: numpy bulk load
            try:
                data = np.loadtxt(path, comments='#')
                if data.ndim == 1:
                    data = data.reshape(-1, 2)
                return data
            except (ValueError, IOError, OSError) as e:
                logger.debug(f"NumPy loadtxt failed for {path}: {e}")

            # Slow path: line-by-line parsing
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            data_start = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith("#") and line.strip():
                    data_start = i
                    break

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
    """Reads .edf detector images as 2D arrays with floor clipping."""

    def _read_impl(self, path: str) -> np.ndarray:
        try:
            arr = np.asarray(fabio.open(path).data.astype(float))
            pos = arr > 0
            floor = float(arr[pos].min()) if pos.any() else 0.0
            return np.clip(arr, floor, None)
        except Exception as e:
            raise IOError(f"Error reading EDF file {path}: {e}")


class XYReader(DataReader):
    """Reads .xy integrated diffraction data as two-column arrays."""

    def _read_impl(self, path: str) -> np.ndarray:
        try:
            data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            return data
        except Exception as e:
            raise IOError(f"Error reading XY file {path}: {e}")


class HDFReader(DataReader):
    """Reads HDF5 detector images with auto-discovery and optional downsampling."""

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
        (2880, 2881),
        (2048, 2048),
        (1024, 1024),
        (512, 512),
    ]

    def __init__(self):
        super().__init__()
        self.original_shape: Optional[Tuple[int, ...]] = None

    def _read_impl(self, path: str) -> np.ndarray:
        try:
            with h5py.File(path, 'r', swmr=True) as f:
                data = self._find_data_array(f)

                if data is None:
                    raise ValueError("No suitable data found in HDF file")

                data = self._process_data_shape(data)
                data = self._apply_floor_clipping(data)
                self.original_shape = data.shape
                data = self._downsample_if_needed(data)

                return data
        except Exception as e:
            raise IOError(f"Error reading HDF file {path}: {e}")

    def _find_data_array(self, h5file: h5py.File) -> Optional[np.ndarray]:
        """Try known paths first, then fall back to largest 2D/3D dataset."""
        for data_path in self.COMMON_DATA_PATHS:
            if data_path in h5file:
                dataset = h5file[data_path]
                if dataset.size > MAX_DATASET_ELEMENTS:
                    return self._sample_large_dataset(dataset)
                else:
                    return np.array(dataset)

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
        """Stride-sample datasets that exceed MAX_DATASET_ELEMENTS."""
        if dataset.ndim == 3:
            slice_size = dataset.shape[1] * dataset.shape[2]
        else:
            slice_size = dataset.shape[0] * dataset.shape[1]

        if slice_size > MAX_DATASET_ELEMENTS:
            step = max(1, int(np.sqrt(slice_size / TARGET_DISPLAY_PIXELS)))
            if dataset.ndim == 2:
                return np.array(dataset[::step, ::step])
            else:
                return np.array(dataset[0, ::step, ::step])

        if dataset.ndim == 3:
            return np.array(dataset[0])
        return np.array(dataset)

    def _process_data_shape(self, data: np.ndarray) -> np.ndarray:
        """Ensure output is 2D: take first frame of 3D, reshape 1D."""
        if data.ndim == 3:
            return data[0]
        elif data.ndim == 1:
            return self._reshape_1d_data(data)
        elif data.ndim == 2:
            return data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

    def _reshape_1d_data(self, data: np.ndarray) -> np.ndarray:
        """Reshape 1D data using known detector dimensions or square root."""
        for height, width in self.DETECTOR_SIZES:
            if data.size == height * width:
                return data.reshape((height, width))

        sqrt_size = int(np.sqrt(data.size))
        if sqrt_size * sqrt_size == data.size:
            return data.reshape((sqrt_size, sqrt_size))

        raise ValueError(f"Cannot determine shape for 1D data of size {data.size}")

    @staticmethod
    def _apply_floor_clipping(data: np.ndarray) -> np.ndarray:
        """Clip negative noise to the minimum positive value."""
        pos = data > 0
        if pos.any():
            floor = float(data[pos].min())
            return np.clip(data, floor, None)
        return data

    def _downsample_if_needed(self, data: np.ndarray) -> np.ndarray:
        max_size = config.synchrotron_max_display_size
        if max_size <= 0:
            return data

        height, width = data.shape
        if height > max_size or width > max_size:
            return self._downsample_for_display(data, max_size)
        return data

    @staticmethod
    def _downsample_for_display(data: np.ndarray, max_size: int) -> np.ndarray:
        """Stride-downsample to fit within max_size × max_size."""
        height, width = data.shape

        scale_factor = max(
            height / max_size,
            width / max_size
        )

        if scale_factor <= 1:
            return data

        step = int(np.ceil(scale_factor))
        return data[::step, ::step]


class DataReaderFactory:
    """Returns the correct reader for a given file extension."""

    READERS: Dict[FileType, DataReader] = {
        FileType.EDF: EDFReader(),
        FileType.DAT: DATReader("xrd"),
        FileType.XY: XYReader(),
        FileType.HDF: HDFReader(),
    }

    NEUTRON_READER = DATReader("neutron")

    @classmethod
    def get_reader(cls, file_path: str, is_neutron: bool = False) -> DataReader:
        if is_neutron and file_path.endswith('.dat'):
            return cls.NEUTRON_READER

        ext = os.path.splitext(file_path)[1].lower()
        file_type = FileType(ext) if ext in SUPPORTED_EXTENSIONS else None

        if file_type and file_type in cls.READERS:
            return cls.READERS[file_type]

        raise ValueError(f"No reader available for file type: {ext}")

    @classmethod
    def read_file(cls, file_path: str, is_neutron: bool = False) -> np.ndarray:
        reader = cls.get_reader(file_path, is_neutron)
        return reader.read(file_path)


# --- File Classification ---

class FileClassifierBase(ABC):
    """Base class: returns (data_type, timestamp, exposure_time) for a file."""

    @abstractmethod
    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        pass


class DATClassifier(FileClassifierBase):
    """Extracts 1D type, timestamp, and exposure from .dat headers."""

    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
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
    """Extracts 2D type, timestamp, and exposure from .edf headers."""

    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
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
    """Classifies .txt as electrochemistry data or neutron logbook metadata."""

    ECHEM_KEYWORDS = ["time", "absolute", "ecell", "voltage", "current", "i/", "ewe", "v/"]

    def classify(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
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

            # Check header for echem keywords
            first_line_lower = lines[0].lower()
            if any(keyword in first_line_lower for keyword in self.ECHEM_KEYWORDS):
                return DataType.ECHEM.value, None, None

            # Check for neutron logbook format: numeric scan ID + weekday timestamps
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
    """Runs classifiers on a DataFrame of files, populating type and metadata columns."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE):
        self.data_source = data_source
        self.classifiers: Dict[FileType, FileClassifierBase] = {
            FileType.DAT: DATClassifier(),
            FileType.EDF: EDFClassifier(),
            FileType.TXT: TXTClassifier()
        }

    def classify_files(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure all classification columns exist
        for col in ["oned", "twod", "echem", "neutron_meta", "source_nxs"]:
            if col not in df.columns:
                df[col] = pd.array([None] * len(df), dtype="object")
            else:
                df[col] = df[col].astype("object")
        if "exposure_time" not in df.columns:
            df["exposure_time"] = pd.array([None] * len(df), dtype="object")
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.array([None] * len(df), dtype="object")

        for idx, row in df.iterrows():
            # Skip already-classified rows
            if pd.notna(row["oned"]) or pd.notna(row["twod"]) or pd.notna(row["neutron_meta"]):
                continue

            file_path = row["path"]
            ext = os.path.splitext(file_path)[1].lower()

            if self.data_source == DataSourceType.NEUTRON:
                # Neutron mode: only classify .txt files
                if ext == '.txt':
                    file_type = FileType(ext) if ext in SUPPORTED_EXTENSIONS else None
                    if file_type and file_type in self.classifiers:
                        classifier = self.classifiers[file_type]
                        data_type, timestamp, exposure_time = classifier.classify(file_path)

                        if data_type == DataType.NEUTRON_META.value:
                            df.at[idx, "neutron_meta"] = file_path
                        elif data_type == DataType.ECHEM.value:
                            df.at[idx, "echem"] = file_path

                        if timestamp:
                            df.at[idx, "timestamp"] = timestamp
                        if exposure_time is not None:
                            df.at[idx, "exposure_time"] = exposure_time

            else:
                file_type = FileType(ext) if ext in SUPPORTED_EXTENSIONS else None
                if file_type and file_type in self.classifiers:
                    classifier = self.classifiers[file_type]
                    data_type, timestamp, exposure_time = classifier.classify(file_path)

                    if data_type:
                        df.at[idx, data_type] = file_path
                        if timestamp:
                            df.at[idx, "timestamp"] = timestamp
                        if exposure_time is not None:
                            df.at[idx, "exposure_time"] = exposure_time

        return df


# --- Global Metadata Extraction ---

def extract_edf_global_metadata(edf_path: str) -> Dict[str, Any]:
    """Extract non-excluded header fields from the first EDF file."""
    try:
        image = fabio.open(edf_path)
        metadata = {}
        for key, value in image.header.items():
            if key in EDF_EXCLUDE_FIELDS:
                continue
            if value is None or str(value).strip() == '':
                continue
            metadata[key] = value
        logger.info(f"Extracted {len(metadata)} EDF metadata fields")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting EDF metadata: {e}")
        return {}


def extract_nxs_global_metadata(nxs_path: str) -> Dict[str, Any]:
    """Flatten all small datasets and attributes from a synchrotron .nxs file."""
    try:
        metadata: Dict[str, Any] = {}

        with h5py.File(nxs_path, 'r') as f:
            def extract_item(name: str, obj: Any) -> None:
                leaf = name.split('/')[-1]
                if leaf in NXS_EXCLUDE_FIELDS:
                    return
                if isinstance(obj, h5py.Dataset):
                    if obj.size > 1000:
                        return
                    if 'data' in name.lower():
                        return

                flat_name = name.replace('/', '_')

                for attr_key, attr_value in obj.attrs.items():
                    meta_key = f"{flat_name}_attr_{attr_key}"
                    decoded_value = _decode_h5_value(attr_value)
                    if decoded_value is not None:
                        metadata[meta_key] = decoded_value

                if isinstance(obj, h5py.Dataset):
                    try:
                        decoded_value = _decode_h5_value(obj[()])
                        if decoded_value is not None:
                            metadata[flat_name] = decoded_value
                    except Exception:
                        pass

            # Root-level attributes
            for root_attr_name, root_attr_val in f.attrs.items():
                decoded_val = _decode_h5_value(root_attr_val)
                if decoded_val is not None:
                    metadata[f"root_attr_{root_attr_name}"] = decoded_val

            f.visititems(extract_item)

        logger.info(f"Extracted {len(metadata)} NeXus metadata fields")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting NeXus metadata: {e}")
        return {}


def _decode_h5_value(value: Any) -> Any:
    """Decode HDF5 bytes/numpy scalars to native Python types; None if too large."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            item = value.item()
            if isinstance(item, bytes):
                return item.decode('utf-8', errors='ignore')
            return item
        elif value.size <= 10:
            decoded = []
            for x in value.flat:
                if isinstance(x, bytes):
                    decoded.append(x.decode('utf-8', errors='ignore'))
                else:
                    decoded.append(x)
            return decoded
        else:
            return None
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


# --- NeXus Metadata Extractor ---

class NexusMetadataExtractor:
    """Extracts timestamps and exposure times from synchrotron .nxs files."""

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
        try:
            with h5py.File(nxs_path, 'r') as f:
                metadata: Dict[str, Any] = {}

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

        except Exception:
            return None

    def _extract_timestamp(self, h5file: h5py.File) -> Optional[str]:
        for ts_path in self.TIMESTAMP_PATHS:
            if ts_path in h5file:
                timestamp_str = self._decode_value(h5file[ts_path][()])
                return self._parse_nexus_timestamp(timestamp_str)
        return None

    def _extract_exposure_time(self, h5file: h5py.File) -> Optional[float]:
        """Try explicit exposure datasets, then compute from start/end times."""
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

                    if 0 < exposure_seconds < 3600:
                        return exposure_seconds

        return None

    def _calculate_midpoint_timestamp(self, h5file: h5py.File) -> Optional[str]:
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
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    @staticmethod
    def _parse_nexus_timestamp(timestamp_str: str) -> str:
        """Normalise ISO/NeXus timestamps to 'YYYY-MM-DD HH:MM:SS'."""
        if 'T' in timestamp_str:
            base_time = timestamp_str.split('+')[0].split('Z')[0]
            if '.' in base_time:
                base_time = base_time.split('.')[0]
            return base_time.replace('T', ' ')
        return timestamp_str


# --- Synchrotron File Grouper ---

class SynchrotronFileGrouper:
    """Groups related .nxs, .hdf, and .xy files by scan ID."""

    def group_files(self, file_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        groups: Dict[str, Dict[str, str]] = {}
        nxs_to_id: Dict[str, str] = {}

        # First pass: index all .nxs files by scan ID
        for extracted_path, original_path in file_dict.items():
            basename = os.path.basename(extracted_path)
            ext = os.path.splitext(basename)[1].lower()

            if ext == '.nxs':
                scan_id = self._extract_scan_id(basename)
                if scan_id:
                    nxs_to_id[extracted_path] = scan_id

        # Second pass: assign each file to its group
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

    @staticmethod
    def _extract_scan_id(filename: str) -> Optional[str]:
        """Extract trailing numeric ID from filename."""
        base_name = os.path.splitext(filename)[0]
        base_name = base_name.replace('_integration', '')

        match = re.search(r'(\d+)(?!.*\d)', base_name)
        return match.group(1) if match else None

    def _determine_group_id(self, basename: str, ext: str, nxs_to_id: Dict[str, str]) -> Optional[str]:
        """Match a file to an existing .nxs scan group."""
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


# --- Neutron Data Processing ---

class NeutronMetadataParser:
    """Parses tab-delimited neutron logbook files into scan DataFrames."""

    @staticmethod
    def parse(path: str) -> Optional[pd.DataFrame]:
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

                        # Expect 5–7 digit numeric scan IDs
                        if not (scan_id.isdigit() and 5 <= len(scan_id) <= 7):
                            continue

                        # Find start and end timestamps by weekday prefix
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
    """Groups .dat neutron files by scan ID → measurement number → data type (tof/d)."""

    def group_neutron_files(self, file_list: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
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

                if scan_id not in groups:
                    groups[scan_id] = {}

                if measurement_num not in groups[scan_id]:
                    groups[scan_id][measurement_num] = {}

                groups[scan_id][measurement_num][data_type] = filepath

        logger.info(f"Grouped {len(groups)} scans")
        return groups

    @staticmethod
    def _extract_neutron_file_info(filename: str) -> Optional[Dict[str, str]]:
        """Parse scan ID, measurement number, and tof/d type from filename conventions."""
        name_no_ext = filename[:-4] if filename.endswith('.dat') else filename
        is_dspacing = '-d-' in name_no_ext or '_d_' in name_no_ext or name_no_ext.endswith(
            '_d') or name_no_ext.endswith('-d')

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

            # Pattern: 12345-1-d.dat (5-7 digit scan IDs).
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


# --- Echem Parser ---

class EchemParser:
    """Parses tab-delimited electrochemistry files into timestamp/voltage/current DataFrames."""

    COLUMN_PATTERNS = {
        "time": ["time", "date"],
        "voltage": ["voltage", "v/", "ecell", "ewe"],
        "current": ["current", "i/"]
    }

    def parse(self, path: str) -> Optional[pd.DataFrame]:
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

        except Exception:
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
                    f"Default column '{col_name}' at index {columns[col_name]} "
                    f"collides with detected column. Disabling."
                )
                columns[col_name] = -1

        return columns

    @staticmethod
    def _parse_data_lines(lines: List[str], columns: Dict[str, int]) -> List[Dict[str, Any]]:
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

            # Skip epoch-zero placeholder rows
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


# --- File Processor ---

class FileProcessor:
    """Collects, extracts, and pre-classifies files from paths, ZIPs, and directories."""

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None,
                 data_source: DataSourceType = DataSourceType.INHOUSE):
        self.progress_callback = progress_callback
        self.data_source = data_source
        self.tempdir: Optional[str] = None
        self.processed_files: set = set()
        self.synchrotron_grouper = SynchrotronFileGrouper()
        self.nexus_extractor = NexusMetadataExtractor()

    def __enter__(self) -> 'FileProcessor':
        self.tempdir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.tempdir and os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir, ignore_errors=True)

    def process_paths(self, selected_paths: List[str]) -> List[FileRecord]:
        """Main entry: collect files, group source-specific ones, process the rest."""
        all_files = self._collect_all_files(selected_paths)

        if not all_files:
            return []

        file_dict = {extracted: original for extracted, original in all_files}

        # Source-specific grouping
        if self.data_source == DataSourceType.NEUTRON:
            records = self._process_neutron_files(all_files)
        elif self.data_source == DataSourceType.SYNCHROTRON:
            synchrotron_groups = self.synchrotron_grouper.group_files(file_dict)
            records = self._process_synchrotron_groups(synchrotron_groups)
        else:
            records = []

        # Process ungrouped files
        remaining_files = []
        for extracted_path, original_path in all_files:
            if extracted_path not in self.processed_files:
                remaining_files.append((extracted_path, original_path))

        if remaining_files:
            if config.parallel_processing and len(remaining_files) > BATCH_SIZE:
                chunk_size = max(BATCH_SIZE, len(remaining_files) // config.max_workers)
                chunks = [remaining_files[i:i + chunk_size]
                          for i in range(0, len(remaining_files), chunk_size)]
                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    for chunk_records in executor.map(self._process_remaining_chunk, chunks):
                        records.extend(chunk_records)
            else:
                records.extend(self._process_remaining_chunk(remaining_files))

        return records

    def _process_neutron_files(self, all_files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Create records for all neutron .txt and .dat files."""
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
        """Recursively collect files from paths/directories, extract ZIPs, convert xlsx/csv."""
        all_files: List[Tuple[str, str]] = []

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

        all_files = self._convert_files_to_txt(all_files)

        logger.info(f"Collected {len(all_files)} total files")
        return all_files

    @staticmethod
    def _convert_files_to_txt(files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Convert .xlsx and .csv to tab-delimited .txt for uniform parsing."""
        converted_files: List[Tuple[str, str]] = []

        for extracted_path, original_path in files:
            lower_path = extracted_path.lower()
            if lower_path.endswith('.xlsx'):
                txt_path = convert_xlsx_to_txt(extracted_path)
                converted_files.append((txt_path, original_path))
            elif lower_path.endswith('.csv'):
                txt_path = convert_csv_to_txt(extracted_path)
                converted_files.append((txt_path, original_path))
            else:
                converted_files.append((extracted_path, original_path))

        return converted_files

    def _extract_zip_files(self, zip_path: str) -> List[Tuple[str, str]]:
        """Extract supported files from ZIP, skipping macOS metadata."""
        extracted: List[Tuple[str, str]] = []

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                members = archive.namelist()
                supported_exts = SUPPORTED_EXTENSIONS - {FileType.ZIP.value}

                for member in members:
                    if (member.endswith("/") or
                            member.startswith("__MACOSX") or
                            member.startswith(".")):
                        continue

                    member_ext = os.path.splitext(member)[1].lower()

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
        """Walk directory tree, collecting supported files and extracting nested ZIPs."""
        files: List[Tuple[str, str]] = []

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
                elif ext in SUPPORTED_EXTENSIONS:
                    files.append((file_path, file_path))

        return files

    def _process_synchrotron_groups(self, synchrotron_groups: Dict[str, Dict[str, str]]) -> List[FileRecord]:
        """Create records for each synchrotron scan group (requires .nxs anchor)."""
        records: List[FileRecord] = []

        for base_id, file_group in synchrotron_groups.items():
            if 'nxs' in file_group:
                record = self._create_synchrotron_record(base_id, file_group)
                if record:
                    records.append(record)
                    for file_path in file_group.values():
                        self.processed_files.add(file_path)

        return records

    def _process_remaining_chunk(self, files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Create simple records for ungrouped .dat, .edf, .txt files."""
        records: List[FileRecord] = []
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
        """Build a FileRecord for a synchrotron group, enriched with .nxs metadata."""
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
                exposure_time=None,
                source_nxs=nxs_path
            )

        return FileRecord(
            path=nxs_path,
            original_path=nxs_path,
            oned=file_group.get('xy'),
            twod=file_group.get('hdf'),
            timestamp=metadata.get('timestamp'),
            exposure_time=metadata.get('exposure_time'),
            source_nxs=nxs_path
        )


# --- Scan Processor ---

class ScanProcessor:
    """Builds the ordered scan list and correlates scans with echem data."""

    def __init__(self, time_method: TimeMethod = TimeMethod.ABSOLUTE,
                 data_source: DataSourceType = DataSourceType.INHOUSE):
        self.time_method = time_method
        self.data_source = data_source
        self.xrd_reference_time: Optional[pd.Timestamp] = None
        self.echem_reference_time: Optional[pd.Timestamp] = None
        self.neutron_reference_time: Optional[pd.Timestamp] = None
        self.echem_parser = EchemParser()
        self.neutron_parser = NeutronMetadataParser()

    def process_scans(self, df: pd.DataFrame) -> Tuple[List[Scan], pd.DataFrame]:
        """Main pipeline: parse echem → build scans → adjust timestamps → correlate."""
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
        """Parse all echem files, concatenate, and sort by timestamp."""
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
        """Match neutron .dat file groups to logbook entries, skip unmatched scans."""
        scan_list: List[Scan] = []

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
        """Build scan list from classified DataFrame, sorted by timestamp."""
        scan_list: List[Scan] = []

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
                    twod_exposure=_nan_to_none(row.get("exposure_time")),
                    source_nxs=_nan_to_none(row.get("path"))
                )
                scan_list.append(scan)
        else:
            # In-house: pair 1D and 2D by matching timestamps
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
                existing = None
                row_ts = _nan_to_none(row.get("timestamp"))
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
        if self.time_method == TimeMethod.RELATIVE:
            self._correlate_relative_time(scan_list, echem_df)
        else:
            self._correlate_absolute_time(scan_list, echem_df)

    def _correlate_relative_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Match scans to nearest echem point by relative offset from respective t=0."""
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

        for scan in scan_list:
            if not scan.timestamp_for_correlation:
                scan.echem_timestamp = None
                continue

            scan_relative_seconds = (scan.timestamp_for_correlation - reference_time).total_seconds()

            time_diffs = np.abs(echem_relative_seconds.values - scan_relative_seconds)
            nearest_idx = int(np.argmin(time_diffs))
            min_diff_seconds = time_diffs[nearest_idx]

            if min_diff_seconds < config.echem_time_tolerance:
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
        """Match scans to nearest echem point by absolute timestamp proximity."""
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

        for scan in scan_list:
            scan_time = scan.timestamp_for_correlation
            if not scan_time:
                scan.echem_timestamp = None
                continue

            if isinstance(scan_time, str):
                scan_time = pd.to_datetime(scan_time)

            # Skip scans outside echem window + tolerance
            if (scan_time < echem_start - pd.Timedelta(seconds=config.echem_time_tolerance) or
                    scan_time > echem_end + pd.Timedelta(seconds=config.echem_time_tolerance)):
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None
                continue

            time_diffs = abs(echem_timestamps - scan_time)
            nearest_idx = int(np.argmin(time_diffs.values))
            min_diff = time_diffs.iloc[nearest_idx]

            if min_diff.total_seconds() < config.echem_time_tolerance:
                scan.echem = float(echem_df.iloc[nearest_idx]["echem_data"])
                current_val = echem_df.iloc[nearest_idx]["current"]
                scan.current = float(current_val) if pd.notna(current_val) else None
                scan.echem_timestamp = str(echem_df.iloc[nearest_idx]["timestamp"])
            else:
                scan.echem = None
                scan.current = None
                scan.echem_timestamp = None

    def _apply_relative_time(self, scan_list: List[Scan], echem_df: pd.DataFrame) -> None:
        """Replace absolute timestamps with HH:MM:SS relative strings for display."""
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
            echem_df["timestamp"] = [self._format_relative_time(sec) for sec in relative_seconds]

    @staticmethod
    def _format_relative_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# --- NeXus File Writer ---

class NXSWriter:
    """Writes the final NeXus/HDF5 file with scan data, echem, and metadata."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE):
        self.data_source = data_source
        self.echem_parser = EchemParser()

    def write(self, output_path: str, scans: List[Scan], echem_df: pd.DataFrame,
              standard_echem_files: Optional[List[str]] = None) -> None:
        reader_factory = DataReaderFactory()

        with h5py.File(output_path, 'w') as f:
            # Root attributes
            f.attrs['NX_class'] = 'NXroot'
            f.attrs['file_name'] = os.path.basename(output_path)
            f.attrs['file_time'] = pd.Timestamp.now().isoformat()

            # --- Per-scan entries ---
            for scan in scans:
                scan_entry = f.create_group(f'scan_{scan.scan_num:04d}')
                scan_entry.attrs['NX_class'] = 'NXentry'
                scan_entry.attrs['scan_number'] = scan.scan_num

                # Scan metadata
                metadata = scan_entry.create_group('metadata')
                metadata.attrs['NX_class'] = 'NXcollection'

                if scan.echem is not None:
                    metadata.create_dataset('voltage (V)', data=scan.echem)

                if scan.current is not None:
                    metadata.create_dataset('current (mA)', data=scan.current)

                if scan.echem_timestamp:
                    metadata.create_dataset('voltage_timestamp', data=str(scan.echem_timestamp))

                if scan.timestamp:
                    metadata.create_dataset('scan_timestamp', data=str(scan.timestamp))

                if scan.timestamp_for_correlation:
                    metadata.create_dataset('midpoint_adjusted_timestamp',
                                            data=str(scan.timestamp_for_correlation))

                if scan.exposure_time is not None:
                    metadata.create_dataset('exposure_time', data=scan.exposure_time)

                # XRD data (1D and/or 2D)
                if scan.oned or scan.twod:
                    xrd_group = scan_entry.create_group('xrd_data')
                    xrd_group.attrs['NX_class'] = 'NXdata'

                    if scan.oned:
                        try:
                            data_1d = reader_factory.read_file(scan.oned)
                            xrd_group.create_dataset('oned_2theta', data=data_1d[:, 0])
                            xrd_group.create_dataset('oned_intensity', data=data_1d[:, 1])
                            xrd_group.attrs['oned_source_file'] = os.path.basename(scan.oned)
                        except Exception as e:
                            logger.error(f"Error reading 1D data for scan {scan.scan_num}: {e}")
                            xrd_group.attrs['oned_source_file'] = os.path.basename(scan.oned)

                    if scan.twod:
                        self._write_2d_data(xrd_group, scan, reader_factory)

                # Neutron data (TOF and d-spacing per bank)
                if scan.neutron_files:
                    neutron_group = scan_entry.create_group('neutron_data')
                    neutron_group.attrs['NX_class'] = 'NXdata'

                    if scan.neutron_start:
                        neutron_group.attrs['start_time'] = scan.neutron_start
                    if scan.neutron_end:
                        neutron_group.attrs['end_time'] = scan.neutron_end

                    for meas_num, meas_files in scan.neutron_files.items():
                        bank_group = neutron_group.create_group(f'bank_{meas_num}')
                        bank_group.attrs['measurement_number'] = meas_num

                        if 'tof' in meas_files:
                            try:
                                tof_data = reader_factory.read_file(meas_files['tof'], is_neutron=True)
                                bank_group.create_dataset('tof', data=tof_data[:, 0])
                                bank_group.create_dataset('tof_intensity', data=tof_data[:, 1])
                                bank_group.attrs['tof_source_file'] = os.path.basename(meas_files['tof'])
                            except Exception as e:
                                logger.error(f"Error reading TOF data: {e}")
                                bank_group.attrs['tof_source_file'] = os.path.basename(meas_files['tof'])

                        if 'd' in meas_files:
                            try:
                                d_data = reader_factory.read_file(meas_files['d'], is_neutron=True)
                                bank_group.create_dataset('d_spacing', data=d_data[:, 0])
                                bank_group.create_dataset('d_intensity', data=d_data[:, 1])
                                bank_group.attrs['d_source_file'] = os.path.basename(meas_files['d'])
                            except Exception as e:
                                logger.error(f"Error reading d-spacing data: {e}")
                                bank_group.attrs['d_source_file'] = os.path.basename(meas_files['d'])

            # --- Global metadata ---
            global_meta = f.create_group('global_metadata')
            global_meta.attrs['NX_class'] = 'NXcollection'
            global_meta.attrs['total_scans'] = len(scans)
            global_meta.attrs['data_source'] = self.data_source.value
            global_meta.attrs['generator'] = APP_NAME
            global_meta.attrs['generator_version'] = APP_VERSION

            if config.include_2d_images:
                global_meta.attrs['twod_included'] = True
                if config.synchrotron_max_display_size > 0:
                    global_meta.attrs['twod_max_display_size'] = config.synchrotron_max_display_size
                else:
                    global_meta.attrs['twod_max_display_size'] = 'full_resolution'
            else:
                global_meta.attrs['twod_included'] = False

            # Extract instrument metadata from first scan's source file
            if scans:
                first_scan = scans[0]

                twod_path = first_scan.twod
                if twod_path and str(twod_path).lower().endswith('.edf'):
                    edf_meta = extract_edf_global_metadata(twod_path)
                    if edf_meta:
                        edf_group = global_meta.create_group('edf_metadata')
                        edf_group.attrs['NX_class'] = 'NXcollection'
                        edf_group.attrs['source_file'] = os.path.basename(twod_path)
                        for key, value in edf_meta.items():
                            if value is not None:
                                try:
                                    edf_group.create_dataset(key, data=value)
                                except Exception as e:
                                    logger.debug(f"Could not write EDF field '{key}': {e}")

                if self.data_source == DataSourceType.SYNCHROTRON:
                    nxs_path = first_scan.source_nxs
                    if nxs_path and os.path.isfile(nxs_path):
                        nxs_meta = extract_nxs_global_metadata(nxs_path)
                        if nxs_meta:
                            nxs_group = global_meta.create_group('synchrotron_metadata')
                            nxs_group.attrs['NX_class'] = 'NXcollection'
                            nxs_group.attrs['source_file'] = os.path.basename(nxs_path)
                            for key, value in nxs_meta.items():
                                if value is not None:
                                    try:
                                        nxs_group.create_dataset(key, data=value)
                                    except Exception as e:
                                        logger.debug(f"Could not write NeXus field '{key}': {e}")

            # --- Operando electrochemistry (time-correlated) ---
            if not echem_df.empty:
                echem_group = f.create_group('operando_electrochemistry')
                echem_group.attrs['NX_class'] = 'NXdata'

                # Use .to_numpy(dtype=object) to avoid PyArrow-backed arrays in pandas >= 3.0
                timestamps = echem_df['timestamp'].astype(str).to_numpy(dtype=object)
                echem_group.create_dataset('timestamps', data=timestamps)
                echem_group.create_dataset('voltage (V)',
                                           data=echem_df['echem_data'].to_numpy(dtype=float))

                if 'current' in echem_df.columns:
                    current_values = echem_df['current'].to_numpy(dtype=float)
                    echem_group.create_dataset('current (mA)', data=current_values)

            # --- Standard electrochemistry (standalone files) ---
            if standard_echem_files:
                std_echem_container = f.create_group('standard_electrochemistry')
                std_echem_container.attrs['NX_class'] = 'NXcollection'
                std_echem_container.attrs['num_files'] = len(standard_echem_files)

                for idx, std_echem_path in enumerate(standard_echem_files, start=1):
                    if not os.path.isfile(std_echem_path):
                        continue

                    standard_echem_df = self.echem_parser.parse(std_echem_path)
                    if standard_echem_df is None or standard_echem_df.empty:
                        continue

                    group_name = f'file_{idx:03d}'
                    std_group = std_echem_container.create_group(group_name)
                    std_group.attrs['NX_class'] = 'NXdata'
                    std_group.attrs['source_file'] = os.path.basename(std_echem_path)

                    std_timestamps = standard_echem_df['timestamp'].astype(str).to_numpy(dtype=object)
                    std_group.create_dataset('timestamps', data=std_timestamps)
                    std_group.create_dataset('voltage (V)',
                                             data=standard_echem_df['echem_data'].to_numpy(dtype=float))

                    if 'current' in standard_echem_df.columns:
                        current_vals = standard_echem_df['current'].to_numpy(dtype=float)
                        if not np.all(np.isnan(current_vals)):
                            std_group.create_dataset('current (mA)', data=current_vals)

                    logger.info(f"Added standard electrochemistry file {idx}:"
                                f" {os.path.basename(std_echem_path)} ({len(standard_echem_df)} points)")

        logger.info(f"NeXus file successfully written to {output_path}")

    @staticmethod
    def _write_2d_data(xrd_group: h5py.Group, scan: Scan,
                       reader_factory: DataReaderFactory) -> None:
        """Embed or reference 2D detector image depending on config."""
        if not scan.twod:
            return

        twod_path = str(scan.twod)
        basename = os.path.basename(twod_path)
        ext = os.path.splitext(basename)[1].lower()

        is_hdf = (ext == ".hdf")
        is_edf = (ext == ".edf")

        # Always store source reference
        xrd_group.attrs["twod_source"] = basename
        xrd_group.attrs["twod_is_hdf"] = is_hdf
        xrd_group.attrs["twod_is_edf"] = is_edf

        if config.synchrotron_max_display_size > 0:
            xrd_group.attrs["twod_max_display_size"] = config.synchrotron_max_display_size
        else:
            xrd_group.attrs["twod_max_display_size"] = "full_resolution"

        if not config.include_2d_images:
            xrd_group.attrs["twod_embedded"] = False
            return

        try:
            if is_hdf:
                hdf_reader = HDFReader()
                data_2d = hdf_reader.read(twod_path)
                xrd_group.attrs["twod_original_shape"] = hdf_reader.original_shape
            else:
                data_2d = reader_factory.read_file(twod_path)
                xrd_group.attrs["twod_original_shape"] = data_2d.shape

            xrd_group.create_dataset("twod_image", data=data_2d)
            xrd_group.attrs["twod_embedded"] = True

        except Exception as e:
            logger.error(f"Error embedding 2D data for scan {scan.scan_num}: {e}")
            xrd_group.attrs["twod_embedded"] = False


# --- NeXus Generator ---

class NXSGenerator:
    """Top-level orchestrator: collect → classify → correlate → write."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE):
        self.data_source = data_source
        self.writer = NXSWriter(data_source)

    def generate_nxs(self, input_paths: List[str],
                     output_path: str,
                     time_method: TimeMethod = TimeMethod.ABSOLUTE,
                     progress_callback: Optional[Callable[[str], None]] = None,
                     standard_echem_files: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        try:
            with FileProcessor(progress_callback, self.data_source) as processor:
                records = processor.process_paths(input_paths)

            if not records:
                return False, ["No files found to process"]

            records_df = pd.DataFrame([r.__dict__ for r in records])

            for col in ["oned", "twod", "echem", "neutron_meta", "timestamp",
                        "source_nxs", "path", "original_path"]:
                if col in records_df.columns:
                    records_df[col] = records_df[col].astype("object")

            if progress_callback:
                progress_callback("Classifying files...")

            classifier = FileClassificationManager(self.data_source)
            sorted_df = classifier.classify_files(records_df)

            if progress_callback:
                progress_callback("Processing scans and correlating with echem...")

            scan_processor = ScanProcessor(time_method, self.data_source)
            scans, echem_df = scan_processor.process_scans(sorted_df)

            validation_errors = self._validate_for_nxs(scans, echem_df)

            if any("No scan data" in e or "No XRD" in e for e in validation_errors):
                return False, validation_errors

            if progress_callback:
                progress_callback("Writing NeXus file...")

            self.writer.write(output_path, scans, echem_df, standard_echem_files)

            return True, validation_errors if validation_errors else ["NeXus file generated successfully"]

        except Exception as e:
            logger.error(f"Error generating NeXus file: {e}")
            return False, [f"Error: {str(e)}"]

    @staticmethod
    def _validate_for_nxs(scans: List[Scan], echem_df: pd.DataFrame) -> List[str]:
        """Return validation errors; empty list means valid."""
        errors: List[str] = []

        if not scans:
            errors.append("No scan data available")

        if echem_df.empty:
            errors.append("Warning: No echem data available")

        has_xrd = any(scan.oned or scan.twod for scan in scans)
        has_neutron = any(scan.neutron_files for scan in scans)

        if not has_xrd and not has_neutron:
            errors.append("No XRD or neutron data found in scans")

        return errors


# --- GUI ---

class NXSGeneratorGUI:
    """Tkinter interface for selecting inputs, options, and triggering generation."""

    def __init__(self, root: tk.Tk):
        self.display_size_combo = None
        self.include_2d_check: Optional[ttk.Checkbutton] = None
        self.progress_label: Optional[ttk.Label] = None
        self.generate_btn: Optional[ttk.Button] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.root = root
        self.root.title("NeXus File Generator")
        self.root.geometry("600x620")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.standard_echem_files: List[str] = []
        self.standard_echem_original_files: List[str] = []
        self.standard_echem_display = tk.StringVar()
        self.data_source = tk.StringVar(value="inhouse")
        self.time_method = tk.StringVar(value="absolute")
        self.progress_text = tk.StringVar(value="Ready")
        self.include_2d_images = tk.BooleanVar(value=False)
        self.max_display_size = tk.StringVar(value="No downsampling")

        self.setup_ui()

    def setup_ui(self) -> None:

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        title = ttk.Label(main_frame, text="NeXus File Generator",
                          font=('Helvetica', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Input path
        ttk.Label(main_frame, text="Input Path:", font=('Helvetica', 10, 'bold')).grid(
            row=1, column=0, sticky=tk.W, pady=(10, 5))

        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)

        ttk.Entry(input_frame, textvariable=self.input_path).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(input_frame, text="Select File/Zip",
                   command=self.select_file).grid(row=0, column=1, padx=(0, 5))

        ttk.Button(input_frame, text="Select Directory",
                   command=self.select_directory).grid(row=0, column=2)

        # Output path
        ttk.Label(main_frame, text="Output NeXus File:", font=('Helvetica', 10, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=(10, 5))

        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)

        ttk.Entry(output_frame, textvariable=self.output_path).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(output_frame, text="Save As...",
                   command=self.select_output).grid(row=0, column=1)

        # Data source
        ttk.Label(main_frame, text="Data Source Type:", font=('Helvetica', 10, 'bold')).grid(
            row=5, column=0, sticky=tk.W, pady=(10, 5))

        source_frame = ttk.Frame(main_frame)
        source_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        ttk.Radiobutton(source_frame, text="In-House",
                        variable=self.data_source, value="inhouse",
                        command=self._on_source_change).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(source_frame, text="Synchrotron",
                        variable=self.data_source, value="synchrotron",
                        command=self._on_source_change).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(source_frame, text="Neutron",
                        variable=self.data_source, value="neutron",
                        command=self._on_source_change).grid(row=0, column=2, padx=5)

        # Time method
        ttk.Label(main_frame, text="Time Sorting Method:", font=('Helvetica', 10, 'bold')).grid(
            row=7, column=0, sticky=tk.W, pady=(10, 5))

        time_frame = ttk.Frame(main_frame)
        time_frame.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        ttk.Radiobutton(time_frame, text="Absolute time (use actual timestamps)",
                        variable=self.time_method, value="absolute").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(time_frame, text="Relative time (start from 00:00:00)",
                        variable=self.time_method, value="relative").grid(row=0, column=1, padx=5)

        # 2D image options
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="2D Image Settings (Optional):", font=('Helvetica', 10, 'bold')).grid(
            row=10, column=0, sticky=tk.W, pady=(5, 5))

        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        self.include_2d_check = ttk.Checkbutton(
            options_frame,
            text="Include 2D detector images in NeXus",
            variable=self.include_2d_images,
            command=self._update_2d_options
        )
        self.include_2d_check.grid(row=0, column=0, padx=5)

        self.display_size_combo = ttk.Combobox(
            options_frame,
            textvariable=self.max_display_size,
            values=["No downsampling", "4096", "2048", "1024", "512"],
            state="readonly"
        )
        self.display_size_combo.grid(row=0, column=1, sticky=tk.W, padx=(7.5, 2.5))

        ttk.Label(options_frame, text="Set max 2D data shape (n x n)").grid(
            row=0, column=2, sticky=tk.W)

        self._update_2d_options()

        # Standard echem (optional)
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=12, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Standard Electrochemistry (Optional):", font=('Helvetica', 10, 'bold')).grid(
            row=13, column=0, sticky=tk.W, pady=(5, 5))

        std_echem_frame = ttk.Frame(main_frame)
        std_echem_frame.grid(row=14, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        std_echem_frame.columnconfigure(0, weight=1)

        ttk.Entry(std_echem_frame, textvariable=self.standard_echem_display).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(std_echem_frame, text="Select Files",
                   command=self.select_standard_echem).grid(row=0, column=1, padx=(0, 5))

        ttk.Button(std_echem_frame, text="Clear",
                   command=self.clear_standard_echem).grid(row=0, column=2)

        # Progress
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=15, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(main_frame, text="Progress:", font=('Helvetica', 10, 'bold')).grid(
            row=16, column=0, sticky=tk.W, pady=(5, 5))

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=17, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_text)
        self.progress_label.grid(row=18, column=0, columnspan=3, pady=(0, 10))

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=19, column=0, columnspan=3, pady=(10, 0))

        self.generate_btn = ttk.Button(button_frame, text="Generate NeXus File",
                                       command=self.generate_nxs)
        self.generate_btn.grid(row=0, column=0, padx=(0, 5))

        ttk.Button(button_frame, text="Exit",
                   command=self.root.quit).grid(row=0, column=1)

    def _on_source_change(self) -> None:
        self._update_2d_options()

    def _update_2d_options(self) -> None:
        """Enable/disable 2D controls based on data source and checkbox state."""
        if self.include_2d_check is None:
            return
        has_2d = self.data_source.get() in ("synchrotron", "inhouse")

        self.include_2d_check.config(state='normal' if has_2d else 'disabled')
        if not has_2d:
            self.include_2d_images.set(False)

        combo_active = has_2d and self.include_2d_images.get()
        if hasattr(self, 'display_size_combo'):
            self.display_size_combo.config(state='readonly' if combo_active else 'disabled')

    def select_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[
                ("All Supported", ("*.dat", "*.edf", "*.txt", "*.xlsx", "*.csv", "*.hdf", "*.nxs", "*.xy", "*.zip")),
                ("DAT files", "*.dat"),
                ("EDF files", "*.edf"),
                ("Text files", "*.txt"),
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("HDF files", "*.hdf"),
                ("NXS files", "*.nxs"),
                ("XY files", "*.xy"),
                ("ZIP files", "*.zip"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            base = os.path.splitext(filename)[0]
            self.output_path.set(f"{base}_output.nxs")

    def select_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_path.set(directory)
            dir_name = os.path.basename(directory)
            parent_dir = os.path.dirname(directory)
            self.output_path.set(os.path.join(parent_dir, f"{dir_name}_output.nxs"))

    def select_output(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save NeXus File As",
            defaultextension=".nxs",
            filetypes=[
                ("NXS files", "*.nxs"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)

    def select_standard_echem(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Select Standard Electrochemistry Files",
            filetypes=[
                ("Supported files", "*.txt *.xlsx *.csv"),
                ("Text files", "*.txt"),
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if not filenames:
            return

        self.clear_standard_echem()

        for filepath in filenames:
            self.standard_echem_original_files.append(filepath)

            lower_path = filepath.lower()
            if lower_path.endswith('.xlsx'):
                converted_path = convert_xlsx_to_txt(filepath)
            elif lower_path.endswith('.csv'):
                converted_path = convert_csv_to_txt(filepath)
            else:
                converted_path = filepath

            self.standard_echem_files.append(converted_path)

        self.standard_echem_display.set(" | ".join(self.standard_echem_original_files))

    def clear_standard_echem(self) -> None:
        self.standard_echem_files = []
        self.standard_echem_original_files = []
        self.standard_echem_display.set("")

    def update_progress(self, message: str) -> None:
        self.root.after(0, lambda: self.progress_text.set(message))

    def generate_nxs(self) -> None:
        """Validate inputs and launch generation in a background thread."""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file or directory")
            return

        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file")
            return

        if self.generate_btn:
            self.generate_btn.config(state='disabled')
        if self.progress_bar:
            self.progress_bar.start()

        thread = threading.Thread(target=self._generate_worker)
        thread.start()

    def _generate_worker(self) -> None:
        """Background thread: run full generation pipeline and report results."""
        try:
            source_type = DataSourceType(self.data_source.get())
            time_method = TimeMethod(self.time_method.get())

            config.include_2d_images = self.include_2d_images.get()

            std_echem_files = self.standard_echem_files if self.standard_echem_files else None

            size_val = self.max_display_size.get()
            if size_val == "No downsampling":
                config.synchrotron_max_display_size = 0
            else:
                config.synchrotron_max_display_size = int(size_val)

            generator = NXSGenerator(source_type)

            success, messages = generator.generate_nxs(
                input_paths=[self.input_path.get()],
                output_path=self.output_path.get(),
                time_method=time_method,
                progress_callback=self.update_progress,
                standard_echem_files=std_echem_files
            )

            if success:
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"NeXus file generated successfully!\n\nOutput: {self.output_path.get()}\n\n"
                ))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    "Failed to generate NeXus file:\n\n" + "\n".join(messages)
                ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error",
                f"An unexpected error occurred:\n\n{str(e)}"
            ))

        finally:
            self.root.after(0, lambda: self.generate_btn.config(state='normal') if self.generate_btn else None)
            self.root.after(0, lambda: self.progress_bar.stop() if self.progress_bar else None)
            self.root.after(0, lambda: self.progress_text.set("Ready"))


# --- Entry Point ---

def main() -> None:
    root = tk.Tk()
    NXSGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
