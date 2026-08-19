"""
Raw data file readers for the core pipeline.

Interface contracts:
- `read_file` flags are keyword-only (`use_cache`, `is_neutron`) so a stray
  positional boolean cannot flip meaning between the two.
- `HDFReader` downsampling is an instance attribute (no mutable module-global).
- The fabio import is guarded so core still imports without it (EDF reads
  raise ImportError).
"""

import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from .config import (
    CACHE_ENABLED,
    MAX_CACHE_SIZE_MB,
    MAX_DATASET_ELEMENTS,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
    TARGET_DISPLAY_PIXELS,
)
from .model import SUPPORTED_EXTENSIONS, FileType

try:
    import fabio

    FABIO_AVAILABLE = True
except ImportError:
    fabio = None
    FABIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Cache
# ============================================================================

class FileCache:
    """Caches file data thread-safely, evicting least-accessed entries
    to stay within a byte-size budget."""

    def __init__(self, max_size_mb: int = MAX_CACHE_SIZE_MB) -> None:
        self.cache: Dict[str, Any] = {}
        self.max_size_bytes: int = max_size_mb * 1024 * 1024
        self.current_size: int = 0
        self.lock: threading.Lock = threading.Lock()
        self.access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Any]:
        """Cached value for key (bumping its access count), or None."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Store value, evicting least-accessed entries to stay within budget."""
        with self.lock:
            size_bytes = sys.getsizeof(value)

            while self.current_size + size_bytes > self.max_size_bytes and self.cache:
                self._evict_lfu()

            self.cache[key] = value
            self.current_size += size_bytes
            self.access_count[key] = 1

    def _evict_lfu(self) -> None:
        """Drop the entry with the lowest access count."""
        if not self.cache:
            return
        lfu_key = min(self.access_count.keys(), key=lambda k: self.access_count.get(k, 0))
        if lfu_key in self.cache:
            value = self.cache.pop(lfu_key)
            self.current_size -= sys.getsizeof(value)
            self.access_count.pop(lfu_key, None)

    def clear(self) -> None:
        """Empty the cache and reset accounting."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.current_size = 0


_file_cache = FileCache()


def clear_global_cache() -> None:
    """Empty the module-wide file cache (e.g. between sessions)."""
    _file_cache.clear()


# ============================================================================
# Readers
# ============================================================================

class DataReader(ABC):
    """Abstract base for file readers with optional caching."""

    @abstractmethod
    def _read_impl(self, path: str) -> np.ndarray:
        """Read the file into an array; raise IOError on failure."""

    def read(self, path: str, *, use_cache: bool = True) -> np.ndarray:
        """Read the file, serving from / storing to the global cache."""
        if not use_cache or not CACHE_ENABLED:
            return self._read_impl(path)

        cache_key = self._get_cache_key(path)
        cached_data = _file_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        data = self._read_impl(path)
        _file_cache.put(cache_key, data)
        return data

    @staticmethod
    def _get_cache_key(path: str) -> str:
        """Cache key from path, size, and mtime (edits never served stale)."""
        stat = os.stat(path)
        return f"{path}_{stat.st_size}_{stat.st_mtime}"


class DATReader(DataReader):
    """Reads .dat files (XRD or neutron) as X/Y(/error) column arrays."""

    def __init__(self, data_type: str = "xrd") -> None:
        super().__init__()
        self.data_type = data_type

    def _read_impl(self, path: str) -> np.ndarray:
        """Bulk numpy load with a tolerant line-parse fallback (keeps errors)."""
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
                        row = [float(parts[0]), float(parts[1])]
                        # Keep the uncertainty column when present (X, Y, E)
                        if len(parts) >= 3:
                            try:
                                row.append(float(parts[2]))
                            except ValueError:
                                pass
                        data.append(row)
                    except ValueError:
                        continue

            if not data:
                file_type = "neutron" if self.data_type == "neutron" else "DAT"
                raise ValueError(f"No valid data found in {file_type} file: {path}")

            # Rows must be rectangular; drop the error column if inconsistent
            ncols = min(len(r) for r in data)
            data = [r[:ncols] for r in data]

            return np.array(data, dtype=float)

        except Exception as e:
            file_type = "neutron DAT" if self.data_type == "neutron" else "DAT"
            raise IOError(f"Error reading {file_type} file {path}: {e}")


class EDFReader(DataReader):
    """Reads .edf detector images as 2D arrays with floor clipping."""

    def _read_impl(self, path: str) -> np.ndarray:
        """Load the image via fabio, clipping to the minimum positive value."""
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
    """Reads .xy integrated diffraction data as X/Y(/error) column arrays."""

    def _read_impl(self, path: str) -> np.ndarray:
        """Load whitespace-delimited columns via numpy."""
        try:
            data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            return data
        except Exception as e:
            raise IOError(f"Error reading XY file {path}: {e}")


class HDFReader(DataReader):
    """Reads HDF5 detector images with auto-discovery and optional downsampling.

    `max_display_size` is per-instance: set it before reading to control
    downsampling (0 disables it). No module-global configuration, but
    `DataReaderFactory.READERS` holds a shared instance — construct your
    own rather than mutating it.
    """

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

    def __init__(self, max_display_size: int = SYNCHROTRON_MAX_DISPLAY_SIZE) -> None:
        super().__init__()
        self.max_display_size = max_display_size
        self.original_shape: Optional[Tuple[int, ...]] = None

    def _get_cache_key(self, path: str) -> str:
        """Base key plus the downsampling setting (part of the result identity)."""
        return f"{super()._get_cache_key(path)}_{self.max_display_size}"

    def _read_impl(self, path: str) -> np.ndarray:
        """Locate the detector image, normalise to 2D, clip, and downsample."""
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
        """Stride-sample datasets that exceed MAX_DATASET_ELEMENTS.

        Works from `.shape` alone so nothing is read until the final slice.
        """
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
        """Apply the per-instance display-size cap (0 disables)."""
        if self.max_display_size <= 0:
            return data

        height, width = data.shape
        if height > self.max_display_size or width > self.max_display_size:
            return self._downsample_for_display(data, self.max_display_size)
        return data

    @staticmethod
    def _downsample_for_display(data: np.ndarray, max_size: int) -> np.ndarray:
        """Stride-downsample to fit within max_size x max_size."""
        height, width = data.shape

        scale_factor = max(
            height / max_size,
            width / max_size
        )

        if scale_factor <= 1:
            return data

        step = int(np.ceil(scale_factor))
        return data[::step, ::step]


# ============================================================================
# Factory
# ============================================================================

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
    def get_reader(cls, file_path: str, *, is_neutron: bool = False) -> DataReader:
        """Reader instance for the file's extension; ValueError if unsupported."""
        if is_neutron and file_path.endswith('.dat'):
            return cls.NEUTRON_READER

        ext = os.path.splitext(file_path)[1].lower()
        file_type = FileType(ext) if ext in SUPPORTED_EXTENSIONS else None

        if file_type and file_type in cls.READERS:
            return cls.READERS[file_type]

        raise ValueError(f"No reader available for file type: {ext}")

    @classmethod
    def read_file(cls, file_path: str, *, use_cache: bool = True,
                  is_neutron: bool = False) -> np.ndarray:
        """Read a data file with the appropriate reader."""
        reader = cls.get_reader(file_path, is_neutron=is_neutron)
        return reader.read(file_path, use_cache=use_cache)
