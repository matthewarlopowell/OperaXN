"""
OperaXN core — the pipeline shared by NeXus generation and the GUI.

Architecture: every raw-data upload is first generated into a canonical .nxs
file (`generate`), and the GUI runs purely by interpreting that file
(`nxs_reader.load`). Uploading a .nxs directly skips generation. One read path.
"""

from .config import (
    ECHEM_TIME_TOLERANCE,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
)
from .model import (
    DataSourceType,
    DataType,
    ExperimentModel,
    FileRecord,
    FileType,
    Scan,
    ScanData,
    TimeMethod,
)
from .readers import (
    DataReaderFactory,
    DATReader,
    EDFReader,
    HDFReader,
    XYReader,
    FABIO_AVAILABLE,
    clear_global_cache,
)
from .classify import (
    DATClassifier,
    EDFClassifier,
    FileClassificationManager,
    NeutronFileGrouper,
    NeutronMetadataParser,
    NexusMetadataExtractor,
    SynchrotronFileGrouper,
    TXTClassifier,
    extract_edf_scan_fields,
    parse_mantid_header,
)
from .profiles import get_profile
from .correlate import (
    EchemParser,
    ScanProcessor,
    format_relative_time,
    parse_relative_time,
)
from .nxs_writer import NXSWriter, write_standard_echem_group
from .nxs_reader import is_canonical_nxs, load
from .generate import (
    FileProcessor,
    cache_dir,
    convert_tabular_to_txt,
    default_cache_path,
    generate,
    process_raw,
    validate,
)

__all__ = [
    # Model
    'DataSourceType', 'DataType', 'ExperimentModel', 'FileRecord', 'FileType',
    'Scan', 'ScanData', 'TimeMethod',
    # Readers
    'DataReaderFactory', 'DATReader', 'EDFReader', 'HDFReader', 'XYReader',
    'FABIO_AVAILABLE', 'clear_global_cache',
    # Classification / harvesting
    'DATClassifier', 'EDFClassifier', 'FileClassificationManager',
    'NeutronFileGrouper', 'NeutronMetadataParser', 'NexusMetadataExtractor',
    'SynchrotronFileGrouper', 'TXTClassifier',
    'extract_edf_scan_fields', 'parse_mantid_header', 'get_profile',
    # Correlation
    'EchemParser', 'ScanProcessor', 'format_relative_time', 'parse_relative_time',
    # NeXus I/O
    'NXSWriter', 'write_standard_echem_group', 'is_canonical_nxs', 'load',
    # Pipeline
    'FileProcessor', 'cache_dir', 'convert_tabular_to_txt',
    'default_cache_path', 'generate', 'process_raw', 'validate',
    # Config
    'ECHEM_TIME_TOLERANCE', 'GENERATOR_NAME', 'GENERATOR_VERSION',
    'SYNCHROTRON_MAX_DISPLAY_SIZE',
]
