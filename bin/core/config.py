"""
Shared configuration for the OperaXN/NexusGen core pipeline.

GUI-specific settings (theme, window sizes, plot appearance) stay in the
respective application packages; only values the core pipeline needs live here.
"""

import multiprocessing

# --- Generator identity ---
# Written into every .nxs as @generator/@generator_version; the file layout
# itself is self-describing (the reader detects it structurally).

GENERATOR_NAME = "operaxn-core"
GENERATOR_VERSION = "2.0.0"

# --- NeXus application definitions ---
# Custom definitions shipped in definitions/ (schema v4 layout).

DEFINITION_VERSION = "1.0"
DEFINITION_URL_BASE = ("https://github.com/matthewarlopowell/OperaXN/"
                       "blob/main/definitions/")

# Route generated files through the pynxtools dataconverter (validating them
# against the NXoperando_* definitions at write time). Requires pynxtools
# with an 'operaxn' reader registered; when anything is missing or fails the
# plain NXSWriter file is kept, so the GUI never breaks.
USE_PYNXTOOLS_WRITER = False

# --- Correlation ---

ECHEM_TIME_TOLERANCE = 300  # seconds, echem-to-scan nearest-neighbour matching
MAX_EXPOSURE_TIME = 3600  # seconds, reject computed exposures above this
ECHEM_LOG_MIN_POINTS = 2  # min echem points in a scan window to emit an NXlog

# --- Scan identification ---

# ISIS-style neutron run numbers; enforced identically by the logbook parser,
# the file grouper, and any display code (see NeutronFileGrouper).
SCAN_ID_MIN_DIGITS = 5
SCAN_ID_MAX_DIGITS = 7

# --- Performance ---

CACHE_ENABLED = True
MAX_CACHE_SIZE_MB = 1000
PARALLEL_PROCESSING = True
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)
BATCH_SIZE = 20
PARALLEL_PROCESSING_THRESHOLD = 20  # min files to trigger parallel processing
FILE_READ_RETRIES = 2
FILE_READ_RETRY_DELAY = 0.5  # seconds

# --- 2D data handling ---

MAX_DATASET_ELEMENTS = 100_000_000  # threshold for sampling large HDF5 datasets
TARGET_DISPLAY_PIXELS = 2048 * 2048  # target pixel count when sampling
SYNCHROTRON_MAX_DISPLAY_SIZE = 4096  # default per-axis cap; 0 disables downsampling
