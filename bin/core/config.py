"""
Shared configuration for the OperaXN core pipeline.

GUI-specific settings (theme, window sizes, plot appearance) stay in the
respective application packages; only values the core pipeline needs live here.
"""

import multiprocessing

# --- Generator identity ---
# Written into every .nxs as a program_name dataset with @version; the file
# layout itself is self-describing (the reader detects it structurally).

GENERATOR_NAME = "operaxn-core"
GENERATOR_VERSION = "2.0.0"

# --- NeXus application definitions ---
# Custom definitions shipped in definitions/ (schema v4 layout).

# 1.0.0: first release, shipped with OperaXN 2.0.0; matches the published
# specification (Tan et al., ACS Energy Lett., Figure 4).
DEFINITION_VERSION = "1.0.0"
DEFINITION_URL_BASE = ("https://github.com/matthewarlopowell/OperaXN/"
                       "blob/main/definitions/")

# After writing, validate and rewrite generated files through the pynxtools
# dataconverter (checking them against the NXoperando_* definitions).
# Requires pynxtools with an 'operaxn' reader registered; when anything is
# missing or fails the plain NXSWriter file is kept, so the GUI never breaks.
USE_PYNXTOOLS_WRITER = False

# --- Correlation ---

ECHEM_TIME_TOLERANCE = 300  # seconds, echem-to-scan nearest-neighbour matching
MAX_EXPOSURE_TIME = 3600  # seconds, reject computed exposures above this
ECHEM_LOG_MIN_POINTS = 2  # min echem points in a scan window to emit an NXlog

# --- Scan identification ---

# ISIS-style neutron run numbers; enforced by the logbook parser and neutron
# file grouper (display code shares the rule via NeutronFileGrouper).
SCAN_ID_MIN_DIGITS = 5
SCAN_ID_MAX_DIGITS = 7

# --- Performance ---

CACHE_ENABLED = True
MAX_CACHE_SIZE_MB = 1000
PARALLEL_PROCESSING = True
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)
BATCH_SIZE = 20
PARALLEL_PROCESSING_THRESHOLD = 20  # min files to trigger parallel processing

# --- 2D data handling ---

MAX_DATASET_ELEMENTS = 100_000_000  # threshold for sampling large HDF5 datasets
TARGET_DISPLAY_PIXELS = 2048 * 2048  # target pixel count when sampling
SYNCHROTRON_MAX_DISPLAY_SIZE = 4096  # default per-axis cap; 0 disables downsampling
