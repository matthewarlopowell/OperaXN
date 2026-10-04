"""
Shared configuration for the OperaXN core pipeline.

GUI-specific settings (theme, window sizes, plot appearance) stay in the
respective application packages; only values the core pipeline needs live here.
"""

import multiprocessing

# --- Generator identity ---
# Written into every .nxs as process/program + process/version; the file
# layout itself is self-describing (the reader detects it structurally).

GENERATOR_NAME = "operaxn"
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

# --- Scan identification ---

# ISIS-style neutron run numbers; enforced by the logbook parser and neutron
# file grouper (display code shares the rule via NeutronFileGrouper).
SCAN_ID_MIN_DIGITS = 5
SCAN_ID_MAX_DIGITS = 7

# --- Instrument profiles ---
# Per-instrument constants that reduced data files do not carry, keyed by
# the harvested instrument name (lowercase). Harvested file metadata always
# wins; a profile only fills the gaps, and the writer ignores keys it does
# not know. Keys (all optional; None means unknown and writes nothing):
#   instrument_name, source_name, source_type, probe
#       entry/instrument/name and instrument/source/{name, type, probe}
#   pre_sample_flightpath_m
#       entry/pre_sample_flightpath (NXoperando_tofnpd), metres
#   detector_description
#       entry/instrument/detector/description (free text)
#   detector_banks  {bank: {"distance_m", "polar_angle_deg", "azimuthal_angle_deg"}}
#       the NXoperando_tofnpd instrument/detector arrays [nBank]: nominal
#       sample-to-bank distance (m), scattering angle 2theta and azimuth
#       (degrees). All-or-skip: each array is written only when EVERY bank
#       supplies that key; detector_number is written iff at least one is.
#   monitor_mode  "timer" | "monitor" | None
#       entry/monitor/mode (NXoperando_tofnpd): whether acquisitions end on
#       clock time or on monitor counts. Not in reduced files; taken from
#       the run records. None writes no monitor group at all.
#   monitors  {detector_number: {"distance_m", "polar_angle_deg", "azimuthal_angle_deg"}}
#       the NXoperando_tofnpd monitor arrays [nMon]: every beam-monitor
#       element of the instrument definition, sample-to-monitor distance
#       (m), 2theta (180 upstream, 0 downstream) and azimuth (degrees).
#       Same all-or-skip rule as detector_banks; written inside the monitor
#       group, so only when monitor_mode is set.
# To add an instrument, derive its block from a Mantid-processed file with
# `operaxn --profile FILE.nxs` and paste it here with a comment saying where
# the numbers came from.

INSTRUMENT_PROFILES = {
    "polaris": {
        "source_name": "ISIS",
        "source_type": "Spallation Neutron Source",
        "probe": "neutron",
        "instrument_name": "POLARIS",
        # Geometry from the Mantid IDF Polaris_upgrade (valid-from
        # 2012-05-01) embedded in POLARIS164798-164802.nxs (Mantid 6.16.1,
        # G. Perez, ISIS): moderator at z = -14.0 m, sample at the origin.
        "pre_sample_flightpath_m": 14.0,
        "detector_description": ("ZnS scintillator elements (4.8 mm x 32-60 mm) in "
                                 "five focussed banks; from the Mantid IDF "
                                 "Polaris_upgrade (valid from 2012-05-01)"),
        # Per bank, the mean over every detector element of the bank in the
        # IDF (bank 5 is the IDF's outer + inner bank5 modules, element IDs
        # 5xxxxx and 6xxxxx). A Mantid-processed file's focussed subset
        # (operaxn --profile) agrees to within 2 mm and 0.06 deg.
        # Ranges and element/module counts, for reference:
        #   1:  220 elements,  4 modules, L2 2.246-2.251 m, 2theta   6.76-14.04 deg
        #   2:  880 elements, 10 modules, L2 1.307-2.359 m, 2theta  19.49-34.10 deg
        #   3:  594 elements,  6 modules, L2 0.925-1.565 m, 2theta  40.38-66.45 deg
        #   4:  660 elements,  6 modules, L2 0.710-1.083 m, 2theta  75.18-112.92 deg
        #   5:  600 elements, 12 modules, L2 0.795-1.540 m, 2theta 134.66-167.42 deg
        # The banks are rings around the beam, so no single azimuth is
        # meaningful: azimuthal_angle_deg stays None and the array is omitted.
        "detector_banks": {
            1: {"distance_m": 2.248, "polar_angle_deg": 10.40, "azimuthal_angle_deg": None},
            2: {"distance_m": 1.783, "polar_angle_deg": 25.99, "azimuthal_angle_deg": None},
            3: {"distance_m": 1.206, "polar_angle_deg": 52.21, "azimuthal_angle_deg": None},
            4: {"distance_m": 0.898, "polar_angle_deg": 92.59, "azimuthal_angle_deg": None},
            5: {"distance_m": 1.251, "polar_angle_deg": 146.72, "azimuthal_angle_deg": None},
        },
        # Counting mode from the run records (Polaris_LDE_Nov25, 108 runs):
        # every run lasted 126 or 127 s while the proton charge varied, and
        # two runs that received 0 uAh still ended on the clock, so runs are
        # time-terminated (to be confirmed with G. Perez, ISIS). The preset
        # is not recorded anywhere; realised durations are per scan in
        # environment/exposure_time.
        "monitor_mode": "timer",
        # Every beam-monitor element of the IDF (physical_monitors in the
        # Mantid file): six upstream positions (2theta 180) and one
        # downstream transmission monitor (2theta 0), two elements each.
        # The reduction normalised to proton charge (NormaliseByCurrent,
        # never NormaliseToMonitor), so none of them was used; which one a
        # monitor normalisation would use is open with G. Perez.
        "monitors": {
            611: {"distance_m": 7.641, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            612: {"distance_m": 7.641, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            621: {"distance_m": 7.356, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            622: {"distance_m": 7.356, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            631: {"distance_m": 4.703, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            632: {"distance_m": 4.703, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            641: {"distance_m": 4.418, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            642: {"distance_m": 4.418, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            651: {"distance_m": 2.773, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            652: {"distance_m": 2.773, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            661: {"distance_m": 2.123, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            662: {"distance_m": 2.123, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
            671: {"distance_m": 3.270, "polar_angle_deg": 0.0, "azimuthal_angle_deg": 0.0},
            672: {"distance_m": 3.270, "polar_angle_deg": 0.0, "azimuthal_angle_deg": 0.0},
        },
    },
    "wish": {
        "source_name": "ISIS",
        "source_type": "Spallation Neutron Source",
        "probe": "neutron",
        "instrument_name": "WISH",
        # Source and name only. The geometry (L1, the five paired banks
        # 1/10 to 5/6, the monitors) waits for a Mantid-processed WISH file
        # (operaxn --profile FILE.nxs). For orientation: the reduction's
        # DIFC per pair, 4997.7 / 10398.8 / 15093.7 / 18640.7 / 20752.1
        # microseconds per angstrom, implies 2theta 27.1 / 58.3 / 90.0 /
        # 121.7 / 153.2 deg at a total flight path of 42.2 m.
    },
    "i11-1": {
        "source_name": "Diamond Light Source",
        "source_type": "Synchrotron X-ray Source",
        "probe": "x-ray",
        "instrument_name": "i11-1",
        # Wavelength comes from the .poni calibration at the beamline and is
        # not stored in the scan .nxs; supply per-experiment when known.
    },
}

# Fallbacks by data source (DataSourceType value) when no instrument name
# could be harvested
DEFAULT_PROFILES = {
    "inhouse": {
        "source_name": "laboratory X-ray source",
        "source_type": "Fixed Tube X-ray",
        "probe": "x-ray",
        "instrument_name": "laboratory diffractometer",
    },
    "synchrotron": {
        "source_name": "synchrotron",
        "source_type": "Synchrotron X-ray Source",
        "probe": "x-ray",
        "instrument_name": "synchrotron beamline",
    },
    "neutron": {
        "source_name": "neutron source",
        "source_type": "Spallation Neutron Source",
        "probe": "neutron",
        "instrument_name": "neutron diffractometer",
    },
}

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
