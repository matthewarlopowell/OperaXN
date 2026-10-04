"""
GUI-side configuration: theme, fonts, plot appearance, and window defaults.

Pipeline constants (correlation tolerance, definition version, performance
limits) live in core.config; the ones the GUI also uses are re-exported here.
"""

import logging
import multiprocessing
import os
import platform

# ============================================================================
# Debug Mode
# ============================================================================

DEBUG_MODE = os.environ.get("OPERAXN_DEBUG", "0") == "1"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "operaxn.log" if DEBUG_MODE else None


# ============================================================================
# Data Source Types
# ============================================================================

# Re-exported from core so enum identity is shared across the whole app:
# a locally defined copy would silently fail == comparisons against core's.
from core import DataSourceType  # noqa: E402,F401


# ============================================================================
# Theme Configuration
# ============================================================================

class OPERAXNTheme:
    """Centralised theme configuration for the entire application."""

    COLORS = {
        'bg_primary': '#1a1f2e',
        'bg_secondary': '#2c3445',
        'bg_tertiary': '#3a4556',
        'accent_primary': '#00d4ff',
        'accent_secondary': '#4ecdc4',
        'accent_hover': '#00a8cc',
        'danger': '#ff6b6b',
        'success': '#51cf66',
        'warning': '#ffd43b',
        'text_primary': '#ffffff',
        'text_secondary': '#a0a8b8',
        'text_dim': '#6c7583',
        'border': '#404859',
        'canvas_bg': '#242938',
        'input_bg': '#2c3445',
        'button_bg': '#00d4ff',
        'button_text': '#1a1f2e',
        'button_hover': '#00a8cc',
        'disabled_bg': '#3a4556',
        'disabled_text': '#6c7583'
    }

    _PLATFORM = platform.system()
    _SANS = (
        'Segoe UI' if _PLATFORM == 'Windows'
        else 'Helvetica Neue' if _PLATFORM == 'Darwin'
        else 'DejaVu Sans'
    )
    _MONO = (
        'Consolas' if _PLATFORM == 'Windows'
        else 'Menlo' if _PLATFORM == 'Darwin'
        else 'DejaVu Sans Mono'
    )

    FONTS = {
        'title': (_SANS, 12, 'bold'),
        'heading': (_SANS, 11, 'bold'),
        'body': (_SANS, 10),
        'small': (_SANS, 9),
        'button': (_SANS, 10, 'bold'),
        'mono': (_MONO, 10)
    }

    PADDING = {
        'small': 5,
        'medium': 10,
        'large': 15
    }


# ============================================================================
# Performance Settings
# ============================================================================

# Cache, processing and the default per-axis 2D display cap (0 disables
# downsampling): re-exported from core so there is a single source
from core.config import (  # noqa: E402,F401
    CACHE_ENABLED,
    MAX_CACHE_SIZE_MB,
    MAX_WORKERS,
    PARALLEL_PROCESSING,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
)

# Display Quality
FIGURE_DPI = 95
EXPORT_DPI = 300
INTERPOLATION_METHOD = "bilinear"

# Data Processing (pipeline constants live in core.config; these are GUI-side)
LARGE_IMAGE_THRESHOLD = 2_000_000
INTENSITY_SAMPLE_SIZE = 20_000
LRU_CACHE_MAXSIZE = 128
SECONDS_PER_HOUR = 3600.0

# Plot Layout
VMIN_FLOOR = 1e-10  # floor value to avoid log(0) in logarithmic plots
DEFAULT_AXIS_PADDING_PERCENT = 5.0

# Axis labels shared by every window and export
TOF_LABEL = "Time of flight (μs)"
DSPACING_LABEL = "d-spacing (Å)"
TWOTHETA_LABEL = "2θ (°)"
TIME_HOURS_LABEL = "Time (h)"

# Plot Appearance
COLORMAP = "YlGnBu_r"
LINE_WIDTH = 1.5
GRID_ALPHA = 0.3
PLOT_UPDATE_DELAY_MS = 50

# Plot Font Sizes (points)
TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 10

# Cropping Defaults
DEFAULT_TWOD_YMIN_PERCENT = 0
DEFAULT_TWOD_YMAX_PERCENT = 100
DEFAULT_TWOD_XMIN_PERCENT = 0
DEFAULT_TWOD_XMAX_PERCENT = 100

# GIF Settings
DEFAULT_GIF_FPS = 5
DEFAULT_GIF_DPI = 100
DEFAULT_GIF_LOOP = 0

# ============================================================================
# Window Sizes
# ============================================================================

# Initial dialog sizes only; BaseDialog.shrink_to_fit resizes to content
WINDOW_SIZES = {
    'export_neutron': "400x375",
    'export_xrd': "400x425",
    'gif': "400x425",
    'progress': "300x100"
}

# ============================================================================
# Scientific Constants
# ============================================================================

XRAY_WAVELENGTH = 1.5406  # Cu K-alpha, angstroms
SYNCHROTRON_WAVELENGTH = 0.495  # angstroms

# ============================================================================
# Export Settings
# ============================================================================

EXCEL_ENGINE = "openpyxl"
CSV_ENCODING = "utf-8"

# ============================================================================
# Application Info
# ============================================================================

APP_NAME = "OperaXN"
# Single source of truth for the app version: core.config.GENERATOR_VERSION
# (also written into every generated .nxs as entry/process/version, which
# the reader maps to the generator_version metadata key)
from core import GENERATOR_VERSION as APP_VERSION  # noqa: E402,F401

APP_AUTHOR = "Matthew Powell"
APP_COPYRIGHT = "2025-2026"
DOCUMENTATION_URL = "https://github.com/matthewarlopowell/OperaXN"
SUPPORT_EMAIL = "Matt.Powell@warwick.ac.uk"

# ============================================================================
# Debug Output
# ============================================================================

def log_config_summary() -> None:
    """Log the runtime configuration at DEBUG level; call once logging has handlers."""
    log = logging.getLogger(__name__)
    log.debug("OperaXN Configuration")
    log.debug(f"Platform: {platform.system()}")
    log.debug(f"CPU Cores: {multiprocessing.cpu_count()}")
    log.debug(f"Max Workers: {MAX_WORKERS}")
    log.debug(f"Cache Size: {MAX_CACHE_SIZE_MB} MB")
    log.debug(f"Display Size: {SYNCHROTRON_MAX_DISPLAY_SIZE}x{SYNCHROTRON_MAX_DISPLAY_SIZE}")
    log.debug(f"Interpolation: {INTERPOLATION_METHOD}")
