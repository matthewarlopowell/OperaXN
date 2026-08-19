"""
Matplotlib plotting layer: themed figure layouts, 1D/2D/echem/neutron
plotters with a plot-object cache, and publication-quality figure export.
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Union, Dict, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .config import (
    CACHE_ENABLED,
    COLORMAP,
    DEFAULT_AXIS_PADDING_PERCENT,
    DEFAULT_TWOD_XMAX_PERCENT,
    DEFAULT_TWOD_XMIN_PERCENT,
    DEFAULT_TWOD_YMAX_PERCENT,
    DEFAULT_TWOD_YMIN_PERCENT,
    EXPORT_DPI,
    FIGURE_DPI,
    GRID_ALPHA,
    INTENSITY_SAMPLE_SIZE,
    INTERPOLATION_METHOD,
    LABEL_FONT_SIZE,
    LARGE_IMAGE_THRESHOLD,
    LINE_WIDTH,
    LRU_CACHE_MAXSIZE,
    MAX_CACHE_SIZE_MB,
    SECONDS_PER_HOUR,
    SYNCHROTRON_WAVELENGTH,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    VMIN_FLOOR,
    XRAY_WAVELENGTH,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models and Enums
# ============================================================================

class LayoutType(Enum):
    """Figure layout arrangement identifiers."""
    ALL = "all"
    ECHEM_SINGLE = "echem_single"
    XRD_BOTH = "xrd_both"
    ONED_ONLY = "oned_only"
    TWOD_ONLY = "twod_only"
    ECHEM_ONLY = "echem_only"
    NEUTRON_GRID = "neutron_grid"
    EMPTY = "empty"


@dataclass
class PlotConfig:
    """Unified configuration dataclass for all plot types."""
    show_voltage: bool = True
    show_current: bool = True
    show_dspacing: bool = False
    show_neutron_dspacing: bool = False
    oned_xmin: Optional[float] = None
    oned_xmax: Optional[float] = None
    oned_ymin: Optional[float] = None
    oned_ymax: Optional[float] = None
    twod_ymin_percent: float = DEFAULT_TWOD_YMIN_PERCENT
    twod_ymax_percent: float = DEFAULT_TWOD_YMAX_PERCENT
    twod_xmin_percent: float = DEFAULT_TWOD_XMIN_PERCENT
    twod_xmax_percent: float = DEFAULT_TWOD_XMAX_PERCENT
    neutron_xmin: Optional[float] = None
    neutron_xmax: Optional[float] = None
    neutron_ymin: Optional[float] = None
    neutron_ymax: Optional[float] = None
    is_synchrotron: bool = False
    is_neutron: bool = False
    use_cache: bool = CACHE_ENABLED

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PlotConfig':
        """Create PlotConfig from dictionary, applying defaults for missing fields."""
        if not config_dict:
            return cls()

        # Set defaults for missing fields
        defaults = {
            'neutron_xmin': None,
            'neutron_xmax': None,
            'neutron_ymin': None,
            'neutron_ymax': None,
            'show_neutron_dspacing': False,
            'twod_xmin_percent': DEFAULT_TWOD_XMIN_PERCENT,
            'twod_xmax_percent': DEFAULT_TWOD_XMAX_PERCENT,
            'twod_ymin_percent': DEFAULT_TWOD_YMIN_PERCENT,
            'twod_ymax_percent': DEFAULT_TWOD_YMAX_PERCENT,
            'use_cache': CACHE_ENABLED,
            'is_neutron': False
        }

        for key, value in defaults.items():
            config_dict.setdefault(key, value)

        return cls(**{k: v for k, v in config_dict.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class FigureLayout:
    """Dataclass holding figure size, DPI, and axes arrangement."""
    figure_size: Tuple[float, float]
    dpi: int
    axes_arrangement: Dict[str, Any]
    height_ratios: Optional[List[float]] = None
    width_ratios: Optional[List[float]] = None
    hspace: float = 0.4
    wspace: float = 0.4


# ============================================================================
# Plot Object Cache
# ============================================================================

class PlotObjectCache:
    """Thread-safe LRU cache for matplotlib plot objects."""

    def __init__(self, max_size_mb: int = MAX_CACHE_SIZE_MB) -> None:
        self.cache = {}
        self.lock = threading.Lock()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.access_count = {}
        self.enabled = CACHE_ENABLED

    def get(self, key: str) -> Optional[Any]:
        """Return the cached object for key; None on miss or when disabled."""
        if not self.enabled:
            return None
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache.get(key)

    def set(self, key: str, obj: Any) -> None:
        """Store an object with LRU eviction; no-op while disabled."""
        if not self.enabled:
            return
        with self.lock:
            if len(self.cache) > 100:
                # LRU eviction
                sorted_keys = sorted(self.access_count.keys(),
                                     key=lambda k: self.access_count.get(k, 0))
                for k in sorted_keys[:20]:
                    self.cache.pop(k, None)
                    self.access_count.pop(k, None)

            self.cache[key] = obj
            self.access_count[key] = 1

    def remove(self, key: str) -> None:
        """Remove cached object by key."""
        with self.lock:
            self.cache.pop(key, None)
            self.access_count.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()


# Global cache instance
_plot_cache = PlotObjectCache()


# ============================================================================
# Utilities
# ============================================================================

class PlotFormatter:
    """Centralised axes formatting helpers."""

    @staticmethod
    def format_axis(ax: plt.Axes,
                    xlabel: str,
                    ylabel: str,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None,
                    grid: bool = True,
                    scientific_y: bool = False) -> None:
        """Apply standard labels, limits, and tick formatting to axes."""
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(grid, alpha=GRID_ALPHA)

        if scientific_y:
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0),
                                useMathText=True)
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    @staticmethod
    def clear_with_message(ax: plt.Axes, message: str) -> None:
        """Clear axes and display a centred text message."""
        ax.clear()
        ax.text(0.5, 0.5, message, ha="center", va="center",
                transform=ax.transAxes, fontsize=LABEL_FONT_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])


class DataTransformer:
    """Unit conversion helpers for diffraction and electrochemistry data."""

    @staticmethod
    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def theta_to_d_spacing_cached(two_theta_tuple: Tuple[float, ...],
                                  wavelength: float = XRAY_WAVELENGTH) -> Tuple[float, ...]:
        """Convert 2-theta tuple to d-spacing using Bragg's law (cached)."""
        two_theta = np.array(two_theta_tuple)
        theta_rad = np.deg2rad(two_theta / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            d_spacing = wavelength / (2 * np.sin(theta_rad))
        return tuple(d_spacing)

    @staticmethod
    def theta_to_d_spacing(two_theta: Union[float, np.ndarray],
                           wavelength: float = XRAY_WAVELENGTH,
                           is_synchrotron: bool = False) -> np.ndarray:
        """Convert 2-theta angles to d-spacing via Bragg's law."""
        if is_synchrotron:
            wavelength = SYNCHROTRON_WAVELENGTH

        # Use cache for small inputs
        if isinstance(two_theta, (int, float)):
            return np.array(DataTransformer.theta_to_d_spacing_cached((two_theta,), wavelength))
        elif isinstance(two_theta, (list, tuple)) and len(two_theta) < 100:
            return np.array(DataTransformer.theta_to_d_spacing_cached(tuple(two_theta), wavelength))

        # Direct calculation for large arrays
        if isinstance(two_theta, (tuple, list)):
            two_theta = np.array(two_theta)

        theta_rad = np.deg2rad(two_theta / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            d_spacing = wavelength / (2 * np.sin(theta_rad))
        return d_spacing

    @staticmethod
    def time_to_hours(time_seconds: np.ndarray) -> np.ndarray:
        """Convert time in seconds to hours."""
        return time_seconds / SECONDS_PER_HOUR


# ============================================================================
# Base Plotter
# ============================================================================

class BasePlotter(ABC):
    """Abstract base for all plot renderers."""

    def __init__(self) -> None:
        self.formatter = PlotFormatter()
        self.transformer = DataTransformer()
        self.cache_enabled = CACHE_ENABLED
        self._line_width = LINE_WIDTH

    @abstractmethod
    def plot(self, ax: plt.Axes, *args: Any, **kwargs: Any) -> Any:
        """Render data onto the given axes."""
        pass

    def clear(self, ax: plt.Axes, cache_prefix: Optional[str] = None) -> None:
        """Clear axes and remove associated cache entry."""
        if cache_prefix:
            cache_key = self._get_cache_key(ax, cache_prefix)
            _plot_cache.remove(cache_key)
        ax.clear()

    def _get_cache_key(self, ax: plt.Axes, prefix: str) -> str:
        """Generate unique cache key from axes identity."""
        return f"{prefix}_{id(ax)}"

    def _get_axis_limits(self, data: np.ndarray,
                         config_min: Optional[float],
                         config_max: Optional[float],
                         is_y: bool = False,
                         padding_percent: float = DEFAULT_AXIS_PADDING_PERCENT) -> Tuple[float, float]:
        """Calculate axis limits from data range with optional padding."""
        if config_min is not None and config_max is not None:
            return config_min, config_max

        data_min = np.nanmin(data) if len(data) > 0 else 0
        data_max = np.nanmax(data) if len(data) > 0 else 1

        if is_y:
            # Add padding to y-axis
            data_range = data_max - data_min
            padding = data_range * padding_percent / 100.0
            data_max = data_max + padding
            if data_min > 0:
                data_min = max(0, data_min - padding)
            else:
                data_min = data_min - padding

        final_min = config_min if config_min is not None else data_min
        final_max = config_max if config_max is not None else data_max

        return final_min, final_max


# ============================================================================
# Specialised Plotters
# ============================================================================

class NeutronPlotter(BasePlotter):
    """Renderer for neutron diffraction bank data in grid layout."""

    def __init__(self) -> None:
        super().__init__()
        self._line_width = 0.5

    def plot_grid(self, fig: plt.Figure, axes_dict: Dict[str, plt.Axes],
                  neutron_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                  scan_num: int,
                  echem_value: Optional[float] = None,
                  current_value: Optional[float] = None,
                  config: Optional[PlotConfig] = None) -> None:
        """Plot all neutron banks in a 2x3 grid."""
        if config is None:
            config = PlotConfig()

        # Clear all neutron axes
        for bank_num in range(1, 6):
            ax_key = f'neutron_{bank_num}'
            if ax_key in axes_dict:
                axes_dict[ax_key].clear()

        # Plot each bank
        for bank_num in range(1, 6):
            self._plot_bank(axes_dict, neutron_data, str(bank_num), config)

    def _plot_bank(self, axes_dict: Dict[str, plt.Axes],
                   neutron_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                   bank_num: str, config: PlotConfig) -> None:
        """Plot a single neutron bank on its axes."""
        ax_key = f'neutron_{bank_num}'
        if ax_key not in axes_dict:
            return

        ax = axes_dict[ax_key]

        if bank_num not in neutron_data:
            self.formatter.clear_with_message(ax, f"No data for bank {bank_num}")
            return

        measurement_data = neutron_data[bank_num]

        # Choose data source based on config
        data, xlabel = self._get_data_source(measurement_data, bank_num, config)

        if data is None:
            self.formatter.clear_with_message(ax, f"No data for bank {bank_num}")
            return

        # Clear cache for redraw
        cache_key = self._get_cache_key(ax, f"neutron_{bank_num}")
        if self.cache_enabled:
            _plot_cache.remove(cache_key)

        # Plot data
        ax.plot(data['x'], data['y'], linewidth=self._line_width, color='blue')

        # Format axis
        x_lim = self._get_axis_limits(data['x'], config.neutron_xmin, config.neutron_xmax)
        y_lim = self._get_axis_limits(data['y'], config.neutron_ymin, config.neutron_ymax, is_y=True)

        self.formatter.format_axis(
            ax, xlabel=xlabel, ylabel="Intensity (arb. units)",
            xlim=x_lim, ylim=y_lim, scientific_y=True
        )

        ax.set_title(f"Bank {bank_num}", fontsize=LABEL_FONT_SIZE - 1)

        # Cache if enabled
        if ax.lines and config.use_cache:
            _plot_cache.set(cache_key, ax.lines[0])

    def _get_data_source(self, measurement_data: Dict[str, Any], bank_num: str,
                         config: PlotConfig) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Select TOF or d-spacing data source for a bank."""
        if config.show_neutron_dspacing:
            if 'd' in measurement_data:
                return measurement_data['d'], "d-spacing (Å)"
            elif 'tof' in measurement_data:
                logger.debug("d-spacing not available for bank %s, showing TOF", bank_num)
                return measurement_data['tof'], "Time of Flight (μs)"
        else:
            if 'tof' in measurement_data:
                return measurement_data['tof'], "Time of Flight (μs)"
            elif 'd' in measurement_data:
                logger.debug("TOF not available for bank %s, showing d-spacing", bank_num)
                return measurement_data['d'], "d-spacing (Å)"

        return None, None

    def plot(self, ax: plt.Axes, *args: Any, **kwargs: Any) -> None:
        """Not used for neutron; delegates to plot_grid instead."""
        pass


class OneDPlotter(BasePlotter):
    """Renderer for 1D XRD diffraction patterns."""

    def plot(self, ax: plt.Axes,
             x_data: np.ndarray,
             y_data: np.ndarray,
             scan_num: int,
             echem_value: Optional[float] = None,
             current_value: Optional[float] = None,
             config: Optional[PlotConfig] = None) -> None:
        """Plot 1D XRD intensity vs angle or d-spacing."""
        if config is None:
            config = PlotConfig()

        cache_key = self._get_cache_key(ax, "oned")
        plot_line = _plot_cache.get(cache_key)

        if plot_line and config.use_cache:
            self._update_plot(ax, plot_line, x_data, y_data, config)
        else:
            self._create_plot(ax, x_data, y_data, config, cache_key)

    def _update_plot(self, ax: plt.Axes, plot_line: Any,
                     x_data: np.ndarray, y_data: np.ndarray,
                     config: PlotConfig) -> None:
        """Update data on an existing cached plot line."""
        if config.show_dspacing:
            d_data = self.transformer.theta_to_d_spacing(
                x_data, is_synchrotron=config.is_synchrotron
            )
            plot_line.set_data(d_data, y_data)
            self._format_dspacing_axis(ax, d_data, y_data, config)
        else:
            plot_line.set_data(x_data, y_data)
            self._format_twotheta_axis(ax, x_data, y_data, config)

    def _create_plot(self, ax: plt.Axes, x_data: np.ndarray,
                     y_data: np.ndarray, config: PlotConfig,
                     cache_key: str) -> None:
        """Create new 1D plot from scratch."""
        self.clear(ax, "oned")

        if config.show_dspacing:
            d_data = self.transformer.theta_to_d_spacing(
                x_data, is_synchrotron=config.is_synchrotron
            )
            ax.plot(d_data, y_data, "b-", linewidth=self._line_width)
            self._format_dspacing_axis(ax, d_data, y_data, config)
        else:
            ax.plot(x_data, y_data, "b-", linewidth=self._line_width)
            self._format_twotheta_axis(ax, x_data, y_data, config)

        if ax.lines and config.use_cache:
            _plot_cache.set(cache_key, ax.lines[0])

    def _format_twotheta_axis(self, ax: plt.Axes, x_data: np.ndarray,
                              y_data: np.ndarray, config: PlotConfig) -> None:
        """Apply 2-theta axis labels and limits."""
        x_lim = self._get_axis_limits(x_data, config.oned_xmin, config.oned_xmax)
        y_lim = self._get_axis_limits(y_data, config.oned_ymin, config.oned_ymax, is_y=True)

        self.formatter.format_axis(
            ax, "2θ (degrees)", "Intensity (counts)",
            xlim=x_lim, ylim=y_lim, scientific_y=True
        )

    def _format_dspacing_axis(self, ax: plt.Axes, d_data: np.ndarray,
                              y_data: np.ndarray, config: PlotConfig) -> None:
        """Apply d-spacing axis labels and limits."""
        x_lim = self._get_axis_limits(d_data, config.oned_xmin, config.oned_xmax)
        y_lim = self._get_axis_limits(y_data, config.oned_ymin, config.oned_ymax, is_y=True)

        self.formatter.format_axis(
            ax, "d-spacing (Å)", "Intensity (counts)",
            xlim=x_lim, ylim=y_lim, scientific_y=True
        )


class TwoDPlotter(BasePlotter):
    """Renderer for 2D XRD detector images."""

    def plot(self, ax: plt.Axes,
             image_data: np.ndarray,
             scan_num: int,
             echem_value: Optional[float] = None,
             current_value: Optional[float] = None,
             intensity_limits: Optional[Tuple[float, float]] = None,
             config: Optional[PlotConfig] = None,
             extent: Optional[Tuple[float, float, float, float]] = None) -> Any:
        """Plot 2D XRD image with log-normalised colour mapping."""
        if config is None:
            config = PlotConfig()

        if extent is None:
            extent = self._calculate_extent(image_data, config)

        cropped_image = self._crop_image(image_data, config)

        cache_key = self._get_cache_key(ax, "twod")
        image_plot = _plot_cache.get(cache_key)

        if image_plot and config.use_cache:
            return self._update_plot(ax, image_plot, cropped_image,
                                     intensity_limits, extent)
        else:
            return self._create_plot(ax, cropped_image, intensity_limits,
                                     extent, cache_key, config)

    def _calculate_extent(self, image_data: np.ndarray,
                          config: PlotConfig) -> Tuple[float, float, float, float]:
        """Calculate image extent after applying crop percentages."""
        base_extent = (0, 1, 0, 1) if config.is_synchrotron else (0, 1 / 2.5, 0, 1)

        x_min, x_max, y_min, y_max = base_extent
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Apply cropping percentages
        x_min_crop = config.twod_xmin_percent / 100.0
        x_max_crop = config.twod_xmax_percent / 100.0
        y_min_crop = config.twod_ymin_percent / 100.0
        y_max_crop = config.twod_ymax_percent / 100.0

        return (
            x_min + x_min_crop * x_range,
            x_min + x_max_crop * x_range,
            y_min + y_min_crop * y_range,
            y_min + y_max_crop * y_range
        )

    def _crop_image(self, image_data: np.ndarray,
                    config: PlotConfig) -> np.ndarray:
        """Crop image array to the configured percentage bounds."""
        height, width = image_data.shape

        # Calculate crop indices
        x_min_idx = int(width * config.twod_xmin_percent / 100)
        x_max_idx = int(width * config.twod_xmax_percent / 100)
        y_min_idx = int(height * config.twod_ymin_percent / 100)
        y_max_idx = int(height * config.twod_ymax_percent / 100)

        # Ensure valid indices
        x_min_idx = max(0, min(x_min_idx, width - 1))
        x_max_idx = max(x_min_idx + 1, min(x_max_idx, width))
        y_min_idx = max(0, min(y_min_idx, height - 1))
        y_max_idx = max(y_min_idx + 1, min(y_max_idx, height))

        return image_data[y_min_idx:y_max_idx, x_min_idx:x_max_idx]

    def _calculate_intensity_limits(self, image_data: np.ndarray,
                                    intensity_limits: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """Derive vmin/vmax intensity limits from image data."""
        if intensity_limits and intensity_limits[0] < intensity_limits[1]:
            vmin, vmax = intensity_limits
        else:
            valid_data = image_data[~np.isnan(image_data)]
            if len(valid_data) == 0:
                return VMIN_FLOOR, 1

            # Sample for large images
            if valid_data.size > LARGE_IMAGE_THRESHOLD:
                sample_size = min(INTENSITY_SAMPLE_SIZE, valid_data.size)
                sample_indices = np.random.choice(len(valid_data), sample_size, replace=False)
                sample = valid_data[sample_indices]
                vmin, vmax = np.percentile(sample, [1, 99])
            else:
                vmin, vmax = valid_data.min(), valid_data.max()

        vmin = max(vmin, VMIN_FLOOR)
        vmax = max(vmax, vmin * 10) if vmax <= vmin else vmax

        return vmin, vmax

    def _update_plot(self, ax: plt.Axes, image_plot: Any, image_data: np.ndarray,
                     intensity_limits: Optional[Tuple[float, float]],
                     extent: Tuple[float, float, float, float]) -> Any:
        """Update cached image data and colour limits."""
        image_plot.set_data(image_data)
        image_plot.set_extent(extent)

        vmin, vmax = self._calculate_intensity_limits(image_data, intensity_limits)
        image_plot.set_clim(vmin, vmax)

        if not hasattr(image_plot, '_last_limits') or image_plot._last_limits != (vmin, vmax):
            image_plot.set_norm(LogNorm(vmin=vmin, vmax=vmax))
            image_plot._last_limits = (vmin, vmax)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        return image_plot

    def _create_plot(self, ax: plt.Axes, image_data: np.ndarray,
                     intensity_limits: Optional[Tuple[float, float]],
                     extent: Tuple[float, float, float, float],
                     cache_key: str, config: PlotConfig) -> Any:
        """Create new 2D image plot from scratch."""
        self.clear(ax, "twod")

        vmin, vmax = self._calculate_intensity_limits(image_data, intensity_limits)

        im = ax.imshow(
            image_data,
            cmap=COLORMAP,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            origin="lower",
            aspect="equal",
            interpolation=INTERPOLATION_METHOD,
            extent=extent
        )

        if config.use_cache:
            _plot_cache.set(cache_key, im)
            im._last_limits = (vmin, vmax)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

        self._add_colorbar(ax, im)

        return im

    def _add_colorbar(self, ax: plt.Axes, im: Any) -> None:
        """Add intensity colour bar to the 2D plot."""
        if hasattr(ax, '_colorbar_ax'):
            return

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = ax.figure.colorbar(im, cax=cax)

        ax._colorbar_ax = cax

        cbar.ax.tick_params(which="major", width=1)
        cbar.ax.tick_params(which="minor", width=1, labelsize=1, labelcolor="white")
        cbar.set_label("Intensity (counts)", fontsize=LABEL_FONT_SIZE)
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)


class EchemPlotter(BasePlotter):
    """Renderer for electrochemistry voltage and current traces."""

    def plot(self, ax: plt.Axes,
             time_data: np.ndarray,
             voltage_data: np.ndarray,
             current_data: Optional[np.ndarray] = None,
             scan_times: Optional[List[float]] = None,
             current_scan_idx: Optional[int] = None,
             config: Optional[PlotConfig] = None) -> None:
        """Plot voltage and/or current vs time."""
        if config is None:
            config = PlotConfig()

        time_hours = self.transformer.time_to_hours(time_data)

        show_voltage = config.show_voltage and len(voltage_data) > 0
        show_current = self._should_show_current(config, current_data)

        if self._needs_recreate(ax, show_voltage, show_current) or not config.use_cache:
            self._recreate_plot(ax, time_hours, voltage_data, current_data,
                                show_voltage, show_current)
        else:
            self._update_existing_plot(ax, time_hours, voltage_data, current_data,
                                       show_voltage, show_current)

        self._update_scan_marker(ax, scan_times, current_scan_idx)

        if show_voltage or show_current:
            ax.set_xlabel("Time (hours)", fontsize=LABEL_FONT_SIZE)
            ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)

    def clear(self, ax: plt.Axes, cache_prefix: Optional[str] = None) -> None:
        """Clear echem axes and all associated cache entries."""
        for suffix in ['v', 'c', 'marker']:
            cache_key = self._get_cache_key(ax, f"echem_{suffix}")
            _plot_cache.remove(cache_key)
        ax.clear()
        self._clear_secondary_axis(ax)

    def _should_show_current(self, config: PlotConfig,
                             current_data: Optional[np.ndarray]) -> bool:
        """Return True if current data is valid and enabled."""
        return (config.show_current and
                current_data is not None and
                len(current_data) > 0 and
                not np.all(np.isnan(current_data)))

    def _needs_recreate(self, ax: plt.Axes, show_voltage: bool,
                        show_current: bool) -> bool:
        """Return True if the plot must be rebuilt from scratch."""
        voltage_line = _plot_cache.get(self._get_cache_key(ax, "echem_v"))
        current_line = _plot_cache.get(self._get_cache_key(ax, "echem_c"))

        has_voltage = voltage_line is not None and voltage_line.get_visible()
        has_current = current_line is not None and current_line.get_visible()
        has_secondary = hasattr(ax, '_ax2')

        return ((show_voltage and show_current and not has_secondary) or
                (not show_voltage and not show_current) or
                (show_voltage != has_voltage) or
                (show_current != has_current and has_secondary))

    def _recreate_plot(self, ax: plt.Axes, time_hours: np.ndarray,
                       voltage_data: np.ndarray, current_data: Optional[np.ndarray],
                       show_voltage: bool, show_current: bool) -> None:
        """Clear and rebuild the echem plot from scratch."""
        self.clear(ax)
        ax.grid(False)

        if show_voltage and show_current:
            self._plot_both(ax, time_hours, voltage_data, current_data)
        elif show_voltage:
            self._plot_voltage_only(ax, time_hours, voltage_data)
        elif show_current:
            self._plot_current_only(ax, time_hours, current_data)
        else:
            self.formatter.clear_with_message(ax, "No data selected for display")

    def _update_existing_plot(self, ax: plt.Axes, time_hours: np.ndarray,
                              voltage_data: np.ndarray, current_data: Optional[np.ndarray],
                              show_voltage: bool, show_current: bool) -> None:
        """Update cached voltage/current line data in place."""
        voltage_line = _plot_cache.get(self._get_cache_key(ax, "echem_v"))
        current_line = _plot_cache.get(self._get_cache_key(ax, "echem_c"))

        if voltage_line and show_voltage:
            voltage_line.set_data(time_hours, voltage_data)
            voltage_line.set_visible(True)
            ax.relim()
            ax.autoscale_view()
        elif voltage_line:
            voltage_line.set_visible(False)

        if show_current:
            if not hasattr(ax, '_ax2'):
                ax2 = ax.twinx()
                ax._ax2 = ax2
            else:
                ax2 = ax._ax2

            if current_line:
                current_line.set_data(time_hours, current_data)
                current_line.set_visible(True)
                ax2.relim()
                ax2.autoscale_view()
            else:
                current_line, = ax2.plot(time_hours, current_data, color="tab:blue",
                                         linewidth=1.5, label="Current")
                _plot_cache.set(self._get_cache_key(ax, "echem_c"), current_line)

            ax2.set_ylabel("Current (mA)", color="tab:blue", fontsize=LABEL_FONT_SIZE)
            ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_FONT_SIZE)
        elif current_line:
            current_line.set_visible(False)
            if hasattr(ax, '_ax2'):
                ax._ax2.set_visible(False)

    def _clear_secondary_axis(self, ax: plt.Axes) -> None:
        """Remove twinx secondary axis if present."""
        if hasattr(ax, '_ax2'):
            ax._ax2.remove()
            delattr(ax, '_ax2')

    def _plot_voltage_only(self, ax: plt.Axes, time_hours: np.ndarray,
                           voltage_data: np.ndarray) -> None:
        """Plot voltage trace on primary axis."""
        color = "tab:red"
        ax.set_ylabel("Voltage (V)", color=color, fontsize=LABEL_FONT_SIZE)
        line, = ax.plot(time_hours, voltage_data, color=color,
                        linewidth=1.5, label="Voltage")
        ax.tick_params(axis="y", labelcolor=color, labelsize=TICK_FONT_SIZE)
        _plot_cache.set(self._get_cache_key(ax, "echem_v"), line)

    def _plot_current_only(self, ax: plt.Axes, time_hours: np.ndarray,
                           current_data: np.ndarray) -> None:
        """Plot current trace on primary axis."""
        color = "tab:blue"
        ax.set_ylabel("Current (mA)", color=color, fontsize=LABEL_FONT_SIZE)
        line, = ax.plot(time_hours, current_data, color=color,
                        linewidth=1.5, label="Current")
        ax.tick_params(axis="y", labelcolor=color, labelsize=TICK_FONT_SIZE)
        _plot_cache.set(self._get_cache_key(ax, "echem_c"), line)

    def _plot_both(self, ax: plt.Axes, time_hours: np.ndarray,
                   voltage_data: np.ndarray, current_data: np.ndarray) -> None:
        """Plot voltage on primary and current on secondary axis."""
        self._plot_voltage_only(ax, time_hours, voltage_data)

        ax2 = ax.twinx()
        ax._ax2 = ax2
        color = "tab:blue"
        ax2.set_ylabel("Current (mA)", color=color, fontsize=LABEL_FONT_SIZE)
        line2, = ax2.plot(time_hours, current_data, color=color,
                          linewidth=1.5, label="Current")
        ax2.tick_params(axis="y", labelcolor=color, labelsize=TICK_FONT_SIZE)

        _plot_cache.set(self._get_cache_key(ax, "echem_c"), line2)

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    def _update_scan_marker(self, ax: plt.Axes, scan_times: Optional[List[float]],
                            current_scan_idx: Optional[int]) -> None:
        """Add or update vertical scan-position marker line."""
        cache_key = self._get_cache_key(ax, "echem_marker")
        marker_line = _plot_cache.get(cache_key)

        if (scan_times and current_scan_idx is not None and
                current_scan_idx < len(scan_times)):
            scan_time_hours = self.transformer.time_to_hours(scan_times[current_scan_idx])

            if marker_line:
                marker_line.set_xdata([scan_time_hours, scan_time_hours])
            else:
                marker_line = ax.axvline(x=scan_time_hours, color="red",
                                         linestyle="--", alpha=1.0, linewidth=2)
                _plot_cache.set(cache_key, marker_line)
        elif marker_line:
            marker_line.set_visible(False)


# ============================================================================
# Figure Layout Manager
# ============================================================================

class FigureLayoutManager:
    """Creates matplotlib figures with data-driven subplot arrangements."""

    LAYOUT_CONFIGS = {
        LayoutType.ALL: FigureLayout(
            figure_size=(14, 10), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 2, "cols": 2},
            height_ratios=[1, 1.5]
        ),
        LayoutType.ECHEM_SINGLE: FigureLayout(
            figure_size=(10, 8), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 2, "cols": 1},
            height_ratios=[1, 1.5]
        ),
        LayoutType.XRD_BOTH: FigureLayout(
            figure_size=(14, 6), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 1, "cols": 2}
        ),
        LayoutType.ONED_ONLY: FigureLayout(
            figure_size=(10, 6), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 1, "cols": 1}
        ),
        LayoutType.TWOD_ONLY: FigureLayout(
            figure_size=(8, 6), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 1, "cols": 1}
        ),
        LayoutType.ECHEM_ONLY: FigureLayout(
            figure_size=(10, 6), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 1, "cols": 1}
        ),
        LayoutType.NEUTRON_GRID: FigureLayout(
            figure_size=(16, 8), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 2, "cols": 3},
            height_ratios=[1, 1], width_ratios=[1, 1, 1],
            hspace=0.35, wspace=0.25
        ),
        LayoutType.EMPTY: FigureLayout(
            figure_size=(10, 8), dpi=FIGURE_DPI,
            axes_arrangement={"rows": 1, "cols": 1}
        )
    }

    def create_layout(self, has_oned: bool, has_twod: bool, has_echem: bool,
                      has_neutron: bool = False, figure_dpi: int = FIGURE_DPI,
                      scan_num: int = 1, echem_value: Optional[float] = None,
                      current_value: Optional[float] = None,
                      config: Optional[PlotConfig] = None) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """Build figure and axes dict for the available data types."""
        plt.rcParams["figure.autolayout"] = False

        layout_type = self._determine_layout_type(has_oned, has_twod, has_echem, has_neutron)

        if has_neutron:
            figure_dpi = max(figure_dpi - 25, 50)

        fig, axes = self._create_figure_for_layout(layout_type, figure_dpi, has_echem)

        title = self._create_title(scan_num, echem_value, current_value, has_neutron)
        fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")

        # Adjust spacing
        if layout_type == LayoutType.NEUTRON_GRID:
            fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.94,
                                hspace=0.35, wspace=0.2)
        else:
            fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92,
                                hspace=0.35, wspace=0.3)

        return fig, axes

    def _determine_layout_type(self, has_oned: bool, has_twod: bool,
                               has_echem: bool, has_neutron: bool) -> LayoutType:
        """Map data availability flags to a LayoutType."""
        if has_neutron:
            return LayoutType.NEUTRON_GRID
        elif has_echem and has_oned and has_twod:
            return LayoutType.ALL
        elif has_echem and (has_oned or has_twod):
            return LayoutType.ECHEM_SINGLE
        elif has_oned and has_twod:
            return LayoutType.XRD_BOTH
        elif has_oned:
            return LayoutType.ONED_ONLY
        elif has_twod:
            return LayoutType.TWOD_ONLY
        elif has_echem:
            return LayoutType.ECHEM_ONLY
        else:
            return LayoutType.EMPTY

    def _create_figure_for_layout(self, layout_type: LayoutType,
                                  dpi: int, has_echem: bool = False) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """Instantiate figure and subplot axes for the given layout type."""
        config = self.LAYOUT_CONFIGS[layout_type]
        fig = plt.figure(figsize=config.figure_size, dpi=dpi)

        if layout_type == LayoutType.ALL:
            gs = fig.add_gridspec(2, 2, height_ratios=config.height_ratios,
                                  hspace=config.hspace, wspace=config.wspace)
            axes = {
                "echem": fig.add_subplot(gs[0, :]),
                "oned": fig.add_subplot(gs[1, 0]),
                "twod": fig.add_subplot(gs[1, 1])
            }
        elif layout_type == LayoutType.ECHEM_SINGLE:
            gs = fig.add_gridspec(2, 1, height_ratios=config.height_ratios,
                                  hspace=config.hspace)
            axes = {"echem": fig.add_subplot(gs[0, 0])}
        elif layout_type == LayoutType.XRD_BOTH:
            gs = fig.add_gridspec(1, 2, wspace=config.wspace)
            axes = {
                "oned": fig.add_subplot(gs[0, 0]),
                "twod": fig.add_subplot(gs[0, 1])
            }
        elif layout_type == LayoutType.NEUTRON_GRID:
            axes = self._create_neutron_grid_axes(fig, config, has_echem)
        elif layout_type in [LayoutType.ONED_ONLY, LayoutType.TWOD_ONLY, LayoutType.ECHEM_ONLY]:
            ax = fig.add_subplot(111)
            axes = {layout_type.value.replace("_only", ""): ax}
        else:
            axes = {}

        return fig, axes

    def _create_neutron_grid_axes(self, fig: plt.Figure, config: FigureLayout,
                                  has_echem: bool) -> Dict[str, plt.Axes]:
        """Create 2x3 grid axes for five neutron banks plus optional echem."""
        gs = fig.add_gridspec(2, 3, height_ratios=config.height_ratios,
                              width_ratios=config.width_ratios,
                              hspace=config.hspace, wspace=config.wspace)

        axes = {}

        if has_echem:
            axes["echem"] = fig.add_subplot(gs[0, 1])
            axes["neutron_1"] = fig.add_subplot(gs[0, 0])
            axes["neutron_2"] = fig.add_subplot(gs[0, 2])
            axes["neutron_3"] = fig.add_subplot(gs[1, 0])
            axes["neutron_4"] = fig.add_subplot(gs[1, 1])
            axes["neutron_5"] = fig.add_subplot(gs[1, 2])
        else:
            axes["neutron_1"] = fig.add_subplot(gs[0, 0])
            axes["neutron_2"] = fig.add_subplot(gs[0, 1])
            axes["neutron_3"] = fig.add_subplot(gs[0, 2])
            axes["neutron_4"] = fig.add_subplot(gs[1, 0])
            axes["neutron_5"] = fig.add_subplot(gs[1, 1])

        return axes

    def _create_title(self, scan_num: int, echem_value: Optional[float],
                      current_value: Optional[float], has_neutron: bool = False) -> str:
        """Build scan title string with optional voltage and current."""
        title = f"{'Neutron ' if has_neutron else ''}Scan {scan_num}"
        if echem_value is not None:
            title += f" (V: {echem_value:.3f} V"
            if current_value is not None:
                title += f", I: {current_value:.3f} mA"
            title += ")"
        return title


# ============================================================================
# Singleton Instances
# ============================================================================

_oned_plotter = OneDPlotter()
_twod_plotter = TwoDPlotter()
_echem_plotter = EchemPlotter()
_neutron_plotter = NeutronPlotter()
_layout_manager = FigureLayoutManager()


# ============================================================================
# Public API Functions
# ============================================================================

def plot_oned_data(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                   scan_num: int, echem_value: Optional[float] = None,
                   current_value: Optional[float] = None,
                   plot_config: Optional[Dict[str, Any]] = None) -> None:
    """Plot 1D XRD intensity pattern on the given axes."""
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()
    _oned_plotter.plot(ax, x_data, y_data, scan_num, echem_value, current_value, config)


def plot_twod_data(ax: plt.Axes, image_data: np.ndarray, scan_num: int,
                   echem_value: Optional[float] = None,
                   current_value: Optional[float] = None,
                   intensity_limits: Optional[Tuple[float, float]] = None,
                   plot_config: Optional[Dict[str, Any]] = None,
                   extent: Optional[Tuple[float, float, float, float]] = None) -> Any:
    """Plot 2D XRD detector image on the given axes."""
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()
    return _twod_plotter.plot(ax, image_data, scan_num, echem_value,
                              current_value, intensity_limits, config, extent)


def plot_echem_data(ax: plt.Axes, time_data: np.ndarray, voltage_data: np.ndarray,
                    current_data: Optional[np.ndarray] = None,
                    scan_times: Optional[List[float]] = None,
                    current_scan_idx: Optional[int] = None,
                    plot_config: Optional[Dict[str, Any]] = None) -> None:
    """Plot electrochemistry voltage/current vs time."""
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()
    _echem_plotter.plot(ax, time_data, voltage_data, current_data,
                        scan_times, current_scan_idx, config)


def plot_neutron_data(fig: plt.Figure, axes_dict: Dict[str, plt.Axes],
                      neutron_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                      scan_num: int,
                      echem_value: Optional[float] = None,
                      current_value: Optional[float] = None,
                      plot_config: Optional[Dict[str, Any]] = None) -> None:
    """Plot neutron bank data across a 2x3 grid layout."""
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()
    _neutron_plotter.plot_grid(fig, axes_dict, neutron_data, scan_num,
                               echem_value, current_value, config)


def create_figure_layout(has_oned: bool, has_twod: bool, has_echem: bool,
                         has_neutron: bool = False,
                         figure_dpi: int = FIGURE_DPI,
                         scan_num: int = 1,
                         echem_value: Optional[float] = None,
                         current_value: Optional[float] = None,
                         plot_config: Optional[Dict[str, Any]] = None) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create figure and axes dict sized for the available data types."""
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()
    fig, axes = _layout_manager.create_layout(
        has_oned, has_twod, has_echem, has_neutron,
        figure_dpi, scan_num, echem_value, current_value, config
    )

    # Handle special case for echem_single layout
    if has_echem and (has_oned or has_twod) and not (has_oned and has_twod) and not has_neutron:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.4)
        if has_oned:
            axes["oned"] = fig.add_subplot(gs[1, 0])
        else:
            axes["twod"] = fig.add_subplot(gs[1, 0])

    return fig, axes


def export_single_scan(scan_data: Dict[str, Any], output_path: str,
                       intensity_limits: Optional[Tuple[float, float]] = None,
                       export_dpi: int = EXPORT_DPI,
                       plot_types: Optional[Dict[str, bool]] = None,
                       plot_config: Optional[Dict[str, Any]] = None,
                       show_scan_markers: bool = False,
                       extent: Optional[Tuple[float, float, float, float]] = None) -> None:
    """Render a single scan to an image file at export resolution."""
    if plot_types is None:
        plot_types = {"oned": True, "twod": True, "echem": True}

    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()
    config.use_cache = False  # Disable cache for export

    has_oned = scan_data.get("oned") is not None and plot_types.get("oned", True)
    has_twod = scan_data.get("twod") is not None and plot_types.get("twod", True)
    has_echem = False  # Single scan export doesn't include echem
    has_neutron = scan_data.get("neutron") is not None

    include_echem_values = plot_types.get("echem", True)
    echem_value = scan_data.get("echem_value") if include_echem_values else None
    current_value = scan_data.get("current_value") if include_echem_values else None

    fig, axes = create_figure_layout(
        has_oned, has_twod, has_echem, has_neutron,
        figure_dpi=150,
        scan_num=scan_data["scan_num"],
        echem_value=echem_value,
        current_value=current_value,
        plot_config=plot_config
    )

    if has_oned and "oned" in axes:
        plot_oned_data(
            axes["oned"], scan_data["oned"]["x"], scan_data["oned"]["y"],
            scan_data["scan_num"], scan_data.get("echem_value"),
            scan_data.get("current_value"), plot_config
        )

    if has_twod and "twod" in axes:
        plot_twod_data(
            axes["twod"], scan_data["twod"], scan_data["scan_num"],
            scan_data.get("echem_value"), scan_data.get("current_value"),
            intensity_limits, plot_config, extent
        )

    if has_neutron and scan_data.get("neutron"):
        plot_neutron_data(
            fig, axes, scan_data["neutron"], scan_data["scan_num"],
            scan_data.get("echem_value"), scan_data.get("current_value"),
            plot_config
        )

    fig.savefig(output_path, dpi=export_dpi, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close(fig)


def clear_plot_cache() -> None:
    """Clear the global plot object cache."""
    _plot_cache.clear()


def get_scan_time_positions(scans: List[Dict[str, Any]],
                            echem_df: pd.DataFrame,
                            time_method: str = "absolute") -> List[float]:
    """Compute scan timestamps in seconds for echem plot markers."""
    if echem_df is None or echem_df.empty:
        return []

    scan_times = []

    if time_method == "relative":
        for scan in scans:
            if scan.get("timestamp"):
                ts = scan["timestamp"]
                if isinstance(ts, str) and ":" in ts:
                    parts = ts.split(":")
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    scan_times.append(seconds)
    else:
        echem_start = pd.to_datetime(echem_df["timestamp"].min())
        for scan in scans:
            if scan.get("timestamp"):
                # Prefer the midpoint-adjusted timestamp so the marker lands on
                # the echem point the scan was correlated against
                scan_time = scan.get("timestamp_for_correlation")
                if scan_time is None:
                    scan_time = pd.to_datetime(scan["timestamp"])
                time_diff = (pd.to_datetime(scan_time) - echem_start).total_seconds()
                scan_times.append(time_diff)

    return scan_times
