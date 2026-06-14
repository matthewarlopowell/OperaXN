"""
OperaXN - gui.py
"""

import logging
import os
import re
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from tkinter import filedialog, messagebox, simpledialog
from typing import Optional, List, Tuple, Callable, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .config import (
    APP_VERSION,
    CACHE_ENABLED,
    CSV_ENCODING,
    DataSourceType,
    DEBUG_MODE,
    DEFAULT_TWOD_XMAX_PERCENT,
    DEFAULT_TWOD_XMIN_PERCENT,
    DEFAULT_TWOD_YMAX_PERCENT,
    DEFAULT_TWOD_YMIN_PERCENT,
    EXCEL_ENGINE,
    FIGURE_DPI,
    INTENSITY_SAMPLE_SIZE,
    LARGE_IMAGE_THRESHOLD,
    MAX_CACHE_SIZE_MB,
    MAX_WORKERS,
    OPERAXNTheme,
    PLOT_UPDATE_DELAY_MS,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
    WINDOW_SIZES,
)

from .dialog import (
    PlotSettingsDialog, ExportOptionsDialog, ExportOptions,
    GIFSettingsDialog, GIFSettings, ProgressDialog,
    DataSourceSelectionDialog, DisplaySizeDialog
)
from .input import (
    process_paths, make_oned_arrays, make_twod_arrays,
    make_echem_arrays, get_correlated_data, make_neutron_arrays
)
from .output import (
    plot_oned_data, plot_twod_data, plot_echem_data,
    create_figure_layout, export_single_scan,
    get_scan_time_positions, plot_neutron_data,
    clear_plot_cache
)

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class UIState(Enum):
    """Possible states of the application UI."""
    IDLE = "idle"
    LOADING = "loading"
    PLOTTING = "plotting"
    EXPORTING = "exporting"


@dataclass
class VisualiserConfig:
    """User-adjustable plot and display configuration."""
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
    data_source: DataSourceType = DataSourceType.INHOUSE
    synchrotron_max_size: int = SYNCHROTRON_MAX_DISPLAY_SIZE
    use_cache: bool = CACHE_ENABLED
    plot_update_delay: int = PLOT_UPDATE_DELAY_MS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with PlotConfig."""
        return {
            'show_voltage': self.show_voltage,
            'show_current': self.show_current,
            'show_dspacing': self.show_dspacing,
            'show_neutron_dspacing': self.show_neutron_dspacing,
            'oned_xmin': self.oned_xmin,
            'oned_xmax': self.oned_xmax,
            'oned_ymin': self.oned_ymin,
            'oned_ymax': self.oned_ymax,
            'twod_ymin_percent': self.twod_ymin_percent,
            'twod_ymax_percent': self.twod_ymax_percent,
            'twod_xmin_percent': self.twod_xmin_percent,
            'twod_xmax_percent': self.twod_xmax_percent,
            'neutron_xmin': self.neutron_xmin,
            'neutron_xmax': self.neutron_xmax,
            'neutron_ymin': self.neutron_ymin,
            'neutron_ymax': self.neutron_ymax,
            'is_synchrotron': (self.data_source == DataSourceType.SYNCHROTRON),
            'is_neutron': (self.data_source == DataSourceType.NEUTRON),
            'use_cache': self.use_cache
        }


@dataclass
class ApplicationState:
    """Mutable runtime state including loaded data and cache."""
    scans: List[Dict[str, Any]] = field(default_factory=list)
    echem_df: Optional[pd.DataFrame] = None
    oned_arrays: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    twod_arrays: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    neutron_arrays: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    echem_arrays: Dict[str, Any] = field(default_factory=dict)
    current_scan_idx: int = 0
    intensity_limits: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    time_method: str = "absolute"
    ui_state: UIState = UIState.IDLE
    data_source: DataSourceType = DataSourceType.INHOUSE
    synchrotron_max_size: int = SYNCHROTRON_MAX_DISPLAY_SIZE
    dialog_windows: List[tk.Toplevel] = field(default_factory=list)
    data_cache: Dict[str, Any] = field(default_factory=dict)
    cache_size_bytes: int = 0
    max_cache_bytes: int = MAX_CACHE_SIZE_MB * 1024 * 1024

    def reset(self) -> None:
        """Reset application state and clear cache."""
        self.__init__()
        self.clear_cache()

    def get_cached_data(self, key: str, loader_func: Callable) -> Any:
        """Get data from cache or load it."""
        if not CACHE_ENABLED:
            return loader_func()

        if key not in self.data_cache:
            data = loader_func()
            self._add_to_cache(key, data)
        return self.data_cache[key]

    def _get_data_size(self, data: Any) -> int:
        """Accurately measure data size."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, dict):
            total_size = sys.getsizeof(data)
            for key, value in data.items():
                total_size += self._get_data_size(key)
                total_size += self._get_data_size(value)
            return total_size
        else:
            return sys.getsizeof(data)

    def _add_to_cache(self, key: str, data: Any) -> None:
        """Add data to cache with size management."""
        try:
            size = self._get_data_size(data)
            if self.cache_size_bytes + size > self.max_cache_bytes:
                self._evict_cache_entries(size)
            self.data_cache[key] = data
            self.cache_size_bytes += size
        except (MemoryError, AttributeError) as e:
            logger.debug("Cache error: %s", e)
            self.data_cache[key] = data

    def _evict_cache_entries(self, needed_size: int) -> None:
        """Evict the oldest cache entries to make room."""
        keys_to_evict = list(self.data_cache.keys())

        for key in keys_to_evict:
            if self.cache_size_bytes + needed_size <= self.max_cache_bytes:
                break
            if key in self.data_cache:
                evicted_data = self.data_cache.pop(key)
                self.cache_size_bytes -= sys.getsizeof(evicted_data)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.data_cache.clear()
        self.cache_size_bytes = 0


# ============================================================================
# Utilities
# ============================================================================

class PerformanceMonitor:
    """Monitor and log timing and memory metrics for operations."""

    def __init__(self) -> None:
        self.metrics = {}
        self.enabled = DEBUG_MODE

    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure operation time."""
        start_time = time.time()

        if self.enabled and psutil:
            start_memory = psutil.virtual_memory().used
        else:
            start_memory = 0

        yield

        elapsed = time.time() - start_time

        if self.enabled and psutil:
            memory_delta = psutil.virtual_memory().used - start_memory
        else:
            memory_delta = 0

        self.metrics[operation] = {'time': elapsed, 'memory': memory_delta}

        if elapsed > 1.0:
            logger.debug("Performance: %s took %.2fs", operation, elapsed)


class DebouncedUpdate:
    """Rate-limit widget updates to avoid redundant redraws."""

    def __init__(self, widget: tk.Widget, delay_ms: int = 50):
        self.widget = widget
        self.delay_ms = delay_ms
        self._after_id = None
        self._pending_func = None

    def schedule(self, func: Callable) -> None:
        """Schedule a debounced update."""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
        self._pending_func = func
        self._after_id = self.widget.after(self.delay_ms, self._execute)

    def _execute(self) -> None:
        """Execute the pending function."""
        if self._pending_func:
            self._pending_func()
            self._pending_func = None
            self._after_id = None

    def cancel(self) -> None:
        """Cancel any pending update."""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
            self._pending_func = None


# ============================================================================
# UI Components
# ============================================================================

class BaseUIComponent(tk.Frame):
    """Base themed frame for all custom UI components."""

    def __init__(self, master: tk.Widget, **kwargs: Any) -> None:
        kwargs.setdefault('bg', OPERAXNTheme.COLORS['bg_primary'])
        super().__init__(master, **kwargs)
        self.master = master


class StyledButton(tk.Button):
    """Themed button with primary/secondary/danger hover effects."""

    STYLES = {
        "primary": {
            "bg": OPERAXNTheme.COLORS['accent_primary'],
            "fg": OPERAXNTheme.COLORS['bg_primary'],
            "hover": OPERAXNTheme.COLORS['accent_hover']
        },
        "secondary": {
            "bg": OPERAXNTheme.COLORS['bg_tertiary'],
            "fg": OPERAXNTheme.COLORS['text_primary'],
            "hover": OPERAXNTheme.COLORS['bg_secondary']
        },
        "danger": {
            "bg": OPERAXNTheme.COLORS['danger'],
            "fg": OPERAXNTheme.COLORS['text_primary'],
            "hover": '#ff5252'
        }
    }

    def __init__(self, master: tk.Widget, text: str = "", command: Optional[Callable] = None, style: str = "primary", **kwargs: Any) -> None:
        style_config = self.STYLES.get(style, self.STYLES["secondary"])

        super().__init__(
            master,
            text=text,
            command=command,
            bg=style_config["bg"],
            fg=style_config["fg"],
            disabledforeground=OPERAXNTheme.COLORS['disabled_text'],
            font=OPERAXNTheme.FONTS['button'],
            relief=tk.FLAT,
            cursor='hand2',
            padx=OPERAXNTheme.PADDING['medium'],
            pady=3,
            **kwargs
        )

        self.default_bg = style_config["bg"]
        self.hover_bg = style_config["hover"]

        self.bind("<Enter>", lambda e: self._set_hover_state(True))
        self.bind("<Leave>", lambda e: self._set_hover_state(False))

    def _set_hover_state(self, hovering: bool) -> None:
        """Set button hover state."""
        if self['state'] != 'disabled':
            self.config(bg=self.hover_bg if hovering else self.default_bg)


class ButtonPanel(BaseUIComponent):
    """Toolbar panel containing the main action buttons."""

    BUTTON_CONFIGS = [
        ("upload_files", "📁 Upload", "normal", "primary"),
        ("plot_data", "📊 Plot", "disabled", "primary"),
        ("export_plots", "💾 Export", "disabled", "secondary"),
        ("create_gif", "🎬 GIF", "disabled", "secondary"),
        ("export_data", "📄 Excel", "disabled", "secondary"),
        ("clear_all", "🗑️ Clear", "normal", "danger"),
    ]

    def __init__(self, master: tk.Widget, callbacks: Dict[str, Callable]) -> None:
        super().__init__(master)
        self.callbacks = callbacks
        self.buttons: Dict[str, StyledButton] = {}
        self._create_buttons()

    def _create_buttons(self) -> None:
        """Create control buttons."""
        container = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        container.pack(fill="x", padx=OPERAXNTheme.PADDING['medium'], pady=3)

        for key, text, state, style in self.BUTTON_CONFIGS:
            if key == "clear_all":
                tk.Frame(container, bg=OPERAXNTheme.COLORS['bg_primary']).pack(
                    side="left", fill="x", expand=True
                )

            btn = StyledButton(
                container,
                text=text,
                command=self.callbacks.get(key),
                style=style,
                state=state,
                width=10
            )
            btn.pack(side="left", padx=2)
            self.buttons[key] = btn

    def update_states(self, states: Dict[str, str]) -> None:
        """Update button states."""
        for key, state in states.items():
            if key in self.buttons:
                self.buttons[key].config(state=state)


class FileListPanel(BaseUIComponent):
    """Scrollable list showing loaded scan files."""

    def __init__(self, master: tk.Widget, on_select: Optional[Callable] = None) -> None:
        super().__init__(master)
        self.on_select = on_select
        self._items_cache: List[str] = []
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create list widgets."""
        list_frame = tk.LabelFrame(
            self,
            text="Loaded Data:",
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            font=OPERAXNTheme.FONTS['body'],
            relief=tk.FLAT,
            borderwidth=1,
            highlightbackground=OPERAXNTheme.COLORS['border'],
            padx=OPERAXNTheme.PADDING['small'],
            pady=1
        )
        list_frame.pack(fill="both", expand=True, padx=OPERAXNTheme.PADDING['small'])

        self.listbox = tk.Listbox(
            list_frame,
            bg=OPERAXNTheme.COLORS['input_bg'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            selectbackground=OPERAXNTheme.COLORS['accent_primary'],
            selectforeground=OPERAXNTheme.COLORS['bg_primary'],
            font=OPERAXNTheme.FONTS['small'],
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            height=3,
            selectmode=tk.SINGLE
        )
        self.listbox.pack(fill="both", expand=True)

        def on_mousewheel(event):
            self.listbox.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        # Bind mousewheel events
        self.listbox.bind("<MouseWheel>", on_mousewheel)  # Windows/Mac
        self.listbox.bind("<Button-4>", lambda e: self.listbox.yview_scroll(-1, "units"))  # Linux up
        self.listbox.bind("<Button-5>", lambda e: self.listbox.yview_scroll(1, "units"))  # Linux down

        if self.on_select:
            self.listbox.bind("<Double-Button-1>", self._on_double_click)

    def _on_double_click(self, event: tk.Event) -> None:
        """Handle double-click event."""
        selection = self.listbox.curselection()
        if selection and self.on_select:
            self.on_select(selection[0])

    def update_items(self, items: List[str]) -> None:
        """Update list items."""
        if items != self._items_cache:
            self.listbox.delete(0, tk.END)
            for item in items:
                self.listbox.insert(tk.END, item)
            self._items_cache = items.copy()

    def show_progress(self, message: str) -> None:
        """Show progress message."""
        count = self.listbox.size()
        if count > 0:
            self.listbox.delete(count - 1)
        self.listbox.insert(tk.END, message)
        self.listbox.see(tk.END)
        self.update_idletasks()


class ScanSelector(BaseUIComponent):
    """Slider widget for navigating between scans."""

    def __init__(self, master: tk.Widget, on_change: Callable) -> None:
        super().__init__(master)
        self.on_change = on_change
        self.debouncer = DebouncedUpdate(self, PLOT_UPDATE_DELAY_MS)
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create selection widgets."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(fill="x", padx=OPERAXNTheme.PADDING['small'], pady=3)

        tk.Label(
            frame,
            text="Scan Number:",
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            font=OPERAXNTheme.FONTS['body']
        ).pack(side="left", padx=(OPERAXNTheme.PADDING['small'] + 1, OPERAXNTheme.PADDING['small']))

        self.scan_var = tk.IntVar(value=1)
        self.slider = tk.Scale(
            frame,
            from_=1,
            to=1,
            orient="horizontal",
            variable=self.scan_var,
            command=self._on_change,
            length=300,
            showvalue=True,
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            troughcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            activebackground=OPERAXNTheme.COLORS['accent_primary'],
            highlightbackground=OPERAXNTheme.COLORS['bg_primary'],
            highlightthickness=0,
            font=OPERAXNTheme.FONTS['small']
        )
        self.slider.pack(side="left", padx=(OPERAXNTheme.PADDING['small'], OPERAXNTheme.PADDING['small'] + 3),
                         fill="x", expand=True)

    def _on_change(self, value: str) -> None:
        """Handle change with debouncing."""
        self.debouncer.schedule(lambda: self.on_change(int(value)))

    def set_range(self, min_val: int, max_val: int) -> None:
        """Set slider range."""
        self.slider.config(from_=min_val, to=max_val)

    def get_value(self) -> int:
        """Get current value."""
        return self.scan_var.get()

    def set_value(self, value: int) -> None:
        """Set current value."""
        self.scan_var.set(value)


class PlotControls(BaseUIComponent):
    """Panel with echem toggles, intensity sliders, and plot settings."""

    def __init__(self, master: tk.Widget, config: VisualiserConfig,
                 on_intensity_update: Callable,
                 on_settings: Callable,
                 on_intensity_apply: Callable) -> None:
        super().__init__(master)
        self.config = config
        self.on_intensity_update = on_intensity_update
        self.on_settings = on_settings
        self.on_intensity_apply = on_intensity_apply
        self.on_echem_update = None
        self.intensity_debouncer = DebouncedUpdate(self, PLOT_UPDATE_DELAY_MS)
        self.echem_debouncer = DebouncedUpdate(self, 10)
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create control widgets."""
        # Settings button
        self.settings_btn = tk.Button(
            self,
            text="Plot Settings",
            command=self.on_settings,
            width=12,
            bg=OPERAXNTheme.COLORS['bg_tertiary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            font=OPERAXNTheme.FONTS['body'],
            relief=tk.FLAT,
            pady=1
        )
        self.settings_btn.pack(side="left",
                               padx=(OPERAXNTheme.PADDING['medium'] + 2, OPERAXNTheme.PADDING['medium']),
                               pady=(0, 3))

        self._create_echem_controls()
        self._create_intensity_controls()
        self._create_data_source_indicator()

    def _create_data_source_indicator(self) -> None:
        """Create indicator showing current data source type."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(side="left", padx=10)

        tk.Label(
            frame,
            text="Data Source:",
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_secondary'],
            font=OPERAXNTheme.FONTS['small']
        ).pack(side="left", padx=2, anchor='s')

        self.source_label = tk.Label(
            frame,
            text="",
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['accent_primary'],
            font=OPERAXNTheme.FONTS['body']
        )
        self.source_label.pack(side="left", padx=2)

    def _create_echem_controls(self) -> None:
        """Create echem display controls."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(side="left")

        tk.Label(frame, text="Echem:",
                 bg=OPERAXNTheme.COLORS['bg_primary'],
                 fg=OPERAXNTheme.COLORS['text_primary'],
                 font=OPERAXNTheme.FONTS['body']).pack(side="left", padx=(0, 3))

        self.show_voltage_var = tk.BooleanVar(value=self.config.show_voltage)
        self.voltage_check = tk.Checkbutton(
            frame,
            text="Voltage",
            variable=self.show_voltage_var,
            command=self._on_echem_change,
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['bg_primary'],
            activeforeground=OPERAXNTheme.COLORS['accent_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            font=OPERAXNTheme.FONTS['body']
        )
        self.voltage_check.pack(side="left", padx=2)

        self.show_current_var = tk.BooleanVar(value=self.config.show_current)
        self.current_check = tk.Checkbutton(
            frame,
            text="Current",
            variable=self.show_current_var,
            command=self._on_echem_change,
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['bg_primary'],
            activeforeground=OPERAXNTheme.COLORS['accent_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            font=OPERAXNTheme.FONTS['body']
        )
        self.current_check.pack(side="left", padx=2)

    def _create_intensity_controls(self) -> None:
        """Create intensity controls for 2D data."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(side="right", padx=OPERAXNTheme.PADDING['medium'] + 2)

        tk.Label(frame, text="2D Intensity:",
                 bg=OPERAXNTheme.COLORS['bg_primary'],
                 fg=OPERAXNTheme.COLORS['text_primary'],
                 font=OPERAXNTheme.FONTS['body']).pack(side="left", padx=2)

        self._create_intensity_control(frame, "Min:", 'intensity_min_var',
                                       'min_entry', 'min_slider', 0)
        self._create_intensity_control(frame, "Max:", 'intensity_max_var',
                                       'max_entry', 'max_slider', 100)

        self.apply_all_btn = tk.Button(
            frame,
            text="Apply All",
            command=self._apply_to_all,
            width=12,
            bg=OPERAXNTheme.COLORS['bg_tertiary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            font=OPERAXNTheme.FONTS['body'],
            relief=tk.FLAT,
            pady=1
        )
        self.apply_all_btn.pack(side="left", padx=(7, 0), pady=(0, 3))

    def _create_intensity_control(self, parent: tk.Frame, label: str, var_name: str, entry_name: str, slider_name: str, default: float) -> None:
        """Create a labelled entry + slider pair for intensity bounds."""
        tk.Label(
            parent,
            text=label,
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            font=OPERAXNTheme.FONTS['body']
        ).pack(side="left", padx=4)

        setattr(self, var_name, tk.DoubleVar(value=default))
        var = getattr(self, var_name)

        entry = tk.Entry(
            parent,
            width=8,
            bg=OPERAXNTheme.COLORS['input_bg'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            insertbackground=OPERAXNTheme.COLORS['accent_primary'],
            font=OPERAXNTheme.FONTS['body'],
            relief=tk.FLAT
        )
        entry.pack(side="left")
        entry.insert(0, str(default))

        def update_from_entry(event=None):
            try:
                value = float(entry.get())
                var.set(value)
                self._on_intensity_change()
            except ValueError:
                pass  # Ignore invalid input

        entry.bind("<Return>", update_from_entry)
        entry.bind("<FocusOut>", update_from_entry)
        setattr(self, entry_name, entry)

        slider = tk.Scale(
            parent,
            from_=0,
            to=0,
            orient="horizontal",
            variable=var,
            command=lambda v: self.intensity_debouncer.schedule(self._update_entry_from_slider),
            length=100,
            showvalue=False,
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            troughcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            activebackground=OPERAXNTheme.COLORS['accent_primary'],
            highlightthickness=0
        )
        slider.pack(side="left", padx=4)
        setattr(self, slider_name, slider)

    def _update_entry_from_slider(self) -> None:
        """Update entry fields from slider values."""
        if hasattr(self, 'min_entry'):
            self.min_entry.delete(0, tk.END)
            self.min_entry.insert(0, f"{self.intensity_min_var.get():.2f}")
            self.max_entry.delete(0, tk.END)
            self.max_entry.insert(0, f"{self.intensity_max_var.get():.2f}")
        self._on_intensity_change()

    def _on_echem_change(self) -> None:
        """Handle echem display change."""
        self.config.show_voltage = self.show_voltage_var.get()
        self.config.show_current = self.show_current_var.get()
        if self.on_echem_update:
            self.echem_debouncer.schedule(self.on_echem_update)

    def _on_intensity_change(self) -> None:
        """Handle intensity change."""
        self.on_intensity_update()

    def _apply_to_all(self) -> None:
        """Apply intensity to all scans."""
        current_min = self.intensity_min_var.get()
        current_max = self.intensity_max_var.get()

        if messagebox.askyesno("Confirm",
                               f"Apply intensity range {current_min:.2f} - {current_max:.2f} to all scans?",
                               parent=self.master):
            self.on_intensity_apply(current_min, current_max)

    def update_data_source(self, source_type: DataSourceType) -> None:
        """Update the data source indicator."""
        if hasattr(self, 'source_label'):
            source_map = {
                DataSourceType.INHOUSE: ("Laboratory X-ray diffraction", OPERAXNTheme.COLORS['text_primary']),
                DataSourceType.SYNCHROTRON: ("Synchrotron X-ray diffraction", OPERAXNTheme.COLORS['accent_primary']),
                DataSourceType.NEUTRON: ("Time-of-flight neutron diffraction", OPERAXNTheme.COLORS['danger'])
            }
            text, color = source_map.get(source_type, ("Unknown", OPERAXNTheme.COLORS['text_primary']))
            self.source_label.config(text=text, fg=color)

    def update_intensity_range(self, vmin: float, vmax: float, current_min: float, current_max: float) -> None:
        """Update intensity slider ranges."""
        if hasattr(self, 'min_slider'):
            self.min_slider.config(from_=vmin, to=vmax, resolution=1)
            self.max_slider.config(from_=vmin, to=vmax, resolution=1)
            self.intensity_min_var.set(current_min)
            self.intensity_max_var.set(current_max)

    def get_intensity_limits(self) -> Tuple[float, float]:
        """Get current intensity limits."""
        if hasattr(self, 'intensity_min_var'):
            return self.intensity_min_var.get(), self.intensity_max_var.get()
        return 0, 100

    def update_echem_states(self, state: str) -> None:
        """Update echem control states."""
        self.voltage_check.config(state=state)
        self.current_check.config(state=state)

    def update_intensity_states(self, state: str) -> None:
        """Update intensity control states."""
        if hasattr(self, 'min_entry'):
            for widget in [self.min_entry, self.min_slider,
                           self.max_entry, self.max_slider,
                           self.apply_all_btn]:
                widget.config(state=state)


# ============================================================================
# Main Application
# ============================================================================

class OPERAXN(tk.Frame):
    """Top-level frame that wires up data loading, plotting, and export."""

    def __init__(self, master: Optional[tk.Tk] = None) -> None:
        # Setup master window
        if master is None:
            master = tk.Tk()
        self.master = master
        self._setup_window()

        # Initialise parent frame
        super().__init__(self.master, bg=OPERAXNTheme.COLORS['bg_primary'])
        self.pack(fill="both", expand=True)

        # Initialise state
        self.state = ApplicationState()
        self.config = VisualiserConfig()
        self.performance = PerformanceMonitor()

        # Initialise plot components
        self.fig = None
        self.axes = {}
        self.canvas = None
        self.twod_image = None

        # Thread management
        self._worker_running = False
        self.worker_thread = None
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # Build UI
        self._build_ui()

        # Start background worker
        self._start_background_worker()

        # Setup cleanup on close
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Update management
        self._update_debouncer = DebouncedUpdate(self, self.config.plot_update_delay)

    def _setup_window(self) -> None:
        """Configure the main window."""
        self.master.configure(bg=OPERAXNTheme.COLORS['bg_primary'])
        self.master.title(f"OperaXN v{APP_VERSION}")

        # Start maximised
        try:
            self.master.state('zoomed')
        except tk.TclError:
            try:
                self.master.attributes('-zoomed', True)
            except tk.TclError:
                # Fallback to screen dimensions
                width = self.master.winfo_screenwidth()
                height = self.master.winfo_screenheight()
                self.master.geometry(f"{width}x{height}+0+0")

    def _build_ui(self) -> None:
        """Build the user interface."""
        # Configure grid weights
        self.grid_rowconfigure(0, weight=0)  # header
        self.grid_rowconfigure(1, weight=0)  # buttons
        self.grid_rowconfigure(2, weight=0)  # file list
        self.grid_rowconfigure(3, weight=1)  # canvas
        self.grid_rowconfigure(4, weight=0)  # scan selector
        self.grid_rowconfigure(5, weight=0)  # controls
        self.grid_columnconfigure(0, weight=1)

        # Create UI components
        self._create_header()
        self._create_button_panel()
        self._create_file_list_panel()
        self._create_canvas_frame()
        self._create_scan_selector()
        self._create_controls_panel()

    def _create_header(self) -> None:
        """Create application header."""
        header = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_secondary'], height=40)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_propagate(False)

        # Title canvas
        title_canvas = tk.Canvas(
            header,
            width=100,
            height=40,
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            highlightthickness=0
        )
        title_canvas.pack(side="left", padx=OPERAXNTheme.PADDING['medium'])

        # Draw OperaXN text
        title_font = ('Segoe UI', 14, 'bold')
        start_x = 4

        # Measure OPERA width
        temp = title_canvas.create_text(0, 0, text="OPERA", font=title_font)
        bbox = title_canvas.bbox(temp)
        opera_width = (bbox[2] - bbox[0]) if bbox else 60
        title_canvas.delete(temp)

        # Draw OPERA
        title_canvas.create_text(
            start_x, 20,
            text="OPERA",
            font=title_font,
            fill=OPERAXNTheme.COLORS['accent_primary'],
            anchor='w'
        )

        # Draw XN
        title_canvas.create_text(
            start_x + opera_width - 2, 20,
            text="XN",
            font=title_font,
            fill=OPERAXNTheme.COLORS['danger'],
            anchor='w'
        )

        # Subtitle
        tk.Label(
            header,
            text="OPERAndo X-ray and Neutron data visualisation tool",
            font=OPERAXNTheme.FONTS['small'],
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            fg=OPERAXNTheme.COLORS['text_secondary']
        ).pack(side="left", padx=0, pady=8, anchor='s')

    def _create_button_panel(self) -> None:
        """Create button panel."""
        callbacks = {
            "upload_files": self._upload_files,
            "plot_data": self._plot_data,
            "export_plots": self._export_data,
            "create_gif": self._create_gif,
            "export_data": self._export_to_excel,
            "clear_all": self._clear_all
        }

        self.button_panel = ButtonPanel(self, callbacks)
        self.button_panel.grid(row=1, column=0, sticky="ew", padx=0, pady=3)

    def _create_file_list_panel(self) -> None:
        """Create file list panel."""
        self.file_list = FileListPanel(self, self._on_file_select)
        self.file_list.grid(row=2, column=0, sticky="ew", padx=0)

    def _create_canvas_frame(self) -> None:
        """Create canvas frame for plots."""
        canvas_container = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        canvas_container.grid(row=3, column=0, sticky="nsew",
                              padx=OPERAXNTheme.PADDING['medium'] + 1,
                              pady=OPERAXNTheme.PADDING['small'])

        canvas_frame = tk.Frame(
            canvas_container,
            bg=OPERAXNTheme.COLORS['canvas_bg'],
            relief=tk.FLAT,
            highlightbackground=OPERAXNTheme.COLORS['border'],
            highlightcolor=OPERAXNTheme.COLORS['accent_primary'],
            highlightthickness=2
        )
        canvas_frame.pack(fill="both", expand=True)

        self.plot_container = tk.Frame(canvas_frame, bg="white")
        self.plot_container.pack(fill="both", expand=True, padx=3)

    def _create_scan_selector(self) -> None:
        """Create scan selector."""
        self.scan_selector = ScanSelector(self, self._on_scan_change)
        self.scan_selector.grid(row=4, column=0, sticky="ew", padx=0, pady=0)

    def _create_controls_panel(self) -> None:
        """Create controls panel."""
        self.controls = self._create_new_controls()
        self.controls.grid(row=5, column=0, sticky="ew", padx=0, pady=0)
        self.controls.grid_remove()

    def _create_new_controls(self) -> PlotControls:
        """Create a new controls panel instance."""
        controls = PlotControls(
            self, self.config,
            self._update_plots_with_intensity,
            self._show_plot_settings,
            self._apply_intensity_to_all
        )
        controls.on_echem_update = self._update_echem_display
        return controls

    def _start_background_worker(self) -> None:
        """Start background worker thread."""
        self._worker_running = True
        self.task_queue = Queue()

        def worker():
            while self._worker_running:
                try:
                    task = self.task_queue.get(timeout=0.1)
                    if task is None:
                        break
                    func, args, kwargs = task
                    func(*args, **kwargs)
                except Empty:
                    continue
                except Exception as e:
                    logger.debug("Worker error: %s", e)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _on_closing(self) -> None:
        """Handle application closing."""
        # Stop worker thread
        self._worker_running = False
        if self.task_queue:
            self.task_queue.put(None)

        # Shutdown thread pool
        self.executor.shutdown(wait=False)

        # Close any open dialogs
        for dialog in self.state.dialog_windows[:]:
            try:
                if dialog.winfo_exists():
                    dialog.destroy()
            except tk.TclError:
                pass

        # Destroy main window
        self.master.destroy()

    # ========================================================================
    # File Operations
    # ========================================================================

    def _upload_files(self) -> None:
        """Upload and process files."""
        selected = self._get_file_selection()
        if not selected:
            return

        self._clear_all()

        # Get data source selection
        data_source = self._get_data_source_selection()
        if not data_source:
            self._show_message("Warning", "Data source selection cancelled.", "warning")
            return

        self.state.data_source = data_source
        self.config.data_source = data_source

        # Prompt for 2D display size when synchrotron is selected
        if data_source == DataSourceType.SYNCHROTRON:
            size_dialog = DisplaySizeDialog(self.master)
            size_result = size_dialog.get_result()
            if size_result is None:
                return
            self.config.synchrotron_max_size = size_result
            self.state.synchrotron_max_size = size_result

        # Show progress
        source_name = {
            DataSourceType.INHOUSE: "in-house",
            DataSourceType.SYNCHROTRON: "synchrotron",
            DataSourceType.NEUTRON: "neutron"
        }.get(data_source, "unknown")

        self.file_list.show_progress(f"Processing {source_name} data...")

        # Process files asynchronously
        self._process_files_async(selected, data_source)

    def _get_file_selection(self) -> Optional[List[str]]:
        """Get file selection from user."""
        dialog = self._create_file_selection_dialog()
        dialog.wait_window()
        return getattr(dialog, 'selected', None)

    def _create_file_selection_dialog(self) -> tk.Toplevel:
        """Create file selection dialog."""
        dialog = tk.Toplevel(self.master)
        dialog.title("Upload Options")
        dialog.geometry(WINDOW_SIZES['upload'])
        dialog.transient(self.master)
        dialog.grab_set()
        dialog.configure(bg=OPERAXNTheme.COLORS['bg_primary'])
        dialog.selected = None

        self._center_window(dialog)

        # Label
        tk.Label(
            dialog,
            text="Choose how to select your data:",
            font=OPERAXNTheme.FONTS['heading'],
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary']
        ).pack(pady=OPERAXNTheme.PADDING['large'])

        # Button frame
        button_frame = tk.Frame(dialog, bg=OPERAXNTheme.COLORS['bg_primary'])
        button_frame.pack(pady=3)

        def handle_files():
            files = filedialog.askopenfilenames(
                title="Select Files",
                filetypes=[
                    ("All Supported", "*.zip;*.dat;*.edf;*.txt;*.hdf;*.nxs;*.xy"),
                    ("ZIP Files", "*.zip"),
                    ("DAT Files", "*.dat"),
                    ("EDF Files", "*.edf"),
                    ("Text Files", "*.txt"),
                    ("HDF Files", "*.hdf"),
                    ("NXS Files", "*.nxs"),
                    ("XY Files", "*.xy"),
                    ("All Files", "*.*")
                ]
            )
            if files:
                dialog.selected = list(files)
                dialog.destroy()

        def handle_directory():
            directory = filedialog.askdirectory(title="Select Directory")
            if directory:
                dialog.selected = [directory]
                dialog.destroy()

        # Create buttons
        StyledButton(button_frame, text="Select Files", command=handle_files,
                     style="primary", width=12, height=2).pack(side="left", padx=OPERAXNTheme.PADDING['medium'])
        StyledButton(button_frame, text="Select Directory", command=handle_directory,
                     style="primary", width=12, height=2).pack(side="left", padx=OPERAXNTheme.PADDING['medium'])
        StyledButton(button_frame, text="Cancel", command=dialog.destroy,
                     style="secondary", width=12, height=2).pack(side="left", padx=OPERAXNTheme.PADDING['medium'])

        return dialog

    def _get_data_source_selection(self) -> Optional[DataSourceType]:
        """Show dialog to select data source type."""
        dialog = DataSourceSelectionDialog(self.master)
        return dialog.get_result()

    def _process_files_async(self, selected_paths: List[str], data_source: DataSourceType) -> None:
        """Process files asynchronously."""
        self.state.ui_state = UIState.LOADING

        def process():
            try:
                with self.performance.measure("file_processing"):
                    self.state.scans, self.state.echem_df, self.state.time_method = process_paths(
                        list(selected_paths),
                        self.file_list.show_progress,
                        data_source=data_source
                    )

                if not self.state.scans:
                    self.after(0, lambda: self._show_message("Warning", "No valid scan data found.", "warning"))
                    return

                self.after(0, self._create_data_arrays)

            except Exception as e:
                logger.debug("Error details: %s", traceback.format_exc())
                self.after(0, lambda: self._show_message("Error", f"Failed to process files: {str(e)}", "error"))
            finally:
                self.state.ui_state = UIState.IDLE

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _create_data_arrays(self) -> None:
        """Create data arrays from scans."""
        with self.performance.measure("array_creation"):
            if self.state.data_source == DataSourceType.NEUTRON:
                self.file_list.show_progress("Creating neutron data arrays...")
                self.state.neutron_arrays = make_neutron_arrays(self.state.scans, self.state)
            else:
                self.file_list.show_progress("Creating 1D data arrays...")
                self.state.oned_arrays = make_oned_arrays(self.state.scans, self.state)

                self.file_list.show_progress("Creating 2D data arrays...")
                self.state.twod_arrays = make_twod_arrays(self.state.scans, self.state)

            self.file_list.show_progress("Processing Echem data...")
            self.state.echem_arrays = make_echem_arrays(self.state.echem_df, self.state.time_method)

        if not self._check_data_availability():
            self._show_message("Warning", "No valid data found in selected files.", "warning")
            return

        self._initialise_plot_limits()
        self._update_file_list()
        self._configure_ui_for_data()

        # Show success message
        source_type = {
            DataSourceType.SYNCHROTRON: "synchrotron",
            DataSourceType.INHOUSE: "in-house",
            DataSourceType.NEUTRON: "neutron"
        }.get(self.state.data_source, "unknown")

        time_info = ("absolute timestamps" if self.state.time_method == "absolute"
                     else "relative time (XRD and echem each start at 00:00:00)")

        message = f"Successfully loaded {len(self.state.scans)} {source_type} scans using {time_info}"

        self._show_message("Success", message, "info")

    def _check_data_availability(self) -> bool:
        """Check if any valid (non-error) data is available."""
        if self.state.data_source == DataSourceType.NEUTRON:
            return bool(self.state.neutron_arrays)
        has_oned = any(not d.get("error") for d in self.state.oned_arrays.values())
        has_twod = any(not d.get("error") for d in self.state.twod_arrays.values())
        return has_oned or has_twod

    def _initialise_plot_limits(self) -> None:
        """Initialise plot limits from data."""
        if self.state.data_source == DataSourceType.NEUTRON:
            return

        if self.state.oned_arrays:
            all_x = []
            all_y = []
            for data in self.state.oned_arrays.values():
                if data.get("error"):
                    continue
                all_x.extend(data["x"])
                all_y.extend(data["y"])

            if all_x and all_y:
                self.config.oned_xmin = float(min(all_x))
                self.config.oned_xmax = float(max(all_x))
                self.config.oned_ymin = float(min(all_y))
                self.config.oned_ymax = float(max(all_y))

    def _update_file_list(self) -> None:
        """Update file list display."""
        items = []
        for scan in self.state.scans:
            text = self._format_scan_text(scan)
            items.append(text)
        self.file_list.update_items(items)

    def _format_scan_text(self, scan: Dict[str, Any]) -> str:
        """Format scan text for display."""
        text = f"Scan {scan['scan_num']}"

        if self.state.data_source == DataSourceType.NEUTRON:
            text = self._format_neutron_scan(scan, text)
        elif self.state.data_source == DataSourceType.SYNCHROTRON:
            text = self._format_synchrotron_scan(scan, text)
        else:
            text = self._format_inhouse_scan(scan, text)

        return text

    def _format_neutron_scan(self, scan: Dict[str, Any], text: str) -> str:
        """Format neutron scan text."""
        # Extract scan ID if available
        if scan.get("neutron_files"):
            for measurement_num, files in scan["neutron_files"].items():
                if files:
                    for file_type, file_path in files.items():
                        basename = os.path.basename(file_path)
                        match = re.search(r'(\d{5})', basename)
                        if match:
                            text += f" - ID: {match.group(1)}"
                            break
                    break

        if scan.get("timestamp"):
            text += f" - {scan['timestamp']}"

        if scan.get("echem") is not None:
            text += f" - {scan['echem']:.3f} V"
            if scan.get("current") is not None:
                text += f" / {scan['current']:.3f} mA"

        text += " [N]"
        return text

    def _format_synchrotron_scan(self, scan: Dict[str, Any], text: str) -> str:
        """Format synchrotron scan text."""
        # Extract group ID
        for file_path in [scan.get("oned"), scan.get("twod")]:
            if file_path:
                basename = os.path.basename(file_path)
                base_no_ext = os.path.splitext(basename)[0]
                base_clean = base_no_ext.replace('_integration', '')

                # Try different patterns
                patterns = [
                    r'[a-z]+\d+-\d+-(\d+)',
                    r'(\d{5,})',
                    r'(\d{3,})(?!.*\d{3,})'
                ]

                for pattern in patterns:
                    match = re.search(pattern, base_clean.lower())
                    if match:
                        text += f" - ID: {match.group(1)}"
                        break
                if " - ID:" in text:
                    break

        if scan.get("timestamp"):
            text += f" - {scan['timestamp']}"

        if scan.get("echem") is not None:
            text += f" - {scan['echem']:.3f} V"
            if scan.get("current") is not None:
                text += f" / {scan['current']:.3f} mA"

        text += " [S]"
        return text

    def _format_inhouse_scan(self, scan: Dict[str, Any], text: str) -> str:
        """Format in-house scan text."""
        filenames = []

        if scan.get("oned"):
            oned_name = os.path.splitext(os.path.basename(scan["oned"]))[0]
            filenames.append(f"1D: {oned_name}")

        if scan.get("twod"):
            twod_name = os.path.splitext(os.path.basename(scan["twod"]))[0]
            filenames.append(f"2D: {twod_name}")

        if filenames:
            text += f" - {' - '.join(filenames)}"

        if scan.get("timestamp"):
            text += f" - {scan['timestamp']}"

        if scan.get("echem") is not None:
            text += f" - {scan['echem']:.3f} V"
            if scan.get("current") is not None:
                text += f" / {scan['current']:.3f} mA"

        return text

    def _configure_ui_for_data(self) -> None:
        """Configure UI after data loading."""
        self.scan_selector.set_range(1, len(self.state.scans))
        self.scan_selector.set_value(1)
        self.state.intensity_limits.clear()

        self.button_panel.update_states({
            "plot_data": "normal",
            "export_data": "normal"
        })

        # Recreate controls
        self.controls.destroy()
        self.controls = self._create_new_controls()
        self.controls.update_data_source(self.state.data_source)
        self.controls.grid(row=5, column=0, sticky="ew", padx=0, pady=0)
        self.controls.grid_remove()

    # ========================================================================
    # Plot Operations
    # ========================================================================

    def _plot_data(self) -> None:
        """Create plots from loaded data."""
        if not self.state.scans:
            return

        try:
            self.state.ui_state = UIState.PLOTTING

            with self.performance.measure("plot_creation"):
                self._setup_plotting()

                # Check data availability
                if self.state.data_source == DataSourceType.NEUTRON:
                    has_oned, has_twod, has_neutron = False, False, bool(self.state.neutron_arrays)
                else:
                    has_oned = bool(self.state.oned_arrays)
                    has_twod = bool(self.state.twod_arrays)
                    has_neutron = False

                has_echem = bool(self.state.echem_arrays.get("x", []).size > 0)

                if not any([has_oned, has_twod, has_echem, has_neutron]):
                    self._show_message("Warning", "No data available to plot.", "warning")
                    return

                # Create figure
                self._create_figure(has_oned, has_twod, has_echem, has_neutron)

                # Update plots
                self.scan_selector.set_value(1)
                self.state.current_scan_idx = 0
                self._update_plots()

                # Enable controls
                self.button_panel.update_states({
                    "export_plots": "normal",
                    "create_gif": "normal"
                })

        except Exception as e:
            logger.debug("Error details: %s", traceback.format_exc())
            self._show_message("Error", f"Failed to create plots: {str(e)}", "error")
        finally:
            self.state.ui_state = UIState.IDLE

    def _setup_plotting(self) -> None:
        """Reset config and clear existing plots before redraw."""
        self.config = VisualiserConfig(data_source=self.state.data_source)
        self.state.intensity_limits.clear()
        self._initialise_plot_limits()

        clear_plot_cache()

        # Clear existing plots
        self.fig = None
        self.axes = {}
        self.twod_image = None

        for widget in self.plot_container.winfo_children():
            widget.destroy()

    def _create_figure(self, has_oned: bool, has_twod: bool, has_echem: bool, has_neutron: bool) -> None:
        """Create the figure and axes."""
        scan_num = self.scan_selector.get_value()
        scan = self.state.scans[scan_num - 1]

        self.fig, self.axes = create_figure_layout(
            has_oned, has_twod, has_echem, has_neutron,
            figure_dpi=FIGURE_DPI,
            scan_num=scan_num,
            echem_value=scan.get("echem"),
            current_value=scan.get("current"),
            plot_config=self.config.to_dict()
        )

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        self.canvas.draw()

        # Setup controls
        self.controls.destroy()
        self.controls = self._create_new_controls()
        self.controls.update_data_source(self.state.data_source)
        self.controls.grid(row=5, column=0, sticky="ew", padx=0, pady=0)
        self.controls.grid()

        # Update control states
        echem_state = "normal" if has_echem else "disabled"
        twod_state = "normal" if has_twod else "disabled"
        self.controls.update_echem_states(echem_state)
        self.controls.update_intensity_states(twod_state)

    def _on_file_select(self, index: int) -> None:
        """Handle file selection."""
        if index < len(self.state.scans):
            self.scan_selector.set_value(index + 1)
            self._update_plots()

    def _on_scan_change(self, value: int) -> None:
        """Handle scan change."""
        self.state.current_scan_idx = value - 1
        self._update_debouncer.schedule(self._update_plots)

    @contextmanager
    def _plot_update_context(self):
        """Context manager for batch plot updates."""
        if self.canvas:
            old_draw = self.canvas.draw_idle
            self.canvas.draw_idle = lambda: None
            yield
            self.canvas.draw_idle = old_draw
            self.canvas.draw_idle()
        else:
            yield

    def _update_plots(self) -> None:
        """Update all plots."""
        if not self.state.scans or not self.fig:
            return

        with self.performance.measure("plot_update"):
            scan_num = self.scan_selector.get_value()
            self.state.current_scan_idx = scan_num - 1
            scan = self.state.scans[self.state.current_scan_idx]

            # Update title
            title = self._format_plot_title(scan_num, scan)
            self.fig.suptitle(title)

            # Update plots
            with self._plot_update_context():
                if self.state.data_source == DataSourceType.NEUTRON:
                    self._update_neutron_plot(scan_num, scan)
                else:
                    self._update_oned_plot(scan_num, scan)
                    self._update_twod_plot(scan_num, scan)
                self._update_echem_plot()

    def _format_plot_title(self, scan_num: int, scan: Dict[str, Any]) -> str:
        """Format plot title."""
        title = f"Scan {scan_num}"

        if scan.get("echem") is not None:
            title += f" (V: {scan['echem']:.3f} V"
            if scan.get("current") is not None:
                title += f", I: {scan['current']:.3f} mA"
            title += ")"

        if self.state.data_source == DataSourceType.NEUTRON:
            title = f"Neutron {title}"

        return title

    def _update_oned_plot(self, scan_num: int, scan: Dict[str, Any]) -> None:
        """Update 1D plot."""
        if "oned" not in self.axes:
            return

        if scan_num in self.state.oned_arrays:
            data = self.state.oned_arrays[scan_num]
            if data.get("error"):
                self.axes["oned"].clear()
                self.axes["oned"].text(
                    0.5, 0.5, "Error reading 1D data — try reopening dataset",
                    ha="center", va="center", transform=self.axes["oned"].transAxes,
                    color="red", fontsize=10
                )
            else:
                plot_oned_data(
                    self.axes["oned"], data["x"], data["y"], scan_num,
                    data.get("echem"), data.get("current"),
                    plot_config=self.config.to_dict()
                )
        else:
            self.axes["oned"].clear()
            self.axes["oned"].text(
                0.5, 0.5, "No 1D data for this scan",
                ha="center", va="center", transform=self.axes["oned"].transAxes,
                color="grey", fontsize=10
            )

    def _update_twod_plot(self, scan_num: int, scan: Dict[str, Any]) -> None:
        """Update 2D plot."""
        if "twod" not in self.axes:
            return

        if scan_num in self.state.twod_arrays:
            data = self.state.twod_arrays[scan_num]
            if data.get("error"):
                self.axes["twod"].clear()
                self.axes["twod"].text(
                    0.5, 0.5, "Error reading 2D data — try reopening dataset",
                    ha="center", va="center", transform=self.axes["twod"].transAxes,
                    color="red", fontsize=10
                )
                self.twod_image = None
            else:
                vmin, vmax = self._calculate_intensity_limits(data["image"])

                if self.state.current_scan_idx not in self.state.intensity_limits:
                    self.state.intensity_limits[self.state.current_scan_idx] = (vmin, vmax)

                cmin, cmax = self.state.intensity_limits[self.state.current_scan_idx]

                if cmin >= cmax or np.isnan(cmin) or np.isnan(cmax):
                    cmin, cmax = vmin, vmax
                    self.state.intensity_limits[self.state.current_scan_idx] = (cmin, cmax)

                self.controls.update_intensity_range(vmin, vmax, cmin, cmax)

                self.twod_image = plot_twod_data(
                    self.axes["twod"], data["image"], scan_num,
                    data.get("echem"), data.get("current"),
                    (cmin, cmax), plot_config=self.config.to_dict()
                )
        else:
            self.axes["twod"].clear()
            self.axes["twod"].text(
                0.5, 0.5, "No 2D data for this scan",
                ha="center", va="center", transform=self.axes["twod"].transAxes,
                color="grey", fontsize=10
            )
            self.twod_image = None

    def _update_neutron_plot(self, scan_num: int, scan: Dict[str, Any]) -> None:
        """Update neutron plot."""
        if not self.state.neutron_arrays or self.state.data_source != DataSourceType.NEUTRON:
            return

        neutron_axes_exist = any(key.startswith('neutron_') for key in self.axes.keys())

        if scan_num in self.state.neutron_arrays and neutron_axes_exist:
            neutron_data = self.state.neutron_arrays[scan_num]

            if neutron_data.get("error"):
                for key in self.axes:
                    if key.startswith('neutron_'):
                        self.axes[key].clear()
                        self.axes[key].text(
                            0.5, 0.5, "Error reading neutron data — try reopening dataset",
                            ha="center", va="center", transform=self.axes[key].transAxes,
                            color="red", fontsize=10
                        )
            else:
                clear_plot_cache()
                plot_neutron_data(
                    self.fig,
                    self.axes,
                    neutron_data,
                    scan_num,
                    scan.get("echem"),
                    scan.get("current"),
                    self.config.to_dict()
                )

            if self.canvas:
                self.canvas.draw_idle()
        elif neutron_axes_exist:
            for key in self.axes:
                if key.startswith('neutron_'):
                    self.axes[key].clear()
                    self.axes[key].text(
                        0.5, 0.5, f"No neutron data for scan {scan_num}",
                        ha="center", va="center", transform=self.axes[key].transAxes,
                        color="grey", fontsize=10
                    )

    def _update_echem_plot(self) -> None:
        """Update echem plot."""
        if "echem" not in self.axes:
            return

        if self.state.echem_arrays.get("x", []).size > 0:
            scan_times = get_scan_time_positions(
                self.state.scans, self.state.echem_df, self.state.time_method
            )
            plot_echem_data(
                self.axes["echem"],
                self.state.echem_arrays["x"],
                self.state.echem_arrays["y"],
                self.state.echem_arrays.get("current"),
                scan_times,
                self.state.current_scan_idx,
                plot_config=self.config.to_dict()
            )
        else:
            self.axes["echem"].clear()
            self.axes["echem"].text(
                0.5, 0.5, "No echem data available",
                ha="center", va="center", transform=self.axes["echem"].transAxes,
                color="grey", fontsize=10
            )

    def _calculate_intensity_limits(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate intensity limits for image."""
        if image.size > LARGE_IMAGE_THRESHOLD:
            # Sample for large images
            sample_size = min(INTENSITY_SAMPLE_SIZE, image.size // 100)
            flat = image.ravel()
            valid_mask = ~np.isnan(flat)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > sample_size:
                sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
                sample = flat[sample_indices]
            else:
                sample = flat[valid_indices]

            if len(sample) > 0:
                return np.percentile(sample, [1, 99])

        valid_data = image[~np.isnan(image)]
        if len(valid_data) == 0:
            return 0, 1
        return np.nanmin(image), np.nanmax(image)

    def _update_echem_display(self) -> None:
        """Update echem display."""
        self.config.show_voltage = self.controls.show_voltage_var.get()
        self.config.show_current = self.controls.show_current_var.get()
        self._update_plots()

    def _update_plots_with_intensity(self) -> None:
        """Update plots with intensity changes."""
        if not self.state.twod_arrays or "twod" not in self.axes:
            self._update_plots()
            return

        vmin, vmax = self.controls.get_intensity_limits()
        self.state.intensity_limits[self.state.current_scan_idx] = (vmin, vmax)

        if self.twod_image is not None:
            try:
                from matplotlib.colors import LogNorm
                vmin = max(vmin, 1e-10)
                vmax = max(vmax, vmin * 10) if vmax <= vmin else vmax

                self.twod_image.set_clim(vmin, vmax)
                self.twod_image.set_norm(LogNorm(vmin=vmin, vmax=vmax))

                if self.canvas:
                    self.canvas.draw_idle()
            except Exception as e:
                logger.debug("Error updating intensity: %s", e)

    def _apply_intensity_to_all(self, current_min: float, current_max: float) -> None:
        """Apply intensity range to all scans."""
        for i in range(len(self.state.scans)):
            self.state.intensity_limits[i] = (current_min, current_max)

        self._update_plots()
        self._show_message("Success",
                           f"Applied intensity range {current_min:.2f} - {current_max:.2f} to all scans",
                           "info")

    def _show_plot_settings(self) -> None:
        """Show plot settings dialog."""
        dialog = PlotSettingsDialog(self.master, self.config, self._on_plot_settings_update)
        self.state.dialog_windows.append(dialog)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(dialog))

    def _on_plot_settings_update(self) -> None:
        """Handle plot settings update."""
        clear_plot_cache()
        self._update_plots()

    # ========================================================================
    # Export Operations
    # ========================================================================

    def _export_data(self) -> None:
        """Export plot data."""
        if not self.state.scans:
            return

        options = self._get_export_options()
        if options:
            self._perform_export(options)

    def _get_export_options(self) -> Optional[ExportOptions]:
        """Get export options from user."""
        is_neutron = (self.state.data_source == DataSourceType.NEUTRON)
        dialog = ExportOptionsDialog(self.master, is_neutron=is_neutron)
        return dialog.get_result()

    def _perform_export(self, options: ExportOptions) -> None:
        """Perform export."""
        try:
            if options.export_type == "current":
                self._export_current_scan(options.dpi, options.plot_types)
            else:
                self._export_multiple_scans(options.dpi, options.plot_types)
        except Exception as e:
            self._show_message("Error", f"Export failed: {str(e)}", "error")

    def _export_current_scan(self, dpi: int, plot_types: Dict[str, bool]) -> None:
        """Export current scan."""
        scan_num = self.scan_selector.get_value()
        scan_data = get_correlated_data(self.state.scans, self.state.echem_df, scan_num, self.state)

        if not scan_data:
            self._show_message("Error", "No data for current scan", "error")
            return

        filename = filedialog.asksaveasfilename(
            parent=self.master,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialfile=f"scan_{scan_num:03d}.png"
        )

        if filename:
            intensity_limits = self.state.intensity_limits.get(self.state.current_scan_idx)
            extent = self._calculate_cropped_extent() if scan_data.get("twod") is not None else None

            export_single_scan(
                scan_data, filename, intensity_limits,
                export_dpi=dpi, plot_types=plot_types,
                plot_config=self.config.to_dict(),
                extent=extent
            )
            self._show_message("Success", f"Exported to {filename}", "info")

    def _export_multiple_scans(self, dpi: int, plot_types: Dict[str, bool]) -> None:
        """Export multiple scans."""
        scan_range = simpledialog.askstring(
            "Scan Range",
            f"Enter scan range (e.g. 1-5 or 1,3,5):\nAvailable: 1-{len(self.state.scans)}",
            parent=self.master
        )

        if not scan_range:
            return

        scan_list = self._parse_scan_range(scan_range)
        if not scan_list:
            self._show_message("Error", "No valid scans in range", "error")
            return

        base_dir = filedialog.askdirectory(
            parent=self.master,
            title="Select Directory for Multiple Exports"
        )
        if not base_dir:
            return

        success_count = 0
        failed_scans = []
        extent = self._calculate_cropped_extent()

        progress = ProgressDialog(self.master, "Exporting Scans", len(scan_list))

        for idx, scan_num in enumerate(scan_list):
            progress.update_progress(idx, f"Exporting scan {scan_num}...")

            try:
                # Get scan data
                scan_data = get_correlated_data(self.state.scans, self.state.echem_df, scan_num, self.state)

                if not scan_data:
                    failed_scans.append((scan_num, "No data available"))
                    continue

                # Get intensity limits for this specific scan
                intensity_limits = self.state.intensity_limits.get(scan_num - 1)

                # Prepare filename
                filename = os.path.join(base_dir, f"scan_{scan_num:03d}.png")

                logger.debug("Exporting scan %d to %s", scan_num, filename)
                logger.debug("  Has oned: %s", scan_data.get('oned') is not None)
                logger.debug("  Has twod: %s", scan_data.get('twod') is not None)
                logger.debug("  Has neutron: %s", scan_data.get('neutron') is not None)
                logger.debug("  Intensity limits: %s", intensity_limits)

                export_single_scan(
                    scan_data=scan_data,
                    output_path=filename,
                    intensity_limits=intensity_limits,
                    export_dpi=dpi,
                    plot_types=plot_types,
                    plot_config=self.config.to_dict(),
                    show_scan_markers=False,
                    extent=extent if scan_data.get("twod") is not None else None
                )

                success_count += 1

            except Exception as e:
                error_msg = f"Scan {scan_num}: {str(e)}"
                failed_scans.append((scan_num, str(e)))
                logger.debug("Failed to export scan %d:\n%s", scan_num, traceback.format_exc())

        progress.destroy()

        message = f"Successfully exported {success_count} scans to:\n{base_dir}"

        if failed_scans:
            message += f"\n\nFailed to export {len(failed_scans)} scans:"
            for scan_num, error in failed_scans[:5]:  # Show first 5 errors
                message += f"\n  Scan {scan_num}: {error}"
            if len(failed_scans) > 5:
                message += f"\n  ... and {len(failed_scans) - 5} more"

        msg_type = "info" if success_count > 0 else "error"
        self._show_message("Export Complete" if success_count > 0 else "Export Failed",
                           message, msg_type)

    def _calculate_cropped_extent(self) -> Tuple[float, float, float, float]:
        """Calculate extent based on cropping settings."""
        if self.config.data_source == DataSourceType.SYNCHROTRON:
            base_extent = (0, 1, 0, 1)
        elif self.config.data_source == DataSourceType.NEUTRON:
            base_extent = (0, 1, 0, 1)
        else:
            base_extent = (0, 1 / 2.5, 0, 1)

        x_range = base_extent[1] - base_extent[0]
        y_range = base_extent[3] - base_extent[2]

        x_min = base_extent[0] + (self.config.twod_xmin_percent / 100.0) * x_range
        x_max = base_extent[0] + (self.config.twod_xmax_percent / 100.0) * x_range
        y_min = base_extent[2] + (self.config.twod_ymin_percent / 100.0) * y_range
        y_max = base_extent[2] + (self.config.twod_ymax_percent / 100.0) * y_range

        return x_min, x_max, y_min, y_max

    def _create_gif(self) -> None:
        """Create animated GIF from scans."""
        if not self.state.scans:
            self._show_message("Warning", "No scans available to create GIF", "warning")
            return

        if not IMAGEIO_AVAILABLE:
            self._show_message("Error",
                               "imageio is required for GIF creation.\n"
                               "Please install it using: pip install imageio",
                               "error")
            return

        gif_dialog = GIFSettingsDialog(self.master, len(self.state.scans))
        gif_settings = gif_dialog.get_result()

        if not gif_settings:
            return

        output_file = filedialog.asksaveasfilename(
            parent=self.master,
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("All files", "*.*")],
            initialfile="xrd_scans_animation.gif"
        )

        if output_file:
            self._generate_gif(output_file, gif_settings)

    def _generate_gif(self, output_file: str, settings: GIFSettings) -> None:
        """Generate GIF file."""
        selected_scans = [self.state.scans[i - 1] for i in settings.scan_list
                          if 1 <= i <= len(self.state.scans)]

        progress = None
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                progress = ProgressDialog(self.master, "Creating GIF", len(selected_scans))

                # Create frames
                frame_files = []
                for idx, scan in enumerate(selected_scans):
                    if not progress.update_progress(idx, f"Processing scan {scan['scan_num']}"):
                        return

                    frame_file = self._create_gif_frame(temp_dir, idx, scan, settings.dpi)
                    if frame_file:
                        frame_files.append(frame_file)

                # Create GIF
                progress.update_progress(len(selected_scans), "Creating animated GIF...")

                writer = imageio.get_writer(output_file, mode='I',
                                            fps=settings.fps,
                                            loop=settings.loop)

                for frame_file in frame_files:
                    frame = imageio.v2.imread(frame_file)
                    writer.append_data(frame)

                writer.close()
                progress.destroy()

                self._show_message("Success",
                                   f"Animated GIF created successfully!\n{output_file}\n\n"
                                   f"Total frames: {len(frame_files)}\n"
                                   f"FPS: {settings.fps}\n"
                                   f"Duration: {len(frame_files) / settings.fps:.1f} seconds",
                                   "info")

            except Exception as e:
                if progress is not None:
                    progress.destroy()
                self._show_message("Error", f"Failed to create GIF: {str(e)}", "error")

    def _create_gif_frame(self, temp_dir: str, index: int, scan: Dict[str, Any], dpi: int) -> Optional[str]:
        """Create a single GIF frame."""
        scan_data = get_correlated_data(self.state.scans, self.state.echem_df, scan['scan_num'], self.state)
        if not scan_data:
            return None

        frame_file = os.path.join(temp_dir, f"frame_{index:04d}.png")
        intensity_limits = self.state.intensity_limits.get(scan['scan_num'] - 1)

        fig = None
        try:
            # Create figure for frame
            if self.state.data_source == DataSourceType.NEUTRON:
                has_oned, has_twod, has_neutron = False, False, scan_data.get("neutron") is not None
            else:
                has_oned = scan_data.get("oned") is not None
                has_twod = scan_data.get("twod") is not None
                has_neutron = False

            has_echem = self.state.echem_arrays.get("x", []).size > 0

            fig, axes = create_figure_layout(
                has_oned, has_twod, has_echem, has_neutron,
                figure_dpi=150,
                scan_num=scan_data["scan_num"],
                echem_value=scan_data.get("echem_value"),
                current_value=scan_data.get("current_value"),
                plot_config=self.config.to_dict()
            )

            # Plot data
            if has_oned and "oned" in axes:
                plot_oned_data(
                    axes["oned"], scan_data["oned"]["x"], scan_data["oned"]["y"],
                    scan_data["scan_num"], scan_data.get("echem_value"),
                    scan_data.get("current_value"), plot_config=self.config.to_dict()
                )

            if has_twod and "twod" in axes:
                plot_twod_data(
                    axes["twod"], scan_data["twod"], scan_data["scan_num"],
                    scan_data.get("echem_value"), scan_data.get("current_value"),
                    intensity_limits, plot_config=self.config.to_dict()
                )

            if has_neutron and "neutron" in scan_data:
                plot_neutron_data(
                    fig, axes, scan_data["neutron"], scan_data["scan_num"],
                    scan_data.get("echem_value"), scan_data.get("current_value"),
                    plot_config=self.config.to_dict()
                )

            if has_echem and "echem" in axes:
                scan_times = get_scan_time_positions(self.state.scans, self.state.echem_df,
                                                     self.state.time_method)
                plot_echem_data(
                    axes["echem"], self.state.echem_arrays["x"], self.state.echem_arrays["y"],
                    self.state.echem_arrays.get("current"), scan_times, index,
                    plot_config=self.config.to_dict()
                )

            fig.savefig(frame_file, dpi=dpi, bbox_inches="tight",
                        facecolor='white', edgecolor='none')

            return frame_file

        finally:
            if fig:
                plt.close(fig)

    def _export_to_excel(self) -> None:
        """Export data to Excel."""
        if not self.state.scans:
            return

        def export_task():
            try:
                with self.performance.measure("excel_export"):
                    summary_df = self._create_summary_dataframe()

                    # Get filename
                    default_filename = {
                        DataSourceType.NEUTRON: "Neutron_Diffraction_Data.xlsx",
                        DataSourceType.SYNCHROTRON: "Synchrotron_XRD_Data.xlsx",
                        DataSourceType.INHOUSE: "In-House_XRD_Data.xlsx"
                    }.get(self.state.data_source, "XRD_Data.xlsx")

                    filename = filedialog.asksaveasfilename(
                        parent=self.master,
                        defaultextension=".xlsx",
                        filetypes=[
                            ("Excel files", "*.xlsx"),
                            ("CSV files", "*.csv"),
                            ("All files", "*.*")
                        ],
                        initialfile=default_filename
                    )

                    if filename:
                        if filename.endswith(".csv"):
                            self._save_to_csv(summary_df, filename)
                        else:
                            self._save_to_excel(summary_df, filename)

                        data_type = {
                            DataSourceType.NEUTRON: "neutron diffraction",
                            DataSourceType.SYNCHROTRON: "synchrotron XRD",
                            DataSourceType.INHOUSE: "in-house XRD"
                        }.get(self.state.data_source, "")

                        self.after(0, lambda: self._show_message(
                            "Success",
                            f"Successfully exported {len(summary_df)} {data_type} scans to:\n{filename}",
                            "info"
                        ))

            except Exception as e:
                logger.debug("Excel export error: %s", traceback.format_exc())
                self.after(0, lambda: self._show_message("Error", f"Excel export failed: {str(e)}", "error"))

        # Run in thread
        thread = threading.Thread(target=export_task, daemon=True)
        thread.start()

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary dataframe."""
        correlated_data_list = []

        for scan in self.state.scans:
            base_data = self._create_scan_summary(scan)
            correlated_data_list.append(base_data)

        df = pd.DataFrame(correlated_data_list)

        if not df.empty:
            df = df.sort_values('scan number').reset_index(drop=True)

        return df

    def _create_scan_summary(self, scan: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for a single scan."""
        original_timestamp = scan.get("original_timestamp", scan.get("timestamp", ""))

        base_data = {
            "scan number": scan["scan_num"],
            "data source": self.state.data_source.value,
            "voltage": scan.get("echem", ""),
            "current": scan.get("current", ""),
            "voltage timestamp": scan.get("echem_timestamp", "")
        }

        if self.state.data_source == DataSourceType.NEUTRON:
            # Neutron-specific data
            for i in range(1, 7):
                base_data[f"bank_{i}_tof"] = ""
                base_data[f"bank_{i}_d"] = ""

            base_data["neutron_start"] = scan.get("neutron_start", "")
            base_data["neutron_end"] = scan.get("neutron_end", "")
            base_data["neutron_timestamp"] = (original_timestamp if self.state.time_method == "absolute"
                                              else scan.get("timestamp", ""))

            if scan.get("neutron_files"):
                for measurement_num, files in scan["neutron_files"].items():
                    if "tof" in files:
                        base_data[f"bank_{measurement_num}_tof"] = os.path.basename(files["tof"])
                    if "d" in files:
                        base_data[f"bank_{measurement_num}_d"] = os.path.basename(files["d"])

            base_data["neutron_metadata"] = os.path.basename(scan["neutron_meta"]) if scan.get("neutron_meta") else ""
        else:
            # XRD data
            base_data["oned"] = os.path.basename(scan["oned"]) if scan.get("oned") else ""
            base_data["twod"] = os.path.basename(scan["twod"]) if scan.get("twod") else ""
            base_data["x-ray timestamp"] = original_timestamp

        return base_data

    def _save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save dataframe to CSV."""
        if self.state.data_source == DataSourceType.NEUTRON:
            # Reorder columns for neutron data
            column_order = self._get_neutron_column_order(df)
            df = df[column_order]

        df.to_csv(filename, index=False, encoding=CSV_ENCODING)

    def _get_neutron_column_order(self, df: pd.DataFrame) -> List[str]:
        """Get optimal column order for neutron data."""
        column_order = ["scan number", "data source"]

        # Add timestamp columns
        for col in ["neutron_timestamp", "neutron_start", "neutron_end"]:
            if col in df.columns:
                column_order.append(col)

        # Add voltage/current columns
        for col in ["voltage", "current", "voltage timestamp"]:
            if col in df.columns:
                column_order.append(col)

        # Add bank columns
        for i in range(1, 7):
            for suffix in ["_tof", "_d"]:
                col = f"bank_{i}{suffix}"
                if col in df.columns:
                    column_order.append(col)

        # Add metadata column
        if "neutron_metadata" in df.columns:
            column_order.append("neutron_metadata")

        # Add any remaining columns
        for col in df.columns:
            if col not in column_order:
                column_order.append(col)

        return column_order

    def _save_to_excel(self, df: pd.DataFrame, filename: str) -> None:
        """Save dataframe to Excel."""
        with pd.ExcelWriter(filename, engine=EXCEL_ENGINE) as writer:
            df.to_excel(writer, index=False, sheet_name="Scan Summary")

            # Auto-adjust column widths
            worksheet = writer.sheets["Scan Summary"]
            for column in df:
                column_length = max(
                    df[column].astype(str).map(len).max(),
                    len(column)
                )

                if "timestamp" in column.lower():
                    column_length += 5
                elif any(x in column.lower() for x in ["tof", "_d", "oned", "twod", "metadata"]):
                    column_length = max(column_length, 20)

                col_idx = df.columns.get_loc(column)
                worksheet.column_dimensions[
                    worksheet.cell(1, col_idx + 1).column_letter
                ].width = min(column_length + 2, 50)

            # Add statistics sheet for neutron data
            if self.state.data_source == DataSourceType.NEUTRON:
                self._add_neutron_statistics_sheet(writer, df)

    def _add_neutron_statistics_sheet(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """Add statistics sheet for neutron data."""
        stats_data = []

        for _, row in df.iterrows():
            stats = self._calculate_neutron_statistics(row)
            stats_data.append(stats)

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, index=False, sheet_name="Neutron Statistics")

            # Auto-adjust column widths
            worksheet = writer.sheets["Neutron Statistics"]
            for column in stats_df:
                column_length = max(
                    stats_df[column].astype(str).map(len).max(),
                    len(column)
                ) + 2
                col_idx = stats_df.columns.get_loc(column)
                worksheet.column_dimensions[
                    worksheet.cell(1, col_idx + 1).column_letter
                ].width = column_length

    def _calculate_neutron_statistics(self, row: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for a neutron scan."""
        measurements_available = 0
        tof_files = 0
        d_files = 0

        for i in range(1, 7):
            if row.get(f"bank_{i}_tof", "") != "":
                tof_files += 1
                measurements_available = max(measurements_available, i)
            if row.get(f"bank_{i}_d", "") != "":
                d_files += 1
                measurements_available = max(measurements_available, i)

        duration = ""
        if row.get("neutron_start") and row.get("neutron_end"):
            try:
                start = pd.to_datetime(row["neutron_start"])
                end = pd.to_datetime(row["neutron_end"])
                duration_seconds = (end - start).total_seconds()
                duration = f"{duration_seconds:.1f} s"
            except (ValueError, TypeError):
                duration = "N/A"

        return {
            "Scan Number": row["scan number"],
            "Measurements": measurements_available,
            "TOF Files": tof_files,
            "D-spacing Files": d_files,
            "Scan Duration": duration,
            "Voltage (V)": row.get("voltage", ""),
            "Current (A)": row.get("current", ""),
            "Has Echem": "Yes" if row.get("voltage", "") != "" else "No"
        }

    def _parse_scan_range(self, scan_range: str) -> List[int]:
        """Parse scan range string."""
        scan_list = []
        try:
            parts = scan_range.replace(" ", "").split(",")

            for part in parts:
                if "-" in part and part.count("-") == 1:
                    start, end = map(int, part.split("-"))
                    scan_list.extend(range(start, end + 1))
                else:
                    scan_list.append(int(part))

            scan_list = sorted(set(scan_list))
            return [s for s in scan_list if 1 <= s <= len(self.state.scans)]
        except (ValueError, TypeError, AttributeError):
            return []

    def _clear_all(self) -> None:
        """Clear all data and reset."""
        if self.state.scans:
            if not messagebox.askyesno("Confirm", "Clear all data?", parent=self.master):
                return

        # Close dialogs
        for dialog in self.state.dialog_windows[:]:
            self._close_dialog(dialog)

        # Reset state
        self.state.reset()
        self.config = VisualiserConfig()

        # Clear UI
        for widget in self.plot_container.winfo_children():
            widget.destroy()

        self.file_list.update_items([])
        self.scan_selector.set_range(1, 1)
        self.scan_selector.set_value(1)

        self.button_panel.update_states({
            "plot_data": "disabled",
            "export_plots": "disabled",
            "create_gif": "disabled",
            "export_data": "disabled"
        })

        self.controls.grid_remove()

        # Reset plot objects
        self.fig = None
        self.axes = {}
        self.canvas = None
        self.twod_image = None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _center_window(self, window: tk.Toplevel) -> None:
        """Center window on screen."""
        window.update_idletasks()

        win_width = window.winfo_width()
        win_height = window.winfo_height()

        main_x = self.master.winfo_x()
        main_y = self.master.winfo_y()
        main_width = self.master.winfo_width()
        main_height = self.master.winfo_height()

        x = main_x + (main_width - win_width) // 2
        y = main_y + (main_height - win_height) // 2

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x = max(0, min(x, screen_width - win_width))
        y = max(0, min(y, screen_height - win_height))

        window.geometry(f"+{x}+{y}")

    def _close_dialog(self, dialog: tk.Toplevel) -> None:
        """Close dialog and remove from tracking."""
        try:
            if dialog in self.state.dialog_windows:
                self.state.dialog_windows.remove(dialog)
            if dialog.winfo_exists():
                dialog.destroy()
        except tk.TclError:
            pass

    def _show_message(self, title: str, message: str, msg_type: str) -> None:
        """Show message dialog."""
        if msg_type == "info":
            messagebox.showinfo(title, message, parent=self.master)
        elif msg_type == "warning":
            messagebox.showwarning(title, message, parent=self.master)
        elif msg_type == "error":
            messagebox.showerror(title, message, parent=self.master)
