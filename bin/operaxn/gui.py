"""
OperaXN main window: toolbar, scan list, matplotlib canvas, scan
navigation, and the figure/Excel/GIF/NeXus export flows.

Consumes scan dictionaries and echem data from the input adapter
(input.py); its own processing is limited to display and export
shaping (intensity percentiles, summary tables, capacity CSV rows) —
the reduction pipeline itself lives in core.
"""

import logging
import os
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
from matplotlib.figure import Figure

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
    MAX_WORKERS,
    OPERAXNTheme,
    PLOT_UPDATE_DELAY_MS,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
)

from .capacity import (
    assign_cycles, compute_capacity,
    plot_capacity_vs_voltage, plot_time_vs_voltage
)
from .dialog import (
    PlotSettingsDialog, ExportOptionsDialog, ExportOptions,
    GIFSettingsDialog, GIFSettings, ProgressDialog,
    UploadOptionsDialog, UploadOptions
)
from .heatmap import HeatmapWindow
from .ici import (
    ICIWindow, echem_source_button, embed_figure, maximise_window,
    sync_figure_to_widget
)
from .input import (
    process_paths, make_oned_arrays, make_twod_arrays,
    make_echem_arrays, get_correlated_data, make_neutron_arrays,
    NeutronFileGrouper, SynchrotronFileGrouper, export_nxs, get_loaded_nxs_path,
    add_standard_echem_files, get_experiment_metadata, get_standard_echem
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
    """Mutable runtime state for the loaded experiment."""
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

    def reset(self) -> None:
        """Restore every field to its dataclass default in place."""
        self.__init__()


# ============================================================================
# Utilities
# ============================================================================

class PerformanceMonitor:
    """Records timing/memory metrics per operation (memory in debug only)."""

    def __init__(self) -> None:
        self.metrics = {}
        self.enabled = DEBUG_MODE

    @contextmanager
    def measure(self, operation: str):
        """Time the wrapped block and record the metrics under `operation`."""
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
    """Coalesces rapid updates so only the last scheduled callback runs."""

    def __init__(self, widget: tk.Widget, delay_ms: int = 50):
        self.widget = widget
        self.delay_ms = delay_ms
        self._after_id = None
        self._pending_func = None

    def schedule(self, func: Callable) -> None:
        """Queue func after the delay, cancelling any pending predecessor."""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
        self._pending_func = func
        self._after_id = self.widget.after(self.delay_ms, self._execute)

    def _execute(self) -> None:
        """Run the pending function once, then clear the pending slot."""
        if self._pending_func:
            self._pending_func()
            self._pending_func = None
            self._after_id = None

    def cancel(self) -> None:
        """Cancel the pending after() callback, if any."""
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
        """Swap the background for hover unless the button is disabled."""
        if self['state'] != 'disabled':
            self.config(bg=self.hover_bg if hovering else self.default_bg)


class ButtonPanel(BaseUIComponent):
    """Toolbar panel containing the main action buttons."""

    BUTTON_CONFIGS = [
        ("upload_files", "📁 Upload", "normal", "primary"),
        ("plot_data", "📊 Plot", "disabled", "primary"),
        ("capacity_plot", "Capacity", "disabled", "primary"),
        ("ici_analysis", "ICI", "disabled", "primary"),
        ("heatmap", "Heatmap", "disabled", "primary"),
        ("export_plots", "Image", "disabled", "secondary"),
        ("create_gif", "GIF", "disabled", "secondary"),
        ("export_data", "Excel", "disabled", "secondary"),
        ("export_nxs", "📄 NeXus", "disabled", "danger"),
        ("clear_all", "🗑 Clear", "normal", "danger"),
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
        """Apply a key-to-tk-state mapping, ignoring unknown keys."""
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
        """Build the listbox with wheel scrolling and double-click select."""
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
            selectmode=tk.SINGLE,
            activestyle="none",
            exportselection=False
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

    def highlight_scan(self, index: int) -> None:
        """Select and reveal the active scan's row."""
        if 0 <= index < self.listbox.size():
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(index)
            self.listbox.see(index)

    def _on_double_click(self, event: tk.Event) -> None:
        """Forward the double-clicked row index to the on_select callback."""
        selection = self.listbox.curselection()
        if selection and self.on_select:
            self.on_select(selection[0])

    def update_items(self, items: List[str]) -> None:
        """Replace the rows, skipping the rebuild when items are unchanged."""
        if items != self._items_cache:
            self.listbox.delete(0, tk.END)
            for item in items:
                self.listbox.insert(tk.END, item)
            self._items_cache = items.copy()

    def show_progress(self, message: str) -> None:
        """Show transient progress only before persistent scan rows exist."""
        if self._items_cache:
            return
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
        """Build the labelled 1-based scan slider."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(fill="x", padx=OPERAXNTheme.PADDING['medium'], pady=(2, 4))

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
        """Debounce slider drags before firing on_change with an int."""
        self.debouncer.schedule(lambda: self.on_change(int(value)))

    def set_range(self, min_val: int, max_val: int) -> None:
        """Set the slider bounds (inclusive, 1-based scan numbers)."""
        self.slider.config(from_=min_val, to=max_val)

    def get_value(self) -> int:
        """The currently selected scan number (1-based)."""
        return self.scan_var.get()

    def set_value(self, value: int) -> None:
        """Move the slider to the given scan number."""
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
        """Assemble settings button, echem, intensity, and source rows."""
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
        """Create the source label; update_data_source sets text/colour."""
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
        """Create voltage/current checkboxes wired to the echem debouncer."""
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
        """Create the 2D intensity row in a frame neutron sources hide."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(side="right", padx=OPERAXNTheme.PADDING['medium'] + 2)
        self.intensity_frame = frame  # hidden for sources without 2D data

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
        """Mirror slider values into the entries, then fire the update."""
        if hasattr(self, 'min_entry'):
            self.min_entry.delete(0, tk.END)
            self.min_entry.insert(0, f"{self.intensity_min_var.get():.2f}")
            self.max_entry.delete(0, tk.END)
            self.max_entry.insert(0, f"{self.intensity_max_var.get():.2f}")
        self._on_intensity_change()

    def _on_echem_change(self) -> None:
        """Sync config from the checkboxes and debounce on_echem_update."""
        self.config.show_voltage = self.show_voltage_var.get()
        self.config.show_current = self.show_current_var.get()
        if self.on_echem_update:
            self.echem_debouncer.schedule(self.on_echem_update)

    def _on_intensity_change(self) -> None:
        """Forward to the owner's intensity-update callback."""
        self.on_intensity_update()

    def _apply_to_all(self) -> None:
        """Confirm, then push the current min/max to every scan."""
        current_min = self.intensity_min_var.get()
        current_max = self.intensity_max_var.get()

        if messagebox.askyesno("Confirm",
                               f"Apply intensity range {current_min:.2f} - {current_max:.2f} to all scans?",
                               parent=self.master):
            self.on_intensity_apply(current_min, current_max)

    def update_data_source(self, source_type: DataSourceType) -> None:
        """Update the data source indicator and source-dependent controls."""
        if hasattr(self, 'source_label'):
            source_map = {
                DataSourceType.INHOUSE: ("Laboratory X-ray diffraction", OPERAXNTheme.COLORS['text_primary']),
                DataSourceType.SYNCHROTRON: ("Synchrotron X-ray diffraction", OPERAXNTheme.COLORS['accent_primary']),
                DataSourceType.NEUTRON: ("Time-of-flight neutron diffraction", OPERAXNTheme.COLORS['danger'])
            }
            text, color = source_map.get(source_type, ("Unknown", OPERAXNTheme.COLORS['text_primary']))
            self.source_label.config(text=text, fg=color)

        # 2D intensity controls are meaningless for neutron (no 2D data)
        if hasattr(self, 'intensity_frame'):
            if source_type == DataSourceType.NEUTRON:
                self.intensity_frame.pack_forget()
            elif not self.intensity_frame.winfo_ismapped():
                self.intensity_frame.pack(
                    side="right", padx=OPERAXNTheme.PADDING['medium'] + 2)

    def update_intensity_range(self, vmin: float, vmax: float, current_min: float, current_max: float) -> None:
        """Rescale both sliders to [vmin, vmax] and set current values."""
        if hasattr(self, 'min_slider'):
            self.min_slider.config(from_=vmin, to=vmax, resolution=1)
            self.max_slider.config(from_=vmin, to=vmax, resolution=1)
            self.intensity_min_var.set(current_min)
            self.intensity_max_var.set(current_max)

    def get_intensity_limits(self) -> Tuple[float, float]:
        """Current (min, max) limits, or (0, 100) before widgets exist."""
        if hasattr(self, 'intensity_min_var'):
            return self.intensity_min_var.get(), self.intensity_max_var.get()
        return 0, 100

    def update_echem_states(self, state: str) -> None:
        """Set both echem checkboxes to the given tk state."""
        self.voltage_check.config(state=state)
        self.current_check.config(state=state)

    def update_intensity_states(self, state: str) -> None:
        """Set every intensity widget to the given tk state, if built."""
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
        """Title the window and start maximised, with platform fallbacks."""
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
        """Create all main-window rows; only the canvas row is stretchable."""
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

        # Size the canvas to the drawn wordmark (a fixed width clips it at
        # higher DPI scaling)
        bbox = title_canvas.bbox("all")
        if bbox:
            title_canvas.config(width=bbox[2] + 6)

        # Subtitle
        tk.Label(
            header,
            text="OPERAndo X-ray and Neutron data visualisation tool",
            font=OPERAXNTheme.FONTS['small'],
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            fg=OPERAXNTheme.COLORS['text_secondary']
        ).pack(side="left", padx=0, pady=8, anchor='s')

    def _create_button_panel(self) -> None:
        """Map toolbar keys to handler methods and grid the ButtonPanel."""
        callbacks = {
            "upload_files": self._upload_files,
            "plot_data": self._plot_data,
            "export_plots": self._export_data,
            "capacity_plot": self._open_capacity_window,
            "ici_analysis": self._open_ici_window,
            "heatmap": self._open_heatmap,
            "create_gif": self._create_gif,
            "export_data": self._export_to_excel,
            "export_nxs": self._export_nxs,
            "clear_all": self._clear_all
        }

        self.button_panel = ButtonPanel(self, callbacks)
        self.button_panel.grid(row=1, column=0, sticky="ew", padx=0, pady=3)

    def _create_file_list_panel(self) -> None:
        """Grid the scan list; double-click jumps plots to that scan."""
        self.file_list = FileListPanel(self, self._on_file_select)
        self.file_list.grid(row=2, column=0, sticky="ew", padx=0)

    def _create_canvas_frame(self) -> None:
        """Build the bordered plot container and show the empty state."""
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

        self.plot_container = tk.Frame(
            canvas_frame, bg=OPERAXNTheme.COLORS['canvas_bg'])
        self.plot_container.pack(fill="both", expand=True, padx=3)
        self._show_empty_state()

    def _show_empty_state(self) -> None:
        """Dark placeholder shown in the plot area when no data is loaded."""
        hint = tk.Frame(self.plot_container, bg=OPERAXNTheme.COLORS['canvas_bg'])
        hint.pack(fill="both", expand=True)
        tk.Label(
            hint, text="No data loaded",
            font=OPERAXNTheme.FONTS['heading'],
            bg=OPERAXNTheme.COLORS['canvas_bg'],
            fg=OPERAXNTheme.COLORS['text_secondary']
        ).pack(expand=True, anchor="s", pady=(0, 2))
        tk.Label(
            hint, text="Upload data or open a NeXus file to begin",
            font=OPERAXNTheme.FONTS['body'],
            bg=OPERAXNTheme.COLORS['canvas_bg'],
            fg=OPERAXNTheme.COLORS['text_dim']
        ).pack(expand=True, anchor="n", pady=(2, 0))

    def _create_scan_selector(self) -> None:
        """Grid the scan slider wired to _on_scan_change."""
        self.scan_selector = ScanSelector(self, self._on_scan_change)
        self.scan_selector.grid(row=4, column=0, sticky="ew", padx=0, pady=0)

    def _create_controls_panel(self) -> None:
        """Grid the plot controls, hidden until data is first plotted."""
        self.controls = self._create_new_controls()
        self.controls.grid(row=5, column=0, sticky="ew", padx=0, pady=0)
        self.controls.grid_remove()

    def _create_new_controls(self) -> PlotControls:
        """Build a PlotControls wired to this frame's update callbacks."""
        controls = PlotControls(
            self, self.config,
            self._update_plots_with_intensity,
            self._show_plot_settings,
            self._apply_intensity_to_all
        )
        controls.on_echem_update = self._update_echem_display
        return controls

    def _start_background_worker(self) -> None:
        """Start the daemon thread that drains task_queue; None stops it."""
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
        """Stop worker and executor, close dialogs, destroy the window."""
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
        """Open the combined upload dialog and process the selection."""
        # Non-modal: lift an already-open dialog instead of duplicating it
        existing = getattr(self, "_upload_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_set()
                    return
            except tk.TclError:
                pass

        self._upload_dialog = UploadOptionsDialog(
            self.master, has_loaded_data=bool(self.state.scans))
        options = self._upload_dialog.get_result()
        self._upload_dialog = None
        if options is None:
            return

        if not options.paths and options.standard_echem_files:
            self._append_standard_echem_async(options.standard_echem_files)
            return

        self._clear_all()

        self.state.data_source = options.data_source
        self.config.data_source = options.data_source
        if options.data_source == DataSourceType.SYNCHROTRON:
            self.config.synchrotron_max_size = options.display_size
            self.state.synchrotron_max_size = options.display_size

        source_name = {
            DataSourceType.INHOUSE: "in-house",
            DataSourceType.SYNCHROTRON: "synchrotron",
            DataSourceType.NEUTRON: "neutron"
        }.get(options.data_source, "unknown")

        self.file_list.show_progress(f"Processing {source_name} data...")

        self._process_files_async(options)

    def _append_standard_echem_async(self, paths: List[str]) -> None:
        """Add analysis-only echem to the session without reloading it."""
        self.state.ui_state = UIState.LOADING
        self.file_list.show_progress("Adding electrochemistry data...")

        def process() -> None:
            try:
                added = add_standard_echem_files(paths)
                if not added:
                    self.after(0, lambda: self._show_message(
                        "No echem data",
                        "No valid electrochemistry data was found in the selected files.",
                        "warning"))
                    return

                def complete() -> None:
                    self.file_list.show_progress(
                        f"Added {len(added)} electrochemistry file(s)")
                    self.button_panel.update_states({
                        "capacity_plot": "normal",
                        "ici_analysis": "normal",
                        "export_nxs": "normal",
                    })
                    sources = self._echem_sources()
                    for attr in ("_ici_window", "_capacity_window"):
                        analysis_window = getattr(self, attr, None)
                        try:
                            if analysis_window is not None and analysis_window.winfo_exists():
                                refresh = getattr(analysis_window, "refresh_sources", None)
                                if refresh is not None:
                                    refresh(sources)
                        except tk.TclError:
                            pass
                    usable = [item for item in added
                              if self._has_analysis_current(item.get("data"))]
                    if not usable:
                        self._show_message(
                            "Echem added without current",
                            "The selected file was loaded, but it has no usable current "
                            "column and therefore cannot be used for ICI or Capacity analysis.",
                            "warning")
                    else:
                        self._show_message(
                            "Electrochemistry added",
                            f"Added {len(usable)} analysis source(s) to the current experiment.",
                            "info")

                self.after(0, complete)
            except Exception as exc:
                logger.debug("Additional echem error: %s", traceback.format_exc())
                self.after(0, lambda error=str(exc): self._show_message(
                    "Error", f"Failed to add electrochemistry: {error}", "error"))
            finally:
                self.state.ui_state = UIState.IDLE

        threading.Thread(target=process, daemon=True).start()

    def _choose_additional_echem(self) -> None:
        """Choose analysis echem and append it to the active experiment."""
        focused = self.master.focus_get()
        parent = focused.winfo_toplevel() if focused is not None else self.master
        paths = filedialog.askopenfilenames(
            parent=parent,
            title="Add Electrochemistry",
            filetypes=[("Supported files", "*.txt *.xlsx *.csv"),
                       ("All files", "*.*")])
        if paths:
            self._append_standard_echem_async(list(paths))

    def _process_files_async(self, options: UploadOptions) -> None:
        """Run process_paths on a worker thread; results return via after()."""
        self.state.ui_state = UIState.LOADING

        # Stored 2D size follows the single display-size setting
        # (synchrotron HDF only; EDF is always full resolution)
        twod_max_size = (options.display_size
                         if options.include_2d
                         and options.data_source == DataSourceType.SYNCHROTRON
                         else 0)

        def process():
            try:
                with self.performance.measure("file_processing"):
                    self.state.scans, self.state.echem_df, self.state.time_method = process_paths(
                        list(options.paths),
                        self.file_list.show_progress,
                        time_method=options.time_method,
                        data_source=options.data_source,
                        standard_echem_files=options.standard_echem_files,
                        include_2d_images=options.include_2d,
                        twod_max_display_size=twod_max_size,
                        title=options.title,
                        sample_name=options.sample_name,
                        sample_description=options.sample_description,
                        sample_preparation_date=options.sample_preparation_date,
                        cycling_protocol=options.cycling_protocol
                    )

                if not self.state.scans:
                    self.after(0, lambda: self._show_message("Warning", "No valid scan data found.", "warning"))
                    return

                self.after(0, self._create_data_arrays)

            except Exception as e:
                logger.debug("Error details: %s", traceback.format_exc())
                # bind now: `e` is unbound once the except block exits, and
                # the lambda only runs later on the Tk main loop
                error_message = f"Failed to process files: {e}"
                self.after(0, lambda: self._show_message("Error", error_message, "error"))
            finally:
                self.state.ui_state = UIState.IDLE

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _create_data_arrays(self) -> None:
        """Build plot arrays on the Tk thread, then configure the data UI."""
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
        """Seed 1D axis limits from the loaded data range; neutron skips it."""
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
        """Rebuild the scan list rows from the loaded scans."""
        items = []
        for scan in self.state.scans:
            text = self._format_scan_text(scan)
            items.append(text)
        self.file_list.update_items(items)

    def _format_scan_text(self, scan: Dict[str, Any]) -> str:
        """Compose a scan's listbox row, dispatching on data source."""
        text = f"Scan {scan['scan_num']}"

        if self.state.data_source == DataSourceType.NEUTRON:
            text = self._format_neutron_scan(scan, text)
        elif self.state.data_source == DataSourceType.SYNCHROTRON:
            text = self._format_synchrotron_scan(scan, text)
        else:
            text = self._format_inhouse_scan(scan, text)

        return text

    def _format_neutron_scan(self, scan: Dict[str, Any], text: str) -> str:
        """Append scan ID, timestamp, and echem readings to a neutron row."""
        # Extract scan ID if available (same rule as file grouping, so the
        # displayed ID always matches the ID used for logbook correlation)
        if scan.get("neutron_files"):
            for measurement_num, files in scan["neutron_files"].items():
                if files:
                    for file_type, file_path in files.items():
                        basename = os.path.basename(file_path)
                        scan_id = NeutronFileGrouper.extract_scan_id(basename)
                        if scan_id:
                            text += f" - ID: {scan_id}"
                            break
                    break

        if scan.get("timestamp"):
            text += f" - {scan['timestamp']}"

        if scan.get("echem") is not None:
            text += f" - {scan['echem']:.3f} V"
            if scan.get("current") is not None:
                text += f" / {scan['current']:.3f} mA"

        return text

    def _format_synchrotron_scan(self, scan: Dict[str, Any], text: str) -> str:
        """Append scan ID, timestamp, and echem to a synchrotron row."""
        # Scan ID via the same rule used for file grouping, so the displayed
        # ID always matches the ID the scan was grouped under
        for file_path in [scan.get("oned"), scan.get("twod")]:
            if file_path:
                scan_id = SynchrotronFileGrouper.extract_scan_id(
                    os.path.basename(file_path))
                if scan_id:
                    text += f" - ID: {scan_id}"
                    break

        if scan.get("timestamp"):
            text += f" - {scan['timestamp']}"

        if scan.get("echem") is not None:
            text += f" - {scan['echem']:.3f} V"
            if scan.get("current") is not None:
                text += f" / {scan['current']:.3f} mA"

        return text

    def _format_inhouse_scan(self, scan: Dict[str, Any], text: str) -> str:
        """Append 1D/2D filenames, timestamp, and echem to an in-house row."""
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
        """Reset slider, enable buttons for the source, rebuild controls."""
        self.scan_selector.set_range(1, len(self.state.scans))
        self.scan_selector.set_value(1)
        self.state.intensity_limits.clear()

        echem_state = "normal" if self._echem_sources() else "disabled"
        # XRD-only view: stays disabled for neutron data
        heatmap_state = "normal" if self.state.oned_arrays else "disabled"
        self.button_panel.update_states({
            "plot_data": "normal",
            "export_data": "normal",
            "export_nxs": "normal",
            "capacity_plot": echem_state,
            "ici_analysis": echem_state,
            "heatmap": heatmap_state
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
        """Rebuild the figure from scratch and show scan 1."""
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
        """Build figure and axes, embed the Tk canvas, reveal controls."""
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
        """Jump slider and plots to the double-clicked list row."""
        if index < len(self.state.scans):
            self.scan_selector.set_value(index + 1)
            self._update_plots()

    def _on_scan_change(self, value: int) -> None:
        """Record the new scan index and debounce a full plot refresh."""
        self.state.current_scan_idx = value - 1
        self._update_debouncer.schedule(self._update_plots)

    @contextmanager
    def _plot_update_context(self):
        """Suppress draw_idle within the block, then draw once on exit."""
        if self.canvas:
            old_draw = self.canvas.draw_idle
            self.canvas.draw_idle = lambda: None
            yield
            self.canvas.draw_idle = old_draw
            self.canvas.draw_idle()
        else:
            yield

    def _update_plots(self) -> None:
        """Redraw every pane for the selected scan; sync list and title."""
        if not self.state.scans or not self.fig:
            return

        with self.performance.measure("plot_update"):
            scan_num = self.scan_selector.get_value()
            self.state.current_scan_idx = scan_num - 1
            self.file_list.highlight_scan(self.state.current_scan_idx)
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
        """Compose the figure suptitle with scan number and echem values."""
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
        """Redraw the 1D axes, or placeholder text on error/missing data."""
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
        """Redraw the 2D image, caching per-scan limits, syncing sliders."""
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
        """Redraw all neutron_* axes for the scan, or placeholder text."""
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
        """Redraw the echem axes with scan-time markers, when data exists."""
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
        """Derive display (vmin, vmax): sampled percentiles when large."""
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
        """Copy checkbox states into config and redraw all plots."""
        self.config.show_voltage = self.controls.show_voltage_var.get()
        self.config.show_current = self.controls.show_current_var.get()
        self._update_plots()

    def _update_plots_with_intensity(self) -> None:
        """Re-clim the 2D image in place (LogNorm), avoiding a full redraw."""
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
        """Overwrite every scan's cached intensity limits, then redraw."""
        for i in range(len(self.state.scans)):
            self.state.intensity_limits[i] = (current_min, current_max)

        self._update_plots()
        self._show_message("Success",
                           f"Applied intensity range {current_min:.2f} - {current_max:.2f} to all scans",
                           "info")

    def _lift_existing_dialog(self, attr: str) -> bool:
        """Focus a still-open dialog on `attr` instead of duplicating it."""
        existing = getattr(self, attr, None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_set()
                    return True
            except tk.TclError:
                pass
        setattr(self, attr, None)
        return False

    def _show_plot_settings(self) -> None:
        """Open the non-modal settings dialog, or lift the existing one."""
        if self._lift_existing_dialog('_plot_settings_dialog'):
            return
        dialog = PlotSettingsDialog(self.master, self.config, self._on_plot_settings_update)
        self._plot_settings_dialog = dialog
        self.state.dialog_windows.append(dialog)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(dialog))

    def _on_plot_settings_update(self) -> None:
        """Invalidate the plot cache and redraw after a settings change."""
        clear_plot_cache()
        self._update_plots()

    # ========================================================================
    # Echem Analysis Windows
    # ========================================================================

    @staticmethod
    def _has_analysis_current(df: Optional[pd.DataFrame]) -> bool:
        """True when df has current values, as capacity/ICI analysis needs."""
        return (df is not None and not df.empty
                and "current" in df.columns
                and bool(df["current"].notna().any()))

    def _echem_sources(self) -> Dict[str, pd.DataFrame]:
        """Echem datasets usable for capacity/ICI analysis, keyed by label.

        The operando echem comes first, followed by any standard echem files
        stored in the loaded .nxs; datasets without current values are
        excluded (nothing to classify or integrate)."""
        sources: Dict[str, pd.DataFrame] = {}
        if self._has_analysis_current(self.state.echem_df):
            sources["Operando echem"] = self.state.echem_df
        for item in get_standard_echem():
            data = item.get("data")
            if not self._has_analysis_current(data):
                continue
            base = (os.path.basename(str(item.get("source_file") or ""))
                    or str(item.get("name") or "standard echem"))
            label = base
            suffix = 2
            while label in sources:
                label = f"{base} ({suffix})"
                suffix += 1
            sources[label] = data
        return sources

    def _open_heatmap(self) -> None:
        """Open the operando heatmap (stacked XRD patterns + voltage)."""
        if self._lift_existing_dialog('_heatmap_window'):
            return
        if not self.state.oned_arrays:
            self._show_message(
                "No data",
                "Load 1D XRD data before opening the heatmap.",
                "warning")
            return

        win = HeatmapWindow(
            self.master,
            scans=self.state.scans,
            oned_arrays=self.state.oned_arrays,
            echem_arrays=self.state.echem_arrays,
            time_method=self.state.time_method)
        self._heatmap_window = win
        self.state.dialog_windows.append(win)
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(win))

    def _open_ici_window(self) -> None:
        """Open the ICI (intermittent current interruption) analysis window."""
        if self._lift_existing_dialog('_ici_window'):
            return
        sources = self._echem_sources()
        if not sources:
            self._show_message(
                "No echem data",
                "Load echem data with current values before opening ICI Analysis.",
                "warning")
            return

        win = ICIWindow(
            self.master, echem_sources=sources,
            on_add_source=self._choose_additional_echem)
        self._ici_window = win
        self.state.dialog_windows.append(win)
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(win))

    def _open_capacity_window(self) -> None:
        """Open capacity analysis (voltage vs time, capacity vs voltage)."""
        if self._lift_existing_dialog('_capacity_window'):
            return
        sources = self._echem_sources()
        if not sources:
            self._show_message(
                "No echem data",
                "Load echem data with current values before opening Capacity Analysis.",
                "warning")
            return

        # Mutable so the source dropdown can swap datasets inside the closures
        current = {
            "name": next(iter(sources)),
            "df": next(iter(sources.values())),
            "available": [],
        }

        win = tk.Toplevel(self.master)
        win.title("OperaXN — Capacity Analysis")
        win.configure(bg=OPERAXNTheme.COLORS['bg_primary'])
        win.minsize(900, 500)
        maximise_window(win)
        win.lift()
        win.focus_force()

        # Compact single-line toolbar matching the main window.
        controls_wrapper = tk.Frame(
            win,
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            highlightbackground=OPERAXNTheme.COLORS['border'],
            highlightthickness=1
        )
        controls_wrapper.pack(side="top", fill="x",
                              padx=OPERAXNTheme.PADDING['small'],
                              pady=(OPERAXNTheme.PADDING['small'], 2))
        controls = tk.Frame(controls_wrapper, bg=OPERAXNTheme.COLORS['bg_secondary'])
        controls.pack(fill="x", padx=OPERAXNTheme.PADDING['small'], pady=4)

        def _label(parent: tk.Widget, text: str, dim: bool = False) -> tk.Label:
            return tk.Label(
                parent, text=text,
                bg=OPERAXNTheme.COLORS['bg_secondary'],
                fg=(OPERAXNTheme.COLORS['text_dim'] if dim
                    else OPERAXNTheme.COLORS['text_primary']),
                font=OPERAXNTheme.FONTS['small'])

        def _nav_button(parent: tk.Widget, text: str, command) -> tk.Button:
            return tk.Button(
                parent, text=text, command=command, width=1,
                bg=OPERAXNTheme.COLORS['bg_tertiary'],
                fg=OPERAXNTheme.COLORS['text_primary'],
                activebackground=OPERAXNTheme.COLORS['bg_secondary'],
                activeforeground=OPERAXNTheme.COLORS['text_primary'],
                disabledforeground=OPERAXNTheme.COLORS['text_dim'],
                font=OPERAXNTheme.FONTS['button'], relief=tk.FLAT,
                cursor='hand2', padx=2, pady=3)

        source_var = tk.StringVar(value=next(iter(sources)))
        source_button = echem_source_button(
            controls,
            source_var,
            list(sources),
            command=lambda name: win.after_idle(_on_source_change, name),
            on_add=self._choose_additional_echem,
        )
        source_button.pack(side="left", padx=(2, OPERAXNTheme.PADDING['small']))

        _label(controls, "Mass (mg):").pack(side="left")
        mass_var = tk.StringVar(value="0")
        mass_entry = tk.Spinbox(
            controls, textvariable=mass_var, from_=0.0, to=99999.0,
            increment=0.1, format="%.2f", width=7,
            command=lambda: _queue_mass_replot(),
            bg=OPERAXNTheme.COLORS['input_bg'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            buttonbackground=OPERAXNTheme.COLORS['bg_secondary'],
            relief=tk.FLAT, font=OPERAXNTheme.FONTS['small'],
            highlightbackground=OPERAXNTheme.COLORS['border'],
            highlightthickness=1)
        mass_entry.pack(side="left", padx=(2, OPERAXNTheme.PADDING['small']))

        _label(controls, "Cycle:").pack(side="left")
        cycle_frame = tk.Frame(controls, bg=OPERAXNTheme.COLORS['bg_secondary'])
        cycle_frame.pack(side="left", padx=(2, OPERAXNTheme.PADDING['small']))
        cycle_var = tk.IntVar(value=1)
        prev_cycle = _nav_button(
            cycle_frame, '◀', lambda: _step_cycle(-1))
        prev_cycle.pack(side="left", padx=1)
        cycle_spin = tk.Spinbox(
            cycle_frame, textvariable=cycle_var, from_=1, to=1, width=4,
            command=lambda: _replot(),
            bg=OPERAXNTheme.COLORS['input_bg'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            buttonbackground=OPERAXNTheme.COLORS['bg_secondary'],
            relief=tk.FLAT, font=OPERAXNTheme.FONTS['small'],
            highlightbackground=OPERAXNTheme.COLORS['border'],
            highlightthickness=1)
        cycle_spin.pack(side="left", padx=2)
        next_cycle = _nav_button(
            cycle_frame, '▶', lambda: _step_cycle(1))
        next_cycle.pack(side="left", padx=1)

        all_cycles_var = tk.BooleanVar(value=True)
        all_cycles_check = tk.Checkbutton(
            controls, text="All cycles", variable=all_cycles_var,
            command=lambda: _toggle_all_cycles(),
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['bg_secondary'],
            activeforeground=OPERAXNTheme.COLORS['text_primary'],
            disabledforeground=OPERAXNTheme.COLORS['text_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            font=OPERAXNTheme.FONTS['small'])
        all_cycles_check.pack(side="left", padx=(0, OPERAXNTheme.PADDING['small']))

        cycles_label = _label(controls, "", dim=True)
        cycles_label.pack(side="left", padx=(0, OPERAXNTheme.PADDING['small']))

        def _refresh_available() -> None:
            try:
                current["available"] = sorted(
                    c for c in assign_cycles(current["df"])["cycle"].unique() if c > 0)
            except Exception:
                logger.debug("Cycle detection failed: %s", traceback.format_exc())
                current["available"] = []
            cycles_label.config(text=f"({len(current['available'])} available)")
            maximum = max(current["available"], default=1)
            cycle_spin.config(to=maximum)
            if cycle_var.get() not in current["available"]:
                cycle_var.set(current["available"][0] if current["available"] else 1)

        _refresh_available()

        def _get_mass() -> float:
            try:
                return float(mass_var.get())
            except (ValueError, tk.TclError):
                return 0.0

        def _get_cycles() -> List[int]:
            if all_cycles_var.get():
                return list(current["available"])
            selected = cycle_var.get()
            return [selected] if selected in current["available"] else []

        # Plot area, bordered like the main-window plot container. The figure
        # has no fixed size — fractional margins keep the layout correct at
        # any window size, unlike a one-shot tight_layout.
        plot_bg = '#ffffff'
        plot_text = '#20242c'
        plot_dim = '#4b5563'
        plot_border = '#9ca3af'
        plot_grid = '#d8dde6'
        plot_frame = tk.Frame(
            win,
            bg=plot_bg,
            relief=tk.FLAT,
            highlightbackground=OPERAXNTheme.COLORS['border'],
            highlightcolor=OPERAXNTheme.COLORS['accent_primary'],
            highlightthickness=2
        )
        plot_frame.pack(fill="both", expand=True,
                        padx=OPERAXNTheme.PADDING['small'],
                        pady=(2, OPERAXNTheme.PADDING['small']))
        plot_frame.pack_propagate(False)

        fig = Figure(facecolor=plot_bg, dpi=FIGURE_DPI)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.12, wspace=0.24)
        ax_time, ax_cap = fig.subplots(1, 2)

        def _style_live_axes() -> None:
            for ax in (ax_time, ax_cap):
                ax.set_facecolor(plot_bg)
                ax.tick_params(colors=plot_dim, labelsize=8)
                ax.xaxis.label.set_color(plot_dim)
                ax.yaxis.label.set_color(plot_dim)
                ax.grid(True, color=plot_grid, alpha=0.75, linewidth=0.5)
                for spine in ax.spines.values():
                    spine.set_color(plot_border)
                legend = ax.get_legend()
                if legend is not None:
                    legend.get_frame().set_facecolor(plot_bg)
                    legend.get_frame().set_edgecolor(plot_border)
                    for text in legend.get_texts():
                        text.set_color(plot_text)

        plot_time_vs_voltage(ax_time, current["df"])
        plot_capacity_vs_voltage(ax_cap, current["df"], mass_mg=0.0)
        _style_live_axes()

        canvas = embed_figure(fig, plot_frame)

        # Watch until the initial layout and DPI adjustment have both settled;
        # their event order is not deterministic, so one-shot timers can miss.
        sync_state = {"attempts": 0}

        def _sync_watchdog() -> None:
            if not win.winfo_exists():
                return
            sync_figure_to_widget(canvas)
            sync_state["attempts"] += 1
            if sync_state["attempts"] < 12:
                win.after(200, _sync_watchdog)

        win.after(150, _sync_watchdog)

        def _replot(*_args) -> None:
            selected = _get_cycles()
            plot_time_vs_voltage(ax_time, current["df"], cycles_to_plot=selected)
            plot_capacity_vs_voltage(ax_cap, current["df"], mass_mg=_get_mass(),
                                     cycles_to_plot=selected)
            _style_live_axes()
            canvas.draw()

        mass_replot_job = {"id": None}

        def _queue_mass_replot(*_args) -> None:
            """Debounce typing while keeping mass changes live."""
            if mass_replot_job["id"] is not None:
                try:
                    win.after_cancel(mass_replot_job["id"])
                except tk.TclError:
                    pass

            def update() -> None:
                mass_replot_job["id"] = None
                _replot()

            mass_replot_job["id"] = win.after(150, update)

        def _step_cycle(delta: int) -> None:
            available = current["available"]
            if not available:
                return
            all_cycles_var.set(False)
            selected = cycle_var.get()
            try:
                index = available.index(selected)
            except ValueError:
                index = 0
            cycle_var.set(available[max(0, min(len(available) - 1, index + delta))])
            _toggle_all_cycles(replot=False)
            _replot()

        def _toggle_all_cycles(replot: bool = True) -> None:
            state = "disabled" if all_cycles_var.get() else "normal"
            for widget in (prev_cycle, cycle_spin, next_cycle):
                widget.config(state=state)
            if replot:
                _replot()

        def _on_source_change(name: str) -> None:
            if name == current["name"] or name not in sources:
                return
            win.configure(cursor="watch")
            win.update_idletasks()
            try:
                current["name"] = name
                current["df"] = sources[name]
                cycle_var.set(1)
                all_cycles_var.set(True)
                _refresh_available()
                _toggle_all_cycles(replot=False)
                _replot()
            finally:
                win.configure(cursor="")

        def _refresh_sources(updated_sources: Dict[str, pd.DataFrame]) -> None:
            sources.clear()
            sources.update(updated_sources)
            source_button.set_values(sources)
            if current["name"] not in sources and sources:
                name = next(iter(sources))
                source_var.set(name)
                _on_source_change(name)

        win.refresh_sources = _refresh_sources

        def _export_csv() -> None:
            selected = _get_cycles()
            if not selected:
                messagebox.showwarning("No cycles", "No cycles selected to export.",
                                       parent=win)
                return
            path = filedialog.asksaveasfilename(
                parent=win, defaultextension=".csv",
                filetypes=[("CSV", "*.csv")], initialfile="capacity_vs_voltage.csv")
            if not path:
                return
            mass = _get_mass()
            df_full = assign_cycles(current["df"])
            rows = []
            for cycle_num in selected:
                cycle_df = df_full[df_full["cycle"] == cycle_num]
                for phase in ("charge", "discharge"):
                    phase_df = cycle_df[cycle_df["phase"] == phase]
                    if phase_df.empty:
                        continue
                    cap = compute_capacity(phase_df)
                    spec_cap = (cap / (mass / 1000.0) if mass > 0
                                else np.full_like(cap, np.nan))
                    for t, v, c, sc in zip(phase_df["t_s"].values,
                                           phase_df["echem_data"].values,
                                           cap, spec_cap):
                        rows.append({
                            "Cycle": cycle_num,
                            "Phase": phase,
                            "Time (s)": t,
                            "Voltage (V)": v,
                            "Capacity (mAh)": c,
                            "Specific Capacity (mAh/g)": sc,
                        })
            pd.DataFrame(rows).to_csv(path, index=False, encoding=CSV_ENCODING)
            messagebox.showinfo("Export complete", f"Saved data to:\n{path}", parent=win)

        export_button = StyledButton(controls, text="Export CSV", command=_export_csv,
                                     style="secondary", width=11)
        export_button.pack(side="left", padx=2)

        mass_entry.bind("<Return>", _replot)
        mass_entry.bind("<FocusOut>", _queue_mass_replot)
        mass_var.trace_add("write", _queue_mass_replot)
        cycle_spin.bind("<Return>", _replot)

        # Opens showing every cycle, so start with the stepper disabled
        _toggle_all_cycles(replot=False)

        self._capacity_window = win
        self.state.dialog_windows.append(win)
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(win))

    # ========================================================================
    # Export Operations
    # ========================================================================

    def _export_data(self) -> None:
        """Collect image-export options, then export one scan or a range."""
        if not self.state.scans:
            return

        options = self._get_export_options()
        if options:
            self._perform_export(options)

    def _get_export_options(self) -> Optional[ExportOptions]:
        """Show the export dialog; None when cancelled or already open."""
        if self._lift_existing_dialog('_export_options_dialog'):
            return None
        is_neutron = (self.state.data_source == DataSourceType.NEUTRON)
        dialog = ExportOptionsDialog(self.master, is_neutron=is_neutron)
        self._export_options_dialog = dialog
        return dialog.get_result()

    def _perform_export(self, options: ExportOptions) -> None:
        """Dispatch to single- or multi-scan export, reporting failures."""
        try:
            if options.export_type == "current":
                self._export_current_scan(options.dpi, options.plot_types)
            else:
                self._export_multiple_scans(options.dpi, options.plot_types)
        except Exception as e:
            self._show_message("Error", f"Export failed: {str(e)}", "error")

    def _export_nxs(self) -> None:
        """Save the canonical NeXus file for the loaded experiment.

        The file already contains everything chosen at generation time
        (standard echem, 2D images or references), so this is a plain save.
        """
        if not self.state.scans:
            self._show_message("Error", "No data loaded", "error")
            return

        loaded_path = get_loaded_nxs_path()
        initial_name = os.path.basename(loaded_path) if loaded_path else "experiment.nxs"

        filename = filedialog.asksaveasfilename(
            parent=self.master,
            title="Export NeXus File",
            defaultextension=".nxs",
            filetypes=[("NeXus files", "*.nxs"), ("All files", "*.*")],
            initialfile=initial_name
        )
        if not filename:
            return

        def export_task():
            try:
                # No listbox progress here: the scan list is populated during
                # exports and progress messages would overwrite its entries
                success, messages = export_nxs(filename)
                if success:
                    self.after(0, lambda: self._show_message(
                        "Success", f"NeXus file exported to:\n{filename}", "info"))
                else:
                    self.after(0, lambda: self._show_message(
                        "Error", "NeXus export failed:\n" + "\n".join(messages), "error"))
            except Exception as e:
                logger.debug("NeXus export error: %s", traceback.format_exc())
                self.after(0, lambda err=e: self._show_message(
                    "Error", f"NeXus export failed: {err}", "error"))

        threading.Thread(target=export_task, daemon=True).start()

    def _export_current_scan(self, dpi: int, plot_types: Dict[str, bool]) -> None:
        """Save the selected scan as PNG/PDF/SVG with cropped 2D extent."""
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
        """Batch-export a typed scan range as PNGs behind a progress bar."""
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
        """Map crop percentages onto the source's base extent rectangle."""
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
        """Collect GIF settings and an output path, then render frames."""
        if not self.state.scans:
            self._show_message("Warning", "No scans available to create GIF", "warning")
            return

        if not IMAGEIO_AVAILABLE:
            self._show_message("Error",
                               "imageio is required for GIF creation.\n"
                               "Please install it using: pip install imageio",
                               "error")
            return

        if self._lift_existing_dialog('_gif_dialog'):
            return
        gif_dialog = GIFSettingsDialog(self.master, len(self.state.scans))
        self._gif_dialog = gif_dialog
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
        """Render frames into a temp dir and assemble them with imageio."""
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
        """Render one scan to a PNG in temp_dir; None if it has no data."""
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
        """Build the summary workbook on a worker thread and save it."""
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
                # bind now: `e` is unbound once the except block exits, and
                # the lambda only runs later on the Tk main loop
                error_message = f"Excel export failed: {e}"
                self.after(0, lambda: self._show_message("Error", error_message, "error"))

        # Run in thread
        thread = threading.Thread(target=export_task, daemon=True)
        thread.start()

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Assemble the per-scan summary table, sorted by scan number."""
        correlated_data_list = []

        for scan in self.state.scans:
            base_data = self._create_scan_summary(scan)
            correlated_data_list.append(base_data)

        df = pd.DataFrame(correlated_data_list)

        if not df.empty:
            df = df.sort_values('scan number').reset_index(drop=True)

        return df

    def _create_scan_summary(self, scan: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten one scan into a row with source-specific columns."""
        original_timestamp = scan.get("original_timestamp", scan.get("timestamp", ""))

        base_data = {
            "scan number": scan["scan_num"],
            "scan ID": "",
            "data source": self.state.data_source.value,
            "voltage (V)": scan.get("echem", ""),
            "current (mA)": scan.get("current", ""),
            "voltage timestamp": scan.get("echem_timestamp", ""),
            "exposure time (s)": scan.get("exposure_time", ""),
        }

        if self.state.data_source == DataSourceType.NEUTRON:
            # Neutron-specific data (banks are limited to 1-5 by the grouper)
            for i in range(1, 6):
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
                    if not base_data["scan ID"]:
                        any_file = files.get("tof") or files.get("d")
                        if any_file:
                            base_data["scan ID"] = NeutronFileGrouper.extract_scan_id(
                                os.path.basename(any_file)) or ""

            base_data["neutron_metadata"] = os.path.basename(scan["neutron_meta"]) if scan.get("neutron_meta") else ""
        else:
            # XRD data
            base_data["oned"] = os.path.basename(scan["oned"]) if scan.get("oned") else ""
            base_data["twod"] = os.path.basename(scan["twod"]) if scan.get("twod") else ""
            base_data["x-ray timestamp"] = original_timestamp

            if self.state.data_source == DataSourceType.SYNCHROTRON:
                for file_path in (scan.get("oned"), scan.get("twod")):
                    if file_path:
                        scan_id = SynchrotronFileGrouper.extract_scan_id(
                            os.path.basename(file_path))
                        if scan_id:
                            base_data["scan ID"] = scan_id
                            break

        return base_data

    def _save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Write the CSV, reordering columns first for neutron data."""
        if self.state.data_source == DataSourceType.NEUTRON:
            # Reorder columns for neutron data
            column_order = self._get_neutron_column_order(df)
            df = df[column_order]

        df.to_csv(filename, index=False, encoding=CSV_ENCODING)

    def _get_neutron_column_order(self, df: pd.DataFrame) -> List[str]:
        """Order columns: ids, times, echem, banks, metadata, then rest."""
        column_order = ["scan number", "scan ID", "data source"]

        # Add timestamp columns
        for col in ["neutron_timestamp", "neutron_start", "neutron_end"]:
            if col in df.columns:
                column_order.append(col)

        # Add voltage/current columns
        for col in ["voltage (V)", "current (mA)", "voltage timestamp",
                    "exposure time (s)"]:
            if col in df.columns:
                column_order.append(col)

        # Add bank columns
        for i in range(1, 6):
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
        """Save the scan summary plus operando echem and experiment sheets."""
        if self.state.data_source == DataSourceType.NEUTRON:
            df = df[self._get_neutron_column_order(df)]

        with pd.ExcelWriter(filename, engine=EXCEL_ENGINE) as writer:
            df.to_excel(writer, index=False, sheet_name="Scan Summary")
            self._format_sheet(writer.sheets["Scan Summary"], df)

            self._add_operando_echem_sheet(writer)
            self._add_experiment_sheet(writer)

            if self.state.data_source == DataSourceType.NEUTRON:
                self._add_neutron_statistics_sheet(writer, df)

    @staticmethod
    def _format_sheet(worksheet, df: pd.DataFrame) -> None:
        """Bold, frozen header row and readable column widths."""
        from openpyxl.styles import Font
        header_font = Font(bold=True)
        for cell in worksheet[1]:
            cell.font = header_font
        worksheet.freeze_panes = "A2"

        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[
                worksheet.cell(1, col_idx + 1).column_letter
            ].width = min(column_length + 3, 50)

    def _add_operando_echem_sheet(self, writer: pd.ExcelWriter) -> None:
        """Full operando echem series so the workbook is self-contained."""
        df = self.state.echem_df
        if df is None or df.empty:
            return

        out = pd.DataFrame({
            "timestamp": df["timestamp"].astype(str),
            "voltage (V)": df["echem_data"],
        })
        if "current" in df.columns:
            out["current (mA)"] = df["current"]

        out.to_excel(writer, index=False, sheet_name="Operando Echem")
        self._format_sheet(writer.sheets["Operando Echem"], out)

    def _add_experiment_sheet(self, writer: pd.ExcelWriter) -> None:
        """Experiment-level metadata from the canonical NeXus file."""
        meta = get_experiment_metadata()
        if not meta:
            return

        rows = []
        for key in ("title", "start_time", "end_time", "experiment_identifier",
                    "data_source", "correlation_method", "generator",
                    "generator_version", "total_scans"):
            if meta.get(key) is not None:
                rows.append({"field": key, "value": str(meta[key])})

        for group in ("instrument", "sample", "user"):
            info = meta.get(group)
            if isinstance(info, dict):
                if info.get("name"):
                    rows.append({"field": group, "value": str(info["name"])})
                source = info.get("source")
                if group == "instrument" and isinstance(source, dict):
                    for key in ("name", "type", "probe"):
                        if source.get(key):
                            rows.append({"field": f"source {key}",
                                         "value": str(source[key])})

        sample = meta.get("sample")
        if isinstance(sample, dict):
            for key in ("description", "preparation_date"):
                if sample.get(key):
                    rows.append({"field": f"sample {key}",
                                 "value": str(sample[key])})

        if meta.get("pre_sample_flightpath") is not None:
            rows.append({"field": "pre_sample_flightpath",
                         "value": str(meta["pre_sample_flightpath"])})

        # Known dataset names only (the flattened group also carries NX_class)
        protocol = meta.get("cycling_protocol")
        if isinstance(protocol, dict):
            for key in ("technique", "voltage_window_lower",
                        "voltage_window_upper", "C_rate", "instrument",
                        "software", "raw_data_file"):
                if protocol.get(key) is not None:
                    rows.append({"field": f"cycling protocol {key}",
                                 "value": str(protocol[key])})

        if not rows:
            return
        out = pd.DataFrame(rows)
        out.to_excel(writer, index=False, sheet_name="Experiment")
        self._format_sheet(writer.sheets["Experiment"], out)

    def _add_neutron_statistics_sheet(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """Append a per-scan statistics sheet with auto-sized columns."""
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
        """Count bank files and derive duration for one statistics row."""
        measurements_available = 0
        tof_files = 0
        d_files = 0

        for i in range(1, 6):
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
        """Parse '1-5' or '1,3,5' into sorted in-range scans; [] on error."""
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
        """Confirm, then close dialogs and reset state and UI to startup."""
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
        self._show_empty_state()

        self.file_list.update_items([])
        self.scan_selector.set_range(1, 1)
        self.scan_selector.set_value(1)

        self.button_panel.update_states({
            "plot_data": "disabled",
            "export_plots": "disabled",
            "capacity_plot": "disabled",
            "ici_analysis": "disabled",
            "heatmap": "disabled",
            "create_gif": "disabled",
            "export_data": "disabled",
            "export_nxs": "disabled"
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
        """Show an info/warning/error messagebox on the main window."""
        if msg_type == "info":
            messagebox.showinfo(title, message, parent=self.master)
        elif msg_type == "warning":
            messagebox.showwarning(title, message, parent=self.master)
        elif msg_type == "error":
            messagebox.showerror(title, message, parent=self.master)
