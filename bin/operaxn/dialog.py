"""
Themed Tkinter dialogs for upload options, plot settings, exports, GIFs and progress.

BaseDialog provides the house style: non-modal, shrink-to-fit sizing, surface-matched
widget helpers, Upload Data style sections, one shared width for the Image, GIF and Plot
Settings pages, and in-dialog progress for the runs started from them.
"""

import ctypes
import logging
import os
import sys
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox
from typing import Dict, List, Callable, Optional, Tuple, Any

import numpy as np
import pandas as pd

from core import ECHEM_TIME_TOLERANCE, TimeMethod, is_canonical_nxs, read_data_source
from core.config import MAX_ECHEM_TIME_TOLERANCE

from .output import clear_plot_cache
from .config import (
    DataSourceType,
    DSPACING_LABEL,
    TOF_LABEL,
    DEFAULT_GIF_DPI,
    DEFAULT_GIF_FPS,
    DEFAULT_GIF_LOOP,
    DEFAULT_TWOD_XMAX_PERCENT,
    DEFAULT_TWOD_XMIN_PERCENT,
    EXPORT_DPI,
    OPERAXNTheme,
    SETTINGS_DIALOG_WIDTH_IN,
    SYNCHROTRON_MAX_DISPLAY_SIZE,
    SYNCHROTRON_WAVELENGTH,
    WINDOW_SIZES,
    XRAY_WAVELENGTH,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class ExportOptions:
    """Structured result from ExportOptionsDialog."""
    export_type: str
    dpi: int
    plot_types: Dict[str, bool]
    scan_list: Optional[List[int]] = None  # range exports only


@dataclass
class GIFSettings:
    """Structured result from GIFSettingsDialog."""
    fps: int
    dpi: int
    loop: int
    scan_list: List[int] = field(default_factory=list)


@dataclass
class UploadOptions:
    """Structured result from UploadOptionsDialog."""
    paths: List[str]
    data_source: "DataSourceType"
    time_method: "TimeMethod"
    display_size: int
    include_2d: bool
    title: str = None
    sample_name: str = None
    sample_description: str = None
    sample_preparation_date: str = None
    cycling_protocol: Dict[str, Any] = None
    standard_echem_files: List[str] = None
    echem_time_tolerance: float = None  # seconds; None = core default


def _style_progressbar() -> None:
    """Theme the app-global ttk progress bar in the accent colour."""
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TProgressbar",
                    background=OPERAXNTheme.COLORS['accent_primary'],
                    troughcolor=OPERAXNTheme.COLORS['bg_tertiary'],
                    borderwidth=0,
                    bordercolor=OPERAXNTheme.COLORS['border'],
                    lightcolor=OPERAXNTheme.COLORS['accent_primary'],
                    darkcolor=OPERAXNTheme.COLORS['accent_primary'])


# ============================================================================
# Base Dialog Class
# ============================================================================

class BaseDialog(tk.Toplevel):
    """Themed non-modal Toplevel, centred on its parent, with surface-matched widget helpers."""

    def __init__(self, master: tk.Misc, title: str, geometry: Optional[str] = None) -> None:
        super().__init__(master)
        self.master = master
        self.result = None
        self.configure(bg=OPERAXNTheme.COLORS['bg_primary'])

        self.title(title)
        if geometry:
            self.geometry(geometry)
        self.transient(master)

        self._center_window()

        # Dialogs are deliberately non-modal (no grab_set): they float above
        # the app (transient) but the main window stays fully interactive.
        self.bind('<Escape>', lambda e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def _work_area(self) -> Optional[Tuple[int, int, int, int]]:
        """Return this dialog's monitor work area (left, top, right, bottom), or None.
        Per monitor, as SPI_GETWORKAREA (the fallback) sees only the primary; no DPI
        awareness is set, so these USER32 values arrive in Tk's virtualised units."""
        if sys.platform != "win32":
            return None
        try:
            monitor = ctypes.windll.user32.MonitorFromWindow(
                self.winfo_id(), 2)  # MONITOR_DEFAULTTONEAREST

            class _MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_ulong),
                            ("rcMonitor", ctypes.c_long * 4),
                            ("rcWork", ctypes.c_long * 4),
                            ("dwFlags", ctypes.c_ulong)]

            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(_MONITORINFO)
            if monitor and ctypes.windll.user32.GetMonitorInfoW(
                    monitor, ctypes.byref(info)):
                return tuple(info.rcWork)

            rect = (ctypes.c_long * 4)()
            if ctypes.windll.user32.SystemParametersInfoW(
                    0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
                return tuple(rect)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug(f"Work area query failed: {exc}")
        return None

    def _frame_overhead(self) -> Tuple[int, int]:
        """Return the window decoration's (horizontal, vertical) size in Tk's own units.
        Tk units keep it in step under DPI virtualisation; valid once mapped. Implausible
        readings give (0, 0), which costs a slightly tall window, never a broken one."""
        side = self.winfo_rootx() - self.winfo_x()
        top = self.winfo_rooty() - self.winfo_y()
        if not (0 <= side <= 200 and 0 <= top <= 200):
            return 0, 0
        return 2 * side, top + side

    @staticmethod
    def _fit_client(work_area: Optional[Tuple[int, int, int, int]],
                    overhead: Tuple[int, int],
                    fallback: Tuple[int, int]) -> Tuple[int, int]:
        """Return the largest client size whose framed window fits the work area.
        An unknown area gives `fallback` (wm_maxsize); static, so testable headless."""
        if work_area is None:
            return fallback
        left, top, right, bottom = work_area
        return (max(200, right - left - overhead[0]),
                max(200, bottom - top - overhead[1]))

    def _center_window(self, width: Optional[int] = None,
                       height: Optional[int] = None) -> None:
        """Centre the dialog over its parent, clamped to the work area."""
        self.update_idletasks()

        # Get parent position and size
        px = self.master.winfo_x()
        py = self.master.winfo_y()
        pw = self.master.winfo_width()
        ph = self.master.winfo_height()

        # Get dialog size
        dw = self.winfo_width() if width is None else width
        dh = self.winfo_height() if height is None else height

        # Calculate centered position
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 2

        # Keep the whole frame on the usable desktop, not merely on screen:
        # clamping to screen height alone parks tall dialogs under the taskbar.
        area = self._work_area()
        left, top, right, bottom = area if area else (
            0, 0, self.winfo_screenwidth(), self.winfo_screenheight())
        h_over, v_over = self._frame_overhead() if area else (0, 0)

        x = max(left, min(x, right - (dw + h_over)))
        y = max(top, min(y, bottom - (dh + v_over)))

        self.geometry(f"+{x}+{y}")

    def shrink_to_fit(self, min_width: int = 340) -> None:
        """Size the window to its content plus margins and re-centre, once the widgets exist.
        Fixed geometries clip under Windows DPI scaling. Clamped to the work area less the
        decoration, so dialogs that can outgrow it must scroll."""
        self.update_idletasks()
        # wm_maxsize is only the fallback: on Windows it is the maximum track
        # size, which overshoots the work area and ignores the title bar.
        max_width, max_height = self._fit_client(
            self._work_area(), self._frame_overhead(), self.wm_maxsize())
        width = min(max(min_width, self.winfo_reqwidth() + 56), max_width)
        height = min(self.winfo_reqheight() + 14, max_height)
        self.geometry(f"{width}x{height}")
        self._center_window(width, height)

    def get_result(self) -> Any:
        """Block until the dialog closes; result is None when cancelled."""
        self.wait_window()
        return self.result

    def cancel(self) -> None:
        """Discard the result and close; bound to Escape and window close."""
        self.result = None
        self.destroy()

    # ========================================================================
    # Common UI Helper Methods
    # ========================================================================

    def _surface_bg(self, parent: tk.Widget) -> str:
        """Return `parent`'s background so themed widgets never show a mismatched box."""
        try:
            return parent.cget('bg')
        except tk.TclError:
            return OPERAXNTheme.COLORS['bg_primary']

    def create_themed_label(self, parent: tk.Widget = None, text: str = "",
                            font_type: str = "body",
                            fg_color: str = None,
                            **pack_options) -> tk.Label:
        """Create a surface-matched label, packed when pack_options given."""
        parent = parent or self
        font = OPERAXNTheme.FONTS.get(font_type, OPERAXNTheme.FONTS['body'])
        fg = fg_color or OPERAXNTheme.COLORS['text_primary']

        label = tk.Label(
            parent,
            text=text,
            font=font,
            bg=self._surface_bg(parent),
            fg=fg
        )

        if pack_options:
            label.pack(**pack_options)

        return label

    def create_themed_button(self, parent: tk.Widget = None, text: str = "",
                             command: Callable = None,
                             style: str = "primary",
                             width: int = 12) -> tk.Button:
        """Create a flat button whose background swaps on hover; unknown styles use secondary."""
        parent = parent or self

        # Style configuration
        styles = {
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

        style_config = styles.get(style, styles["secondary"])

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=style_config["bg"],
            fg=style_config["fg"],
            font=OPERAXNTheme.FONTS['button'],
            relief=tk.FLAT,
            cursor='hand2'
        )

        # Add hover effects
        btn.bind("<Enter>", lambda e: btn.config(bg=style_config["hover"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=style_config["bg"]))

        return btn

    def create_button_frame(self, buttons: List[Tuple[str, Callable, str]],
                            pady: int = 20, side: str = "top") -> tk.Frame:
        """Create and pack a button row from (text, command, style) tuples.
        side='bottom', packed before any expanding content, keeps the row visible on overflow."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(pady=pady, side=side)
        self._button_frame = frame
        self._buttons = []

        for text, command, style in buttons:
            btn = self.create_themed_button(frame, text, command, style)
            btn.pack(side="left", padx=5)
            self._buttons.append(btn)

        return frame

    def create_themed_entry(self, parent: tk.Widget = None, textvariable: tk.Variable = None,
                            width: int = 12, **kwargs) -> tk.Entry:
        """Create a themed entry; keyword args override the style defaults."""
        parent = parent or self

        defaults = {
            'textvariable': textvariable,
            'width': width,
            'bg': OPERAXNTheme.COLORS['input_bg'],
            'fg': OPERAXNTheme.COLORS['text_primary'],
            'insertbackground': OPERAXNTheme.COLORS['accent_primary'],
            'font': OPERAXNTheme.FONTS['body'],
            'relief': tk.FLAT,
            'borderwidth': 1,
            'highlightbackground': OPERAXNTheme.COLORS['border'],
            'highlightcolor': OPERAXNTheme.COLORS['accent_primary'],
            'highlightthickness': 1
        }
        defaults.update(kwargs)

        return tk.Entry(parent, **defaults)

    def create_themed_scale(self, parent: tk.Widget = None, variable: tk.Variable = None,
                            from_: float = 0, to: float = 100,
                            orient: str = "horizontal",
                            length: int = 200,
                            showvalue: bool = True,
                            command: Callable = None) -> tk.Scale:
        """Create a surface-matched scale with accent active colour."""
        parent = parent or self
        bg_color = self._surface_bg(parent)

        scale = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient=orient,
            variable=variable,
            length=length,
            showvalue=showvalue,
            command=command,
            bg=bg_color,
            fg=OPERAXNTheme.COLORS['text_primary'],
            troughcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            activebackground=OPERAXNTheme.COLORS['accent_primary'],
            highlightthickness=0
        )

        return scale

    def create_themed_radiobutton(self, parent: tk.Widget = None, text: str = "",
                                  variable: tk.Variable = None, value: Any = None,
                                  command: Callable = None,
                                  fg_color: str = None) -> tk.Radiobutton:
        """Create a surface-matched radio button with accent active text."""
        parent = parent or self
        bg_color = self._surface_bg(parent)
        fg = fg_color or OPERAXNTheme.COLORS['text_primary']

        rb = tk.Radiobutton(
            parent,
            text=text,
            variable=variable,
            value=value,
            command=command,
            font=OPERAXNTheme.FONTS['body'],
            bg=bg_color,
            fg=fg,
            activebackground=bg_color,
            activeforeground=OPERAXNTheme.COLORS['accent_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            highlightthickness=0
        )

        return rb

    def fit_height_in_place(self) -> None:
        """Refit the height to the content, within the work area, without moving the window."""
        self.update_idletasks()
        _, max_height = self._fit_client(
            self._work_area(), self._frame_overhead(), self.wm_maxsize())
        height = min(self.winfo_reqheight() + 14, max_height)
        self.geometry(f"{self.winfo_width()}x{height}")

    def fit_standard_size(self) -> None:
        """Size to the shared Image, GIF and Plot Settings width, the height fitted to the content.
        The width is in inches, so it holds at any display scaling."""
        self.shrink_to_fit(
            min_width=round(SETTINGS_DIALOG_WIDTH_IN * self.winfo_fpixels('1i')))

    # ========================================================================
    # Upload Data Layout Blocks
    # ========================================================================

    def create_section(self, parent: tk.Widget, title: str) -> tk.Frame:
        """Pack a section heading and return the centred block beneath it, as in Upload Data.
        The heading label is kept on the block as `heading`."""
        heading = self.create_themed_label(parent, title, font_type="heading", pady=(14, 4))
        block = tk.Frame(parent, bg=self._surface_bg(parent))
        block.pack()
        block.heading = heading
        return block

    def create_body(self) -> tk.Frame:
        """Pack and return the frame the sections go in, above the pinned button row."""
        body = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        body.pack(side="top", fill="both", expand=True)
        return body

    def create_themed_checkbutton(self, parent: tk.Widget, text: str, variable: tk.Variable,
                                  command: Callable = None) -> tk.Checkbutton:
        """Create a surface-matched check box with accent active text."""
        bg = self._surface_bg(parent)
        return tk.Checkbutton(
            parent, text=text, variable=variable, command=command,
            font=OPERAXNTheme.FONTS['body'],
            bg=bg,
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=bg,
            activeforeground=OPERAXNTheme.COLORS['accent_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            disabledforeground=OPERAXNTheme.COLORS['text_dim'],
            highlightthickness=0)

    def create_scale_row(self, parent: tk.Widget, row: int, label: str, variable: tk.Variable,
                         from_: float, to: float, resolution: float = 1, unit: str = "",
                         command: Callable = None) -> tk.Scale:
        """Grid a labelled slider and its live value on one row; command runs after each move."""
        self.create_themed_label(parent, label).grid(
            row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        value_label = self.create_themed_label(parent, "")
        value_label.config(width=5, anchor="w")

        def show_value() -> None:
            value_label.config(text=f"{float(variable.get()):g}{unit}")

        def moved(_value: str) -> None:
            show_value()
            if command:
                command()

        scale = self.create_themed_scale(parent, variable, from_, to, showvalue=False)
        if resolution != 1:
            scale.config(resolution=resolution)
        scale.config(command=moved)
        scale.grid(row=row, column=1, sticky="w", pady=2)
        value_label.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=2)
        show_value()
        return scale

    # ========================================================================
    # In-dialog Progress
    # ========================================================================

    def begin_progress(self, maximum: int, text: str = "Processing...") -> None:
        """Show a progress bar above the buttons for a run started from this dialog.
        The first button locks and the last becomes Stop; update_progress then reports progress."""
        bg = OPERAXNTheme.COLORS['bg_primary']
        self._stopped = False
        if getattr(self, "_progress_frame", None) is None:
            self._progress_frame = tk.Frame(self, bg=bg)
            self._progress_label = tk.Label(
                self._progress_frame, font=OPERAXNTheme.FONTS['small'], bg=bg,
                fg=OPERAXNTheme.COLORS['text_secondary'])
            self._progress_label.pack()
            _style_progressbar()
            self._progress_bar = ttk.Progressbar(
                self._progress_frame, mode='determinate', style="TProgressbar")
            self._progress_bar.pack(fill="x", padx=40, pady=(4, 0))
        self._progress_bar.config(maximum=max(1, maximum), value=0)
        self._progress_label.config(text=text)
        self._progress_frame.pack(side="bottom", fill="x", after=self._button_frame)
        first, last = self._buttons[0], self._buttons[-1]
        first.config(state="disabled")
        last.config(text="Stop", command=self._stop_run)
        # The height is fitted to the content, so grow it in place for the bar
        self.fit_height_in_place()
        self.update()

    def _stop_run(self) -> None:
        """Ask the running export to stop after its current item."""
        self._stopped = True

    def update_progress(self, value: int, text: str) -> bool:
        """Set the bar and status text and repaint; False once stopped or the dialog is closed."""
        if getattr(self, "_stopped", False) or not self.winfo_exists():
            return False
        self._progress_label.config(text=text)
        self._progress_bar['value'] = value
        self.update()
        return not self._stopped and bool(self.winfo_exists())


# ============================================================================
# Plot Settings Dialog
# ============================================================================

class PlotSettingsDialog(BaseDialog):
    """Axis-range and display-mode settings, applied live to the shared config and plots."""

    X_TWOTHETA_HEADING = "X-axis range (2θ [°]):"
    X_DSPACING_HEADING = "X-axis range (d-spacing [Å]):"

    def __init__(self, master: tk.Misc, config: Any, on_update: Callable) -> None:
        super().__init__(master, "Plot Settings")
        self.config = config
        self.on_update = on_update

        self._create_widgets()
        self.fit_standard_size()

    def _create_widgets(self) -> None:
        """Build the sections: neutron display mode and ranges, else 1D ranges and 2D crop."""
        is_neutron = getattr(self.config, 'data_source', None) == DataSourceType.NEUTRON
        apply = self._apply_neutron_settings if is_neutron else self._apply_settings
        self.create_button_frame([
            ("Apply", apply, "primary"),
            ("Close", self.destroy, "secondary")
        ], pady=14, side="bottom")
        self.create_themed_label(self, "Leave a field blank for automatic scaling.",
                                 font_type="small",
                                 fg_color=OPERAXNTheme.COLORS['text_secondary'],
                                 side="bottom", pady=(10, 0))

        body = self.create_body()
        if is_neutron:
            self._create_neutron_sections(body)
        else:
            self._create_oned_sections(body)
            self._create_twod_section(body)

    def _range_row(self, block: tk.Frame, min_var: tk.StringVar, max_var: tk.StringVar,
                   keys: Tuple[str, str], on_change: Callable) -> None:
        """Grid Min and Max entries side by side: Return applies, leaving a field stores it."""
        for col, (label, var, key) in enumerate(zip(("Min:", "Max:"), (min_var, max_var), keys)):
            self.create_themed_label(block, label).grid(
                row=0, column=2 * col, sticky="e", padx=(16 if col else 0, 6), pady=2)
            entry = self.create_themed_entry(block, var, width=10)
            entry.grid(row=0, column=2 * col + 1, pady=2)
            entry.bind("<Return>", lambda _e: on_change())
            entry.bind("<FocusOut>", lambda _e, k=key, v=var: self._validate_entry(k, v))

    def _validate_entry(self, key: str, var: tk.StringVar) -> None:
        """Store an entry's limit in config; blank or 'auto' means None (auto-scale).
        Anything else unparseable is cleared to blank and stored as None."""
        try:
            value_str = var.get().strip()

            if not value_str or value_str.lower() == 'auto':
                self._set_config_value(key, None)
                if not value_str:
                    var.set("")
            else:
                self._set_config_value(key, float(value_str))

        except (ValueError, tk.TclError):
            var.set("")
            self._set_config_value(key, None)

    def _set_config_value(self, key: str, value: Any) -> None:
        """Set the config attribute a field key maps to; unknown attributes are ignored."""
        mapping = {
            'xmin': 'oned_xmin', 'xmax': 'oned_xmax',
            'ymin': 'oned_ymin', 'ymax': 'oned_ymax',
        }

        config_key = mapping.get(key, key)
        if hasattr(self.config, config_key):
            setattr(self.config, config_key, value)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a limit for its entry field in full; None becomes blank (auto)."""
        return "" if value is None else str(value)

    # --- neutron ---

    def _create_neutron_sections(self, body: tk.Frame) -> None:
        """Build the display-mode choice and the X and intensity ranges."""
        mode = self.create_section(body, "Display mode:")
        self.neutron_display_var = tk.BooleanVar(value=self.config.show_neutron_dspacing)
        for text, value in ((TOF_LABEL, False), (DSPACING_LABEL, True)):
            self.create_themed_radiobutton(
                mode, text, self.neutron_display_var, value,
                self._on_neutron_mode_change).pack(anchor="w", pady=1)

        self.neutron_xmin_var = tk.StringVar(value=self._format_value(self.config.neutron_xmin))
        self.neutron_xmax_var = tk.StringVar(value=self._format_value(self.config.neutron_xmax))
        self._range_row(self.create_section(body, "X-axis range:"),
                        self.neutron_xmin_var, self.neutron_xmax_var,
                        ('neutron_xmin', 'neutron_xmax'), self._apply_neutron_settings)

        self.neutron_ymin_var = tk.StringVar(value=self._format_value(self.config.neutron_ymin))
        self.neutron_ymax_var = tk.StringVar(value=self._format_value(self.config.neutron_ymax))
        self._range_row(self.create_section(body, "Y-axis range (intensity):"),
                        self.neutron_ymin_var, self.neutron_ymax_var,
                        ('neutron_ymin', 'neutron_ymax'), self._apply_neutron_settings)

    def _on_neutron_mode_change(self) -> None:
        """Clear the X limits (TOF and d-spacing units differ) and apply."""
        self.neutron_xmin_var.set("")
        self.neutron_xmax_var.set("")
        self.config.neutron_xmin = None
        self.config.neutron_xmax = None

        self._apply_neutron_settings()

    def _apply_neutron_settings(self) -> None:
        """Validate neutron fields into config, clear plot cache, redraw."""
        if hasattr(self, 'neutron_display_var'):
            self.config.show_neutron_dspacing = self.neutron_display_var.get()

            for key in ['neutron_xmin', 'neutron_xmax', 'neutron_ymin', 'neutron_ymax']:
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    self._validate_entry(key, getattr(self, var_name))

            clear_plot_cache()

            if self.on_update:
                self.on_update()

    # --- X-ray 1D and 2D ---

    def _create_oned_sections(self, body: tk.Frame) -> None:
        """Build the 1D X range with its d-spacing toggle, and the intensity range."""
        x_block = self.create_section(body, self.X_TWOTHETA_HEADING)
        self.x_frame_label = x_block.heading
        # Stored limits are in the units shown when they were applied, so opening
        # in d-spacing mode only relabels them
        if self.config.show_dspacing:
            self.x_frame_label.config(text=self.X_DSPACING_HEADING)
        self.xmin_var = tk.StringVar(value=self._format_value(self.config.oned_xmin))
        self.xmax_var = tk.StringVar(value=self._format_value(self.config.oned_xmax))
        self._range_row(x_block, self.xmin_var, self.xmax_var, ('xmin', 'xmax'),
                        self._apply_settings)
        self.dspacing_var = tk.BooleanVar(value=self.config.show_dspacing)
        self.create_themed_checkbutton(
            x_block, "Show d-spacing instead of 2θ", self.dspacing_var,
            command=self._on_dspacing_toggle).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.ymin_var = tk.StringVar(value=self._format_value(self.config.oned_ymin))
        self.ymax_var = tk.StringVar(value=self._format_value(self.config.oned_ymax))
        self._range_row(self.create_section(body, "Y-axis range (intensity):"),
                        self.ymin_var, self.ymax_var, ('ymin', 'ymax'), self._apply_settings)

    def _create_twod_section(self, body: tk.Frame) -> None:
        """Build the 2D crop sliders, applied as they move."""
        crop = self.create_section(body, "2D image crop (%):")
        rows = [
            ("Y min:", 'ymin_2d_var', self.config.twod_ymin_percent),
            ("Y max:", 'ymax_2d_var', self.config.twod_ymax_percent),
            ("X min:", 'xmin_2d_var',
             getattr(self.config, 'twod_xmin_percent', DEFAULT_TWOD_XMIN_PERCENT)),
            ("X max:", 'xmax_2d_var',
             getattr(self.config, 'twod_xmax_percent', DEFAULT_TWOD_XMAX_PERCENT)),
        ]
        for row, (label, var_name, value) in enumerate(rows):
            var = tk.DoubleVar(value=value)
            setattr(self, var_name, var)
            self.create_scale_row(crop, row, label, var, 0, 100, unit="%",
                                  command=self._apply_settings)

    def _on_dspacing_toggle(self) -> None:
        """Convert entered X limits between 2-theta and d-spacing, then apply."""
        new_dspacing = self.dspacing_var.get()

        if new_dspacing != self.config.show_dspacing:
            if new_dspacing:
                self.x_frame_label.config(text=self.X_DSPACING_HEADING)
                self._convert_xaxis_to_dspacing()
            else:
                self.x_frame_label.config(text=self.X_TWOTHETA_HEADING)
                self._convert_xaxis_to_twotheta()

        self._apply_settings()

    def _get_wavelength(self) -> float:
        """Return the X-ray wavelength for the current data source."""
        if getattr(self.config, 'data_source', None) == DataSourceType.SYNCHROTRON:
            return SYNCHROTRON_WAVELENGTH
        return XRAY_WAVELENGTH

    def _convert_xaxis_to_dspacing(self) -> None:
        """Convert X-axis entry values from 2-theta to d-spacing."""
        wavelength = self._get_wavelength()

        for var in (self.xmin_var, self.xmax_var):
            value_str = var.get().strip()
            if value_str:
                try:
                    two_theta = float(value_str)
                    theta_rad = np.deg2rad(two_theta / 2.0)
                    if theta_rad > 0:
                        d_val = wavelength / (2 * np.sin(theta_rad))
                        var.set(f"{d_val:.4f}" if np.isfinite(d_val) else "")
                    else:
                        var.set("")
                except (ValueError, TypeError) as e:
                    logger.debug("Could not convert 2θ to d-spacing: %s", e)

    def _convert_xaxis_to_twotheta(self) -> None:
        """Convert X-axis entry values from d-spacing to 2-theta."""
        wavelength = self._get_wavelength()

        for var in (self.xmin_var, self.xmax_var):
            value_str = var.get().strip()
            if value_str:
                try:
                    d_val = float(value_str)
                    sin_theta = wavelength / (2 * d_val)
                    if 0 < sin_theta <= 1:
                        theta_rad = np.arcsin(sin_theta)
                        two_theta = 2 * np.rad2deg(theta_rad)
                        var.set(f"{two_theta:.2f}")
                    else:
                        var.set("")
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logger.debug("Could not convert d-spacing to 2θ: %s", e)

    def _apply_settings(self) -> None:
        """Validate every field into config and redraw; a d-spacing toggle clears the plot cache."""
        try:
            old_dspacing = self.config.show_dspacing if hasattr(self, 'dspacing_var') else False
            if hasattr(self, 'dspacing_var'):
                self.config.show_dspacing = self.dspacing_var.get()

            # Validate all 1D fields
            for key in ['xmin', 'xmax', 'ymin', 'ymax']:
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    self._validate_entry(key, getattr(self, var_name))

            # Update 2D settings
            for key, var_name in [('twod_ymin_percent', 'ymin_2d_var'),
                                  ('twod_ymax_percent', 'ymax_2d_var'),
                                  ('twod_xmin_percent', 'xmin_2d_var'),
                                  ('twod_xmax_percent', 'xmax_2d_var')]:
                if hasattr(self, var_name):
                    setattr(self.config, key, getattr(self, var_name).get())

            if self.on_update:
                self.on_update()

            # Clear cache if d-spacing changed
            if old_dspacing != getattr(self.config, 'show_dspacing', False):
                clear_plot_cache()

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e), parent=self)


# ============================================================================
# Scan Ranges
# ============================================================================

# Scan-range grammar shared by the Image and GIF pages (see parse_scan_range)
SCAN_RANGE_HINT = "e.g. 1-5, 1,3,5 or 1-5,10-15"


def _scan_range_row(dialog: BaseDialog, parent: tk.Widget, var: tk.StringVar,
                    indent: int = 0) -> tk.Entry:
    """Pack a scan-range entry with its grammar hint into parent and return the entry."""
    row = tk.Frame(parent, bg=dialog._surface_bg(parent))
    row.pack(anchor="w", pady=(6, 0))
    entry = dialog.create_themed_entry(
        row, var, width=16,
        disabledbackground=OPERAXNTheme.COLORS['bg_tertiary'],
        disabledforeground=OPERAXNTheme.COLORS['text_dim'])
    entry.pack(side="left", padx=(indent, 8))
    dialog.create_themed_label(row, SCAN_RANGE_HINT, font_type="small",
                               fg_color=OPERAXNTheme.COLORS['text_secondary'],
                               side="left")
    return entry


def parse_scan_range(text: str, max_scan: Optional[int] = None) -> List[int]:
    """Parse "1-5", "1,3,5" or "1-5,10-15" into sorted unique scan numbers from 1
    (up to max_scan when given); malformed input yields []."""
    try:
        scans = set()
        for part in text.replace(" ", "").split(","):
            if part.count("-") == 1:
                start, end = map(int, part.split("-"))
                scans.update(range(start, end + 1))
            else:
                scans.add(int(part))
    except (ValueError, TypeError, AttributeError):
        return []
    return [s for s in sorted(scans) if s >= 1 and (max_scan is None or s <= max_scan)]


# ============================================================================
# Export Options Dialog
# ============================================================================

class ExportOptionsDialog(BaseDialog):
    """Image export page: scans, plots and resolution.
    With on_export(options, dialog) the dialog stays open and the run shows its progress here."""

    def __init__(self, master: tk.Misc, is_neutron: bool = False, num_scans: int = 1,
                 on_export: Optional[Callable] = None) -> None:
        super().__init__(master, "Export Image")
        self.is_neutron = is_neutron
        self.num_scans = num_scans
        self.on_export = on_export
        self._create_widgets()
        self._update_range_state()
        self.fit_standard_size()

    def _create_widgets(self) -> None:
        """Build the scan and plot sections, resolution last; neutron offers echem only."""
        self.create_button_frame([
            ("Export", self._do_export, "primary"),
            ("Cancel", self.cancel, "secondary")
        ], pady=14, side="bottom")
        body = self.create_body()

        scans = self.create_section(body, "Scans:")
        self.export_var = tk.StringVar(value="current")
        for value, text in (("current", "Current scan only"),
                            ("multiple", "Scan range (one file per scan)")):
            self.create_themed_radiobutton(
                scans, text, self.export_var, value,
                command=self._update_range_state).pack(anchor="w", pady=1)
        self.range_var = tk.StringVar(value=f"1-{self.num_scans}")
        self.range_entry = _scan_range_row(self, scans, self.range_var, indent=24)

        plots = self.create_section(body, "Plot:")
        self.plot_vars = {}
        if self.is_neutron:
            plot_types = [("echem", "Echem values")]
        else:
            plot_types = [
                ("oned", "1D XRD data"),
                ("twod", "2D XRD data"),
                ("echem", "Echem values")
            ]
        for key, text in plot_types:
            self.plot_vars[key] = tk.BooleanVar(value=True)
            self.create_themed_checkbutton(
                plots, text, self.plot_vars[key]).pack(anchor="w", pady=1)

        # Resolution sits under the plot choices, gridded in its own row frame
        resolution = tk.Frame(plots, bg=self._surface_bg(plots))
        resolution.pack(anchor="w", pady=(6, 0))
        self.dpi_var = tk.IntVar(value=EXPORT_DPI)
        self.create_scale_row(resolution, 0, "Resolution (DPI):", self.dpi_var, 100, 600)

    def _update_range_state(self) -> None:
        """Enable the scan range only for a range export."""
        state = "normal" if self.export_var.get() == "multiple" else "disabled"
        self.range_entry.config(state=state)

    def _do_export(self) -> None:
        """Validate the choices, then hand them to on_export (or store them and close)."""
        plot_types = {k: v.get() for k, v in self.plot_vars.items()}
        if not any(plot_types.values()) and not self.is_neutron:
            messagebox.showwarning("No Selection",
                                   "Please select at least one plot type to export",
                                   parent=self)
            return

        scan_list = None
        if self.export_var.get() == "multiple":
            scan_list = parse_scan_range(self.range_var.get(), self.num_scans)
            if not scan_list:
                messagebox.showwarning("Invalid Range",
                                       "Please enter a valid scan range",
                                       parent=self)
                return

        self.result = ExportOptions(
            export_type=self.export_var.get(),
            dpi=self.dpi_var.get(),
            plot_types=plot_types,
            scan_list=scan_list,
        )
        if self.on_export is None:
            self.destroy()
        else:
            self.on_export(self.result, self)


# ============================================================================
# GIF Settings Dialog
# ============================================================================

class GIFSettingsDialog(BaseDialog):
    """GIF page: scan range and animation settings.
    With on_create(settings, dialog) the dialog stays open and the build shows its progress here."""

    def __init__(self, master: tk.Misc, num_scans: int,
                 on_create: Optional[Callable] = None) -> None:
        super().__init__(master, "Export GIF")
        self.num_scans = num_scans
        self.on_create = on_create
        self._create_widgets()
        self.fit_standard_size()

    def _create_widgets(self) -> None:
        """Build the scan-range and animation sections."""
        self.create_button_frame([
            ("Export", self._create, "primary"),
            ("Cancel", self.cancel, "secondary")
        ], pady=14, side="bottom")
        body = self.create_body()

        scans = self.create_section(body, "Scans:")
        self.range_var = tk.StringVar(value=f"1-{self.num_scans}")
        _scan_range_row(self, scans, self.range_var)

        animation = self.create_section(body, "Animation:")
        self.fps_var = tk.IntVar(value=DEFAULT_GIF_FPS)
        self.dpi_var = tk.IntVar(value=DEFAULT_GIF_DPI)
        self.loop_var = tk.IntVar(value=DEFAULT_GIF_LOOP)
        for row, (label, var, from_, to, resolution) in enumerate([
                ("Frames per second:", self.fps_var, 1, 20, 1),
                ("Resolution (DPI):", self.dpi_var, 50, 300, 10),
                ("Loop count (0 = infinite):", self.loop_var, 0, 10, 1)]):
            self.create_scale_row(animation, row, label, var, from_, to, resolution)

    def _create(self) -> None:
        """Validate the range, then hand the settings to on_create (or store them and close)."""
        scan_list = parse_scan_range(self.range_var.get(), self.num_scans)
        if not scan_list:
            messagebox.showwarning("Invalid Range",
                                   "Please enter a valid scan range",
                                   parent=self)
            return

        self.result = GIFSettings(
            fps=self.fps_var.get(),
            dpi=self.dpi_var.get(),
            loop=self.loop_var.get(),
            scan_list=scan_list,
        )
        if self.on_create is None:
            self.destroy()
        else:
            self.on_create(self.result, self)


# ============================================================================
# Upload Options Dialog
# ============================================================================

class UploadOptionsDialog(BaseDialog):
    """Single upload window for every load and generation option. A directly opened .nxs
    locks the source to the one it records and greys out generation-only options."""

    DISPLAY_OPTIONS = ["No downsampling", "4096", "2048", "1024", "512"]

    SOURCES = [
        ("inhouse", "Laboratory X-ray diffraction"),
        ("synchrotron", "Synchrotron X-ray diffraction"),
        ("neutron", "Time-of-flight neutron diffraction"),
    ]

    def __init__(self, master: tk.Misc, has_loaded_data: bool = False) -> None:
        super().__init__(master, "Upload Data", "540x700")
        self._has_loaded_data = has_loaded_data
        self._paths: List[str] = []
        self._std_echem: List[str] = []
        self._metadata_open = False
        # (path, is canonical, recorded data source) for the last single .nxs checked
        self._canonical_cache: Tuple[str, bool, Optional[str]] = ("", False, None)
        # The user's source choice, held while a .nxs's recorded source overrides it
        self._source_before_nxs: Optional[str] = None
        self._create_widgets()
        self._update_option_states()
        self._resize_to_content()

    # --- scrollable body (the button row stays pinned below it) ---

    def _build_scroll_area(self) -> tk.Frame:
        """Build the scrollable area and return the frame the sections go in.
        Its outer frame is packed last (see _create_widgets) so the button row keeps its space."""
        bg = OPERAXNTheme.COLORS['bg_primary']
        outer = tk.Frame(self, bg=bg)
        self._scroll_outer = outer

        self._canvas = tk.Canvas(outer, bg=bg, highlightthickness=0, bd=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(outer, orient="vertical",
                                       command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        body = tk.Frame(self._canvas, bg=bg)
        self._body_window = self._canvas.create_window((0, 0), window=body,
                                                       anchor="n")
        body.bind("<Configure>", lambda e: self._on_body_configure())
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>",
                                                             self._on_wheel))
        self._canvas.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))
        return body

    def _on_body_configure(self) -> None:
        """Keep the scroll region and scrollbar in step with the content."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        overflows = self._body.winfo_reqheight() > self._canvas.winfo_height() + 1
        if overflows and not self._scrollbar.winfo_ismapped():
            self._scrollbar.pack(side="right", fill="y")
        elif not overflows and self._scrollbar.winfo_ismapped():
            self._scrollbar.pack_forget()
            self._canvas.yview_moveto(0)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Centre the content and match its width to the viewport."""
        self._canvas.coords(self._body_window, event.width // 2, 0)
        self._canvas.itemconfigure(self._body_window, width=event.width)
        self._on_body_configure()

    def _on_wheel(self, event: tk.Event) -> None:
        """Scroll the content on the mouse wheel, only while it overflows."""
        if self._scrollbar.winfo_ismapped():
            self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _resize_to_content(self) -> None:
        """Resize to the content, clamped to the work area by shrink_to_fit."""
        self.update_idletasks()
        self._canvas.configure(height=self._body.winfo_reqheight())
        self.shrink_to_fit(min_width=540)
        self.update_idletasks()
        self._on_body_configure()

    def _toggle_metadata(self) -> None:
        """Expand or collapse the optional experiment-metadata section."""
        self._metadata_open = not self._metadata_open
        arrow = "▾" if self._metadata_open else "▸"
        self._meta_toggle.configure(
            text=f"{arrow}  Experiment metadata (optional)")
        if self._metadata_open:
            self._meta_frame.pack(after=self._meta_toggle)
        else:
            self._meta_frame.pack_forget()
        self._resize_to_content()

    # --- layout ---

    def _create_widgets(self) -> None:
        """Build all sections of the combined upload dialog."""
        self._body = self._build_scroll_area()

        # --- Input selection ---
        self.create_themed_label(self._body, "Input data:", font_type="heading",
                                 pady=(14, 4))
        input_frame = self._block()
        self.input_var = tk.StringVar(value="No data selected")
        self._entry(input_frame, self.input_var, width=20, readonly=True
                    ).pack(side="left", padx=(0, 8))
        self._small_button(input_frame, "Files", self._select_files)
        self._small_button(input_frame, "Directory", self._select_directory)

        # --- Data source ---
        self.create_themed_label(self._body, "Data source type:",
                                 font_type="heading", pady=(14, 4))
        source_frame = self._block()
        self.source_var = tk.StringVar(value="inhouse")
        self._source_radios: List[tk.Widget] = []
        for value, label in self.SOURCES:
            radio = self.create_themed_radiobutton(
                source_frame, label, self.source_var, value,
                command=self._update_option_states)
            radio.pack(anchor="w", pady=1)
            self._source_radios.append(radio)

        # --- Time correlation ---
        self.create_themed_label(self._body, "Time correlation:",
                                 font_type="heading", pady=(14, 4))
        time_frame = self._block()
        self.time_var = tk.StringVar(value=TimeMethod.ABSOLUTE.value)
        for value, label in [
            (TimeMethod.ABSOLUTE.value, "Absolute time (use actual timestamps)"),
            (TimeMethod.RELATIVE.value, "Relative time (re-zero unsynchronised clocks)"),
        ]:
            self.create_themed_radiobutton(
                time_frame, label, self.time_var, value).pack(anchor="w", pady=1)

        # Nearest-neighbour match bound, applied on both sides of the echem
        # record; generation-only and recorded in the file
        tolerance_row = tk.Frame(time_frame, bg=OPERAXNTheme.COLORS['bg_primary'])
        tolerance_row.pack(anchor="w", pady=(6, 0))
        self.create_themed_label(tolerance_row, "Echem time tolerance (s):",
                                 side="left", padx=(0, 8))
        self.tolerance_var = tk.StringVar(value=f"{ECHEM_TIME_TOLERANCE:g}")
        self.tolerance_entry = self._entry(tolerance_row, self.tolerance_var, width=8)
        self.tolerance_entry.pack(side="left")

        # --- 2D settings ---
        self.create_themed_label(self._body, "2D detector images:",
                                 font_type="heading", pady=(14, 4))
        twod_frame = self._block()
        size_row = tk.Frame(twod_frame, bg=OPERAXNTheme.COLORS['bg_primary'])
        size_row.pack(anchor="w")
        self.create_themed_label(size_row, "Max 2D size (n x n):",
                                 side="left", padx=(0, 8))
        self.size_var = tk.StringVar(value=str(SYNCHROTRON_MAX_DISPLAY_SIZE))
        self.size_menu = tk.OptionMenu(size_row, self.size_var, *self.DISPLAY_OPTIONS)
        self.size_menu.config(
            bg=OPERAXNTheme.COLORS['bg_tertiary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['bg_secondary'],
            activeforeground=OPERAXNTheme.COLORS['text_primary'],
            disabledforeground=OPERAXNTheme.COLORS['text_dim'],
            highlightthickness=0, relief=tk.FLAT, width=14,
            indicatoron=0,
            font=OPERAXNTheme.FONTS['body'], cursor='hand2')
        self.size_menu["menu"].config(
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['accent_primary'],
            activeforeground=OPERAXNTheme.COLORS['bg_primary'],
            font=OPERAXNTheme.FONTS['body'])
        self.size_menu.pack(side="left")

        self.include_2d_var = tk.BooleanVar(value=False)
        self.include_2d_check = self.create_themed_checkbutton(
            twod_frame, "Include 2D images in the NeXus file", self.include_2d_var)
        self.include_2d_check.pack(anchor="w", pady=(6, 0))

        # --- Experiment metadata: collapsed by default so the essentials
        # (input, source, correlation, 2D) fit without scrolling ---
        self._meta_toggle = self.create_themed_label(
            self._body, "▸  Experiment metadata (optional)",
            font_type="heading", fg_color=OPERAXNTheme.COLORS['accent_primary'],
            pady=(14, 4))
        self._meta_toggle.configure(cursor="hand2")
        self._meta_toggle.bind("<Button-1>", lambda e: self._toggle_metadata())
        self._meta_frame = tk.Frame(self._body,
                                    bg=OPERAXNTheme.COLORS['bg_primary'])

        self.create_themed_label(self._meta_frame, "Experiment details:",
                                 font_type="body", pady=(0, 4))
        # Both metadata grids stretch to the width of the widest one and
        # weight their value column, so every entry shares one right edge.
        details = self._block(self._meta_frame)
        details.pack_configure(fill="x")
        details.grid_columnconfigure(1, weight=1)
        # Kept so _update_option_states can grey the whole block: metadata is
        # generation-only and a directly opened .nxs would silently discard it.
        self._meta_entries: List[tk.Widget] = []
        self.title_var = tk.StringVar()
        self.sample_var = tk.StringVar()
        self.sample_desc_var = tk.StringVar()
        self.prep_date_var = tk.StringVar()
        for row, (label, var) in enumerate([
                ("Title:", self.title_var),
                ("Sample name:", self.sample_var),
                ("Cell description:", self.sample_desc_var),
                ("Preparation date (YYYY-MM-DD):", self.prep_date_var)]):
            self.create_themed_label(details, label).grid(
                row=row, column=0, sticky="e", padx=(0, 8), pady=2)
            entry = self._entry(details, var, width=28)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self._meta_entries.append(entry)

        # --- Cycling protocol (NXnote entry/cycling_protocol; generation-only) ---
        self.create_themed_label(self._meta_frame, "Cycling protocol:",
                                 font_type="body", pady=(12, 4))
        proto = self._block(self._meta_frame)
        # Only the fields an experimentalist knows offhand are asked for; the
        # definition also reserves instrument/software (settable through the
        # generate() API) and raw_data_file, which the writer fills from the
        # electrochemistry files actually ingested.
        self.proto_technique_var = tk.StringVar()
        self.proto_v_lower_var = tk.StringVar()
        self.proto_v_upper_var = tk.StringVar()
        self.proto_c_rate_var = tk.StringVar()

        def proto_label(parent: tk.Widget, text: str, row: int, col: int) -> None:
            self.create_themed_label(parent, text, font_type="body").grid(
                row=row, column=col, sticky="e", padx=(0, 6), pady=2)

        proto.pack_configure(fill="x")
        proto.grid_columnconfigure(1, weight=1)

        proto_label(proto, "Technique:", 0, 0)
        technique_entry = self._entry(proto, self.proto_technique_var, width=38)
        technique_entry.grid(row=0, column=1, sticky="ew", pady=2)
        self._meta_entries.append(technique_entry)

        # The two bounds share the value cell, each taking half of it.
        proto_label(proto, "Voltage window (V):", 1, 0)
        window_row = tk.Frame(proto, bg=OPERAXNTheme.COLORS['bg_primary'])
        window_row.grid(row=1, column=1, sticky="ew", pady=2)
        window_row.grid_columnconfigure(1, weight=1)
        window_row.grid_columnconfigure(3, weight=1)
        proto_label(window_row, "lower", 0, 0)
        v_lower_entry = self._entry(window_row, self.proto_v_lower_var, width=8)
        v_lower_entry.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        proto_label(window_row, "upper", 0, 2)
        v_upper_entry = self._entry(window_row, self.proto_v_upper_var, width=8)
        v_upper_entry.grid(row=0, column=3, sticky="ew")
        self._meta_entries.extend([v_lower_entry, v_upper_entry])

        proto_label(proto, "C-rate:", 2, 0)
        c_rate_entry = self._entry(proto, self.proto_c_rate_var, width=12)
        c_rate_entry.grid(row=2, column=1, sticky="ew", pady=2)
        self._meta_entries.append(c_rate_entry)

        # --- Standard echem ---
        self.create_themed_label(self._body,
                                 "Standard electrochemistry (optional):",
                                 font_type="heading", pady=(14, 4))
        echem_frame = self._block()
        self.std_echem_var = tk.StringVar(value="None")
        self._entry(echem_frame, self.std_echem_var, width=20, readonly=True
                    ).pack(side="left", padx=(0, 8))
        self.std_select_btn = self._small_button(
            echem_frame, "Select", self._select_std_echem)
        self.std_clear_btn = self._small_button(
            echem_frame, "Clear", self._clear_std_echem)

        # The button row and note are pinned to the bottom edge and packed
        # before the scrolling content, so they always keep their space.
        self.create_button_frame([
            ("Upload", self._confirm, "primary"),
            ("Cancel", self.cancel, "secondary")
        ], pady=14, side="bottom")

        # Note shown when generation options do not apply; packed above the
        # buttons only while it has text, so no blank line is reserved for it
        self.note_var = tk.StringVar(value="")
        self._note_label = tk.Label(self, textvariable=self.note_var,
                                    font=OPERAXNTheme.FONTS['small'],
                                    bg=OPERAXNTheme.COLORS['bg_primary'],
                                    fg=OPERAXNTheme.COLORS['text_secondary'])

        self._scroll_outer.pack(side="top", fill="both", expand=True)

    def _block(self, parent: tk.Widget = None) -> tk.Frame:
        """Pack and return a centred content block; widgets inside anchor west."""
        frame = tk.Frame(parent or self._body,
                         bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack()
        return frame

    def _entry(self, parent: tk.Widget, var: tk.StringVar, width: int,
               readonly: bool = False) -> tk.Entry:
        """Create a themed entry, optionally read-only (for picker-filled fields)."""
        kwargs = ({"state": "readonly",
                   "readonlybackground": OPERAXNTheme.COLORS['bg_tertiary']}
                  if readonly else {})
        return self.create_themed_entry(parent, var, width, **kwargs)

    @staticmethod
    def _small_button(parent: tk.Widget, text: str, command) -> tk.Button:
        """Create a compact themed picker button, packed to the left."""
        btn = tk.Button(parent, text=text, command=command, width=9,
                        bg=OPERAXNTheme.COLORS['button_bg'],
                        fg=OPERAXNTheme.COLORS['button_text'],
                        font=OPERAXNTheme.FONTS['small'],
                        relief=tk.FLAT, cursor='hand2')
        btn.bind("<Enter>", lambda e: btn.config(bg=OPERAXNTheme.COLORS['button_hover']))
        btn.bind("<Leave>", lambda e: btn.config(bg=OPERAXNTheme.COLORS['button_bg']))
        btn.pack(side="left", padx=2)
        return btn

    # --- selection handlers ---

    def _select_files(self) -> None:
        """Pick input data files and refresh the option states."""
        files = filedialog.askopenfilenames(
            parent=self, title="Select Files",
            filetypes=[
                ("All Supported",
                 "*.zip;*.dat;*.edf;*.txt;*.xlsx;*.csv;*.hdf;*.nxs;*.xy"),
                ("NeXus files", "*.nxs"),
                ("ZIP files", "*.zip"),
                ("All files", "*.*"),
            ])
        if files:
            self._paths = list(files)
            if len(files) == 1:
                self.input_var.set(os.path.basename(files[0]))
            else:
                self.input_var.set(f"{len(files)} files selected")
            self._update_option_states()

    def _select_directory(self) -> None:
        """Pick an input directory and refresh the option states."""
        directory = filedialog.askdirectory(parent=self, title="Select Directory")
        if directory:
            self._paths = [directory]
            self.input_var.set(directory)
            self._update_option_states()

    def _select_std_echem(self) -> None:
        """Pick standard (non-operando) echem files to store in the .nxs."""
        files = filedialog.askopenfilenames(
            parent=self, title="Select Standard Electrochemistry Files",
            filetypes=[("Supported files", "*.txt *.xlsx *.csv"),
                       ("All files", "*.*")])
        if files:
            self._std_echem = list(files)
            self.std_echem_var.set(f"{len(files)} file(s)")

    def _clear_std_echem(self) -> None:
        """Drop the selected standard echem files."""
        self._std_echem = []
        self.std_echem_var.set("None")

    # --- dynamic enabling ---

    def _is_direct_nxs(self) -> bool:
        """True when the selection is one canonical .nxs, as in input.process_paths.
        A third-party .nxs is regenerated, so its options stay live. Cached per path, as
        every source radio click re-runs the state update."""
        if len(self._paths) != 1 or not self._paths[0].lower().endswith(".nxs"):
            return False
        path = self._paths[0]
        if self._canonical_cache[0] != path:
            canonical = is_canonical_nxs(path)
            self._canonical_cache = (path, canonical,
                                     read_data_source(path) if canonical else None)
        return self._canonical_cache[1]

    def _sync_source_with_file(self, direct_nxs: bool) -> bool:
        """Show and lock a directly opened .nxs's recorded source, restoring the user's choice
        after. True while the file sets it; an unrecorded or unknown source stays selectable."""
        recorded = self._canonical_cache[2] if direct_nxs else None
        from_file = recorded in {value for value, _ in self.SOURCES}
        if from_file:
            if self._source_before_nxs is None:
                self._source_before_nxs = self.source_var.get()
            self.source_var.set(recorded)
        elif self._source_before_nxs is not None:
            self.source_var.set(self._source_before_nxs)
            self._source_before_nxs = None
        for radio in self._source_radios:
            radio.config(state="disabled" if from_file else "normal")
        return from_file

    def _update_option_states(self) -> None:
        """Enable/disable generation-only options and lock the source to match the selection."""
        direct_nxs = self._is_direct_nxs()
        source_from_file = self._sync_source_with_file(direct_nxs)
        source = self.source_var.get()

        # Size applies to synchrotron HDF data (display, and storage if included)
        self.size_menu.config(
            state="normal" if source == "synchrotron" else "disabled")

        # Generation-only options are fixed inside a directly opened .nxs
        gen_state = "disabled" if direct_nxs else "normal"
        has_2d_source = source in ("inhouse", "synchrotron")
        self.include_2d_check.config(
            state="normal" if (has_2d_source and not direct_nxs) else "disabled")
        if not has_2d_source or direct_nxs:
            self.include_2d_var.set(False)
        self.std_select_btn.config(state=gen_state)
        self.std_clear_btn.config(state=gen_state)
        self.tolerance_entry.config(state=gen_state)

        # Experiment metadata is generation-only too: _confirm drops it for a
        # direct .nxs, so leaving it typeable would silently discard the input.
        for entry in self._meta_entries:
            entry.config(state=gen_state)

        note = ("" if not direct_nxs
                else "Opening a NeXus file: source and generation options are fixed in the file"
                if source_from_file
                else "Opening a NeXus file: generation options are fixed in the file")
        self.note_var.set(note)
        if bool(note) != self._note_label.winfo_manager().startswith("pack"):
            if note:
                self._note_label.pack(side="bottom", pady=(10, 0), after=self._button_frame)
            else:
                self._note_label.pack_forget()
            # The first fit happens in __init__; later changes refit where the window is
            if self.winfo_ismapped():
                self.fit_height_in_place()

    # --- result ---

    def _confirm(self) -> None:
        """Build the UploadOptions result and close.
        An empty selection is valid only to append standard echem to already-loaded data."""
        if not self._paths and not (self._has_loaded_data and self._std_echem):
            messagebox.showerror("Error", "Please select files or a directory",
                                 parent=self)
            return

        size_val = self.size_var.get()
        display_size = 0 if size_val == "No downsampling" else int(size_val)

        # The whole metadata block is generation-only: a directly opened .nxs
        # ignores it, so the fields are greyed there and nothing is collected
        # (and their validation is skipped). Appending standard echem alone
        # ignores it too.
        if self._is_direct_nxs() or not self._paths:
            title = sample_name = sample_description = None
            prep_date, protocol, tolerance = None, None, None
        else:
            title = self.title_var.get().strip() or None
            sample_name = self.sample_var.get().strip() or None
            sample_description = self.sample_desc_var.get().strip() or None
            prep_date = self._validated_prep_date()
            if prep_date is False:
                return
            protocol = self._validated_protocol()
            if protocol is False:
                return
            tolerance = self._validated_tolerance()
            if tolerance is False:
                return

        self.result = UploadOptions(
            paths=list(self._paths),
            data_source=DataSourceType(self.source_var.get()),
            time_method=TimeMethod(self.time_var.get()),
            display_size=display_size,
            include_2d=self.include_2d_var.get(),
            title=title,
            sample_name=sample_name,
            sample_description=sample_description,
            sample_preparation_date=prep_date,
            cycling_protocol=protocol,
            standard_echem_files=list(self._std_echem) or None,
            echem_time_tolerance=tolerance,
        )
        self.destroy()

    def _validated_prep_date(self):
        """Return the stripped preparation date, or None when blank.
        False (after an error box) means it does not parse as ISO 8601."""
        text = self.prep_date_var.get().strip()
        if not text:
            return None
        try:
            pd.to_datetime(text, format='ISO8601')
        except (ValueError, TypeError):
            messagebox.showerror(
                "Invalid preparation date",
                f"'{text}' is not an ISO 8601 date (use YYYY-MM-DD).",
                parent=self)
            return False
        return text

    def _validated_tolerance(self):
        """Return the echem time tolerance in seconds; None (the core default) when blank.
        False (after an error box) means the field is not a positive finite number;
        anything above core's limit falls back to that limit."""
        text = self.tolerance_var.get().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            value = None
        if value is None or not np.isfinite(value) or value <= 0:
            messagebox.showerror(
                "Echem time tolerance",
                f"'{text}' is not a positive number of seconds",
                parent=self)
            return False
        return min(value, MAX_ECHEM_TIME_TOLERANCE)

    def _validated_protocol(self):
        """Return the cycling-protocol dict (blank fields omitted), or None when all are blank.
        False (after an error box) means a field is invalid."""
        fields = [
            ("technique", self.proto_technique_var),
            ("voltage_window_lower", self.proto_v_lower_var),
            ("voltage_window_upper", self.proto_v_upper_var),
            ("c_rate", self.proto_c_rate_var),
        ]
        values = {key: var.get().strip() for key, var in fields}
        values = {key: text for key, text in values.items() if text}
        if not values:
            return None
        if not values.get("technique"):
            messagebox.showerror(
                "Cycling protocol",
                "Technique is required when cycling protocol details are given",
                parent=self)
            return False
        for key, label in (("voltage_window_lower", "lower"),
                           ("voltage_window_upper", "upper")):
            if key in values:
                try:
                    values[key] = float(values[key])
                except ValueError:
                    messagebox.showerror(
                        "Cycling protocol",
                        f"Voltage window {label} bound '{values[key]}' "
                        "is not a number",
                        parent=self)
                    return False
        return values


# ============================================================================
# Progress Dialog
# ============================================================================

class ProgressDialog(BaseDialog):
    """Progress bar dialog for long-running operations."""

    def __init__(self, master: tk.Misc, title: str, maximum: int) -> None:
        super().__init__(master, title, WINDOW_SIZES['progress'])
        self.maximum = maximum
        self._create_widgets()
        self.shrink_to_fit()

    def _create_widgets(self) -> None:
        """Build the determinate bar and status label, forcing a paint."""
        self.label = self.create_themed_label(
            text="Processing...",
            font_type="body",
            pady=10
        )

        _style_progressbar()

        self.bar = ttk.Progressbar(
            self,
            length=250,
            mode='determinate',
            maximum=self.maximum,
            style="TProgressbar"
        )
        self.bar.pack(pady=10)
        self.update()

    def update_progress(self, value: int, text: str) -> bool:
        """Set the bar value and label text and repaint; return False once the dialog is closed."""
        if not self.winfo_exists():
            return False

        self.label.config(text=text)
        self.bar['value'] = value
        self.update()
        return True
