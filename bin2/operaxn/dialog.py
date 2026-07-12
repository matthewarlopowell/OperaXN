"""
Themed Tkinter dialogs. BaseDialog provides the house style (non-modal,
shrink-to-fit sizing, surface-matched widget helpers); concrete dialogs
cover upload options, plot settings, exports, GIF creation, and progress.
"""

import logging
import os
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox
from typing import Dict, List, Callable, Tuple, Any

import numpy as np

from core.model import TimeMethod

from .output import clear_plot_cache
from .config import (
    DataSourceType,
    DEFAULT_GIF_DPI,
    DEFAULT_GIF_FPS,
    DEFAULT_GIF_LOOP,
    DEFAULT_TWOD_XMAX_PERCENT,
    DEFAULT_TWOD_XMIN_PERCENT,
    EXPORT_DPI,
    OPERAXNTheme,
    SYNCHROTRON_WAVELENGTH,
    WINDOW_SIZES,
    XRAY_WAVELENGTH,
)


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class ExportOptions:
    """Structured result from ExportOptionsDialog."""
    export_type: str
    dpi: int
    plot_types: Dict[str, bool]


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
    standard_echem_files: List[str] = None



# ============================================================================
# Base Dialog Class
# ============================================================================

class BaseDialog(tk.Toplevel):
    """Base dialog with common functionality and theming."""

    def __init__(self, master: tk.Misc, title: str, geometry: str) -> None:
        super().__init__(master)
        self.master = master
        self.result = None

        # Apply theme
        self.configure(bg=OPERAXNTheme.COLORS['bg_primary'])

        self.title(title)
        self.geometry(geometry)
        self.transient(master)

        self._center_window()

        # Dialogs are deliberately non-modal (no grab_set): they float above
        # the app (transient) but the main window stays fully interactive.

        # Bind escape key to cancel
        self.bind('<Escape>', lambda e: self.cancel())

        # Protocol for window close
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def _center_window(self) -> None:
        """Center dialog on parent window."""
        self.update_idletasks()

        # Get parent position and size
        px = self.master.winfo_x()
        py = self.master.winfo_y()
        pw = self.master.winfo_width()
        ph = self.master.winfo_height()

        # Get dialog size
        dw = self.winfo_width()
        dh = self.winfo_height()

        # Calculate centered position
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 2

        # Ensure dialog is on screen
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        x = max(0, min(x, sw - dw))
        y = max(0, min(y, sh - dh))

        self.geometry(f"+{x}+{y}")

    def shrink_to_fit(self, min_width: int = 340) -> None:
        """Size the window to its content (plus margins) and re-center.

        Call at the end of _create_widgets: fixed pixel geometries clip
        content under Windows DPI scaling; content-driven sizing does not.
        """
        self.update_idletasks()
        width = max(min_width, self.winfo_reqwidth() + 56)
        height = self.winfo_reqheight() + 14
        self.geometry(f"{width}x{height}")
        self._center_window()

    def get_result(self) -> Any:
        """Get dialog result after closing."""
        self.wait_window()
        return self.result

    def cancel(self) -> None:
        """Cancel dialog and discard result."""
        self.result = None
        self.destroy()

    # ========================================================================
    # Common UI Helper Methods
    # ========================================================================

    def _surface_bg(self, parent: tk.Widget) -> str:
        """Background of the surface a widget sits on (its parent's bg), so
        themed widgets never show a mismatched box behind them."""
        try:
            return parent.cget('bg')
        except tk.TclError:
            return OPERAXNTheme.COLORS['bg_primary']

    def create_themed_label(self, parent: tk.Widget = None, text: str = "",
                            font_type: str = "body",
                            fg_color: str = None,
                            **pack_options) -> tk.Label:
        """Create a themed label."""
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
        """Create a themed button with hover effects."""
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
                            pady: int = 20) -> tk.Frame:
        """Create a button row from (text, command, style) tuples."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack(pady=pady)

        for text, command, style in buttons:
            btn = self.create_themed_button(frame, text, command, style)
            btn.pack(side="left", padx=5)

        return frame

    def create_themed_labelframe(self, parent: tk.Widget = None, text: str = "", **kwargs) -> tk.LabelFrame:
        """Create a themed label frame."""
        parent = parent or self
        bg_color = self._surface_bg(parent)

        defaults = {
            'text': text,
            'bg': bg_color,
            'fg': OPERAXNTheme.COLORS['text_primary'],
            'font': OPERAXNTheme.FONTS['heading'],
            'relief': tk.FLAT,
            'borderwidth': 1,
            'highlightbackground': OPERAXNTheme.COLORS['border'],
            'padx': 10,
            'pady': 10
        }
        defaults.update(kwargs)

        return tk.LabelFrame(parent, **defaults)

    def create_themed_entry(self, parent: tk.Widget = None, textvariable: tk.Variable = None,
                            width: int = 12, **kwargs) -> tk.Entry:
        """Create a themed entry widget."""
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
        """Create a themed scale widget."""
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
        """Create a themed radio button."""
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


# ============================================================================
# Plot Settings Dialog
# ============================================================================

class PlotSettingsDialog(BaseDialog):
    """Plot settings configuration dialog."""

    def __init__(self, master: tk.Misc, config: Any, on_update: Callable) -> None:
        is_neutron = (hasattr(config, 'data_source') and
                      config.data_source == DataSourceType.NEUTRON)
        height = "475" if is_neutron else "375"

        super().__init__(master, "Plot Settings", f"400x{height}")
        self.config = config
        self.on_update = on_update
        self.main_app = master

        self._create_widgets()
        self.shrink_to_fit()

    def _create_widgets(self) -> None:
        """Build plot settings tabs and controls."""
        # Setup notebook style
        self._setup_notebook_style()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        if hasattr(self.config, 'data_source'):
            if self.config.data_source == DataSourceType.NEUTRON:
                self._create_neutron_tab(notebook)
            else:
                self._create_oned_tab(notebook)
                self._create_twod_tab(notebook)
        else:
            self._create_oned_tab(notebook)
            self._create_twod_tab(notebook)

        # Close button
        close_btn = self.create_themed_button(
            text="Close",
            command=self.destroy,
            style="secondary"
        )
        close_btn.pack(pady=(0, 10))

    def _setup_notebook_style(self) -> None:
        """Configure themed notebook style."""
        C = OPERAXNTheme.COLORS
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=C['bg_primary'],
                        bordercolor=C['border'],
                        lightcolor=C['bg_primary'], darkcolor=C['bg_primary'])
        # clam tabs draw their face from light/dark edge colours too — set all
        # of them or unselected tabs keep the default light face
        style.configure('TNotebook.Tab',
                        background=C['bg_secondary'],
                        foreground=C['text_secondary'],
                        bordercolor=C['border'],
                        lightcolor=C['bg_secondary'], darkcolor=C['bg_secondary'],
                        padding=[12, 6], font=OPERAXNTheme.FONTS['body'])
        style.map('TNotebook.Tab',
                  background=[('selected', C['bg_tertiary']),
                              ('!selected', C['bg_secondary'])],
                  foreground=[('selected', C['accent_primary']),
                              ('!selected', C['text_secondary'])],
                  lightcolor=[('selected', C['bg_tertiary']),
                              ('!selected', C['bg_secondary'])])

    def _create_range_entry_row(self, parent: tk.Widget, label: str, var: tk.StringVar, row: int,
                                col: int, key: str, on_change: Callable = None) -> tk.Entry:
        """Create a labeled range entry row."""
        self.create_themed_label(parent, label, font_type="body").grid(
            row=row, column=col, padx=5, pady=5
        )

        entry = self.create_themed_entry(parent, var, width=12)
        entry.grid(row=row, column=col + 1, padx=5, pady=5)

        if not on_change:
            on_change = self._apply_settings

        entry.bind("<Return>", lambda e: on_change())
        entry.bind("<FocusOut>", lambda e: self._validate_entry(key, var))

        return entry

    def _create_scale_with_label(self, parent: tk.Widget, label: str, var_name: str,
                                 value: float, row: int) -> None:
        """Create scale with percentage value label."""
        self.create_themed_label(parent, label, font_type="body").grid(
            row=row, column=0, padx=5, pady=5, sticky='w'
        )

        setattr(self, var_name, tk.DoubleVar(value=value))
        var = getattr(self, var_name)

        scale = self.create_themed_scale(parent, var, 0, 100, showvalue=False)
        scale.grid(row=row, column=1, padx=5, pady=5)

        # Value label
        label_var_name = f"{var_name}_label"
        value_label = self.create_themed_label(
            parent, f"{value:.0f}%", font_type="body"
        )
        value_label.grid(row=row, column=2, padx=5, pady=5)
        setattr(self, label_var_name, value_label)

        # Update label on change
        scale.config(command=lambda v: [
            self._update_scale_label(var_name, v),
            self._apply_settings()
        ])

    def _update_scale_label(self, var_name: str, value: str) -> None:
        """Update scale value label."""
        label_var_name = f"{var_name}_label"
        if hasattr(self, label_var_name):
            label = getattr(self, label_var_name)
            label.config(text=f"{float(value):.0f}%")

    def _validate_entry(self, key: str, var: tk.StringVar) -> None:
        """Validate entry field; empty treated as auto-scale."""
        try:
            value_str = var.get().strip()

            if not value_str or value_str.lower() == 'auto':
                # Set to None for auto-scaling
                self._set_config_value(key, None)
                if not value_str:
                    var.set("")
            else:
                # Try to parse as float
                float_val = float(value_str)
                self._set_config_value(key, float_val)

        except (ValueError, tk.TclError):
            var.set("")
            self._set_config_value(key, None)

    def _set_config_value(self, key: str, value: Any) -> None:
        """Set config attribute by key name."""
        mapping = {
            'xmin': 'oned_xmin', 'xmax': 'oned_xmax',
            'ymin': 'oned_ymin', 'ymax': 'oned_ymax',
            'neutron_xmin': 'neutron_xmin', 'neutron_xmax': 'neutron_xmax',
            'neutron_ymin': 'neutron_ymin', 'neutron_ymax': 'neutron_ymax'
        }

        config_key = mapping.get(key, key)
        if hasattr(self.config, config_key):
            setattr(self.config, config_key, value)

    def _create_neutron_tab(self, notebook: ttk.Notebook) -> None:
        """Create neutron settings tab."""
        tab = tk.Frame(notebook, bg=OPERAXNTheme.COLORS['bg_secondary'])
        notebook.add(tab, text="Neutron Settings")

        # Display mode
        mode_frame = self.create_themed_labelframe(tab, "Display Mode")
        mode_frame.pack(fill="x", padx=10, pady=10)

        self.neutron_display_var = tk.BooleanVar(value=self.config.show_neutron_dspacing)

        self.create_themed_radiobutton(
            mode_frame, "Time of Flight (TOF) (μs)",
            self.neutron_display_var, False,
            self._on_neutron_mode_change
        ).pack(anchor="w", pady=5)

        self.create_themed_radiobutton(
            mode_frame, "d-spacing (Å)",
            self.neutron_display_var, True,
            self._on_neutron_mode_change
        ).pack(anchor="w", pady=5)

        # X-axis range
        x_frame = self.create_themed_labelframe(tab, "X-Axis Range")
        x_frame.pack(fill="x", padx=10, pady=10)

        self.neutron_xmin_var = tk.StringVar(value=self._format_value(self.config.neutron_xmin))
        self.neutron_xmax_var = tk.StringVar(value=self._format_value(self.config.neutron_xmax))

        self._create_range_entry_row(x_frame, "Min:", self.neutron_xmin_var,
                                     0, 0, 'neutron_xmin', self._apply_neutron_settings)
        self._create_range_entry_row(x_frame, "Max:", self.neutron_xmax_var,
                                     0, 2, 'neutron_xmax', self._apply_neutron_settings)

        # Y-axis range
        y_frame = self.create_themed_labelframe(tab, "Y-Axis Range (Intensity)")
        y_frame.pack(fill="x", padx=10, pady=10)

        self.neutron_ymin_var = tk.StringVar(value=self._format_value(self.config.neutron_ymin, True))
        self.neutron_ymax_var = tk.StringVar(value=self._format_value(self.config.neutron_ymax, True))

        self._create_range_entry_row(y_frame, "Min:", self.neutron_ymin_var,
                                     0, 0, 'neutron_ymin', self._apply_neutron_settings)
        self._create_range_entry_row(y_frame, "Max:", self.neutron_ymax_var,
                                     0, 2, 'neutron_ymax', self._apply_neutron_settings)

        # Info label
        self.create_themed_label(
            tab,
            "Note: X-axis units change based on display mode.\nLeave fields blank for automatic scaling.",
            font_type="small",
            fg_color=OPERAXNTheme.COLORS['text_secondary'],
            pady=10
        )

    def _format_value(self, value: Any, scientific: bool = False) -> str:
        """Format value for display."""
        if value is None:
            return ""
        if scientific:
            return f"{value:.1e}"
        return str(value)

    def _on_neutron_mode_change(self) -> None:
        """Handle neutron display mode change."""
        # Clear X-axis limits when switching modes
        self.neutron_xmin_var.set("")
        self.neutron_xmax_var.set("")
        self.config.neutron_xmin = None
        self.config.neutron_xmax = None

        self._apply_neutron_settings()

    def _apply_neutron_settings(self) -> None:
        """Apply neutron settings and refresh plots."""
        if hasattr(self, 'neutron_display_var'):
            self.config.show_neutron_dspacing = self.neutron_display_var.get()

            # Validate all neutron fields
            for key in ['neutron_xmin', 'neutron_xmax', 'neutron_ymin', 'neutron_ymax']:
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    self._validate_entry(key, getattr(self, var_name))

            # Clear cache and update
            clear_plot_cache()

            if self.on_update:
                self.on_update()

            # Force redraw
            if hasattr(self.main_app, '_update_plots'):
                self.main_app._update_plots()

            if hasattr(self.main_app, 'canvas') and self.main_app.canvas:
                self.main_app.canvas.draw_idle()

    def _create_oned_tab(self, notebook: ttk.Notebook) -> None:
        """Create 1D settings tab."""
        tab = tk.Frame(notebook, bg=OPERAXNTheme.COLORS['bg_secondary'])
        notebook.add(tab, text="1D Settings")

        # X-axis range
        self.x_frame_label = self.create_themed_labelframe(
            tab, "X-Axis Range (2θ degrees)", pady=5
        )
        self.x_frame_label.pack(fill="x", padx=10, pady=10)

        self.xmin_var = tk.StringVar(value=self._format_value(self.config.oned_xmin))
        self.xmax_var = tk.StringVar(value=self._format_value(self.config.oned_xmax))

        self._create_range_entry_row(self.x_frame_label, "Min:", self.xmin_var, 0, 0, 'xmin')
        self._create_range_entry_row(self.x_frame_label, "Max:", self.xmax_var, 0, 2, 'xmax')

        # Y-axis range
        y_frame = self.create_themed_labelframe(tab, "Y-Axis Range (Intensity)", pady=5)
        y_frame.pack(fill="x", padx=10, pady=10)

        self.ymin_var = tk.StringVar(value=self._format_value(self.config.oned_ymin, True))
        self.ymax_var = tk.StringVar(value=self._format_value(self.config.oned_ymax, True))

        self._create_range_entry_row(y_frame, "Min:", self.ymin_var, 0, 0, 'ymin')
        self._create_range_entry_row(y_frame, "Max:", self.ymax_var, 0, 2, 'ymax')

        # D-spacing option
        self.dspacing_var = tk.BooleanVar(value=self.config.show_dspacing)
        dspacing_check = tk.Checkbutton(
            tab,
            text="Show d-spacing instead of 2θ",
            variable=self.dspacing_var,
            command=self._on_dspacing_toggle,
            bg=OPERAXNTheme.COLORS['bg_secondary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['bg_secondary'],
            activeforeground=OPERAXNTheme.COLORS['accent_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            font=OPERAXNTheme.FONTS['body']
        )
        dspacing_check.pack(pady=10)

        # Update label if in d-spacing mode
        if self.config.show_dspacing:
            self.x_frame_label.config(text="X-Axis Range (d-spacing [Å])")
            self._convert_xaxis_to_dspacing()

    def _on_dspacing_toggle(self) -> None:
        """Handle d-spacing toggle."""
        new_dspacing = self.dspacing_var.get()

        if new_dspacing != self.config.show_dspacing:
            if new_dspacing:
                self.x_frame_label.config(text="X-Axis Range (d-spacing [Å])")
                self._convert_xaxis_to_dspacing()
            else:
                self.x_frame_label.config(text="X-Axis Range (2θ degrees)")
                self._convert_xaxis_to_twotheta()

        self._apply_settings()

    def _get_wavelength(self) -> float:
        """Return the X-ray wavelength for the current data source."""
        if (hasattr(self.config, 'data_source') and
                self.config.data_source == DataSourceType.SYNCHROTRON):
            return SYNCHROTRON_WAVELENGTH
        return XRAY_WAVELENGTH

    def _convert_xaxis_to_dspacing(self) -> None:
        """Convert X-axis entry values from 2-theta to d-spacing."""
        wavelength = self._get_wavelength()

        for var, _ in [(self.xmin_var, 'xmin'), (self.xmax_var, 'xmax')]:
            value_str = var.get().strip()
            if value_str:
                try:
                    two_theta = float(value_str)
                    theta_rad = np.deg2rad(two_theta / 2.0)
                    if theta_rad > 0:
                        d_val = wavelength / (2 * np.sin(theta_rad))
                        if not np.isinf(d_val) and not np.isnan(d_val):
                            var.set(f"{d_val:.4f}")
                        else:
                            var.set("")
                    else:
                        var.set("")
                except (ValueError, TypeError) as e:
                    logging.getLogger(__name__).debug(
                        "Could not convert 2θ to d-spacing: %s", e)

    def _convert_xaxis_to_twotheta(self) -> None:
        """Convert X-axis entry values from d-spacing to 2-theta."""
        wavelength = self._get_wavelength()

        for var, _ in [(self.xmin_var, 'xmin'), (self.xmax_var, 'xmax')]:
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
                    logging.getLogger(__name__).debug(
                        "Could not convert d-spacing to 2θ: %s", e)

    def _apply_settings(self) -> None:
        """Apply all settings and refresh plots."""
        try:
            # Store old d-spacing setting
            old_dspacing = self.config.show_dspacing if hasattr(self, 'dspacing_var') else False

            # Update d-spacing
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

            # Trigger update
            if self.on_update:
                self.on_update()

            # Clear cache if d-spacing changed
            if old_dspacing != getattr(self.config, 'show_dspacing', False):
                clear_plot_cache()

                if hasattr(self.main_app, '_update_plots'):
                    self.main_app._update_plots()

            # Redraw
            if hasattr(self.main_app, 'canvas') and self.main_app.canvas:
                self.main_app.canvas.draw_idle()

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e), parent=self)

    def _create_twod_tab(self, notebook: ttk.Notebook) -> None:
        """Create 2D settings tab."""
        tab = tk.Frame(notebook, bg=OPERAXNTheme.COLORS['bg_secondary'])
        notebook.add(tab, text="2D Settings")

        # Y-axis cropping
        y_frame = self.create_themed_labelframe(tab, "Y-Axis Range (%)")
        y_frame.pack(fill="x", padx=10, pady=10)

        self._create_scale_with_label(y_frame, "Y Min:", 'ymin_2d_var',
                                      self.config.twod_ymin_percent, 0)
        self._create_scale_with_label(y_frame, "Y Max:", 'ymax_2d_var',
                                      self.config.twod_ymax_percent, 1)

        # X-axis cropping
        x_frame = self.create_themed_labelframe(tab, "X-Axis Range (%)")
        x_frame.pack(fill="x", padx=10, pady=10)

        xmin_default = getattr(self.config, 'twod_xmin_percent', DEFAULT_TWOD_XMIN_PERCENT)
        xmax_default = getattr(self.config, 'twod_xmax_percent', DEFAULT_TWOD_XMAX_PERCENT)

        self._create_scale_with_label(x_frame, "X Min:", 'xmin_2d_var', xmin_default, 0)
        self._create_scale_with_label(x_frame, "X Max:", 'xmax_2d_var', xmax_default, 1)


# ============================================================================
# Export Options Dialog
# ============================================================================

class ExportOptionsDialog(BaseDialog):
    """Export type and format selection dialog."""

    def __init__(self, master: tk.Misc, is_neutron: bool = False) -> None:
        self.is_neutron = is_neutron
        if is_neutron:
            window_size = WINDOW_SIZES['export_neutron']
        else:
            window_size = WINDOW_SIZES['export_xrd']
        super().__init__(master, "Export Options", window_size)
        self._create_widgets()
        self.shrink_to_fit()

    def _create_widgets(self) -> None:
        """Build export options widgets."""
        # Export type
        self.create_themed_label(text="Select export type:", font_type="heading", pady=10)

        self.export_var = tk.StringVar(value="current")
        options = [
            ("current", "Current scan only"),
            ("multiple", "Multiple scans (individual files)")
        ]

        for value, text in options:
            self.create_themed_radiobutton(
                text=text, variable=self.export_var, value=value
            ).pack(anchor="w", padx=20)

        # Plot types
        self.create_themed_label(
            text="Select plots to export:",
            font_type="heading",
            pady=(20, 10)
        )

        plot_frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        plot_frame.pack(pady=5)

        self.plot_vars = {}

        if self.is_neutron:
            # Neutron mode
            plot_types = [
                ("echem", "Echem Values")
            ]
        else:
            # X-ray mode
            plot_types = [
                ("oned", "1D XRD Data"),
                ("twod", "2D XRD Data"),
                ("echem", "Echem Values")
                ]

        for key, text in plot_types:
            self.plot_vars[key] = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(
                plot_frame,
                text=text,
                variable=self.plot_vars[key],
                bg=OPERAXNTheme.COLORS['bg_primary'],
                fg=OPERAXNTheme.COLORS['text_primary'],
                activebackground=OPERAXNTheme.COLORS['bg_primary'],
                activeforeground=OPERAXNTheme.COLORS['accent_primary'],
                selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
                font=OPERAXNTheme.FONTS['body']
            )
            cb.pack(anchor="w")

        # DPI setting
        self.create_themed_label(text="Export DPI:", font_type="heading", pady=(20, 5))

        self.dpi_var = tk.IntVar(value=EXPORT_DPI)
        dpi_scale = self.create_themed_scale(
            variable=self.dpi_var,
            from_=100, to=600,
            showvalue=True
        )
        dpi_scale.pack()

        # Buttons
        self.create_button_frame([
            ("Export", self._do_export, "primary"),
            ("Cancel", self.cancel, "secondary")
        ])

    def _do_export(self) -> None:
        """Validate selections and set export result."""
        if not any(var.get() for var in self.plot_vars.values()) and not self.is_neutron:
            messagebox.showwarning("No Selection",
                                   "Please select at least one plot type to export",
                                   parent=self)
            return

        self.result = ExportOptions(
            export_type=self.export_var.get(),
            dpi=self.dpi_var.get(),
            plot_types={k: v.get() for k, v in self.plot_vars.items()},
        )
        self.destroy()


# ============================================================================
# GIF Settings Dialog
# ============================================================================

class GIFSettingsDialog(BaseDialog):
    """GIF creation settings dialog."""

    def __init__(self, master: tk.Misc, num_scans: int) -> None:
        super().__init__(master, "Create GIF", WINDOW_SIZES['gif'])
        self.num_scans = num_scans
        self._create_widgets()
        self.shrink_to_fit()

    def _create_widgets(self) -> None:
        """Build GIF settings widgets."""
        # Scan range
        range_frame = self.create_themed_labelframe(self, "Scan Range:")
        range_frame.pack(fill="x", padx=20, pady=10)

        self.create_themed_label(
            range_frame, "Enter scan range:", font_type="body"
        ).pack(anchor="w")

        self.create_themed_label(
            range_frame,
            "Examples: 1-5 or 1,3,5 or 1-5,10-15",
            font_type="small",
            fg_color=OPERAXNTheme.COLORS['text_secondary']
        ).pack(anchor="w")

        self.range_var = tk.StringVar(value=f"1-{self.num_scans}")
        entry = self.create_themed_entry(range_frame, self.range_var, width=30)
        entry.pack(fill="x", pady=5)

        # GIF settings
        settings_frame = self.create_themed_labelframe(self, "GIF Settings:")
        settings_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.fps_var = tk.IntVar(value=DEFAULT_GIF_FPS)
        self.dpi_var = tk.IntVar(value=DEFAULT_GIF_DPI)
        self.loop_var = tk.IntVar(value=DEFAULT_GIF_LOOP)

        settings = [
            ("Frames per second:", self.fps_var, 1, 20, 1),
            ("Image quality (DPI):", self.dpi_var, 50, 300, 10),
            ("Loop count (0 = infinite):", self.loop_var, 0, 10, 1)
        ]

        for label, var, from_, to, resolution in settings:
            row = tk.Frame(settings_frame, bg=OPERAXNTheme.COLORS['bg_primary'])
            row.pack(fill="x", pady=5)

            self.create_themed_label(row, label, font_type="body").pack(side="left", padx=5)

            scale = self.create_themed_scale(
                row, var, from_, to, showvalue=True
            )
            if resolution > 1:
                scale.config(resolution=resolution)
            scale.pack(side="left", padx=5)

        # Buttons
        self.create_button_frame([
            ("Create GIF", self._create, "primary"),
            ("Cancel", self.cancel, "secondary")
        ])

    def _create(self) -> None:
        """Validate and set GIF creation result."""
        scan_list = self._parse_scan_range(self.range_var.get())
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
        self.destroy()

    def _parse_scan_range(self, scan_range: str) -> List[int]:
        """Parse scan range string."""
        try:
            scan_list = []
            parts = scan_range.replace(" ", "").split(",")

            for part in parts:
                if "-" in part and part.count("-") == 1:
                    start, end = map(int, part.split("-"))
                    scan_list.extend(range(start, end + 1))
                else:
                    scan_list.append(int(part))

            scan_list = sorted(set(scan_list))
            return [s for s in scan_list if 1 <= s <= self.num_scans]
        except (ValueError, TypeError, AttributeError):
            return []


# ============================================================================
# Upload Options Dialog
# ============================================================================

class UploadOptionsDialog(BaseDialog):
    """Single upload window collecting every load/generation option at once
    (data selection, source, time correlation, 2D handling, experiment
    details, standard echem) — replaces the old sequential dialog chain."""

    DISPLAY_OPTIONS = ["No downsampling", "4096", "2048", "1024", "512"]

    SOURCES = [
        ("inhouse", "Laboratory X-ray diffraction"),
        ("synchrotron", "Synchrotron X-ray diffraction"),
        ("neutron", "Time-of-flight neutron diffraction"),
    ]

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, "Load Data", "540x700")
        self._paths: List[str] = []
        self._std_echem: List[str] = []
        self._create_widgets()
        self._update_option_states()
        self.shrink_to_fit(min_width=540)

    # --- layout ---

    def _create_widgets(self) -> None:
        # --- Input selection ---
        self.create_themed_label(text="Input data:", font_type="heading", pady=(14, 4))
        input_frame = self._block()
        self.input_var = tk.StringVar(value="No data selected")
        self._entry(input_frame, self.input_var, width=20, readonly=True
                    ).pack(side="left", padx=(0, 8))
        self._small_button(input_frame, "Files", self._select_files)
        self._small_button(input_frame, "Directory", self._select_directory)

        # --- Data source ---
        self.create_themed_label(text="Data source type:", font_type="heading",
                                 pady=(14, 4))
        source_frame = self._block()
        self.source_var = tk.StringVar(value="inhouse")
        for value, label in self.SOURCES:
            self.create_themed_radiobutton(
                source_frame, label, self.source_var, value,
                command=self._update_option_states).pack(anchor="w", pady=1)

        # --- Time correlation ---
        self.create_themed_label(text="Time correlation:", font_type="heading",
                                 pady=(14, 4))
        time_frame = self._block()
        self.time_var = tk.StringVar(value=TimeMethod.ABSOLUTE.value)
        for value, label in [
            (TimeMethod.ABSOLUTE.value, "Absolute time (use actual timestamps)"),
            (TimeMethod.RELATIVE.value, "Relative time (re-zero unsynchronised clocks)"),
        ]:
            self.create_themed_radiobutton(
                time_frame, label, self.time_var, value).pack(anchor="w", pady=1)

        # --- 2D settings ---
        self.create_themed_label(text="2D detector images:", font_type="heading",
                                 pady=(14, 4))
        twod_frame = self._block()
        size_row = tk.Frame(twod_frame, bg=OPERAXNTheme.COLORS['bg_primary'])
        size_row.pack(anchor="w")
        tk.Label(size_row, text="Max 2D size (n x n):",
                 font=OPERAXNTheme.FONTS['body'],
                 bg=OPERAXNTheme.COLORS['bg_primary'],
                 fg=OPERAXNTheme.COLORS['text_primary']).pack(side="left", padx=(0, 8))
        self.size_var = tk.StringVar(value="4096")
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
        self.include_2d_check = tk.Checkbutton(
            twod_frame, text="Include 2D images in the NeXus file",
            variable=self.include_2d_var,
            font=OPERAXNTheme.FONTS['body'],
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary'],
            activebackground=OPERAXNTheme.COLORS['bg_primary'],
            activeforeground=OPERAXNTheme.COLORS['accent_primary'],
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary'],
            disabledforeground=OPERAXNTheme.COLORS['text_dim'],
            highlightthickness=0)
        self.include_2d_check.pack(anchor="w", pady=(6, 0))

        # --- Experiment details ---
        self.create_themed_label(text="Experiment details (optional):",
                                 font_type="heading", pady=(14, 4))
        details = self._block()
        self.title_var = tk.StringVar()
        self.sample_var = tk.StringVar()
        for row, (label, var) in enumerate([("Title:", self.title_var),
                                            ("Sample name:", self.sample_var)]):
            tk.Label(details, text=label,
                     font=OPERAXNTheme.FONTS['body'],
                     bg=OPERAXNTheme.COLORS['bg_primary'],
                     fg=OPERAXNTheme.COLORS['text_primary']
                     ).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
            self._entry(details, var, width=28).grid(row=row, column=1, pady=2)

        # --- Standard echem ---
        self.create_themed_label(text="Standard electrochemistry (optional):",
                                 font_type="heading", pady=(14, 4))
        echem_frame = self._block()
        self.std_echem_var = tk.StringVar(value="None")
        self._entry(echem_frame, self.std_echem_var, width=20, readonly=True
                    ).pack(side="left", padx=(0, 8))
        self.std_select_btn = self._small_button(
            echem_frame, "Select", self._select_std_echem)
        self.std_clear_btn = self._small_button(
            echem_frame, "Clear", self._clear_std_echem)

        # Note shown when generation options do not apply
        self.note_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.note_var,
                 font=OPERAXNTheme.FONTS['small'],
                 bg=OPERAXNTheme.COLORS['bg_primary'],
                 fg=OPERAXNTheme.COLORS['warning']).pack(pady=(10, 0))

        self.create_button_frame([
            ("Load", self._confirm, "primary"),
            ("Cancel", self.cancel, "secondary")
        ], pady=14)

    def _block(self) -> tk.Frame:
        """Centered content block; widgets inside anchor west (house style)."""
        frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        frame.pack()
        return frame

    def _entry(self, parent: tk.Widget, var: tk.StringVar, width: int,
               readonly: bool = False) -> tk.Entry:
        kwargs = ({"state": "readonly",
                   "readonlybackground": OPERAXNTheme.COLORS['bg_tertiary']}
                  if readonly else {})
        return self.create_themed_entry(parent, var, width, **kwargs)

    @staticmethod
    def _small_button(parent: tk.Widget, text: str, command) -> tk.Button:
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
        directory = filedialog.askdirectory(parent=self, title="Select Directory")
        if directory:
            self._paths = [directory]
            self.input_var.set(directory)
            self._update_option_states()

    def _select_std_echem(self) -> None:
        files = filedialog.askopenfilenames(
            parent=self, title="Select Standard Electrochemistry Files",
            filetypes=[("Supported files", "*.txt *.xlsx *.csv"),
                       ("All files", "*.*")])
        if files:
            self._std_echem = list(files)
            self.std_echem_var.set(f"{len(files)} file(s)")

    def _clear_std_echem(self) -> None:
        self._std_echem = []
        self.std_echem_var.set("None")

    # --- dynamic enabling ---

    def _is_direct_nxs(self) -> bool:
        return len(self._paths) == 1 and self._paths[0].lower().endswith(".nxs")

    def _update_option_states(self) -> None:
        direct_nxs = self._is_direct_nxs()
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

        self.note_var.set(
            "Opening a NeXus file: generation options are fixed in the file"
            if direct_nxs else "")

    # --- result ---

    def _confirm(self) -> None:
        if not self._paths:
            messagebox.showerror("Error", "Please select files or a directory",
                                 parent=self)
            return

        size_val = self.size_var.get()
        display_size = 0 if size_val == "No downsampling" else int(size_val)

        source_map = {
            "inhouse": DataSourceType.INHOUSE,
            "synchrotron": DataSourceType.SYNCHROTRON,
            "neutron": DataSourceType.NEUTRON,
        }

        self.result = UploadOptions(
            paths=list(self._paths),
            data_source=source_map[self.source_var.get()],
            time_method=TimeMethod(self.time_var.get()),
            display_size=display_size,
            include_2d=self.include_2d_var.get(),
            title=self.title_var.get().strip() or None,
            sample_name=self.sample_var.get().strip() or None,
            standard_echem_files=list(self._std_echem) or None,
        )
        self.destroy()


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
        """Build progress bar and label."""
        self.label = self.create_themed_label(
            text="Processing...",
            font_type="body",
            pady=10
        )

        # Style the progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar",
                        background=OPERAXNTheme.COLORS['accent_primary'],
                        troughcolor=OPERAXNTheme.COLORS['bg_tertiary'],
                        borderwidth=0,
                        lightcolor=OPERAXNTheme.COLORS['accent_primary'],
                        darkcolor=OPERAXNTheme.COLORS['accent_primary'])

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
        """Update progress bar and label."""
        if not self.winfo_exists():
            return False

        self.label.config(text=text)
        self.bar['value'] = value
        self.update()
        return True
