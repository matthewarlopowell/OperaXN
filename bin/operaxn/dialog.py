"""
Dialog Module for OperaXN
"""

import logging
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field
from tkinter import messagebox
from typing import Dict, List, Callable, Tuple, Any

import numpy as np

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

        self.grab_set()

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
            bg=OPERAXNTheme.COLORS.get('bg_secondary' if parent != self else 'bg_primary'),
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
        bg_color = OPERAXNTheme.COLORS.get('bg_secondary' if parent != self else 'bg_primary')

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
        bg_color = OPERAXNTheme.COLORS.get('bg_secondary' if parent != self else 'bg_primary')

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
        bg_color = OPERAXNTheme.COLORS.get('bg_secondary' if parent != self else 'bg_primary')
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
            selectcolor=OPERAXNTheme.COLORS['bg_tertiary']
        )

        return rb


# ============================================================================
# Data Source Selection Dialog
# ============================================================================

class DataSourceSelectionDialog(BaseDialog):
    """Data source type selection dialog."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, "Select Data Source", "400x275")
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Build data source selection widgets."""
        # Title
        self.create_themed_label(
            text="Select Your Data Source Type",
            font_type="heading",
            pady=(20, 10)
        )

        # Radio button frame
        radio_frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        radio_frame.pack(pady=10)

        self.source_var = tk.StringVar(value="inhouse")

        # Data source options
        sources = [
            ("inhouse", "Laboratory X-ray diffraction", OPERAXNTheme.COLORS['text_primary']),
            ("synchrotron", "Synchrotron X-ray diffraction", OPERAXNTheme.COLORS['accent_primary']),
            ("neutron", "Time-of-flight neutron diffraction", OPERAXNTheme.COLORS['danger'])
        ]

        for value, label, color in sources:
            option_frame = tk.Frame(radio_frame, bg=OPERAXNTheme.COLORS['bg_primary'])
            option_frame.pack(anchor="w", pady=5, padx=20, fill="x")

            rb = self.create_themed_radiobutton(
                option_frame, label, self.source_var, value, fg_color=color
            )
            rb.pack(side="left")

        # Buttons
        self.create_button_frame([
            ("OK", self._confirm, "primary"),
            ("Cancel", self.cancel, "secondary")
        ])

    def _confirm(self) -> None:
        """Confirm selection and close dialog."""
        source_map = {
            "inhouse": DataSourceType.INHOUSE,
            "synchrotron": DataSourceType.SYNCHROTRON,
            "neutron": DataSourceType.NEUTRON
        }

        self.result = source_map.get(self.source_var.get(), DataSourceType.INHOUSE)
        self.destroy()


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
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=OPERAXNTheme.COLORS['bg_primary'])
        style.configure('TNotebook.Tab',
                        background=OPERAXNTheme.COLORS['bg_secondary'],
                        foreground=OPERAXNTheme.COLORS['text_primary'],
                        padding=[10, 6])
        style.map('TNotebook.Tab',
                  background=[('selected', OPERAXNTheme.COLORS['bg_tertiary'])],
                  foreground=[('selected', OPERAXNTheme.COLORS['accent_primary'])])

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
# Display Size Dialog
# ============================================================================

class DisplaySizeDialog(BaseDialog):
    """Max display size selection for synchrotron 2D data downsampling."""

    DISPLAY_OPTIONS = ["No downsampling", "4096", "2048", "1024", "512"]

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, "2D Display Size", "350x175")
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Build display size selection widgets."""
        self.create_themed_label(
            text="Max 2D image display size:",
            font_type="heading",
            pady=(20, 5)
        )

        self.create_themed_label(
            text="Larger images will be stride-downsampled to this size.",
            font_type="small",
            fg_color=OPERAXNTheme.COLORS['text_secondary'],
            pady=(0, 10)
        )

        # Combobox
        combo_frame = tk.Frame(self, bg=OPERAXNTheme.COLORS['bg_primary'])
        combo_frame.pack(pady=5)

        self.size_var = tk.StringVar(value="4096")
        style = ttk.Style()
        style.configure("Display.TCombobox", padding=5)
        self.combo = ttk.Combobox(
            combo_frame,
            textvariable=self.size_var,
            values=self.DISPLAY_OPTIONS,
            state="readonly",
            width=20
        )
        self.combo.pack()

        # Buttons
        self.create_button_frame([
            ("OK", self._confirm, "primary"),
            ("Cancel", self.cancel, "secondary")
        ], pady=15)

    def _confirm(self) -> None:
        """Parse selection and set result as int (0 = no downsampling)."""
        val = self.size_var.get()
        if val == "No downsampling":
            self.result = 0
        else:
            self.result = int(val)
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
