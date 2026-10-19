"""
Operando heatmap window: stacked 1D patterns as an intensity map.

2θ or d-spacing (X-ray) or time of flight / d-spacing (neutron, one map per
detector bank) on x, scan number or elapsed time on y, with the correlated voltage
track on a shared y axis so peak shifts line up with the electrochemistry.
Patterns are interpolated onto the first usable scan's x grid when grids differ;
the window also exports a publication figure (PNG, PDF or SVG).
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .config import (
    COLORMAP, DSPACING_LABEL, EXPORT_DPI, FIGURE_DPI, SECONDS_PER_HOUR,
    TIME_HOURS_LABEL, TOF_LABEL, TWOTHETA_LABEL, VMIN_FLOOR, XRAY_WAVELENGTH,
)
from .dialog import parse_scan_range
from .ici import (
    BG, BG2, BORDER, INPUT, TEXT, _FONT,
    _PAD_S, CHARGE_COLOR,
    PLOT_BG, PLOT_BORDER, PLOT_DIM, PLOT_GRID,
    _Btn, _canvas_frame, _label, _separator, _spinbox, _style_ax, _stub,
    _themed_menubutton, embed_figure, maximise_window, watch_figure_sync,
)


# Colour maps offered in the window; the application default comes first
COLORMAPS = (COLORMAP, 'viridis', 'inferno', 'magma', 'plasma', 'cividis',
             'turbo', 'jet', 'Greys')

# Neutron x axes in the order they are offered, with their plot labels
NEUTRON_AXES = {'tof': TOF_LABEL, 'd': DSPACING_LABEL}
# X-ray x axes; d-spacing is converted from 2θ and drawn decreasing, in 2θ's order,
# like the main window's d-spacing view
XRD_AXES = {'twotheta': TWOTHETA_LABEL, 'd': DSPACING_LABEL}
INTENSITY_LABEL = 'Intensity (arb. units)'
VOLTAGE_LABEL = 'Voltage (V)'

EXPORT_FILETYPES = (('PNG image', '*.png'), ('PDF document', '*.pdf'),
                    ('SVG image', '*.svg'))
PUBLICATION_SIZE = (7.2, 5.0)  # inches; a two-column journal figure
LOG_FLOOR_PERCENTILE = 1.0  # of the positive values, the auto log-scale floor
SCAN_LIST_HINT = 'e.g. 1-5, 103'  # the GIF dialog's scan-range grammar


# ============================================================================
# Matrix Builders
# ============================================================================

def build_xrd_matrix(oned_arrays: Dict[int, Dict[str, Any]]):
    """Stack 1D patterns as (x_grid, matrix, scan_nums) on the first usable scan's grid.
    Other grids are interpolated; entries without 1-D x and y are skipped;
    (None, None, []) when nothing is usable."""
    scan_nums = sorted(oned_arrays)
    rows = []
    x_grid = None
    used = []
    for num in scan_nums:
        entry = oned_arrays[num]
        if entry.get("x") is None or entry.get("y") is None:
            continue
        x = np.asarray(entry["x"], dtype=float)
        y = np.asarray(entry["y"], dtype=float)
        if x.ndim != 1 or x.shape != y.shape or x.size < 2:
            continue
        if x_grid is None:
            x_grid = x
        if not np.array_equal(x, x_grid):
            # np.interp needs ascending sample points; a descending grid would give junk
            order = np.argsort(x, kind='stable')
            y = np.interp(x_grid, x[order], y[order])
        rows.append(y)
        used.append(num)
    if not rows:
        return None, None, []
    return x_grid, np.vstack(rows), used


def dspacing_columns(x, matrix, wavelength: float):
    """Re-express a 2θ map's columns as ascending d-spacing (Bragg's law) for wavelength (Å).
    Columns outside 0 < 2θ < 180° have no d and are dropped."""
    if matrix is None:
        return x, matrix
    two_theta = np.asarray(x, dtype=float)
    keep = (two_theta > 0) & (two_theta < 180)
    d = wavelength / (2 * np.sin(np.deg2rad(two_theta[keep] / 2)))
    order = np.argsort(d)
    return d[order], np.asarray(matrix)[:, keep][:, order]


def neutron_banks(neutron_arrays: Dict[int, Dict[str, Any]]) -> List[str]:
    """Return the bank keys recorded in any scan, numeric banks first in order."""
    banks = {str(bank) for banks in neutron_arrays.values() for bank in banks}
    return sorted(banks, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else b))


def neutron_axes(neutron_arrays: Dict[int, Dict[str, Any]], bank: str) -> List[str]:
    """Return the axis keys ('tof', 'd') recorded for a bank in any scan."""
    keys = set()
    for banks in neutron_arrays.values():
        keys.update(banks.get(bank, {}))
    return [key for key in NEUTRON_AXES if key in keys]


def neutron_bank_arrays(neutron_arrays: Dict[int, Dict[str, Any]],
                        bank: str, key: str) -> Dict[int, Dict[str, Any]]:
    """Return {scan_num: {x, y}} for one bank and axis, skipping scans without it."""
    arrays = {}
    for num, banks in neutron_arrays.items():
        data = banks.get(bank, {}).get(key)
        if data is not None:
            arrays[num] = {"x": data["x"], "y": data["y"]}
    return arrays


def _keep_rows(x, matrix, scan_nums: List[int], keep):
    """Return the rows whose scan number passes keep(num); (None, None, []) when none do."""
    idx = [i for i, num in enumerate(scan_nums) if keep(num)]
    if not idx:
        return None, None, []
    return x, matrix[idx], [scan_nums[i] for i in idx]


def correlated_rows(x, matrix, scan_nums: List[int], correlated):
    """Keep the rows whose scan has correlated echem."""
    return _keep_rows(x, matrix, scan_nums, lambda num: num in correlated)


def exclude_rows(x, matrix, scan_nums: List[int], excluded):
    """Drop the rows of the excluded scan numbers."""
    return _keep_rows(x, matrix, scan_nums, lambda num: num not in excluded)


def scan_positions_seconds(scans: List[Dict[str, Any]],
                           time_method: str = "absolute",
                           reference: Optional[pd.Timestamp] = None) -> Dict[int, float]:
    """Return elapsed seconds per scan_num on the echem time axis, omitting untimed scans.
    reference (absolute mode) is the echem start; it defaults to the first timed scan."""
    positions: Dict[int, float] = {}
    for scan in scans:
        ts = scan.get("timestamp")
        if not ts:
            continue
        if time_method == "relative":
            if isinstance(ts, str) and ts.count(":") == 2:
                h, m, s = ts.split(":")
                try:
                    positions[scan["scan_num"]] = int(h) * 3600 + int(m) * 60 + int(s)
                except ValueError:
                    continue
        else:
            when = scan.get("timestamp_for_correlation")
            try:
                when = pd.to_datetime(when if when is not None else ts)
            except (ValueError, TypeError):
                continue
            if reference is None:
                reference = when
            positions[scan["scan_num"]] = (when - reference).total_seconds()
    return positions


def echem_scan_curve(scan_nums, y_vals, positions, echem_seconds,
                     echem_voltage):
    """Map a full-resolution echem trace onto the scan axis as (volts, y).
    Only the span bracketed by scans is kept, so there are no flat clipped tails;
    empty arrays when fewer than two scans or echem points qualify."""
    scan_y = np.asarray(y_vals, dtype=float)
    scan_t = np.asarray([positions.get(num, np.nan) for num in scan_nums],
                        dtype=float)
    ex = np.asarray(echem_seconds, dtype=float)
    ev = np.asarray(echem_voltage, dtype=float)

    scan_count = min(scan_y.size, scan_t.size)
    echem_count = min(ex.size, ev.size)
    scan_y, scan_t = scan_y[:scan_count], scan_t[:scan_count]
    ex, ev = ex[:echem_count], ev[:echem_count]

    scan_valid = np.isfinite(scan_t) & np.isfinite(scan_y)
    echem_valid = np.isfinite(ex) & np.isfinite(ev)
    scan_t, scan_y = scan_t[scan_valid], scan_y[scan_valid]
    ex, ev = ex[echem_valid], ev[echem_valid]
    if scan_t.size < 2 or ex.size < 2:
        return np.array([]), np.array([])

    scan_order = np.argsort(scan_t)
    scan_t, scan_y = scan_t[scan_order], scan_y[scan_order]
    scan_t, unique_indices = np.unique(scan_t, return_index=True)
    scan_y = scan_y[unique_indices]
    if scan_t.size < 2:
        return np.array([]), np.array([])

    echem_order = np.argsort(ex)
    ex, ev = ex[echem_order], ev[echem_order]
    within_scans = (ex >= scan_t[0]) & (ex <= scan_t[-1])
    if np.count_nonzero(within_scans) < 2:
        return np.array([]), np.array([])

    ex, ev = ex[within_scans], ev[within_scans]
    return ev, np.interp(ex, scan_t, scan_y)


# ============================================================================
# Display Helpers
# ============================================================================

def colormap_name(name: str, reverse: bool = False) -> str:
    """Return the Matplotlib map name, flipping the '_r' suffix when reversed."""
    if not reverse:
        return name
    return name[:-2] if name.endswith('_r') else name + '_r'


def intensity_limits(matrix: np.ndarray, ceiling: float = 100.0,
                     vmin: Optional[float] = None, vmax: Optional[float] = None,
                     log: bool = False) -> Tuple[float, float]:
    """Return (vmin, vmax): entered values win, else data minimum to ceiling percentile.
    Log ignores non-positive values and floors at LOG_FLOOR_PERCENTILE of the rest,
    so zeros in a background-subtracted pattern cannot flatten the scale."""
    data = np.asarray(matrix, dtype=float)
    data = data[np.isfinite(data)]
    if log:
        data = data[data > 0]
    if data.size == 0:
        data = np.array([VMIN_FLOOR])
    if vmin is not None:
        lo = float(vmin)
    elif log:
        lo = float(np.percentile(data, LOG_FLOOR_PERCENTILE))
    else:
        lo = float(data.min())
    hi = (float(np.percentile(data, min(max(ceiling, 50.0), 100.0)))
          if vmax is None else float(vmax))
    if hi <= lo:
        hi = float(data.max()) if vmax is None and float(data.max()) > lo else lo + 1.0
    if log:
        lo = max(lo, VMIN_FLOOR)
        if hi <= lo:
            hi = lo * 10.0
    return lo, hi


def cell_edges(values: np.ndarray) -> np.ndarray:
    """Return the n + 1 cell edges of n centres: midpoints, half a step beyond each end.
    A lone centre gets a unit-wide cell, which 'nearest' shading would draw zero high."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        centre = float(v[0]) if v.size else 0.0
        return np.array([centre - 0.5, centre + 0.5])
    return np.concatenate(([v[0] - (v[1] - v[0]) / 2.0], (v[:-1] + v[1:]) / 2.0,
                           [v[-1] + (v[-1] - v[-2]) / 2.0]))


def row_edges(y_vals: np.ndarray) -> Tuple[float, float]:
    """Return the outer cell edges of the rows (half a step beyond each end)."""
    edges = cell_edges(y_vals)
    return float(edges[0]), float(edges[-1])


def publication_figure(x, y_vals, matrix, *, x_label: str, y_label: str,
                       cmap: str, vmin: float, vmax: float, log: bool = False,
                       x_range: Tuple[Optional[float], Optional[float]] = (None, None),
                       invert_x: bool = False,
                       voltage=None, title: str = '',
                       dpi: int = EXPORT_DPI) -> Figure:
    """Build the publication figure: voltage track (left, shared y), map, colour bar.
    voltage is (volts, y) on the row axis or None for no track; invert_x draws x decreasing;
    the mesh is rasterised so PDF and SVG exports keep vector axes and text."""
    fig = Figure(figsize=PUBLICATION_SIZE, dpi=dpi, facecolor='white')
    has_voltage = voltage is not None and len(voltage[0]) > 1
    widths = (1.0, 3.4, 0.1) if has_voltage else (3.4, 0.1)
    gs = GridSpec(1, len(widths), figure=fig, width_ratios=widths,
                  wspace=0.06, left=0.1, right=0.9, top=0.94, bottom=0.11)

    ax_v = fig.add_subplot(gs[0, 0]) if has_voltage else None
    ax = fig.add_subplot(gs[0, 1], sharey=ax_v) if has_voltage else fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, -1])

    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(cell_edges(x), cell_edges(y_vals), matrix, norm=norm,
                         cmap=cmap, shading='flat', rasterized=True)
    ax.set_xlabel(x_label, fontsize=10, color='black')
    if x_range[0] is not None or x_range[1] is not None:
        ax.set_xlim(x_range[0] if x_range[0] is not None else ax.get_xlim()[0],
                    x_range[1] if x_range[1] is not None else ax.get_xlim()[1])
    if invert_x:
        ax.invert_xaxis()
    if title:
        ax.set_title(title, fontsize=10, color='black')

    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label(INTENSITY_LABEL, fontsize=10, color='black')

    if has_voltage:
        ax_v.plot(voltage[0], voltage[1], color=CHARGE_COLOR, lw=1.0)
        ax_v.set_xlabel(VOLTAGE_LABEL, fontsize=10, color='black')
        ax_v.set_ylabel(y_label, fontsize=10, color='black')
        ax_v.grid(True, color=PLOT_GRID, linewidth=0.5)
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(y_label, fontsize=10, color='black')
    ax.set_ylim(*row_edges(y_vals))

    # Every piece of text black, whatever the screen theme or rcParams say
    for axis in (ax, ax_v, cax):
        if axis is None:
            continue
        axis.tick_params(which='both', labelsize=9, colors='black', width=0.8)
        axis.xaxis.get_offset_text().set_color('black')
        axis.yaxis.get_offset_text().set_color('black')
        for spine in axis.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)
    return fig


def _entry(master, var, width=7):
    """Build a flat entry in the theme's input colours and border."""
    return tk.Entry(
        master, textvariable=var, width=width,
        bg=INPUT, fg=TEXT, insertbackground=TEXT,
        relief=tk.FLAT, font=_FONT,
        highlightbackground=BORDER, highlightthickness=1,
    )


def _menu_button(master, var, values, prefix, command=None) -> tk.Menubutton:
    """Build a compact themed menu button that sets var to one of values."""
    display = tk.StringVar(value=f'{prefix}{var.get()}  ▾')
    button, menu = _themed_menubutton(master, display)
    for value in values:
        menu.add_radiobutton(
            label=str(value), value=value, variable=var,
            command=(lambda selected=value: command(selected)) if command else None)
    button.config(menu=menu)
    var.trace_add('write', lambda *_: display.set(f'{prefix}{var.get()}  ▾'))
    return button


def _parse_float(text: str) -> Optional[float]:
    """Parse an entry's text as a float; None when blank or not a finite number."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


# ============================================================================
# Heatmap Window
# ============================================================================

class HeatmapWindow(tk.Toplevel):
    """Operando heatmap window; opens maximised.
    Takes oned_arrays (X-ray, 2θ / d at wavelength Å) or neutron_arrays (per bank, TOF / d);
    dspacing picks the start axis, x_range seeds the range, exports are <export_stem>_heatmap*."""

    def __init__(self, parent: tk.Widget,
                 scans: List[Dict[str, Any]],
                 oned_arrays: Optional[Dict[int, Dict[str, Any]]] = None,
                 echem_arrays: Optional[Dict[str, Any]] = None,
                 time_method: str = "absolute",
                 neutron_arrays: Optional[Dict[int, Dict[str, Any]]] = None,
                 dspacing: bool = False,
                 x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                 wavelength: float = XRAY_WAVELENGTH,
                 export_stem: str = 'experiment') -> None:
        super().__init__(parent)
        self.title('OperaXN: Operando Heatmap')
        self.configure(bg=BG)
        self.minsize(900, 500)
        maximise_window(self)

        self._oned_arrays = oned_arrays or {}
        self._neutron = {num: {str(bank): data for bank, data in banks.items()}
                         for num, banks in (neutron_arrays or {}).items()}
        self._banks = neutron_banks(self._neutron)
        self._echem_arrays = echem_arrays or {}
        self._scan_echem = {scan["scan_num"]: scan.get("echem") for scan in scans
                            if scan.get("echem") is not None}
        # Scans with a correlated echem value (the Correlated filter keeps these)
        self._correlated = set(self._scan_echem) | {
            num for num, entry in self._oned_arrays.items()
            if entry.get("echem") is not None}
        self._positions = scan_positions_seconds(
            scans, time_method, reference=self._echem_reference(time_method))
        self._replot_job = None
        self._matrices: Dict[Tuple[str, str], tuple] = {}
        self._excluded: set = set()
        self._x = self._matrix = None
        self._scan_nums: List[int] = []
        self._wavelength = wavelength
        self._file_stem = export_stem

        self.bank_var = tk.StringVar(value=self._banks[0] if self._banks else '')
        if self.is_neutron:
            first_axes = neutron_axes(self._neutron, self.bank_var.get())
            axis = ('d' if dspacing and 'd' in first_axes
                    else (first_axes[0] if first_axes else ''))
        else:
            axis = 'd' if dspacing else 'twotheta'
        self.xaxis_var = tk.StringVar(value=axis)

        self._build_ui()
        self._refresh_axis_buttons()
        # The main window's x range is in its own axis units; seed it only on that axis
        if x_range is not None and (self.xaxis_var.get() == 'd') == dspacing:
            self._set_entries(self.xmin_var, self.xmax_var, *x_range)
        self._load_matrix()
        self._replot()

        watch_figure_sync(self, (self._canvas,))

    @property
    def is_neutron(self) -> bool:
        """True when the window shows per-bank neutron maps."""
        return bool(self._banks)

    def _echem_reference(self, time_method: str) -> Optional[pd.Timestamp]:
        """Return the echem start (absolute mode) so scan rows share the voltage axis."""
        timestamps = self._echem_arrays.get('timestamps') or []
        if time_method == 'absolute' and timestamps:
            try:
                # The earliest point, as make_echem_arrays measures the curve from it
                return pd.to_datetime(timestamps).min()
            except (ValueError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the ICI-style layout: selector strip above the figure, display bar below."""
        # Display bar, packed from the bottom first like the ICI regression bar
        bar_wrapper = tk.Frame(self, bg=BG2, highlightbackground=BORDER,
                               highlightthickness=1)
        bar_wrapper.pack(side='bottom', fill='x', padx=_PAD_S, pady=(2, _PAD_S))
        bar = tk.Frame(bar_wrapper, bg=BG2)
        bar.pack(fill='x', padx=2, pady=4)

        # Selector strip
        strip_wrapper = tk.Frame(self, bg=BG2, highlightbackground=BORDER,
                                 highlightthickness=1)
        strip_wrapper.pack(side='top', fill='x', padx=_PAD_S, pady=(_PAD_S, 2))
        strip = tk.Frame(strip_wrapper, bg=BG2)
        strip.pack(fill='x', padx=2, pady=4)

        self._bank_btns = {}
        if self.is_neutron:
            self._bank_btns = self._toggle_group(
                self._group(strip), 'Bank:', [(b, b) for b in self._banks], self.bank_var)
            self._sep(strip)
        x_options = ([('tof', 'TOF'), ('d', 'd-spacing')] if self.is_neutron
                     else [('twotheta', '2θ'), ('d', 'd-spacing')])
        self._x_btns = self._toggle_group(self._group(strip), 'X axis:', x_options,
                                          self.xaxis_var)
        self._sep(strip)

        # Time is the default row axis whenever the scans carry positions; the buttons
        # keep one order (Time first) so they never swap places between datasets
        self.yaxis_var = tk.StringVar(value='time' if self._positions else 'scan')
        self._y_btns = self._toggle_group(
            self._group(strip), 'Y axis:', [('time', 'Time'), ('scan', 'Scan')],
            self.yaxis_var)
        # Time mode needs scan positions; _y_values falls back when a plotted scan lacks one
        if not self._positions:
            self._y_btns['time'].config(state='disabled')
        self._sep(strip)

        self.scans_var = tk.StringVar(value='all')
        self._scans_btns = self._toggle_group(
            self._group(strip), 'Scans:', [('all', 'All'), ('correlated', 'Correlated')],
            self.scans_var)
        if not self._correlated:
            self._scans_btns['correlated'].config(state='disabled')
        self._sep(strip)

        # Excluded scan numbers, in the GIF dialog's scan-range grammar
        exclude = self._group(strip)
        _label(exclude, 'Exclude:').pack(side='left', padx=(0, 2))
        self.exclude_var = tk.StringVar()
        entry = _entry(exclude, self.exclude_var, width=14)
        entry.pack(side='left', padx=2)
        entry.bind('<Return>', lambda _e: self._apply_exclusions())
        entry.bind('<FocusOut>', lambda _e: self._apply_exclusions())
        _label(exclude, SCAN_LIST_HINT, dim=True).pack(side='left')

        # Display bar: scale, colour map, ranges, export
        self.scale_var = tk.StringVar(value='linear')
        self._scale_btns = self._toggle_group(
            self._group(bar), 'Scale:', [('linear', 'Linear'), ('log', 'Log')],
            self.scale_var)
        self._sep(bar)

        cmap_group = self._group(bar)
        self.cmap_var = tk.StringVar(value=COLORMAPS[0])
        _menu_button(cmap_group, self.cmap_var, COLORMAPS, 'Colour map: ',
                     command=lambda _v: self._replot()).pack(side='left', padx=2)
        self.reverse_var = tk.BooleanVar(value=False)
        self._reverse_btn = _Btn(cmap_group, 'Reverse', style='toggle_off', padx=6,
                                 command=self._toggle_reverse)
        self._reverse_btn.pack(side='left', padx=2)
        self._sep(bar)

        self.xmin_var, self.xmax_var = tk.StringVar(), tk.StringVar()
        self._range_pair(self._group(bar), 'X range:', self.xmin_var, self.xmax_var)
        self._sep(bar)

        intensity = self._group(bar)
        self.zmin_var, self.zmax_var = tk.StringVar(), tk.StringVar()
        self._range_pair(intensity, 'Intensity:', self.zmin_var, self.zmax_var)
        # Ceiling percentile clips hot pixels so weak peaks show
        _label(intensity, 'ceiling (%):', dim=True).pack(side='left', padx=(_PAD_S, 0))
        self.ceiling_var = tk.DoubleVar(value=100.0)
        _spinbox(intensity, self.ceiling_var, from_=50.0, to=100.0,
                 increment=0.5, format='%.1f', width=6).pack(side='left', padx=2)
        self.ceiling_var.trace_add('write', lambda *_: self._queue_replot())
        _Btn(intensity, 'Reset', padx=6,
             command=self._reset_ranges).pack(side='left', padx=2)
        self._sep(bar)

        export = self._group(bar)
        _Btn(export, 'Export figure...', style='primary', padx=6,
             command=self._export_figure).pack(side='left', padx=2)
        if self.is_neutron:
            _Btn(export, 'Export all banks...', padx=6,
                 command=self._export_all_banks).pack(side='left', padx=2)

        # Figure: heatmap and voltage track on a shared y axis
        plot_frame = _canvas_frame(self)
        plot_frame.pack(fill='both', expand=True, padx=_PAD_S, pady=2)

        self._fig = Figure(facecolor=PLOT_BG, dpi=FIGURE_DPI)
        # Keep a dedicated gutter after the colourbar so its tick labels and
        # scientific-notation offset cannot collide with the voltage panel.
        gs = GridSpec(1, 4, figure=self._fig,
                      width_ratios=(4, 0.12, 0.22, 1.1),
                      wspace=0.05, left=0.07, right=0.98,
                      top=0.94, bottom=0.09)
        self.ax_map = self._fig.add_subplot(gs[0, 0])
        self.cax = self._fig.add_subplot(gs[0, 1])
        self.ax_v = self._fig.add_subplot(gs[0, 3], sharey=self.ax_map)
        self._canvas = embed_figure(self._fig, plot_frame)
        # The toolbar's save icon renders the publication figure too, so
        # every export from this window carries the same black-text styling.
        # The button keeps the method bound at construction, so reroute both.
        toolbar = self._canvas.toolbar
        toolbar.save_figure = lambda *_a, **_k: self._export_figure()
        if 'Save' in getattr(toolbar, '_buttons', {}):
            toolbar._buttons['Save'].config(command=self._export_figure)

    @staticmethod
    def _group(master) -> tk.Frame:
        """Return a control group frame packed along a strip, as in the ICI window."""
        frame = tk.Frame(master, bg=BG2)
        frame.pack(side='left', padx=2)
        return frame

    @staticmethod
    def _sep(master) -> None:
        """Pack a vertical separator between control groups."""
        _separator(master, horizontal=False).pack(side='left', fill='y', padx=2, pady=3)

    def _toggle_group(self, master, label, options, var) -> dict:
        """Build labelled toggle buttons bound to var; return them keyed by value."""
        _label(master, label).pack(side='left', padx=(0, 2))
        buttons = {}
        for value, text in options:
            btn = _Btn(master, text, padx=6,
                       style='toggle_on' if value == var.get() else 'toggle_off',
                       command=lambda v=value: self._set_toggle(var, buttons, v))
            btn.pack(side='left', padx=2)
            buttons[value] = btn
        return buttons

    def _range_pair(self, master, label, min_var, max_var) -> None:
        """Build labelled min-to-max entries that replot on Return or focus loss."""
        _label(master, label).pack(side='left', padx=(0, 2))
        for i, var in enumerate((min_var, max_var)):
            if i:
                _label(master, 'to', dim=True).pack(side='left', padx=2)
            entry = _entry(master, var)
            entry.pack(side='left')
            entry.bind('<Return>', lambda _e: self._replot())
            entry.bind('<FocusOut>', lambda _e: self._replot())

    def _set_toggle(self, var: tk.StringVar, buttons: dict, value: str) -> None:
        """Switch a toggle group and replot; bank, x axis and scan filter reload the rows."""
        var.set(value)
        for val, btn in buttons.items():
            btn.set_selected(val == value)
        if var is self.xaxis_var:
            self._set_entries(self.xmin_var, self.xmax_var)
        if var is self.bank_var or var is self.xaxis_var:
            self._refresh_axis_buttons()
        if var is self.bank_var or var is self.xaxis_var or var is self.scans_var:
            self._load_matrix()
        self._replot()
        self._forget_zoom_history()

    def _toggle_reverse(self) -> None:
        """Flip the colour map direction."""
        self.reverse_var.set(not self.reverse_var.get())
        self._reverse_btn.set_selected(self.reverse_var.get())
        self._replot()

    def _refresh_axis_buttons(self) -> None:
        """Enable only the axes recorded for the selected bank, moving off one it lacks."""
        if not self.is_neutron:
            return
        available = neutron_axes(self._neutron, self.bank_var.get())
        if available and self.xaxis_var.get() not in available:
            self.xaxis_var.set(available[0])
            # A range typed on the old axis is in the wrong units for the new one
            self._set_entries(self.xmin_var, self.xmax_var)
            for val, btn in self._x_btns.items():
                btn.set_selected(val == available[0])
        for val, btn in self._x_btns.items():
            btn.config(state='normal' if val in available else 'disabled')

    @staticmethod
    def _set_entries(min_var, max_var, lo=None, hi=None) -> None:
        """Fill a min/max entry pair (blank for None)."""
        min_var.set('' if lo is None else f'{lo:g}')
        max_var.set('' if hi is None else f'{hi:g}')

    def _reset_ranges(self) -> None:
        """Clear the x and intensity ranges and reset the ceiling."""
        self._set_entries(self.xmin_var, self.xmax_var)
        self._set_entries(self.zmin_var, self.zmax_var)
        self.ceiling_var.set(100.0)
        self._replot()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _matrix_for(self, bank: Optional[str] = None) -> tuple:
        """Return the cached (x, matrix, scan_nums) for a bank on the selected axis.
        X-ray data ignores bank."""
        if self.is_neutron:
            key = (bank or self.bank_var.get(), self.xaxis_var.get())
            if key not in self._matrices:
                self._matrices[key] = build_xrd_matrix(
                    neutron_bank_arrays(self._neutron, *key))
        else:
            key = ('xrd', self.xaxis_var.get())
            if key not in self._matrices:
                x, matrix, scan_nums = build_xrd_matrix(self._oned_arrays)
                if key[1] == 'd':
                    x, matrix = dspacing_columns(x, matrix, self._wavelength)
                self._matrices[key] = (x, matrix, scan_nums)
        return self._matrices[key]

    def _rows_for(self, bank: Optional[str] = None) -> tuple:
        """Return a bank's (x, matrix, scan_nums) after the scan filter and exclusions."""
        x, matrix, scan_nums = self._matrix_for(bank)
        if matrix is not None and self.scans_var.get() == 'correlated':
            x, matrix, scan_nums = correlated_rows(x, matrix, scan_nums, self._correlated)
        if matrix is not None and self._excluded:
            x, matrix, scan_nums = exclude_rows(x, matrix, scan_nums, self._excluded)
        return x, matrix, scan_nums

    def _apply_exclusions(self) -> None:
        """Re-read the Exclude entry and reload the rows when it changed."""
        excluded = set(parse_scan_range(self.exclude_var.get()))
        if excluded != self._excluded:
            self._excluded = excluded
            self._load_matrix()
            self._replot()
            self._forget_zoom_history()

    def _load_matrix(self) -> None:
        """Point the window at the rows for the current selection."""
        self._x, self._matrix, self._scan_nums = self._rows_for()

    def _x_label(self) -> str:
        """Return the axis label for the selected x quantity."""
        axes = NEUTRON_AXES if self.is_neutron else XRD_AXES
        return axes.get(self.xaxis_var.get(), TWOTHETA_LABEL)

    def _x_inverted(self) -> bool:
        """True when x is drawn decreasing: X-ray d-spacing, kept in 2θ's order."""
        return not self.is_neutron and self.xaxis_var.get() == 'd'

    def _map_title(self, bank: Optional[str] = None) -> str:
        """Return 'Bank N' for neutron maps, empty for X-ray."""
        return f'Bank {bank or self.bank_var.get()}' if self.is_neutron else ''

    def _y_values(self, scan_nums: Optional[List[int]] = None):
        """Return (row values, label) for the map's rows.
        Elapsed hours in Time mode when every scan has a rising position, else scan numbers."""
        scan_nums = self._scan_nums if scan_nums is None else scan_nums
        if self.yaxis_var.get() == 'time':
            hours = [self._positions.get(num) for num in scan_nums]
            if hours and all(h is not None for h in hours):
                hours = np.asarray(hours, dtype=float) / SECONDS_PER_HOUR
                if np.all(np.diff(hours) > 0):
                    return hours, TIME_HOURS_LABEL
        return np.asarray(scan_nums, dtype=float), 'Scan number'

    def _x_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Return the entered x limits (None where blank or invalid), low before high."""
        lo, hi = _parse_float(self.xmin_var.get()), _parse_float(self.xmax_var.get())
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return lo, hi

    def _colour_scale(self, x, matrix) -> Tuple[np.ndarray, float, float, bool]:
        """Return (matrix, vmin, vmax, log) for the current controls.
        Limits come from the columns inside the x range; log mode clips at VMIN_FLOOR."""
        log = self.scale_var.get() == 'log'
        matrix = np.asarray(matrix, dtype=float)
        lo, hi = self._x_range()
        visible = np.ones(len(x), dtype=bool)
        if lo is not None:
            visible &= x >= lo
        if hi is not None:
            visible &= x <= hi
        sample = matrix[:, visible] if visible.any() else matrix
        try:
            ceiling = float(self.ceiling_var.get())
        except (ValueError, tk.TclError):
            ceiling = 100.0
        vmin, vmax = intensity_limits(
            sample, ceiling, _parse_float(self.zmin_var.get()),
            _parse_float(self.zmax_var.get()), log)
        if log:
            matrix = np.clip(matrix, VMIN_FLOOR, None)
        return matrix, vmin, vmax, log

    def _cmap(self) -> str:
        """Return the selected colour map name, reversed when requested."""
        return colormap_name(self.cmap_var.get(), self.reverse_var.get())

    def _voltage_curve(self, scan_nums, y_vals, time_axis: bool):
        """Return (volts, y) on the row axis, or None when there is no echem.
        Uses the full echem trace, falling back to the per-scan correlated values."""
        ex = np.asarray(self._echem_arrays.get('x', []), dtype=float)
        ev = np.asarray(self._echem_arrays.get('y', []), dtype=float)
        if ex.size and ev.size:
            if time_axis:
                return ev, ex / SECONDS_PER_HOUR
            curve_v, curve_y = echem_scan_curve(
                scan_nums, y_vals, self._positions, ex, ev)
            if curve_v.size:
                return curve_v, curve_y

        volts, ys = [], []
        for num, y in zip(scan_nums, y_vals):
            v = self._scan_echem.get(num)
            if v is None:
                v = self._oned_arrays.get(num, {}).get('echem')
            if v is not None:
                volts.append(v)
                ys.append(y)
        if not volts:
            return None
        volts = np.asarray(volts, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if len(volts) > 1:
            dense_y = np.linspace(ys[0], ys[-1], max(200, len(ys) * 8))
            volts = np.interp(dense_y, ys, volts)
            ys = dense_y
        return volts, ys

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _queue_replot(self, *_):
        """Debounce ceiling-spinbox edits into one replot."""
        if self._replot_job is not None:
            self.after_cancel(self._replot_job)
        self._replot_job = self.after(250, self._replot)

    def _replot(self):
        """Redraw the heatmap and voltage track for the current controls."""
        self._replot_job = None
        if not self.winfo_exists():
            return

        ax = self.ax_map
        ax.cla()
        self.cax.cla()

        y_vals, y_label = self._y_values()
        _style_ax(ax, self._x_label(), y_label, self._map_title())
        ax.grid(False)

        if self._matrix is None:
            filtered = self.scans_var.get() == 'correlated' or self._excluded
            _stub(ax, 'No scans left after filtering' if filtered
                  else 'No patterns loaded')
            self.cax.set_axis_off()
            self._plot_voltage(None, False)
            self._canvas.draw_idle()
            return
        self.cax.set_axis_on()

        matrix, vmin, vmax, log = self._colour_scale(self._x, self._matrix)
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
        mesh = ax.pcolormesh(cell_edges(self._x), cell_edges(y_vals), matrix,
                             norm=norm, cmap=self._cmap(), shading='flat')
        lo, hi = self._x_range()
        if lo is not None or hi is not None:
            xlim = ax.get_xlim()
            ax.set_xlim(lo if lo is not None else xlim[0],
                        hi if hi is not None else xlim[1])
        if self._x_inverted():
            ax.invert_xaxis()

        colorbar = self._fig.colorbar(mesh, cax=self.cax)
        colorbar.ax.set_facecolor(PLOT_BG)
        colorbar.ax.tick_params(which='both', colors=PLOT_DIM, labelsize=7)
        colorbar.outline.set_edgecolor(PLOT_BORDER)

        self._plot_voltage(y_vals, y_label == TIME_HOURS_LABEL)
        ax.set_ylim(*row_edges(y_vals))
        self._canvas.draw_idle()

    def _forget_zoom_history(self) -> None:
        """Clear the toolbar's view history after new rows, bank or axis, so Home cannot
        return to a view of the old ones. Range-entry replots keep it: they also fire on
        focus loss, which a zoom or pan click causes."""
        self._canvas.toolbar.update()

    def _plot_voltage(self, y_vals, time_axis: bool) -> None:
        """Draw the voltage track on the map's shared y axis.
        time_axis is the resolved mode, as Time can fall back to scan numbers."""
        ax = self.ax_v
        ax.cla()
        _style_ax(ax, VOLTAGE_LABEL, '')
        ax.tick_params(labelleft=False)

        if y_vals is None:
            _stub(ax, 'No data')
            return
        curve = self._voltage_curve(self._scan_nums, y_vals, time_axis)
        if curve is None:
            _stub(ax, 'No echem')
            return
        ax.plot(curve[0], curve[1], color=CHARGE_COLOR, lw=1.2)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_stem(self, bank: Optional[str] = None) -> str:
        """Return the default file stem for an export."""
        name = f'{self._file_stem}_heatmap'
        return f'{name}_bank{bank or self.bank_var.get()}' if self.is_neutron else name

    def render_figure(self, bank: Optional[str] = None,
                      dpi: int = EXPORT_DPI) -> Optional[Figure]:
        """Build the publication figure of a bank's map with the current settings.
        None when no rows remain on the selected axis after the filters."""
        x, matrix, scan_nums = self._rows_for(bank)
        if matrix is None:
            return None
        y_vals, y_label = self._y_values(scan_nums)
        matrix, vmin, vmax, log = self._colour_scale(x, matrix)
        return publication_figure(
            x, y_vals, matrix, x_label=self._x_label(), y_label=y_label,
            cmap=self._cmap(), vmin=vmin, vmax=vmax, log=log,
            x_range=self._x_range(), invert_x=self._x_inverted(),
            voltage=self._voltage_curve(scan_nums, y_vals, y_label == TIME_HOURS_LABEL),
            title=self._map_title(bank), dpi=dpi)

    def save_figure(self, path: str, bank: Optional[str] = None,
                    dpi: int = EXPORT_DPI) -> bool:
        """Write the publication figure to path (format from its extension).
        False when the bank has nothing to plot."""
        fig = self.render_figure(bank, dpi)
        if fig is None:
            return False
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white',
                    edgecolor='none')
        return True

    def _export_figure(self) -> None:
        """Save the displayed map as a publication figure."""
        if self._matrix is None:
            messagebox.showwarning('No data', 'Nothing to export.', parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Export heatmap figure',
            defaultextension='.png', filetypes=EXPORT_FILETYPES,
            initialfile=self._export_stem() + '.png')
        if not path:
            return
        try:
            self.save_figure(path)
        except Exception as exc:
            messagebox.showerror('Export failed', str(exc), parent=self)
            return
        messagebox.showinfo('Export complete', f'Saved {path}', parent=self)

    def _export_all_banks(self) -> None:
        """Save one publication figure per bank, bank number appended."""
        path = filedialog.asksaveasfilename(
            parent=self, title='Export all banks (the bank number is appended)',
            defaultextension='.png', filetypes=EXPORT_FILETYPES,
            initialfile=f'{self._file_stem}_heatmap.png')
        if not path:
            return
        stem, ext = os.path.splitext(path)
        written = []
        try:
            for bank in self._banks:
                out = f'{stem}_bank{bank}{ext or ".png"}'
                if self.save_figure(out, bank):
                    written.append(out)
        except Exception as exc:
            messagebox.showerror('Export failed', str(exc), parent=self)
            return
        if not written:
            messagebox.showwarning('No data', 'No bank has patterns on this axis.',
                                   parent=self)
            return
        messagebox.showinfo('Export complete',
                            'Saved:\n' + '\n'.join(written), parent=self)
