"""
Operando heatmap window: stacked 1D XRD patterns as an intensity map.

The canonical operando figure — 2θ on x, scan number or elapsed time on y,
intensity as colour — with the correlated voltage track alongside on a
shared y axis, so peak shifts line up with the electrochemistry. Patterns
are interpolated onto the first scan's x grid when grids differ.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .config import COLORMAP, FIGURE_DPI, VMIN_FLOOR
from .ici import (
    BG, BG2, BORDER, _PAD_M, _PAD_S, CHARGE_COLOR,
    PLOT_BG, PLOT_BORDER, PLOT_DIM,
    _Btn, _canvas_frame, _label, _separator, _spinbox, _style_ax, _stub,
    embed_figure, maximise_window, sync_figure_to_widget,
)


# ============================================================================
# Matrix Builders
# ============================================================================

def build_xrd_matrix(oned_arrays: Dict[int, Dict[str, Any]]):
    """Stack 1D patterns as (x_grid, matrix, scan_nums), interpolating onto
    the first scan's x grid; (None, None, []) when nothing usable."""
    scan_nums = sorted(oned_arrays)
    rows = []
    x_grid = None
    used = []
    for num in scan_nums:
        entry = oned_arrays[num]
        x = np.asarray(entry.get("x"), dtype=float)
        y = np.asarray(entry.get("y"), dtype=float)
        if x is None or y is None or len(x) != len(y) or len(x) < 2:
            continue
        if x_grid is None:
            x_grid = x
        rows.append(y if len(x) == len(x_grid) and np.array_equal(x, x_grid)
                    else np.interp(x_grid, x, y))
        used.append(num)
    if x_grid is None or not rows:
        return None, None, []
    return x_grid, np.vstack(rows), used


def scan_positions_seconds(scans: List[Dict[str, Any]],
                           time_method: str = "absolute",
                           reference: Optional[pd.Timestamp] = None) -> Dict[int, float]:
    """Elapsed seconds per scan_num on the echem time axis; missing scans are
    omitted. `reference` is the echem start (defaults to the first scan)."""
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
    """Map a full-resolution echem trace onto the heatmap's scan axis.

    Only the time range bracketed by scans is returned, avoiding flat clipped
    tails before the first scan or after the final scan.
    """
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
# Heatmap Window
# ============================================================================

class HeatmapWindow(tk.Toplevel):
    """Operando heatmap window; opens maximised.

        HeatmapWindow(parent, scans, oned_arrays=..., echem_arrays=...)
    """

    def __init__(self, parent: tk.Widget,
                 scans: List[Dict[str, Any]],
                 oned_arrays: Optional[Dict[int, Dict[str, Any]]] = None,
                 echem_arrays: Optional[Dict[str, Any]] = None,
                 time_method: str = "absolute") -> None:
        super().__init__(parent)
        self.title('OperaXN — Operando Heatmap')
        self.configure(bg=BG)
        self.minsize(900, 500)
        maximise_window(self)

        self._oned_arrays = oned_arrays or {}
        self._echem_arrays = echem_arrays or {}
        self._positions = scan_positions_seconds(
            scans, time_method, reference=self._echem_reference(time_method))
        self._replot_job = None
        self._x, self._matrix, self._scan_nums = build_xrd_matrix(self._oned_arrays)

        self._build_ui()
        self._replot()

        self._sync_attempts = 0
        self.after(150, self._sync_watchdog)

    def _echem_reference(self, time_method: str) -> Optional[pd.Timestamp]:
        """Echem start time, so scan rows share the voltage curve's axis."""
        timestamps = self._echem_arrays.get('timestamps') or []
        if time_method == 'absolute' and timestamps:
            try:
                return pd.to_datetime(timestamps[0])
            except (ValueError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the controls strip and the shared-axis figure."""
        controls_wrapper = tk.Frame(self, bg=BG2,
                                    highlightbackground=BORDER,
                                    highlightthickness=1)
        controls_wrapper.pack(side='top', fill='x', padx=_PAD_S, pady=(_PAD_S, 2))
        controls = tk.Frame(controls_wrapper, bg=BG2)
        controls.pack(fill='x', padx=_PAD_S, pady=4)

        # Y axis: scan number vs elapsed time
        _label(controls, 'Y axis:').pack(side='left', padx=(2, _PAD_S))
        self.yaxis_var = tk.StringVar(value='scan')
        self._y_btns = {}
        for value, text in (('scan', 'Scan'), ('time', 'Time')):
            btn = _Btn(controls, text,
                       style='toggle_on' if value == 'scan' else 'toggle_off',
                       command=lambda v=value: self._set_toggle(
                           self.yaxis_var, self._y_btns, v))
            btn.pack(side='left', padx=2)
            self._y_btns[value] = btn

        _separator(controls, horizontal=False).pack(
            side='left', fill='y', padx=_PAD_M, pady=3)

        # Intensity scale
        _label(controls, 'Scale:').pack(side='left', padx=(0, _PAD_S))
        self.scale_var = tk.StringVar(value='linear')
        self._scale_btns = {}
        for value, text in (('linear', 'Linear'), ('log', 'Log')):
            btn = _Btn(controls, text,
                       style='toggle_on' if value == 'linear' else 'toggle_off',
                       command=lambda v=value: self._set_toggle(
                           self.scale_var, self._scale_btns, v))
            btn.pack(side='left', padx=2)
            self._scale_btns[value] = btn

        _separator(controls, horizontal=False).pack(
            side='left', fill='y', padx=_PAD_M, pady=3)

        # Intensity ceiling percentile (clips hot pixels so weak peaks show)
        _label(controls, 'Ceiling (%):', dim=True).pack(side='left')
        self.ceiling_var = tk.DoubleVar(value=100.0)
        ceiling = _spinbox(controls, self.ceiling_var, from_=50.0, to=100.0,
                           increment=0.5, format='%.1f', width=6)
        ceiling.pack(side='left', padx=(2, _PAD_S))
        self.ceiling_var.trace_add('write', lambda *_: self._queue_replot())

        # Time mode needs a position for every plotted scan
        if not self._positions:
            self._y_btns['time'].config(state='disabled')

        # Figure: heatmap and voltage track on a shared y axis
        plot_frame = _canvas_frame(self)
        plot_frame.pack(fill='both', expand=True, padx=_PAD_S, pady=(2, _PAD_S))

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

    def _set_toggle(self, var: tk.StringVar, buttons: dict, value: str) -> None:
        """Switch a toggle group and replot."""
        var.set(value)
        for val, btn in buttons.items():
            btn.set_selected(val == value)
        self._replot()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _y_values(self):
        """Row coordinates: scan numbers, or elapsed hours when they exist
        for every plotted scan and increase monotonically."""
        if self.yaxis_var.get() == 'time':
            hours = [self._positions.get(num) for num in self._scan_nums]
            if all(h is not None for h in hours):
                hours = np.asarray(hours, dtype=float) / 3600.0
                if np.all(np.diff(hours) > 0):
                    return hours, 'Time (h)'
        return np.asarray(self._scan_nums, dtype=float), 'Scan number'

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _queue_replot(self, *_):
        """Debounce ceiling-spinbox edits."""
        if self._replot_job is not None:
            self.after_cancel(self._replot_job)
        self._replot_job = self.after(250, self._replot)

    def _replot(self):
        """Redraw the heatmap and voltage track for the current controls."""
        self._replot_job = None

        ax = self.ax_map
        ax.cla()
        self.cax.cla()

        y_vals, y_label = self._y_values()
        _style_ax(ax, '2θ (°)', y_label)

        if self._matrix is None:
            _stub(ax, 'No 1D patterns loaded')
            self.cax.set_axis_off()
            self._plot_voltage(None, False)
            self._canvas.draw_idle()
            return
        self.cax.set_axis_on()

        matrix = np.clip(self._matrix, VMIN_FLOOR, None)
        try:
            ceiling = float(self.ceiling_var.get())
        except (ValueError, tk.TclError):
            ceiling = 100.0
        vmax = float(np.percentile(matrix, min(max(ceiling, 50.0), 100.0)))
        vmin = float(matrix.min())
        if vmax <= vmin:
            vmax = float(matrix.max())

        if self.scale_var.get() == 'log':
            norm = LogNorm(vmin=max(vmin, VMIN_FLOOR), vmax=max(vmax, VMIN_FLOOR * 10))
            mesh = ax.pcolormesh(self._x, y_vals, matrix, norm=norm,
                                 cmap=COLORMAP, shading='nearest')
        else:
            mesh = ax.pcolormesh(self._x, y_vals, matrix, vmin=vmin, vmax=vmax,
                                 cmap=COLORMAP, shading='nearest')

        colorbar = self._fig.colorbar(mesh, cax=self.cax)
        colorbar.ax.set_facecolor(PLOT_BG)
        colorbar.ax.tick_params(colors=PLOT_DIM, labelsize=7)
        colorbar.outline.set_edgecolor(PLOT_BORDER)

        self._plot_voltage(y_vals, y_label == 'Time (h)')
        self._canvas.draw_idle()

    def _plot_voltage(self, y_vals, time_axis: bool) -> None:
        """Voltage track sharing the map's y axis.

        `time_axis` is the resolved mode (the Time toggle can fall back to
        scan numbers when positions are missing)."""
        ax = self.ax_v
        ax.cla()
        _style_ax(ax, 'Voltage (V)', '')
        ax.tick_params(labelleft=False)

        if y_vals is None:
            _stub(ax, 'No data')
            return

        ex = np.asarray(self._echem_arrays.get('x', []), dtype=float)
        ev = np.asarray(self._echem_arrays.get('y', []), dtype=float)
        if ex.size and ev.size:
            if time_axis:
                # Full echem curve on the shared elapsed-time axis.
                ax.plot(ev, ex / 3600.0, color=CHARGE_COLOR, lw=1.2)
                return

            # Preserve the full echem resolution in scan mode by mapping its
            # elapsed-time samples onto the neighbouring scan positions.
            curve_v, curve_y = echem_scan_curve(
                self._scan_nums, y_vals, self._positions, ex, ev)
            if curve_v.size:
                ax.plot(curve_v, curve_y, color=CHARGE_COLOR, lw=1.2)
                return

        # Per-scan correlated voltage fallback when scan timestamps or the full
        # echem trace are unavailable. Draw a continuous marker-free curve.
        volts, ys = [], []
        for num, y in zip(self._scan_nums, y_vals):
            v = self._oned_arrays.get(num, {}).get('echem')
            if v is not None:
                volts.append(v)
                ys.append(y)
        if volts:
            volts = np.asarray(volts, dtype=float)
            ys = np.asarray(ys, dtype=float)
            if len(volts) > 1:
                dense_y = np.linspace(ys[0], ys[-1], max(200, len(ys) * 8))
                volts = np.interp(dense_y, ys, volts)
                ys = dense_y
            ax.plot(volts, ys, color=CHARGE_COLOR, lw=1.2)
        else:
            _stub(ax, 'No echem')

    # ------------------------------------------------------------------
    # Display sync (see ici.sync_figure_to_widget)
    # ------------------------------------------------------------------

    def _sync_watchdog(self):
        """Re-fit the figure to its widget while the window is settling."""
        if not self.winfo_exists():
            return
        sync_figure_to_widget(self._canvas)
        self._sync_attempts += 1
        if self._sync_attempts < 12:
            self.after(200, self._sync_watchdog)
