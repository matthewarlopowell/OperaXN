"""
Echem capacity analysis and window: charge/discharge phases, cycles, cumulative capacity and
voltage plots.

The analysis operates on the input adapter's echem DataFrame (timestamp, echem_data, current)
and is shared with the ICI window; CapacityWindow is built from the ICI module's widgets so
the analysis windows look alike. Ported from the OperaXN-ICI fork (Joe Arroyo).
"""

import logging
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .config import CSV_ENCODING, FIGURE_DPI, SECONDS_PER_HOUR, TIME_HOURS_LABEL
from .ici import (
    BG, BG2, BG3, BORDER, DIM, INPUT, TEXT, _FONT, _PAD_S,
    PLOT_BG, PLOT_BORDER, PLOT_TEXT,
    _Btn, _canvas_frame, _label, _separator, _spinbox, _style_ax,
    echem_source_button, embed_figure, maximise_window, watch_figure_sync,
)

logger = logging.getLogger(__name__)


def _assign_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a t_s column (elapsed seconds) unless present, falling back to the sample index."""
    if "t_s" not in df.columns:
        try:
            # In relative mode the timestamp column is rewritten to HH:MM:SS
            # display strings; original_timestamp keeps full precision.
            ts_col = "original_timestamp" if "original_timestamp" in df.columns else "timestamp"
            ts = pd.to_datetime(df[ts_col])
            df["t_s"] = (ts - ts.iloc[0]).dt.total_seconds()
        except Exception:
            df["t_s"] = np.arange(len(df), dtype=float)
    return df


def _label_phases(current: np.ndarray) -> np.ndarray:
    """Label samples by current sign; zero current inherits the prior label, or 'rest' if none."""
    labels = np.empty(len(current), dtype=object)
    for i in range(len(current)):
        if current[i] > 0:
            labels[i] = "charge"
        elif current[i] < 0:
            labels[i] = "discharge"
        else:
            labels[i] = (
                labels[i - 1]
                if i > 0 and labels[i - 1] in ("charge", "discharge")
                else "rest"
            )
    return labels


def assign_cycles(echem_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with t_s, phase and cycle columns; each new charge block starts a cycle.
    Cycles are 1-indexed; samples before the first charge stay cycle 0."""
    df = _assign_time(echem_df.copy())
    labels = _label_phases(df["current"].fillna(0).values)
    df["phase"] = labels

    cycles = np.zeros(len(df), dtype=int)
    cycle_num = 0
    prev = None

    for i in range(len(labels)):
        if labels[i] == "charge" and prev != "charge":
            cycle_num += 1
        if labels[i] in ("charge", "discharge"):
            cycles[i] = cycle_num
        # leading rests stay cycle 0
        prev = labels[i]

    df["cycle"] = cycles
    return df


def compute_capacity(df: pd.DataFrame) -> np.ndarray:
    """Return cumulative capacity (mAh) by trapezoidal integration of |current| (mA) over t_s.
    Starts at 0 on the segment's first row."""
    if df.empty:
        return np.array([])

    t = df["t_s"].values
    i_mA = np.abs(df["current"].fillna(0).values)

    dt = np.diff(t, prepend=t[0])
    i_mid = np.concatenate(([i_mA[0]], (i_mA[:-1] + i_mA[1:]) / 2))
    return np.cumsum(i_mid * dt) / SECONDS_PER_HOUR


def _start_cycle_plot(ax, echem_df, xlabel, cycles_to_plot):
    """Reset ax for a voltage plot; return (df, cycles, colours), or None after a no-data note."""
    ax.clear()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    if echem_df is None or echem_df.empty:
        ax.text(0.5, 0.5, "No echem data loaded", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=11)
        return None
    df = assign_cycles(echem_df)
    all_cycles = sorted([c for c in df["cycle"].unique() if c > 0])
    cycles = cycles_to_plot if cycles_to_plot is not None else all_cycles
    cmap = cm.tab10 if len(all_cycles) <= 10 else cm.viridis
    colors = {c: cmap(i / max(len(all_cycles) - 1, 1)) for i, c in enumerate(all_cycles)}
    return df, cycles, colors


def plot_capacity_vs_voltage(ax, echem_df, mass_mg=0.0, cycles_to_plot=None):
    """Plot voltage against capacity (mAh/g when mass_mg > 0), one colour per cycle.
    Charge and discharge legs share the cycle colour; only the charge leg is labelled."""
    use_specific = mass_mg > 0
    setup = _start_cycle_plot(
        ax, echem_df, "Specific Capacity (mAh/g)" if use_specific else "Capacity (mAh)",
        cycles_to_plot)
    if setup is None:
        return
    df, cycles, colors = setup
    for cycle_num in cycles:
        cycle_df = df[df["cycle"] == cycle_num]
        color = colors.get(cycle_num, "grey")
        for phase in ("charge", "discharge"):
            phase_df = cycle_df[cycle_df["phase"] == phase]
            if phase_df.empty:
                continue
            cap = compute_capacity(phase_df)
            if use_specific:
                cap = cap / (mass_mg / 1000.0)
            ax.plot(cap, phase_df["echem_data"].values, color=color, linewidth=1.5,
                    label=f"Cycle {cycle_num}" if phase == "charge" else None)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")


def plot_time_vs_voltage(ax, echem_df, cycles_to_plot=None):
    """Plot voltage against elapsed time (h), one colour per cycle."""
    setup = _start_cycle_plot(ax, echem_df, TIME_HOURS_LABEL, cycles_to_plot)
    if setup is None:
        return
    df, cycles, colors = setup
    for cycle_num in cycles:
        cycle_df = df[df["cycle"] == cycle_num]
        ax.plot(cycle_df["t_s"].values / SECONDS_PER_HOUR, cycle_df["echem_data"].values,
                color=colors.get(cycle_num, "grey"), linewidth=1.5, label=f"Cycle {cycle_num}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")


# ============================================================================
# Capacity Window
# ============================================================================

MASS_REPLOT_DELAY_MS = 150  # debounce for live redraws while the mass is typed


class CapacityWindow(tk.Toplevel):
    """Capacity analysis window (voltage against time and capacity); opens maximised.
    Mass and cycle edits redraw live, with no Apply step; CSVs default to <export_stem>_capacity."""

    def __init__(self, parent: tk.Widget, echem_sources: Dict[str, pd.DataFrame],
                 on_add_source=None, export_stem: str = 'experiment') -> None:
        super().__init__(parent)
        self.title('OperaXN: Capacity Analysis')
        self.configure(bg=BG)
        self.minsize(900, 500)
        maximise_window(self)

        self._sources = dict(echem_sources)
        self._source_name = next(iter(self._sources))
        self._df = self._sources[self._source_name]
        self._available: List[int] = []
        self._on_add_source = on_add_source
        self._export_stem = export_stem
        self._mass_job = None

        self._build_ui()
        self._refresh_available()
        # Opens showing every cycle, so the stepper starts disabled
        self._toggle_all_cycles(replot=False)
        self._replot()
        watch_figure_sync(self, (self._canvas,))
        self.lift()
        self.focus_force()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the ICI-style layout: selector strip above the figure, mass and export below."""
        # Bottom bar, packed first like the ICI regression bar and the heatmap display bar
        bar = self._strip('bottom', (2, _PAD_S))
        _label(bar, 'Mass (mg):').pack(side='left', padx=(2, 0))
        self.mass_var = tk.StringVar(value='0')
        mass_entry = _spinbox(bar, self.mass_var, from_=0.0, to=99999.0,
                              increment=0.1, format='%.2f', width=7,
                              command=self._queue_mass_replot)
        mass_entry.pack(side='left', padx=(2, _PAD_S))
        mass_entry.bind('<Return>', self._replot)
        mass_entry.bind('<FocusOut>', self._queue_mass_replot)
        self.mass_var.trace_add('write', self._queue_mass_replot)
        _separator(bar, horizontal=False).pack(side='left', fill='y', padx=_PAD_S, pady=3)
        _Btn(bar, 'Export CSV', command=self._export_csv).pack(side='left', padx=2)

        # Selector strip: echem source, then the cycle stepper
        strip = self._strip('top', (_PAD_S, 2))
        self.source_var = tk.StringVar(value=self._source_name)
        # Swap datasets only once the dropdown has closed
        self.source_button = echem_source_button(
            strip, self.source_var, list(self._sources),
            command=lambda name: self.after_idle(self._on_source_change, name),
            on_add=self._on_add_source)
        self.source_button.pack(side='left', padx=2)
        _separator(strip, horizontal=False).pack(side='left', fill='y', padx=2, pady=3)

        _label(strip, 'Cycle:').pack(side='left', padx=(2, 0))
        self.cycle_var = tk.IntVar(value=1)
        self._prev_btn = _Btn(strip, '◀', width=1, padx=2,
                              command=lambda: self._step_cycle(-1))
        self._prev_btn.pack(side='left', padx=1)
        self._cycle_spin = _spinbox(strip, self.cycle_var, from_=1, to=1, width=4,
                                    command=self._replot,
                                    disabledbackground=INPUT, disabledforeground=DIM)
        self._cycle_spin.pack(side='left', padx=2)
        self._cycle_spin.bind('<Return>', self._replot)
        self._next_btn = _Btn(strip, '▶', width=1, padx=2,
                              command=lambda: self._step_cycle(1))
        self._next_btn.pack(side='left', padx=1)

        self.all_cycles_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            strip, text='All cycles', variable=self.all_cycles_var,
            command=self._toggle_all_cycles,
            bg=BG2, fg=TEXT,
            activebackground=BG2, activeforeground=TEXT,
            disabledforeground=TEXT,
            selectcolor=BG3,
            font=_FONT,
        ).pack(side='left', padx=(_PAD_S, 2))
        self._cycles_label = _label(strip, '', dim=True)
        self._cycles_label.pack(side='left')

        # Figure: voltage against time and against capacity
        plot_frame = _canvas_frame(self)
        plot_frame.pack(fill='both', expand=True, padx=_PAD_S, pady=2)
        plot_frame.pack_propagate(False)
        self._fig = Figure(facecolor=PLOT_BG, dpi=FIGURE_DPI)
        gs = GridSpec(1, 2, figure=self._fig, wspace=0.22,
                      left=0.07, right=0.98, top=0.93, bottom=0.10)
        self.ax_time = self._fig.add_subplot(gs[0, 0])
        self.ax_cap = self._fig.add_subplot(gs[0, 1])
        self._canvas = embed_figure(self._fig, plot_frame)

    def _strip(self, side: str, pady: Tuple[int, int]) -> tk.Frame:
        """Pack a bordered control strip on one edge of the window and return its row."""
        wrapper = tk.Frame(self, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        wrapper.pack(side=side, fill='x', padx=_PAD_S, pady=pady)
        row = tk.Frame(wrapper, bg=BG2)
        row.pack(fill='x', padx=2, pady=4)
        return row

    @staticmethod
    def _style_axes(ax, title: str) -> None:
        """Apply the analysis-figure theme, keeping the plot's own labels, and theme its legend."""
        _style_ax(ax, ax.get_xlabel(), ax.get_ylabel(), title)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(PLOT_BG)
            legend.get_frame().set_edgecolor(PLOT_BORDER)
            for text in legend.get_texts():
                text.set_color(PLOT_TEXT)

    # ------------------------------------------------------------------
    # Cycles, mass and plotting
    # ------------------------------------------------------------------

    def _refresh_available(self) -> None:
        """Detect the dataset's cycles and bound the cycle stepper to them."""
        try:
            self._available = sorted(
                c for c in assign_cycles(self._df)['cycle'].unique() if c > 0)
        except Exception:
            logger.debug('Cycle detection failed: %s', traceback.format_exc())
            self._available = []
        self._cycles_label.config(text=f'({len(self._available)} available)')
        self._cycle_spin.config(to=max(self._available, default=1))
        if self._selected_cycle() not in self._available:
            self.cycle_var.set(self._available[0] if self._available else 1)

    def _selected_cycle(self) -> Optional[int]:
        """Return the stepper's cycle, or None while its text is not a number."""
        try:
            return self.cycle_var.get()
        except tk.TclError:
            return None

    def _get_mass(self) -> float:
        """Return the entered sample mass in mg (0 while blank or invalid)."""
        try:
            return float(self.mass_var.get())
        except (ValueError, tk.TclError):
            return 0.0

    def _get_cycles(self) -> List[int]:
        """Return every available cycle, or the selected one when stepping."""
        if self.all_cycles_var.get():
            return list(self._available)
        selected = self._selected_cycle()
        return [selected] if selected in self._available else []

    def _replot(self, *_args) -> None:
        """Redraw both panels for the selected cycles and mass."""
        selected = self._get_cycles()
        plot_time_vs_voltage(self.ax_time, self._df, cycles_to_plot=selected)
        plot_capacity_vs_voltage(self.ax_cap, self._df, mass_mg=self._get_mass(),
                                 cycles_to_plot=selected)
        self._style_axes(self.ax_time, 'Voltage vs time')
        self._style_axes(self.ax_cap, 'Voltage vs capacity')
        self._canvas.draw()

    def _queue_mass_replot(self, *_args) -> None:
        """Redraw shortly after the last mass edit, so typing stays responsive."""
        if self._mass_job is not None:
            try:
                self.after_cancel(self._mass_job)
            except tk.TclError:
                pass
        self._mass_job = self.after(MASS_REPLOT_DELAY_MS, self._run_mass_replot)

    def _run_mass_replot(self) -> None:
        """Run the queued mass redraw."""
        self._mass_job = None
        self._replot()

    def _step_cycle(self, delta: int) -> None:
        """Step to the previous or next available cycle, leaving all-cycles mode."""
        if not self._available:
            return
        self.all_cycles_var.set(False)
        try:
            index = self._available.index(self._selected_cycle())
        except ValueError:
            index = 0
        self.cycle_var.set(self._available[max(0, min(len(self._available) - 1, index + delta))])
        self._toggle_all_cycles(replot=False)
        self._replot()

    def _toggle_all_cycles(self, replot: bool = True) -> None:
        """Enable the cycle stepper only while a single cycle is shown."""
        state = 'disabled' if self.all_cycles_var.get() else 'normal'
        for widget in (self._prev_btn, self._cycle_spin, self._next_btn):
            widget.config(state=state)
        if replot:
            self._replot()

    # ------------------------------------------------------------------
    # Sources and export
    # ------------------------------------------------------------------

    def _on_source_change(self, name: str) -> None:
        """Switch to another echem dataset, showing all its cycles."""
        if name == self._source_name or name not in self._sources:
            return
        self.configure(cursor='watch')
        self.update_idletasks()
        try:
            self._source_name = name
            self._df = self._sources[name]
            self.cycle_var.set(1)
            self.all_cycles_var.set(True)
            self._refresh_available()
            self._toggle_all_cycles(replot=False)
            self._replot()
        finally:
            self.configure(cursor='')

    def refresh_sources(self, sources: Dict[str, pd.DataFrame]) -> None:
        """Expose newly appended echem files; fall back to the first if the shown one is gone."""
        self._sources = dict(sources)
        self.source_button.set_values(self._sources)
        if self._source_name not in self._sources and self._sources:
            name = next(iter(self._sources))
            self.source_var.set(name)
            self._on_source_change(name)

    def _export_csv(self) -> None:
        """Write the selected cycles' voltage and (specific) capacity to a CSV file."""
        selected = self._get_cycles()
        if not selected:
            messagebox.showwarning('No cycles', 'No cycles selected to export.', parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            filetypes=[('CSV', '*.csv')], initialfile=f'{self._export_stem}_capacity.csv')
        if not path:
            return
        mass = self._get_mass()
        df_full = assign_cycles(self._df)
        rows = []
        for cycle_num in selected:
            cycle_df = df_full[df_full['cycle'] == cycle_num]
            for phase in ('charge', 'discharge'):
                phase_df = cycle_df[cycle_df['phase'] == phase]
                if phase_df.empty:
                    continue
                cap = compute_capacity(phase_df)
                spec_cap = (cap / (mass / 1000.0) if mass > 0
                            else np.full_like(cap, np.nan))
                for t, v, c, sc in zip(phase_df['t_s'].values,
                                       phase_df['echem_data'].values,
                                       cap, spec_cap):
                    rows.append({
                        'Cycle': cycle_num,
                        'Phase': phase,
                        'Time (s)': t,
                        'Voltage (V)': v,
                        'Capacity (mAh)': c,
                        'Specific Capacity (mAh/g)': sc,
                    })
        pd.DataFrame(rows).to_csv(path, index=False, encoding=CSV_ENCODING)
        messagebox.showinfo('Export complete', f'Saved data to:\n{path}', parent=self)
