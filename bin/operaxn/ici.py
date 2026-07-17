"""
ICI (intermittent current interruption) analysis window.

Pulse detection and regression follow the pyICI conventions
(https://github.com/joe-arroyo/pyICI): a pulse is an active period
(current != 0) followed by a rest (current == 0) no longer than max_rest;
for each pulse a linear fit of ΔV against √Δt over a configurable time
window yields R = -intercept/I and k = -slope/I, with errors propagated
from the fit covariance.

Panels: overview (V vs t, phase-coloured), pulse zoom (V + I), ICI fit
(ΔV vs √Δt with Δt secondary axis), R² vs pulse, and R/k vs voltage,
capacity, or specific capacity for charge and discharge.

Ported from the OperaXN-ICI fork (Joe Arroyo).
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .capacity import assign_cycles, compute_capacity
from .config import FIGURE_DPI, OPERAXNTheme

# Theme shorthands (module-local; every widget below uses them)
_C = OPERAXNTheme.COLORS
BG = _C['bg_primary']
BG2 = _C['bg_secondary']
BG3 = _C['bg_tertiary']
CANVAS = _C['canvas_bg']
ACCENT = _C['accent_primary']
TEXT = _C['text_primary']
DIM = _C['text_dim']
BORDER = _C['border']
INPUT = _C['input_bg']

PLOT_BG = '#ffffff'
PLOT_TEXT = '#20242c'
PLOT_DIM = '#4b5563'
PLOT_BORDER = '#9ca3af'
PLOT_GRID = '#d8dde6'

CHARGE_COLOR = '#008fb3'
DISCHARGE_COLOR = '#d84b4b'
CURRENT_COLOR = '#9a7400'
HIGHLIGHT_COLOR = '#4b5563'

_FONT = OPERAXNTheme.FONTS['small']
_FONT_BTN = OPERAXNTheme.FONTS['button']
_PAD_S = OPERAXNTheme.PADDING['small']
_PAD_M = OPERAXNTheme.PADDING['medium']


# ============================================================================
# Window Helpers
# ============================================================================

def maximise_window(win: tk.Toplevel) -> None:
    """Open a toplevel maximised, with fallbacks per window manager."""
    try:
        win.state('zoomed')
    except tk.TclError:
        try:
            win.attributes('-zoomed', True)
        except tk.TclError:
            width = win.winfo_screenwidth()
            height = win.winfo_screenheight()
            win.geometry(f'{width}x{height}+0+0')


# ============================================================================
# Data Helpers
# ============================================================================

def _prepare_df(echem_df: pd.DataFrame) -> pd.DataFrame:
    """Return echem_df with t_s/phase/cycle columns for ICI analysis."""
    return assign_cycles(echem_df)


def _detect_pulses(phase_df: pd.DataFrame, max_rest: float = 1800.0) -> list[dict]:
    """
    Mirrors pyICI's assign_valid_pulses logic.

    A pulse = active period (current != 0) followed by rest (current == 0).
    Valid only if 0 < rest_duration <= max_rest. This naturally excludes the
    CV step (rest too long) and inter-cycle rests.
    """
    if phase_df.empty:
        return []

    df = phase_df.reset_index(drop=True)
    current = df['current'].fillna(0).values
    t_s = df['t_s'].values
    n = len(current)

    pulses = []
    i = 0
    while i < n:
        if current[i] == 0:
            i += 1
            continue

        # Active period
        pulse_start = i
        while i < n and current[i] != 0:
            i += 1
        pulse_end = i - 1

        # Rest period
        rest_start = i
        while i < n and current[i] == 0:
            i += 1
        rest_end = i - 1

        # Valid only if rest exists and is within limit
        if rest_end < rest_start:
            continue
        rest_duration = t_s[rest_end] - t_s[rest_start]
        if not (0 < rest_duration <= max_rest):
            continue

        pulse_mask = np.zeros(n, dtype=bool)
        pulse_mask[pulse_start:pulse_end + 1] = True
        relax_mask = np.zeros(n, dtype=bool)
        relax_mask[rest_start:rest_end + 1] = True

        pulses.append({
            'pulse_mask': pulse_mask,
            'relax_mask': relax_mask,
            'pulse_start_t': t_s[pulse_start],
            'pulse_end_t': t_s[pulse_end],
            'relax_end_t': t_s[rest_end],
        })

    return pulses


def _compute_fit(seg: pd.DataFrame, pulse: dict, r1_start: float, r1_length: float,
                 cap_arr: np.ndarray | None = None) -> dict | None:
    """ΔV vs √Δt linear fit for one pulse.

    r1_start/r1_length are a time window (seconds) into the rest period:
    fit points with r1_start <= Δt < r1_start + r1_length. Returns None if
    rest data is insufficient.

    cap_arr, if given, is the cumulative capacity (mAh) for `seg` (same
    length, same row order — see compute_capacity), used to attach a capacity
    value to this pulse's V0 point by position rather than by voltage
    (voltage can repeat across pulses in flat regions, capacity is monotonic).
    """
    active = seg[pulse['pulse_mask']]
    rest = seg[pulse['relax_mask']]
    if active.empty or rest.empty:
        return None

    V0 = active['echem_data'].iloc[-1]
    I_A = active['current'].iloc[-1] / 1000.0  # mA -> A

    if cap_arr is not None and len(cap_arr) == len(seg):
        pulse_cap = cap_arr[pulse['pulse_mask']]
        capacity = float(pulse_cap[-1]) if len(pulse_cap) else np.nan
    else:
        capacity = np.nan

    delta_V = rest['echem_data'].values - V0
    elapsed = rest['t_s'].values - rest['t_s'].values[0]
    sqrt_dt = np.sqrt(np.maximum(elapsed, 0.0))

    win_mask = (elapsed >= r1_start) & (elapsed < r1_start + r1_length)
    if win_mask.sum() < 2:
        return None

    X = sqrt_dt[win_mask]
    y = delta_V[win_mask]

    X_mean, y_mean = np.mean(X), np.mean(y)
    dX = X - X_mean
    denom = np.dot(dX, dX)
    if denom == 0:
        return None
    slope = np.dot(dX, y - y_mean) / denom
    intercept = y_mean - slope * X_mean
    y_pred = slope * X + intercept

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    n_pts = len(X)
    s2 = ss_res / max(n_pts - 2, 1)
    var_slope = s2 / denom
    var_intercept = s2 * (1.0 / n_pts + X_mean ** 2 / denom)

    R = -intercept / I_A if I_A != 0 else np.nan
    k = -slope / I_A if I_A != 0 else np.nan
    R_err = abs(R) * np.sqrt(var_intercept) / abs(intercept) if (I_A and intercept) else np.nan
    k_err = abs(k) * np.sqrt(var_slope) / abs(slope) if (I_A and slope) else np.nan

    return dict(
        sqrt_dt=sqrt_dt, delta_V=delta_V,
        X_fit=X, y_fit=y_pred,
        slope=slope, intercept=intercept, r2=r2,
        V0=V0, I_A=I_A, R=R, k=k, R_err=R_err, k_err=k_err,
        capacity=capacity,
    )


def _compute_all_pulses(seg: pd.DataFrame, pulses: list[dict], get_r1) -> list[dict]:
    """Fit every pulse in seg; get_r1(pulse_idx) -> (r1_start_s, r1_length_s)."""
    cap_arr = compute_capacity(seg) if not seg.empty else np.array([])
    results = []
    for i, p in enumerate(pulses):
        r1_start, r1_length = get_r1(i + 1)
        fit = _compute_fit(seg, p, r1_start, r1_length, cap_arr=cap_arr)
        if fit is not None:
            fit['pulse_idx'] = i + 1
            results.append(fit)
    return results


# ============================================================================
# Shared Widget Helpers
# ============================================================================

class _Btn(tk.Button):
    """Small themed button matching the main-window StyledButton look."""

    _STYLES = {
        'primary': (ACCENT, _C['button_text']),
        'secondary': (BG3, TEXT),
        'toggle_on': (ACCENT, _C['button_text']),
        'toggle_off': (BG3, DIM),
    }

    def __init__(self, master, text, command=None, style='secondary', **kw):
        bg, fg = self._STYLES.get(style, self._STYLES['secondary'])
        padx = kw.pop('padx', 10)
        pady = kw.pop('pady', 3)
        super().__init__(
            master, text=text, command=command,
            bg=bg, fg=fg,
            activebackground=_C['accent_hover'],
            activeforeground=_C['button_text'],
            font=_FONT_BTN, relief=tk.FLAT, cursor='hand2',
            padx=padx, pady=pady, **kw,
        )
        self._bg = bg
        self._hover = _C['accent_hover'] if style == 'primary' else BG2
        self.bind('<Enter>', lambda _: self._set_hover_state(True))
        self.bind('<Leave>', lambda _: self._set_hover_state(False))

    def _set_hover_state(self, hovering: bool) -> None:
        """Set button hover state."""
        if self['state'] != 'disabled':
            self.config(bg=self._hover if hovering else self._bg)

    def set_selected(self, selected: bool) -> None:
        """Update a toggle button without losing its state on mouse leave."""
        self._bg = ACCENT if selected else BG3
        self.config(
            bg=self._bg,
            fg=_C['button_text'] if selected else DIM,
        )


def _spinbox(master, var, **kw):
    """Themed spinbox."""
    return tk.Spinbox(
        master, textvariable=var,
        bg=INPUT, fg=TEXT, buttonbackground=BG2,
        relief=tk.FLAT, font=_FONT,
        highlightbackground=BORDER, highlightthickness=1,
        **kw,
    )


def _label(master, text, dim=False, **kw):
    """Themed label; dim=True uses the dimmed text colour."""
    return tk.Label(
        master, text=text,
        bg=kw.pop('bg', BG2),
        fg=DIM if dim else TEXT,
        font=_FONT, **kw,
    )


def _separator(master, horizontal=True):
    """One-pixel border-coloured separator frame."""
    if horizontal:
        return tk.Frame(master, bg=BORDER, height=1)
    return tk.Frame(master, bg=BORDER, width=1)


def _canvas_frame(master):
    """Styled border frame that matches the main window's plot container."""
    return tk.Frame(
        master,
        bg=PLOT_BG,
        relief=tk.FLAT,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        highlightthickness=2,
        takefocus=True,
    )


def echem_source_button(master, var, values, command=None,
                        on_add=None) -> tk.Menubutton:
    """Compact themed menu button for choosing an echem dataset.

    Shared by the ICI and Capacity analysis windows.
    """
    values = list(values)

    def display_text(value) -> str:
        value = str(value)
        short = value if len(value) <= 16 else f'{value[:13]}...'
        return f'Echem: {short}  ▾'

    display_var = tk.StringVar(value=display_text(var.get()))
    button = tk.Menubutton(master, textvariable=display_var)
    button.config(
        bg=BG3, fg=TEXT,
        activebackground=_C['accent_hover'],
        activeforeground=_C['button_text'],
        font=_FONT, relief=tk.FLAT, cursor='hand2',
        highlightthickness=1, highlightbackground=BORDER,
        indicatoron=False, width=24, anchor='w',
    )
    source_menu = tk.Menu(button, tearoff=False)
    source_menu.config(
        bg=BG2, fg=TEXT,
        activebackground=ACCENT,
        activeforeground=_C['button_text'],
        font=_FONT,
    )

    def select(value) -> None:
        var.set(value)
        if command is not None:
            command(value)

    def set_values(new_values) -> None:
        source_menu.delete(0, 'end')
        new_values = list(new_values)
        for value in new_values:
            source_menu.add_radiobutton(
                label=str(value), value=value, variable=var,
                command=lambda selected=value: select(selected),
            )
        if on_add is not None:
            if new_values:
                source_menu.add_separator()
            source_menu.add_command(label='Add echem...', command=on_add)
        button.config(
            state='normal' if len(new_values) > 1 or on_add is not None else 'disabled',
            disabledforeground=DIM,
        )

    set_values(values)
    button.config(menu=source_menu)
    var.trace_add('write', lambda *_: display_var.set(display_text(var.get())))
    button.set_values = set_values
    return button


# ============================================================================
# Selector Strip (Cycle | Pulse | Phase | Scope)
# ============================================================================

class _SelectorStrip(tk.Frame):
    """Navigation strip: echem source, cycle/pulse spinboxes, phase toggle,
    R² scope."""

    def __init__(self, master, on_change, source_names=None,
                 on_source_change=None, on_add_source=None):
        super().__init__(master, bg=BG2)
        self._on_change = on_change
        self._source_names = list(source_names or [])
        self._on_source_change = on_source_change
        self._on_add_source = on_add_source
        self._build()

    def _build(self):
        """Create the selector widgets."""
        row = tk.Frame(self, bg=BG2)
        row.pack(fill='x', padx=1, pady=4)

        # Echem source (only when there is a choice to make)
        if self._source_names and self._on_source_change is not None:
            source_frame = tk.Frame(row, bg=BG2)
            source_frame.pack(side='left', padx=(1, 2))
            self.source_var = tk.StringVar(value=self._source_names[0])
            self.source_button = echem_source_button(
                source_frame, self.source_var, self._source_names,
                command=self._on_source_change,
                on_add=self._on_add_source,
            )
            self.source_button.pack(side='left', padx=2)
            _separator(row, horizontal=False).pack(
                side='left', fill='y', padx=2, pady=3)

        # Cycle
        cycle_frame = tk.Frame(row, bg=BG2)
        cycle_frame.pack(side='left', padx=2)
        _label(cycle_frame, 'Cycle:').pack(side='left')
        self.cycle_var = tk.IntVar(value=1)
        _Btn(cycle_frame, '◀', width=1, padx=2,
             command=self._prev_cycle).pack(side='left', padx=1)
        self._cycle_spin = _spinbox(cycle_frame, self.cycle_var, from_=1, to=999, width=4,
                                    command=self._on_change)
        self._cycle_spin.pack(side='left', padx=2)
        _Btn(cycle_frame, '▶', width=1, padx=2,
             command=self._next_cycle).pack(side='left', padx=1)
        self.cycle_var.trace_add('write', lambda *_: self._on_change())

        _separator(row, horizontal=False).pack(
            side='left', fill='y', padx=2, pady=3)

        # Pulse
        pulse_frame = tk.Frame(row, bg=BG2)
        pulse_frame.pack(side='left', padx=2)
        _label(pulse_frame, 'Pulse:').pack(side='left')
        self.pulse_var = tk.IntVar(value=1)
        _Btn(pulse_frame, '◀', width=1, padx=2,
             command=self._prev_pulse).pack(side='left', padx=1)
        self._pulse_spin = _spinbox(pulse_frame, self.pulse_var, from_=1, to=999, width=4,
                                    command=self._on_change)
        self._pulse_spin.pack(side='left', padx=2)
        _Btn(pulse_frame, '▶', width=1, padx=2,
             command=self._next_pulse).pack(side='left', padx=1)
        self.pulse_var.trace_add('write', lambda *_: self._on_change())

        _separator(row, horizontal=False).pack(
            side='left', fill='y', padx=2, pady=3)

        # Phase toggle
        phase_frame = tk.Frame(row, bg=BG2)
        phase_frame.pack(side='left', padx=2)
        _label(phase_frame, 'Phase:').pack(side='left')
        self.phase_var = tk.StringVar(value='charge')
        self._btn_chg = _Btn(phase_frame, 'Charge', style='toggle_on', padx=4,
                             command=lambda: self._set_phase('charge'))
        self._btn_chg.pack(side='left', padx=2)
        self._btn_dis = _Btn(phase_frame, 'Discharge', style='toggle_off', padx=4,
                             command=lambda: self._set_phase('discharge'))
        self._btn_dis.pack(side='left', padx=2)

        _separator(row, horizontal=False).pack(
            side='left', fill='y', padx=2, pady=3)

        # R² scope
        scope_frame = tk.Frame(row, bg=BG2)
        scope_frame.pack(side='left', padx=2)
        _label(scope_frame, 'R²:').pack(side='left')
        self.scope_var = tk.StringVar(value='phase')
        self._scope_btns = {}
        for value, text in (('phase', 'Phase'), ('cycle', 'Cycle'), ('all', 'All')):
            btn = _Btn(scope_frame, text, padx=4,
                       style='toggle_on' if value == 'phase' else 'toggle_off',
                       command=lambda v=value: self._set_scope(v))
            btn.pack(side='left', padx=2)
            self._scope_btns[value] = btn

        self.show_all_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row, text='All cycles',
            variable=self.show_all_var,
            command=self._on_change,
            bg=BG2, fg=TEXT,
            activebackground=BG2, activeforeground=TEXT,
            disabledforeground=TEXT,
            selectcolor=BG3,
            font=_FONT,
        ).pack(side='left', padx=2)

    def _prev_cycle(self):
        """Step to the previous cycle."""
        value = self.cycle_var.get()
        if value > 1:
            self.cycle_var.set(value - 1)

    def _next_cycle(self):
        """Step to the next cycle."""
        value = self.cycle_var.get()
        if value < int(self._cycle_spin.cget('to')):
            self.cycle_var.set(value + 1)

    def _prev_pulse(self):
        """Step to the previous pulse."""
        value = self.pulse_var.get()
        if value > 1:
            self.pulse_var.set(value - 1)

    def _next_pulse(self):
        """Step to the next pulse."""
        value = self.pulse_var.get()
        if value < int(self._pulse_spin.cget('to')):
            self.pulse_var.set(value + 1)

    def set_pulse_range(self, max_pulse: int) -> None:
        """Bound the pulse spinbox to the pulses in the current selection."""
        clamped = max(1, max_pulse)
        self._pulse_spin.config(to=clamped)
        if self.pulse_var.get() > clamped:
            self.pulse_var.set(clamped)

    def set_cycle_range(self, max_cycle: int) -> None:
        """Bound the cycle spinbox to the cycles in the loaded data."""
        self._cycle_spin.config(to=max_cycle)
        self.cycle_var.set(1)

    def set_sources(self, source_names) -> None:
        """Refresh the echem menu after files are appended to the session."""
        self._source_names = list(source_names)
        if hasattr(self, 'source_button'):
            self.source_button.set_values(self._source_names)
        if self._source_names and self.source_var.get() not in self._source_names:
            self.source_var.set(self._source_names[0])

    def _set_phase(self, phase):
        """Switch the phase toggle and refresh."""
        self.phase_var.set(phase)
        self._btn_chg.set_selected(phase == 'charge')
        self._btn_dis.set_selected(phase == 'discharge')
        self._on_change()

    def _set_scope(self, scope):
        """Switch the R² scope toggle and refresh."""
        self.scope_var.set(scope)
        for value, btn in self._scope_btns.items():
            btn.set_selected(value == scope)
        self._on_change()


# ============================================================================
# Regression Bar (bottom)
# ============================================================================

class _RegressionBar(tk.Frame):
    """Regression window and pulse-detection controls.

    Edits apply to the selected pulse automatically (like the main window's
    intensity controls); Apply All pushes the current values to every pulse."""

    def __init__(self, master, on_change, on_apply_all, on_export):
        super().__init__(master, bg=BG2)
        self._on_change = on_change
        self._on_apply_all = on_apply_all
        self._on_export = on_export
        self._build()

    def _build(self):
        """Create the regression bar widgets."""
        row = tk.Frame(self, bg=BG2)
        row.pack(fill='x', padx=2, pady=4)

        _label(row, 'Regression:').pack(side='left', padx=(2, _PAD_S))
        _label(row, 'Start (s):', dim=True).pack(side='left')
        self.start_var = tk.DoubleVar(value=0.5)
        _spinbox(row, self.start_var, from_=0.0, to=9999.0,
                 increment=0.1, format='%.2f', width=5).pack(
            side='left', padx=(2, _PAD_S))
        _label(row, 'Length (s):', dim=True).pack(side='left')
        self.length_var = tk.DoubleVar(value=1.0)
        _spinbox(row, self.length_var, from_=0.1, to=9999.0,
                 increment=0.1, format='%.2f', width=5).pack(
            side='left', padx=(2, _PAD_S))

        _label(row, 'Max rest (s):', dim=True).pack(side='left')
        self.max_rest_var = tk.DoubleVar(value=300.0)
        _spinbox(row, self.max_rest_var, from_=1.0, to=99999.0,
                 increment=10.0, format='%.0f', width=6).pack(
            side='left', padx=(2, _PAD_S))

        # Any edit (arrows or typing) applies live to the selected pulse
        for var in (self.start_var, self.length_var, self.max_rest_var):
            var.trace_add('write', lambda *_: self._on_change())

        _Btn(row, 'Apply All', style='primary',
             command=self._on_apply_all).pack(side='left', padx=2)

        _separator(row, horizontal=False).pack(
            side='left', fill='y', padx=_PAD_S, pady=3)
        _Btn(row, 'Export CSV', style='secondary',
             command=self._on_export).pack(side='left', padx=2)


# ============================================================================
# R/k Axis Bar (mass input + x-axis selection, header of the right panel)
# ============================================================================

class _RKAxisBar(tk.Frame):
    """Header strip above the R/k panel: sample mass and x-axis selection."""

    def __init__(self, master, on_change):
        super().__init__(master, bg=BG2)
        self._on_change = on_change
        self._build()

    def _build(self):
        """Create the mass entry and x-axis toggle buttons."""
        row = tk.Frame(self, bg=BG2)
        row.pack(fill='x', padx=2, pady=4)

        _label(row, 'Mass (mg):').pack(side='left', padx=(2, 0))
        self.mass_var = tk.StringVar(value='0')
        mass_entry = _spinbox(row, self.mass_var, from_=0.0, to=99999.0,
                              increment=0.1, format='%.2f', width=7,
                              command=self._on_change)
        mass_entry.pack(side='left', padx=(2, _PAD_S))
        mass_entry.bind('<Return>', lambda *_: self._on_change())
        mass_entry.bind('<FocusOut>', lambda *_: self._on_change())
        self.mass_var.trace_add('write', lambda *_: self._on_change())

        _separator(row, horizontal=False).pack(
            side='left', fill='y', padx=_PAD_S, pady=3)
        _label(row, 'R/k vs:').pack(side='left', padx=(0, 2))
        self.xaxis_var = tk.StringVar(value='voltage')
        self._btns = {}
        for value, text in (('voltage', 'Voltage'),
                            ('capacity', 'Capacity'),
                            ('specific_capacity', 'Specific cap.')):
            btn = _Btn(row, text, padx=6,
                       style='toggle_on' if value == 'voltage' else 'toggle_off',
                       command=lambda v=value: self._set_xaxis(v))
            btn.pack(side='left', padx=2)
            self._btns[value] = btn

    def _set_xaxis(self, value):
        """Switch the R/k x-axis toggle and refresh."""
        self.xaxis_var.set(value)
        on = dict(bg=ACCENT, fg=_C['button_text'])
        off = dict(bg=BG3, fg=DIM)
        for val, btn in self._btns.items():
            btn.config(**(on if val == value else off))
        self._on_change()


# ============================================================================
# Matplotlib Helpers
# ============================================================================

def _style_ax(ax, xlabel='', ylabel='', title=''):
    """Apply the light analysis-figure theme to an axes."""
    ax.set_facecolor(PLOT_BG)
    for spine in ax.spines.values():
        spine.set_color(PLOT_BORDER)
    ax.tick_params(colors=PLOT_DIM, labelsize=8)
    ax.xaxis.label.set_color(PLOT_DIM)
    ax.yaxis.label.set_color(PLOT_DIM)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, color=PLOT_TEXT, fontsize=9, pad=4)
    ax.grid(True, color=PLOT_GRID, alpha=0.75, linewidth=0.5)


def _stub(ax, msg='No data'):
    """Centred placeholder message on an empty axes."""
    ax.text(0.5, 0.5, msg, color=PLOT_DIM, fontsize=9,
            ha='center', va='center', transform=ax.transAxes, style='italic')


def embed_figure(fig, parent) -> FigureCanvasTkAgg:
    """Embed a matplotlib Figure in a Tk parent with a themed toolbar.

    Shared by the ICI and Capacity analysis windows.
    """
    canvas = FigureCanvasTkAgg(fig, master=parent)
    toolbar_frame = tk.Frame(parent, bg=BG2)
    toolbar_frame.pack(fill='x', side='bottom')
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.config(bg=BG2)
    for child in toolbar.winfo_children():
        try:
            child.config(bg=BG2, fg=TEXT)
        except tk.TclError:
            pass
    # Coordinate readout resizes the toolbar on hover; keep it quiet
    toolbar.set_message = lambda msg: None
    toolbar.update()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas._operaxn_parent = parent
    canvas._operaxn_toolbar = toolbar_frame
    canvas.draw()

    # Every resize on a scaled display can re-run matplotlib's device-pixel-
    # ratio race (see sync_figure_to_widget) — not just the opening layout
    # the windows' startup watchdogs cover. Reconcile after each resize
    # settles; the sync is a no-op when figure and widget already agree.
    def _sync_after_resize(_event=None):
        job = getattr(canvas, '_operaxn_sync_job', None)
        if job is not None:
            try:
                parent.after_cancel(job)
            except tk.TclError:
                pass

        def run():
            canvas._operaxn_sync_job = None
            sync_figure_to_widget(canvas, force=True)

        canvas._operaxn_sync_job = parent.after(180, run)

    parent.bind('<Configure>', _sync_after_resize, add='+')
    return canvas


def sync_figure_to_widget(canvas: FigureCanvasTkAgg, force: bool = False) -> None:
    """Resize a canvas's figure to its widget's current size.

    On scaled displays matplotlib's one-off DPI adjustment can land after
    the widget's last <Configure> event; the figure is then larger than the
    widget and renders cropped, with no further event to reconcile them.
    Replays the resize handler with the widget's actual size (which also
    recreates the Tk photo buffer). No-op when the two already agree —
    pass force=True after a window resize, where the sizes can compare
    equal while the photo buffer is still stale.
    """
    widget = canvas.get_tk_widget()
    parent = getattr(canvas, '_operaxn_parent', widget.master)
    toolbar = getattr(canvas, '_operaxn_toolbar', None)
    width = parent.winfo_width()
    height = parent.winfo_height()
    if toolbar is not None:
        height -= toolbar.winfo_height()
    # the canvas widget sits inside two 1px borders (frame highlight +
    # canvas border): its actual size is parent - 4 in each dimension
    width = max(1, width - 4)
    height = max(1, height - 4)
    if width <= 1 or height <= 1:
        return
    dpi = canvas.figure.dpi
    fig_w, fig_h = canvas.figure.get_size_inches()
    if (force or round(fig_w * dpi) != width
            or round(fig_h * dpi) != height):
        widget.configure(width=width, height=height)
        canvas.resize(SimpleNamespace(width=width, height=height))
        canvas.draw_idle()


# ============================================================================
# ICI Window
# ============================================================================

class ICIWindow(tk.Toplevel):
    """ICI analysis window; opens maximised.

        ICIWindow(parent, echem_df)                    # single dataset
        ICIWindow(parent, echem_sources={name: df})    # selectable datasets
    """

    def __init__(self, parent: tk.Widget,
                 echem_df: Optional[pd.DataFrame] = None,
                 echem_sources: Optional[dict] = None,
                 on_add_source=None) -> None:
        super().__init__(parent)
        self.title('OperaXN — ICI Analysis')
        self.configure(bg=BG)
        self.minsize(1000, 600)
        maximise_window(self)

        if echem_sources:
            self._sources = dict(echem_sources)
        elif echem_df is not None and not echem_df.empty:
            self._sources = {'Operando echem': echem_df}
        else:
            self._sources = {}

        self._df: Optional[pd.DataFrame] = None
        self._pulses: dict = {}
        self._ici_cache: dict = {}
        self._ax_pulse_twin = None
        self._updating = False
        self._source_change_job = None
        self._reg_change_job = None
        self._last_max_rest = 300.0
        self._on_add_source = on_add_source
        self._last_cycle_phase: tuple = (None, None)

        # Regression-window overrides, resolved pulse > phase > defaults.
        # Bar edits write the pulse level; Apply All writes the phase level.
        self._r1_pulse_overrides: dict = {}   # (cycle, phase, pulse_idx) -> (start_s, length_s)
        self._r1_phase_overrides: dict = {}   # phase                     -> (start_s, length_s)

        self._build_ui()
        self.after_idle(self._focus_primary_panel)

        if self._sources:
            self._load(next(iter(self._sources.values())))

        # Watch until the initial layout and DPI adjustment have both settled;
        # their event order is not deterministic, so one-shot timers can miss.
        self._sync_attempts = 0
        self.after(150, self._sync_watchdog)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the window layout: figures, selector strip, regression bar."""
        # Regression bar — packed from bottom first
        _separator(self).pack(fill='x', side='bottom')
        self.reg_bar = _RegressionBar(self, on_change=self._on_regression_change,
                                      on_apply_all=self._on_apply_all,
                                      on_export=self._export_csv)
        self.reg_bar.pack(fill='x', side='bottom')

        # Content area: left 2/3 | right 1/3. The right panel's width is
        # pinned and propagation disabled: otherwise the figure canvas's
        # requested size feeds back into the layout when matplotlib rescales
        # for display DPI, and the panel grows until it swallows the window.
        content = tk.Frame(self, bg=BG)
        content.pack(fill='both', expand=True)
        content.pack_propagate(False)

        left = tk.Frame(content, bg=BG)
        right = tk.Frame(content, bg=BG)
        right.pack_propagate(False)
        right.pack(side='right', fill='y')
        _separator(content, horizontal=False).pack(side='right', fill='y', padx=2)
        left.pack(side='left', fill='both', expand=True)
        left.grid_propagate(False)

        # R/k axis controls head the right panel they act on
        rk_wrapper = tk.Frame(right, bg=BG2,
                              highlightbackground=BORDER,
                              highlightthickness=1)
        rk_wrapper.pack(fill='x', padx=_PAD_S, pady=(_PAD_S, 0))
        self.rk_axis_bar = _RKAxisBar(rk_wrapper, on_change=self._on_rk_axis_changed)
        self.rk_axis_bar.pack(fill='x')

        # Left panel: top figure / selector strip / bottom figure
        left.grid_rowconfigure(0, weight=1)   # top figure
        left.grid_rowconfigure(1, weight=0)   # selector strip
        left.grid_rowconfigure(2, weight=1)   # bottom figure
        left.grid_columnconfigure(0, weight=1)

        # Top figure (Overview + Pulse zoom)
        top_frame = _canvas_frame(left)
        self._primary_canvas_frame = top_frame
        top_frame.grid(row=0, column=0, sticky='nsew', padx=_PAD_S, pady=(_PAD_S, 2))
        top_frame.pack_propagate(False)
        self._fig_top = Figure(facecolor=PLOT_BG, dpi=FIGURE_DPI)
        self._build_top_axes()
        self._canvas_top = embed_figure(self._fig_top, top_frame)
        self._canvas_top.get_tk_widget().bind(
            '<Button-1>', lambda _event: top_frame.focus_set(), add='+')

        # Selector strip
        strip_wrapper = tk.Frame(left, bg=BG2,
                                 highlightbackground=BORDER,
                                 highlightthickness=1)
        strip_wrapper.grid(row=1, column=0, sticky='ew', padx=_PAD_S, pady=2)
        self.selector = _SelectorStrip(strip_wrapper,
                                       on_change=self._on_selection_change,
                                       source_names=list(self._sources),
                                       on_source_change=self._on_source_change,
                                       on_add_source=self._on_add_source)
        self.selector.pack(fill='x')

        # Bottom figure (ICI fit + R²)
        bottom_frame = _canvas_frame(left)
        bottom_frame.grid(row=2, column=0, sticky='nsew', padx=_PAD_S, pady=(2, _PAD_S))
        bottom_frame.pack_propagate(False)
        self._fig_bot = Figure(facecolor=PLOT_BG, dpi=FIGURE_DPI)
        self._build_bot_axes()
        self._canvas_bot = embed_figure(self._fig_bot, bottom_frame)

        # Right panel: R / k figures (4 stacked)
        right_frame = _canvas_frame(right)
        right_frame.pack(fill='both', expand=True, padx=_PAD_S, pady=(2, _PAD_S))
        right_frame.pack_propagate(False)
        self._fig_right = Figure(figsize=(5.0, 10), facecolor=PLOT_BG, dpi=FIGURE_DPI)
        self._build_right_axes()
        self._canvas_right = embed_figure(self._fig_right, right_frame)

        # A third of the screen, or enough for the R/k axis controls
        # (idle tasks first so the axis bar's requested width is computed)
        self.update_idletasks()
        right.configure(width=max(int(self.winfo_screenwidth() * 0.30),
                                  self.rk_axis_bar.winfo_reqwidth() + 4 * _PAD_S))

    def _focus_primary_panel(self) -> None:
        """Start with the overview/pulse panel as the active plot region."""
        frame = getattr(self, '_primary_canvas_frame', None)
        if frame is not None and frame.winfo_exists():
            frame.focus_set()

    def _build_top_axes(self):
        """Create the overview and pulse-zoom axes."""
        gs = GridSpec(1, 2, figure=self._fig_top,
                      wspace=0.35,
                      left=0.10, right=0.90,
                      top=0.88, bottom=0.18)
        self.ax_overview = self._fig_top.add_subplot(gs[0, 0])
        self.ax_pulse = self._fig_top.add_subplot(gs[0, 1])
        _style_ax(self.ax_overview, 't (h)', 'Voltage (V)', 'Overview')
        _style_ax(self.ax_pulse, 't (h)', 'Voltage (V)', 'Pulse zoom')

    def _build_bot_axes(self):
        """Create the ICI-fit and R² axes."""
        gs = GridSpec(1, 2, figure=self._fig_bot,
                      wspace=0.35,
                      left=0.10, right=0.90,
                      top=0.80, bottom=0.18)
        self.ax_dv = self._fig_bot.add_subplot(gs[0, 0])
        self.ax_r2 = self._fig_bot.add_subplot(gs[0, 1])
        _style_ax(self.ax_dv, '√Δt (s¹ᐟ²)', 'ΔV (V)', 'ICI fit')
        _style_ax(self.ax_r2, 'Pulse #', 'R²', 'R² vs Pulse')
        _stub(self.ax_dv)
        _stub(self.ax_r2)

    def _build_right_axes(self):
        """Create the four stacked R/k axes."""
        gs = GridSpec(4, 1, figure=self._fig_right,
                      hspace=0.40,
                      left=0.14, right=0.97,
                      top=0.95, bottom=0.04)
        self.ax_r_chg = self._fig_right.add_subplot(gs[0])
        self.ax_r_dis = self._fig_right.add_subplot(gs[1])
        self.ax_k_chg = self._fig_right.add_subplot(gs[2])
        self.ax_k_dis = self._fig_right.add_subplot(gs[3])
        _style_ax(self.ax_r_chg, '', 'R (Ω)', 'R – Charge')
        _style_ax(self.ax_r_dis, '', 'R (Ω)', 'R – Discharge')
        _style_ax(self.ax_k_chg, '', 'k', 'k – Charge')
        _style_ax(self.ax_k_dis, 'V (V)', 'k', 'k – Discharge')
        for ax in (self.ax_r_chg, self.ax_r_dis, self.ax_k_chg, self.ax_k_dis):
            _stub(ax, 'No data')

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load(self, echem_df: pd.DataFrame):
        """Load echem data, reset caches, and redraw everything."""
        self._updating = True
        try:
            self._df = _prepare_df(echem_df)
            self._pulses = {}
            self._ici_cache = {}
            self.selector.set_cycle_range(max(1, self._max_cycle()))
            self.selector.pulse_var.set(1)
        finally:
            self._updating = False
        self._update_all()

    def _on_source_change(self, name):
        """Queue a source switch after the dropdown has closed."""
        if name not in self._sources:
            return
        if self._source_change_job is not None:
            self.after_cancel(self._source_change_job)
        self._source_change_job = self.after_idle(self._apply_source_change, name)

    def _apply_source_change(self, name):
        """Switch datasets once Tk is idle, avoiding menu callback re-entry."""
        self._source_change_job = None
        df = self._sources.get(name)
        if df is None:
            return
        self.configure(cursor='watch')
        self.update_idletasks()
        try:
            # Overrides are indexed by pulse/cycle of the previous dataset.
            self._r1_pulse_overrides.clear()
            self._r1_phase_overrides.clear()
            self._last_cycle_phase = (None, None)
            self._load(df)
        finally:
            self.configure(cursor='')

    def refresh_sources(self, sources) -> None:
        """Expose newly appended echem files without rebuilding the window."""
        self._sources = dict(sources)
        self.selector.set_sources(self._sources)

    def _max_cycle(self) -> int:
        """Highest detected cycle number in the loaded data (0 if none)."""
        if self._df is None:
            return 0
        cycles = self._df[self._df['cycle'] > 0]['cycle']
        return int(cycles.max()) if not cycles.empty else 0

    def _get_pulses(self, cycle: int, phase: str) -> list[dict]:
        """Detected pulses for (cycle, phase), cached until the bar changes."""
        key = (cycle, phase)
        if key not in self._pulses:
            if self._df is None:
                return []
            try:
                self._last_max_rest = float(self.reg_bar.max_rest_var.get())
            except (ValueError, tk.TclError):
                pass  # entry is mid-edit; keep the last valid threshold
            mask = (self._df['cycle'] == cycle) & (self._df['phase'] == phase)
            self._pulses[key] = _detect_pulses(
                self._df[mask], max_rest=self._last_max_rest)
        return self._pulses[key]

    # ------------------------------------------------------------------
    # Plots: Overview and Pulse Zoom
    # ------------------------------------------------------------------

    def _plot_overview(self, cycle, pulse_idx, phase):
        """Voltage vs time for the selected cycle with pulse highlight."""
        ax = self.ax_overview
        ax.cla()
        _style_ax(ax, 't (h)', 'Voltage (V)', 'Overview')
        if self._df is None:
            _stub(ax, 'No data loaded')
            return

        df = self._df
        t_h = df['t_s'].values / 3600.0
        voltage = df['echem_data'].values

        if self.selector.show_all_var.get():
            for c in sorted(df[df['cycle'] > 0]['cycle'].unique()):
                mask = df['cycle'] == c
                ax.plot(t_h[mask], voltage[mask], color=PLOT_BORDER, lw=0.8, zorder=1)

        cycle_mask = df['cycle'] == cycle
        for ph, colour in (('charge', CHARGE_COLOR), ('discharge', DISCHARGE_COLOR)):
            mask = cycle_mask & (df['phase'] == ph)
            if mask.any():
                ax.plot(t_h[mask], voltage[mask], color=colour, lw=1.8, zorder=2,
                        label=ph.capitalize())

        pulses = self._get_pulses(cycle, phase)
        if 1 <= pulse_idx <= len(pulses):
            pulse = pulses[pulse_idx - 1]
            t0 = pulse['pulse_start_t'] / 3600.0
            t1 = pulse['relax_end_t'] / 3600.0
            ax.axvspan(t0, t1, alpha=0.25, color=HIGHLIGHT_COLOR, zorder=3)
            ax.axvline(t0, color=HIGHLIGHT_COLOR, lw=1.0, ls='--', zorder=4)

        ax.legend(fontsize=7, loc='best',
                  labelcolor=PLOT_TEXT, facecolor=PLOT_BG, edgecolor=PLOT_BORDER)

    def _plot_pulse_zoom(self, cycle, pulse_idx, phase):
        """Voltage and current for the selected pulse and its relaxation."""
        if self._ax_pulse_twin is not None:
            self._fig_top.delaxes(self._ax_pulse_twin)
            self._ax_pulse_twin = None

        ax = self.ax_pulse
        ax.cla()
        _style_ax(ax, 't (h)', 'Voltage (V)', 'Pulse zoom')

        if self._df is None:
            _stub(ax, 'No data loaded')
            return

        pulses = self._get_pulses(cycle, phase)
        if not pulses:
            _stub(ax, f'No pulses found\n(cycle {cycle}, {phase})')
            return

        pulse_idx = max(1, min(pulse_idx, len(pulses)))
        pulse = pulses[pulse_idx - 1]
        mask = (self._df['cycle'] == cycle) & (self._df['phase'] == phase)
        seg = self._df[mask]

        combined = pd.concat([seg[pulse['pulse_mask']], seg[pulse['relax_mask']]])
        if combined.empty:
            _stub(ax, 'Empty pulse segment')
            return

        t_h = combined['t_s'].values / 3600.0
        voltage = combined['echem_data'].values
        current = combined['current'].fillna(0).values
        colour = CHARGE_COLOR if phase == 'charge' else DISCHARGE_COLOR

        # V0 = voltage at end of active period (pyICI convention)
        pulse_seg = seg[pulse['pulse_mask']]
        relax_seg = seg[pulse['relax_mask']]
        t0_h = pulse['pulse_end_t'] / 3600.0

        ax.plot(t_h, voltage, color=colour, lw=1.8, label='Voltage', zorder=3)

        # Shade rest period + V0 marker (matches pyICI)
        if not relax_seg.empty:
            ax.axvspan(t0_h, pulse['relax_end_t'] / 3600.0,
                       color=colour, alpha=0.12, zorder=1)
        if not pulse_seg.empty:
            V0 = pulse_seg['echem_data'].iloc[-1]
            ax.plot(t0_h, V0, 'o', color=_C['success'], markersize=7,
                    label=f'V₀ = {V0:.3f} V', zorder=5)

        self._ax_pulse_twin = ax.twinx()
        ax2 = self._ax_pulse_twin
        ax2.plot(t_h, current, color=CURRENT_COLOR, lw=1.2, ls='--',
                 label='Current', alpha=0.8, zorder=2)
        ax2.set_ylabel('Current (mA)', color=PLOT_DIM, fontsize=9)
        ax2.tick_params(colors=PLOT_DIM, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color(PLOT_BORDER)
        ax2.set_facecolor('none')

        # Only include lines with real labels (avoids _child1 artifact)
        lines_v = [ln for ln in ax.get_lines() if not ln.get_label().startswith('_')]
        lines_i = [ln for ln in ax2.get_lines() if not ln.get_label().startswith('_')]
        legend_position = (
            dict(loc='lower center', bbox_to_anchor=(0.5, 0.02))
            if phase == 'charge'
            else dict(loc='upper center', bbox_to_anchor=(0.5, 0.98))
        )
        ax.legend(lines_v + lines_i, [ln.get_label() for ln in lines_v + lines_i],
                  fontsize=7, labelcolor=PLOT_TEXT, facecolor=PLOT_BG,
                  edgecolor=PLOT_BORDER,
                  **legend_position)
        ax.set_title(
            f'Cycle {cycle}  |  {phase.capitalize()}  |  '
            f'Pulse {pulse_idx}/{len(pulses)}',
            color=PLOT_TEXT, fontsize=9, pad=4,
        )

    # ------------------------------------------------------------------
    # ICI Analysis
    # ------------------------------------------------------------------

    def _get_r1(self, cycle: int = None, phase: str = None, pulse_idx: int = None):
        """Resolve the regression window (start_s, length_s) for a pulse,
        falling back pulse override -> phase override -> the bar defaults."""
        if cycle is not None and phase is not None and pulse_idx is not None:
            override = self._r1_pulse_overrides.get((cycle, phase, pulse_idx))
            if override is not None:
                return override
        return self._r1_phase_overrides.get(phase, (0.5, 1.0))

    def _compute_ici(self, cycle: int, phase: str) -> list[dict]:
        """Fit results for (cycle, phase), cached until Apply/reload."""
        key = (cycle, phase)
        if key not in self._ici_cache:
            pulses = self._get_pulses(cycle, phase)
            if not pulses or self._df is None:
                self._ici_cache[key] = []
            else:
                mask = (self._df['cycle'] == cycle) & (self._df['phase'] == phase)
                self._ici_cache[key] = _compute_all_pulses(
                    self._df[mask], pulses,
                    lambda idx: self._get_r1(cycle, phase, idx))
        return self._ici_cache[key]

    # ------------------------------------------------------------------
    # Plots: ICI Fit and R²
    # ------------------------------------------------------------------

    def _plot_ici_fit(self, cycle: int, pulse_idx: int, phase: str):
        """Plot ΔV against square-root elapsed time with its regression."""
        ax = self.ax_dv
        ax.cla()
        _style_ax(ax, '√Δt (√s)', 'ΔV (V)', 'ICI fit')

        results = self._compute_ici(cycle, phase)
        fit = next((r for r in results if r['pulse_idx'] == pulse_idx), None)

        if fit is None:
            _stub(ax, 'Too few rest points in the regression window\n'
                      '(increase Length and press Apply)')
            return

        colour = CHARGE_COLOR if phase == 'charge' else DISCHARGE_COLOR
        ax.scatter(fit['sqrt_dt'], fit['delta_V'],
                   color=colour, s=12, alpha=0.7, zorder=3, label='ΔV')

        ax.plot(fit['X_fit'], fit['y_fit'],
                color=CHARGE_COLOR, lw=2, zorder=4,
                label=f'fit  R²={fit["r2"]:.3f}')

        ax.legend(fontsize=7, loc='best',
                  labelcolor=PLOT_TEXT, facecolor=PLOT_BG, edgecolor=PLOT_BORDER)

        ax.set_title(
            f'ICI fit  |  C{cycle}  {phase[:3]}  |  Pulse {pulse_idx}  '
            f'|  R={fit["R"]:.4f} Ω  k={fit["k"]:.4f} Ω·s⁻¹ᐟ²',
            color=PLOT_TEXT, fontsize=9, pad=4)

    def _plot_r2_panel(self, cycle: int, pulse_idx: int, phase: str, scope: str):
        """R² of every fit in the selected scope, star on the current pulse."""
        ax = self.ax_r2
        ax.cla()
        _style_ax(ax, 'Pulse #', 'R²', 'R² vs Pulse')

        if self._df is None:
            _stub(ax, 'No data')
            return

        datasets = []
        if scope == 'phase':
            colour = CHARGE_COLOR if phase == 'charge' else DISCHARGE_COLOR
            datasets.append((self._compute_ici(cycle, phase), phase.capitalize(), colour, 0.85))
        elif scope == 'cycle':
            datasets.append((self._compute_ici(cycle, 'charge'), 'Charge', CHARGE_COLOR, 0.85))
            datasets.append((self._compute_ici(cycle, 'discharge'), 'Discharge', DISCHARGE_COLOR, 0.85))
        else:
            for c in range(1, self._max_cycle() + 1):
                alpha = 1.0 if c == cycle else 0.35   # same shading rule as the R/k panels
                for ph, colour in (('charge', CHARGE_COLOR), ('discharge', DISCHARGE_COLOR)):
                    datasets.append((self._compute_ici(c, ph), f'C{c} {ph[:3]}', colour, alpha))

        for results, label, colour, alpha in datasets:
            if not results:
                continue
            xs = [r['pulse_idx'] for r in results]
            ys = [r['r2'] for r in results]
            ax.scatter(xs, ys, color=colour, s=18, alpha=alpha, label=label, zorder=3)

        # Star on selected pulse
        current_fits = self._compute_ici(cycle, phase)
        selected = next((r for r in current_fits if r['pulse_idx'] == pulse_idx), None)
        if selected and not np.isnan(selected['r2']):
            ax.scatter([pulse_idx], [selected['r2']],
                       color=HIGHLIGHT_COLOR, s=70, marker='*', zorder=5)

        if any(results for results, _, _, _ in datasets):
            ax.legend(fontsize=7, loc='best',
                      labelcolor=PLOT_TEXT, facecolor=PLOT_BG,
                      edgecolor=PLOT_BORDER)

    # ------------------------------------------------------------------
    # Plots: R and k vs Voltage / Capacity / Specific Capacity
    # ------------------------------------------------------------------

    _XAXIS_LABELS = {
        'voltage': 'V (V)',
        'capacity': 'Capacity (mAh)',
        'specific_capacity': 'Specific Capacity (mAh/g)',
    }

    def _get_mass_mg(self) -> float:
        """Sample mass from the R/k axis bar, 0.0 when unset or invalid."""
        try:
            return float(self.rk_axis_bar.mass_var.get())
        except (ValueError, tk.TclError):
            return 0.0

    def _update_right_plots(self, cycle: int, phase: str, scope: str):
        """Redraw the four stacked R/k panels for the selected scope."""
        xaxis = self.rk_axis_bar.xaxis_var.get()
        mass_mg = self._get_mass_mg()
        xlabel = self._XAXIS_LABELS[xaxis]

        panels = (
            (self.ax_r_chg, 'R (Ω)', 'R – Charge'),
            (self.ax_r_dis, 'R (Ω)', 'R – Discharge'),
            (self.ax_k_chg, 'k (Ω·s⁻¹ᐟ²)', 'k – Charge'),
            (self.ax_k_dis, 'k (Ω·s⁻¹ᐟ²)', 'k – Discharge'),
        )
        for index, (ax, ylabel, title) in enumerate(panels):
            ax.cla()
            _style_ax(ax, xlabel if index == len(panels) - 1 else '', ylabel, title)

        if self._df is None:
            for ax, _, _ in panels:
                _stub(ax)
            self._canvas_right.draw_idle()
            return

        if xaxis == 'specific_capacity' and mass_mg <= 0:
            for ax, _, _ in panels:
                _stub(ax, 'Enter mass (mg) > 0')
            self._canvas_right.draw_idle()
            return

        cycles = ([cycle] if scope in ('phase', 'cycle')
                  else list(range(1, self._max_cycle() + 1)))

        for c in cycles:
            alpha = 1.0 if c == cycle else 0.5
            cycle_suffix = f' C{c}' if len(cycles) > 1 else ''
            for ph, ax_r, ax_k, colour in (
                ('charge', self.ax_r_chg, self.ax_k_chg, CHARGE_COLOR),
                ('discharge', self.ax_r_dis, self.ax_k_dis, DISCHARGE_COLOR),
            ):
                results = self._compute_ici(c, ph)
                if not results:
                    continue
                if xaxis == 'voltage':
                    xs = np.array([r['V0'] for r in results])
                elif xaxis == 'capacity':
                    xs = np.array([r['capacity'] for r in results])
                else:
                    xs = np.array([r['capacity'] for r in results]) / (mass_mg / 1000.0)
                R_values = np.array([r['R'] for r in results])
                k_values = np.array([r['k'] for r in results])
                R_errors = np.array([r['R_err'] for r in results])
                k_errors = np.array([r['k_err'] for r in results])
                label = f'{ph.capitalize()}{cycle_suffix}'
                valid_R = ~np.isnan(xs) & ~np.isnan(R_values)
                valid_k = ~np.isnan(xs) & ~np.isnan(k_values)
                if valid_R.any():
                    ax_r.errorbar(xs[valid_R], R_values[valid_R], yerr=R_errors[valid_R],
                                  fmt='o', color=colour, alpha=alpha,
                                  ms=5, lw=1, capsize=3, label=label)
                if valid_k.any():
                    ax_k.errorbar(xs[valid_k], k_values[valid_k], yerr=k_errors[valid_k],
                                  fmt='o', color=colour, alpha=alpha,
                                  ms=5, lw=1, capsize=3, label=label)

        for ax, _, _ in panels:
            if ax.get_lines() or ax.collections:
                ax.legend(fontsize=7, loc='best',
                          labelcolor=PLOT_TEXT, facecolor=PLOT_BG,
                          edgecolor=PLOT_BORDER)
            else:
                _stub(ax, 'No data')

        self._canvas_right.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_selection_change(self):
        """Handle cycle/pulse/phase/scope changes from the selector strip."""
        if getattr(self, '_updating', False):
            return
        self._updating = True
        try:
            cycle = self.selector.cycle_var.get()
            phase = self.selector.phase_var.get()
            pulses = self._get_pulses(cycle, phase)
            self.selector.set_pulse_range(len(pulses))
            self._last_cycle_phase = (cycle, phase)
            self._update_live_plots()
        finally:
            self._updating = False

    def _export_csv(self):
        """Write ici_charge.csv and ici_discharge.csv covering every cycle."""
        if self._df is None:
            return
        directory = filedialog.askdirectory(title='Select export folder', parent=self)
        if not directory:
            return

        max_cycle = self._max_cycle()
        mass_mg = self._get_mass_mg()
        columns = ['cycle', 'voltage (V)', 'capacity (mAh)', 'specific_capacity (mAh/g)',
                   'R (Ohm)', 'R_error (Ohm)',
                   'k (Ohm.s^-1/2)', 'k_error (Ohm.s^-1/2)', 'R2']

        for phase in ('charge', 'discharge'):
            rows = []
            for cycle in range(1, max_cycle + 1):
                for r in self._compute_ici(cycle, phase):
                    rows.append({
                        'cycle': cycle,
                        'voltage (V)': r['V0'],
                        'capacity (mAh)': r['capacity'],
                        'specific_capacity (mAh/g)': (r['capacity'] / (mass_mg / 1000.0)
                                                      if mass_mg > 0 else np.nan),
                        'R (Ohm)': r['R'],
                        'R_error (Ohm)': r['R_err'],
                        'k (Ohm.s^-1/2)': r['k'],
                        'k_error (Ohm.s^-1/2)': r['k_err'],
                        'R2': r['r2'],
                    })
            out = pd.DataFrame(rows, columns=columns)
            out.to_csv(os.path.join(directory, f'ici_{phase}.csv'), index=False)

        messagebox.showinfo(
            'Export complete',
            f'Saved ici_charge.csv and ici_discharge.csv to:\n{directory}',
            parent=self
        )

    def _on_rk_axis_changed(self, *_):
        """Redraw the right panel when the mass or x-axis selection changes."""
        if self._df is None:
            return
        cycle = self.selector.cycle_var.get()
        phase = self.selector.phase_var.get()
        scope = self.selector.scope_var.get()
        self._update_right_plots(cycle, phase, scope)

    def _reg_values(self):
        """Bar values as (start_s, length_s), or None while an edit is invalid."""
        try:
            return (float(self.reg_bar.start_var.get()),
                    float(self.reg_bar.length_var.get()))
        except (ValueError, tk.TclError):
            return None

    def _on_regression_change(self, *_):
        """Debounce bar edits; each settles onto the selected pulse."""
        if self._reg_change_job is not None:
            self.after_cancel(self._reg_change_job)
        self._reg_change_job = self.after(250, self._apply_regression_change)

    def _apply_regression_change(self):
        """Apply the bar's values to the selected pulse and refit."""
        self._reg_change_job = None
        if self._df is None:
            return
        value = self._reg_values()
        if value is None:
            return
        try:
            cycle = self.selector.cycle_var.get()
            phase = self.selector.phase_var.get()
            pulse = self.selector.pulse_var.get()
        except tk.TclError:
            return
        self._r1_pulse_overrides[(cycle, phase, pulse)] = value
        self._pulses.clear()
        self._ici_cache.clear()
        self._update_live_plots()

    def _on_apply_all(self):
        """Apply the bar's values to every pulse, cycle, and phase."""
        if self._df is None:
            return
        value = self._reg_values()
        if value is None:
            return
        self._r1_pulse_overrides.clear()
        self._r1_phase_overrides['charge'] = value
        self._r1_phase_overrides['discharge'] = value
        self._pulses.clear()
        self._ici_cache.clear()
        self._update_live_plots()

    def _update_live_plots(self):
        """Redraw all panels for the current selection."""
        if self._df is None:
            return
        cycle = self.selector.cycle_var.get()
        pulse = self.selector.pulse_var.get()
        phase = self.selector.phase_var.get()
        scope = self.selector.scope_var.get()
        self._plot_overview(cycle, pulse, phase)
        self._plot_pulse_zoom(cycle, pulse, phase)
        self._fig_top.canvas.draw_idle()
        self._plot_ici_fit(cycle, pulse, phase)
        self._plot_r2_panel(cycle, pulse, phase, scope)
        self._canvas_bot.draw_idle()
        self._update_right_plots(cycle, phase, scope)

    def _update_all(self):
        """Full refresh after a data load."""
        self._update_live_plots()

    def _sync_figures(self):
        """Re-fit all three figures to their widgets (see sync_figure_to_widget)."""
        if not self.winfo_exists():
            return
        for canvas in (self._canvas_top, self._canvas_bot, self._canvas_right):
            sync_figure_to_widget(canvas)

    def _sync_watchdog(self):
        """Re-fit the figures repeatedly while the window is settling."""
        if not self.winfo_exists():
            return
        self._sync_figures()
        self._sync_attempts += 1
        if self._sync_attempts < 12:
            self.after(200, self._sync_watchdog)
