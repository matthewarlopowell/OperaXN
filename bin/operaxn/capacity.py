"""
Echem capacity analysis: charge/discharge phases, cycles, cumulative capacity and voltage plots.

Operates on the input adapter's echem DataFrame (timestamp, echem_data, current) and is
shared by the capacity and ICI windows. Ported from the OperaXN-ICI fork (Joe Arroyo).
"""

import numpy as np
import pandas as pd
from matplotlib import cm

from .config import SECONDS_PER_HOUR, TIME_HOURS_LABEL


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
