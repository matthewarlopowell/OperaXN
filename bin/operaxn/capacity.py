"""
Capacity analysis over operando echem data: charge/discharge/rest phase
classification, cycle detection, and cumulative capacity integration.

Operates on the echem DataFrame provided by the input adapter
(columns: timestamp, echem_data, current); shared by the analysis
windows.

Ported from the OperaXN-ICI fork (Joe Arroyo).
"""

import numpy as np
import pandas as pd
from matplotlib import cm


def _assign_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add t_s (elapsed seconds) column if not already present."""
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
    """
    Assign 'charge', 'discharge', or 'rest' to each sample based on
    current sign. Rest points inherit the preceding phase label.
    """
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


def classify_phases(echem_df: pd.DataFrame):
    """
    Split echem_df into charge and discharge DataFrames.
    Uses current sign: positive = charge, negative = discharge.
    Returns (charge_df, discharge_df), each with a 't_s' column
    (elapsed seconds from start of the full dataset).
    """
    df = _assign_time(echem_df.copy())
    df["phase"] = _label_phases(df["current"].fillna(0).values)
    return df[df["phase"] == "charge"].copy(), df[df["phase"] == "discharge"].copy()


def assign_cycles(echem_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'phase' and 'cycle' columns (cycle 1-indexed).
    A cycle starts with charge and ends after the following discharge.
    """
    df = _assign_time(echem_df.copy())
    labels = _label_phases(df["current"].fillna(0).values)
    df["phase"] = labels

    cycles = np.zeros(len(df), dtype=int)
    cycle_num = 0
    prev = None

    for i in range(len(labels)):
        if labels[i] == "charge" and prev != "charge":
            cycle_num += 1          # new charge block = new cycle
        if labels[i] in ("charge", "discharge"):
            cycles[i] = cycle_num
        # rest before any cycle starts stays 0
        prev = labels[i]

    df["cycle"] = cycles
    return df


def compute_capacity(df: pd.DataFrame) -> np.ndarray:
    """
    Cumulative capacity (mAh) via trapezoidal integration of |I| over time.
    Resets to 0 at the start of the passed segment.
    """
    if df.empty:
        return np.array([])

    t = df["t_s"].values
    i_mA = np.abs(df["current"].fillna(0).values)

    dt = np.diff(t, prepend=t[0])
    i_mid = np.concatenate(([i_mA[0]], (i_mA[:-1] + i_mA[1:]) / 2))
    capacity = np.cumsum(i_mid * dt) / 3600.0  # mAh
    return capacity


def plot_capacity_vs_voltage(ax, echem_df, mass_mg=0.0, cycles_to_plot=None):
    """Plot capacity (or specific capacity when mass_mg > 0) against voltage,
    one colour per cycle; charge and discharge legs share the cycle colour."""
    ax.clear()
    use_specific = mass_mg > 0
    ax.set_xlabel("Specific Capacity (mAh/g)" if use_specific else "Capacity (mAh)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    if echem_df is None or echem_df.empty:
        ax.text(0.5, 0.5, "No echem data loaded", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=11)
        return
    df = assign_cycles(echem_df)
    all_cycles = sorted([c for c in df["cycle"].unique() if c > 0])
    cycles = cycles_to_plot if cycles_to_plot is not None else all_cycles
    cmap = cm.tab10 if len(all_cycles) <= 10 else cm.viridis
    colors = {c: cmap(i / max(len(all_cycles) - 1, 1)) for i, c in enumerate(all_cycles)}
    for cycle_num in cycles:
        cycle_df = df[df["cycle"] == cycle_num]
        color = colors.get(cycle_num, "grey")
        charge_df = cycle_df[cycle_df["phase"] == "charge"]
        if not charge_df.empty:
            cap = compute_capacity(charge_df)
            if use_specific:
                cap = cap / (mass_mg / 1000.0)
            ax.plot(cap, charge_df["echem_data"].values, color=color,
                    linewidth=1.5, label=f"Cycle {cycle_num}")
        discharge_df = cycle_df[cycle_df["phase"] == "discharge"]
        if not discharge_df.empty:
            cap = compute_capacity(discharge_df)
            if use_specific:
                cap = cap / (mass_mg / 1000.0)
            ax.plot(cap, discharge_df["echem_data"].values, color=color, linewidth=1.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")


def plot_time_vs_voltage(ax, echem_df, cycles_to_plot=None):
    """Plot voltage against elapsed time (hours), one colour per cycle."""
    ax.clear()
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    if echem_df is None or echem_df.empty:
        ax.text(0.5, 0.5, "No echem data loaded", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=11)
        return
    df = assign_cycles(echem_df)
    all_cycles = sorted([c for c in df["cycle"].unique() if c > 0])
    cycles = cycles_to_plot if cycles_to_plot is not None else all_cycles
    cmap = cm.tab10 if len(all_cycles) <= 10 else cm.viridis
    colors = {c: cmap(i / max(len(all_cycles) - 1, 1)) for i, c in enumerate(all_cycles)}
    for cycle_num in cycles:
        cycle_df = df[df["cycle"] == cycle_num]
        ax.plot(cycle_df["t_s"].values / 3600.0, cycle_df["echem_data"].values,
                color=colors.get(cycle_num, "grey"), linewidth=1.5, label=f"Cycle {cycle_num}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")
