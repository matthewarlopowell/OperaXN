"""Echem analysis windows: Capacity, ICI and heatmap maths, their windows and main-window wiring."""
import builtins
import gc
import glob
import inspect
import os
import re
import warnings
import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text

pytestmark = pytest.mark.gui

tk = pytest.importorskip("tkinter")
from tkinter import filedialog, messagebox  # noqa: E402

import operaxn
import operaxn.capacity as capacity_module
import operaxn.config as cfg
import operaxn.dialog as dialog_module
import operaxn.gui as gui_module
import operaxn.heatmap as heatmap_module
import operaxn.ici as ici_module
from operaxn import core
from operaxn.capacity import (
    _assign_time, assign_cycles, compute_capacity,
    plot_capacity_vs_voltage, plot_time_vs_voltage, CapacityWindow,
)
from operaxn.dialog import parse_scan_range
from operaxn.gui import OPERAXN
from operaxn.heatmap import (
    NEUTRON_AXES, XRD_AXES,
    HeatmapWindow, build_xrd_matrix, cell_edges, colormap_name, correlated_rows,
    echem_scan_curve, exclude_rows, intensity_limits, neutron_axes,
    neutron_bank_arrays, neutron_banks,
    publication_figure, row_edges, scan_positions_seconds, title_text,
)
from operaxn.ici import (
    ICIWindow, PLOT_BG, _RKAxisBar, _SelectorStrip, _compute_all_pulses, _compute_fit,
    _detect_pulses, _style_ax, embed_figure, watch_figure_sync,
)
from operaxn.input import get_standard_echem

from _helpers import background_errors, pending_after_ids, pump

R_TRUE, K_TRUE, I_MA = 50.0, 5.0, 2.0
I_A = I_MA / 1000.0
V0 = 3.8


@pytest.fixture(autouse=True)
def _clean_plot_cache():
    """Clear the id(ax)-keyed plot cache around each test, as test axes are freed constantly."""
    operaxn.clear_plot_cache()
    yield
    operaxn.clear_plot_cache()


def build_cycling_df(n_cycles=2, active_s=10, rest_s=5):
    """Build constant-current cycles: +1 mA charge, rest, -1 mA discharge, rest."""
    current = []
    for _ in range(n_cycles):
        current += [1.0] * active_s + [0.0] * rest_s
        current += [-1.0] * active_s + [0.0] * rest_s
    n = len(current)
    t0 = pd.Timestamp("2026-06-01 10:00:00")
    return pd.DataFrame({
        "timestamp": [t0 + pd.Timedelta(seconds=s) for s in range(n)],
        "echem_data": np.linspace(3.0, 4.0, n),
        "current": current,
    })


@pytest.fixture
def fit_df():
    """Return an exact dV = -I(R + k*sqrt(dt)) relaxation: a 5-point pulse then 40 rest points."""
    active_t = np.arange(5, dtype=float)
    rest_t = 4.0 + 0.25 * np.arange(1, 41)
    rest_v = V0 - I_A * (R_TRUE + K_TRUE * np.sqrt(rest_t - rest_t[0]))
    return pd.DataFrame({
        "t_s": np.concatenate([active_t, rest_t]),
        "current": np.concatenate([np.full(5, I_MA), np.zeros(40)]),
        "echem_data": np.concatenate([np.full(5, V0), rest_v]),
    })


def build_oned_arrays():
    """Build three flat 1D scans on a 50-point grid; scan 3 uses a coarser grid."""
    hm_x = np.linspace(10, 80, 50)
    return {
        1: {"x": hm_x, "y": np.full(50, 1.0), "timestamp": "2026-06-01 10:00:00", "echem": 3.2},
        2: {"x": hm_x, "y": np.full(50, 2.0), "timestamp": "2026-06-01 10:05:00", "echem": 3.4},
        3: {"x": np.linspace(10, 80, 25), "y": np.full(25, 3.0),
            "timestamp": "2026-06-01 10:10:00", "echem": 3.6},
    }


def build_neutron_arrays():
    """Build three scans of banks 1 and 3 on TOF and d grids; scan 2 lacks bank 3 d."""
    tof = np.linspace(1000.0, 20000.0, 40)
    d = np.linspace(0.3, 6.0, 40)
    arrays = {}
    for num in (1, 2, 3):
        arrays[num] = {
            "1": {"tof": {"x": tof, "y": np.full(40, float(num))},
                  "d": {"x": d, "y": np.full(40, 10.0 * num)}},
            "3": {"tof": {"x": tof, "y": np.full(40, 2.0 * num)},
                  "d": {"x": d, "y": np.full(40, 20.0 * num)}},
        }
    del arrays[2]["3"]["d"]
    return arrays


def build_hm_scans():
    """Build three scans with display timestamps and correlation midpoints 30 s later."""
    return [
        {"scan_num": 1, "timestamp": "2026-06-01 10:00:00",
         "timestamp_for_correlation": pd.Timestamp("2026-06-01 10:00:30")},
        {"scan_num": 2, "timestamp": "2026-06-01 10:05:00",
         "timestamp_for_correlation": pd.Timestamp("2026-06-01 10:05:30")},
        {"scan_num": 3, "timestamp": "2026-06-01 10:10:00",
         "timestamp_for_correlation": pd.Timestamp("2026-06-01 10:10:30")},
    ]


def test_assign_cycles_rests_inherit_preceding_phase():
    """Rests are folded into the preceding charge/discharge phase."""
    phases = assign_cycles(build_cycling_df())["phase"]
    assert (phases == "charge").sum() == 2 * 15, f"got {(phases == 'charge').sum()}"
    assert (phases == "discharge").sum() == 2 * 15, f"got {(phases == 'discharge').sum()}"


def test_assign_cycles_detects_cycles_and_time_axis():
    """Two cycles detected on a zero-based 1 s time axis."""
    cyc = assign_cycles(build_cycling_df())
    cycles_found = sorted(c for c in cyc["cycle"].unique() if c > 0)
    assert cycles_found == [1, 2], f"got {cycles_found}"
    assert cyc["t_s"].iloc[0] == 0.0
    assert np.allclose(np.diff(cyc["t_s"].values), 1.0)


def test_leading_rest_stays_cycle_zero():
    """A rest before the first charge is never assigned to a cycle."""
    lead_rest = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-06-01 09:59:57") + pd.Timedelta(seconds=s)
                      for s in range(3)],
        "echem_data": [3.0] * 3,
        "current": [0.0] * 3,
    })
    lead = pd.concat([lead_rest, build_cycling_df()], ignore_index=True)
    assert (assign_cycles(lead)["cycle"].values[:3] == 0).all()


def test_discharge_first_record_starts_cycle_one_on_its_discharge():
    """A record that starts on discharge counts cycles from that discharge, none left in cycle 0."""
    df = build_cycling_df()
    cyc = assign_cycles(df.assign(current=-df["current"]))
    assert (cyc["cycle"] > 0).all(), "a discharge-first record lost samples to cycle 0"
    assert sorted(cyc["cycle"].unique()) == [1, 2], f"got {sorted(cyc['cycle'].unique())}"
    assert list(cyc[cyc["cycle"] == 1]["phase"].unique()) == ["discharge", "charge"]


def test_compute_capacity_trapezoidal_integral():
    """1 mA over 9 one-second intervals integrates to exactly 9/3600 mAh."""
    cyc = assign_cycles(build_cycling_df())
    c1_charge = cyc[(cyc["cycle"] == 1) & (cyc["phase"] == "charge")]
    cap = compute_capacity(c1_charge[c1_charge["current"] != 0])
    assert np.isclose(cap[-1], 9.0 / 3600.0), f"got {cap[-1]}"
    assert cap[0] == 0.0, "capacity must reset at segment start"


def test_capacity_plots_draw_one_labelled_line_set_per_cycle():
    """Both plots draw per-cycle lines on a headless Figure with correct labels."""
    df = build_cycling_df()
    fig = Figure()
    ax_time, ax_cap = fig.subplots(1, 2)
    plot_time_vs_voltage(ax_time, df)
    plot_capacity_vs_voltage(ax_cap, df, mass_mg=10.0)
    assert len(ax_time.get_lines()) == 2
    assert [t.get_text() for t in ax_cap.get_legend().get_texts()] == ["Cycle 1", "Cycle 2"]
    assert ax_cap.get_xlabel() == "Specific capacity [mAh/g]"


@pytest.mark.parametrize("n_cycles, expected", [
    (10, [f"Cycle {c}" for c in range(1, 11)]),
    (12, ["Cycle 1", "Cycle 12"]),
])
def test_capacity_legend_names_the_ends_past_ten_cycles(n_cycles, expected):
    """Up to ten cycles every cycle is named; past that only the first and last, all still drawn."""
    df = build_cycling_df(n_cycles=n_cycles)
    fig = Figure()
    ax_time, ax_cap = fig.subplots(1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plot_time_vs_voltage(ax_time, df)
        plot_capacity_vs_voltage(ax_cap, df)
    for ax in (ax_time, ax_cap):
        assert [t.get_text() for t in ax.get_legend().get_texts()] == expected
    assert len(ax_time.get_lines()) == n_cycles, "every cycle must still be drawn"
    cmap = cm.tab10 if n_cycles <= 10 else cm.viridis
    assert ax_time.get_lines()[-1].get_color() == cmap(1.0), "tab10 up to ten cycles, viridis above"
    plot_time_vs_voltage(ax_time, df, cycles_to_plot=[5])
    assert [t.get_text() for t in ax_time.get_legend().get_texts()] == ["Cycle 5"]


def test_no_legend_warning_when_nothing_plotted():
    """Rest-only data starts no cycle; no legend warning is emitted."""
    rest_only = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-06-01 10:00:00") + pd.Timedelta(seconds=s)
                      for s in range(6)],
        "echem_data": [3.5] * 6,
        "current": [0.0] * 6,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig = Figure()
        ax1, ax2 = fig.subplots(1, 2)
        plot_time_vs_voltage(ax1, rest_only)
        plot_capacity_vs_voltage(ax2, rest_only)


def test_relative_mode_keeps_subsecond_precision():
    """original_timestamp wins over rounded HH:MM:SS display strings."""
    rel = pd.DataFrame({
        "timestamp": ["00:00:00", "00:00:00", "00:00:01", "00:00:02"],
        "original_timestamp": [pd.Timestamp("2026-06-01 10:00:00") + pd.Timedelta(seconds=0.5 * s)
                               for s in range(4)],
        "echem_data": [3.7] * 4,
        "current": [1.0] * 4,
    })
    rel_t = _assign_time(rel.copy())["t_s"].values
    assert np.allclose(np.diff(rel_t), 0.5), f"got {np.diff(rel_t)}"


def test_detect_pulses_respects_max_rest():
    """Three valid pulses detected; the long-rest fourth is excluded by max_rest."""
    seg_current, seg_t = [], []
    t = 0.0
    for rest_len in (10, 10, 10, 400):
        for _ in range(5):
            seg_current.append(2.0)
            seg_t.append(t)
            t += 1.0
        for _ in range(rest_len):
            seg_current.append(0.0)
            seg_t.append(t)
            t += 1.0
    pulse_df = pd.DataFrame({
        "t_s": seg_t, "current": seg_current, "echem_data": [3.8] * len(seg_t),
    })
    pulses = _detect_pulses(pulse_df, max_rest=300.0)
    assert len(pulses) == 3, f"got {len(pulses)}"
    assert pulses[0]["pulse_mask"].sum() == 5 and pulses[0]["relax_mask"].sum() == 10, \
        "pulse masks must cover the active period"
    assert all(p["relax_end_t"] - p["pulse_end_t"] < 300 for p in pulses)


def test_compute_fit_recovers_parameters(fit_df):
    """Exact relaxation data recovers R, k, r2 = 1, V0 and pulse-end capacity."""
    fit_pulses = _detect_pulses(fit_df, max_rest=300.0)
    assert len(fit_pulses) == 1, "fit dataset must yield one pulse"
    cap_arr = compute_capacity(_assign_time(fit_df.copy()))
    fit = _compute_fit(fit_df, fit_pulses[0], r1_start=0.5, r1_length=3.0, cap_arr=cap_arr)
    assert fit is not None
    assert np.isclose(fit["R"], R_TRUE), f"got {fit['R']}"
    assert np.isclose(fit["k"], K_TRUE), f"got {fit['k']}"
    assert np.isclose(fit["r2"], 1.0), f"got {fit['r2']}"
    assert fit["V0"] == V0, "V0 must be the last active voltage"
    assert np.isclose(fit["capacity"], 2.0 * 4.0 / 3600.0), f"got {fit['capacity']}"


def test_narrow_fit_window_rejected(fit_df):
    """A regression window holding one or two samples yields no fit; three samples fit."""
    pulse = _detect_pulses(fit_df, max_rest=300.0)[0]
    assert _compute_fit(fit_df, pulse, r1_start=0.5, r1_length=0.1) is None
    # 0.5 and 0.75 s: a two-point line would report R² 1 with zero error bars
    assert _compute_fit(fit_df, pulse, r1_start=0.5, r1_length=0.5) is None
    fit = _compute_fit(fit_df, pulse, r1_start=0.5, r1_length=0.75)
    assert fit is not None and len(fit["X_fit"]) == 3
    assert np.isfinite(fit["R_err"]) and np.isfinite(fit["k_err"])


def _ten_hz_pulse(lead_s):
    """Return a 10 Hz log from a 5 s pulse lead_s into the record, timed as the app times it."""
    n_lead, n_rest = int(lead_s * 10), 100
    current = [0.0] * n_lead + [2.0] * 50 + [0.0] * n_rest
    df = _assign_time(pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(
            np.arange(len(current)) * 100, unit="ms"),
        "echem_data": np.r_[np.full(n_lead + 50, 3.8),
                            3.7 - 0.001 * np.sqrt(np.arange(n_rest) / 10)],
        "current": current,
    }))
    return df.iloc[n_lead:].reset_index(drop=True)


@pytest.mark.parametrize("case", ["a sample 1 us early", "a pulse 3600 s into a 10 Hz log"])
def test_fit_window_ignores_timestamp_noise(fit_df, case):
    """A window takes the samples it covers, whatever float noise their timestamps carry."""
    if case == "a sample 1 us early":
        rest = np.flatnonzero(_detect_pulses(fit_df, max_rest=300.0)[0]["relax_mask"])
        t_s = fit_df["t_s"].to_numpy().copy()
        t_s[rest[[2, 6]]] -= 1e-6  # the 0.5 s and 1.5 s rest samples
        log, window, taken = fit_df.assign(t_s=t_s), (0.5, 1.0), [0.5, 0.75, 1.0, 1.25]
    else:
        log, window, taken = _ten_hz_pulse(3600.0), (0.1, 0.9), np.arange(1, 10) / 10
    fit = _compute_fit(log, _detect_pulses(log, max_rest=300.0)[0], *window)
    got = fit["X_fit"] ** 2
    assert len(got) == len(taken) and np.allclose(got, taken, atol=1e-5), got


def test_compute_all_pulses_returns_one_indexed_fit_per_pulse(fit_df):
    """_compute_all_pulses yields one fit per detected pulse, tagged with a 1-based pulse_idx."""
    fit_pulses = _detect_pulses(fit_df, max_rest=300.0)
    results = _compute_all_pulses(fit_df, fit_pulses, lambda idx: (0.5, 3.0))
    assert len(results) == 1 and results[0]["pulse_idx"] == 1


def test_shared_plot_palette_is_white():
    """Shared ICI/Heatmap axes styling paints a white palette."""
    fig = Figure()
    ax = fig.subplots()
    _style_ax(ax, "x", "y", "Light figure")
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert PLOT_BG == "#ffffff"


def test_build_xrd_matrix_stacks_and_interpolates():
    """Matrix stacks one row per scan in order, interpolating differing grids."""
    mx, matrix, used = build_xrd_matrix(build_oned_arrays())
    assert matrix.shape == (3, 50), f"got {matrix.shape}"
    assert used == [1, 2, 3], "rows must keep scan order"
    assert np.allclose(matrix[2], 3.0), "differing grid must be interpolated"


def test_build_xrd_matrix_skips_entries_without_data():
    """Entries with missing or non 1-D x/y are skipped instead of crashing the heatmap."""
    x = np.linspace(10, 80, 50)
    _, matrix, used = build_xrd_matrix({1: {"x": None, "y": None}, 2: {"x": x, "y": np.ones(50)},
                                        3: {"error": True}, 4: {"x": 5.0, "y": 1.0}})
    assert used == [2] and matrix.shape == (1, 50)


def test_build_xrd_matrix_interpolates_a_descending_grid():
    """A scan recorded on a descending grid lands on the first scan's grid in the right order."""
    x = np.linspace(1.0, 5.0, 5)
    _, matrix, _ = build_xrd_matrix({1: {"x": x, "y": x * 10.0},
                                     2: {"x": x[::-1] + 0.5, "y": (x[::-1] + 0.5) * 10.0}})
    assert np.allclose(matrix[1][1:], matrix[0][1:]), "np.interp needs ascending sample points"


def test_axis_labels_come_from_one_set():
    """Every window takes its TOF, d-spacing, 2θ and time labels from operaxn.config."""
    assert (cfg.TOF_LABEL, cfg.DSPACING_LABEL, cfg.TWOTHETA_LABEL, cfg.TIME_HOURS_LABEL) == (
        "Time of flight [μs]", "$d$-spacing [Å]", r"2$\theta$ [°]", "Time [h]")
    assert (cfg.TOF_TEXT, cfg.DSPACING_TEXT) == ("Time of flight (μs)", "d-spacing (Å)")
    stray = ("Time of Flight", "d spacing (", "2θ (degrees)", "Time (hours)", "'t (h)'",
             "Time (h)")
    for path in glob.glob(os.path.join(os.path.dirname(cfg.__file__), "*.py")):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for variant in stray:
            assert variant not in text, f"{os.path.basename(path)} still writes {variant!r}"


def test_figure_text_puts_units_in_square_brackets():
    """Main-window panes, the heatmap export and Capacity write [units] and an italic θ."""
    t = np.linspace(0.0, 7200.0, 50)
    d = np.linspace(1.0, 5.0, 40)
    bank = {"tof": {"x": d * 5000.0, "y": d}, "d": {"x": d, "y": d}}
    figures = []
    for dspacing in (False, True):
        fig = Figure()
        ax_oned, ax_twod, ax_echem, ax_current = fig.subplots(1, 4)
        operaxn.plot_oned_data(ax_oned, np.linspace(10, 80, 50), np.arange(50.0),
                               plot_config={"show_dspacing": dspacing})
        operaxn.plot_twod_data(ax_twod, np.ones((8, 8)))
        for _ in range(2):  # the second call takes the cached update path
            operaxn.plot_echem_data(ax_echem, t, 3.7 + 0.0 * t, 0.1 + 0.0 * t)
        operaxn.plot_echem_data(ax_current, t, np.array([]), 0.1 + 0.0 * t)
        fig_n = Figure()
        banks = dict(zip([f"neutron_{n}" for n in range(1, 6)], fig_n.subplots(1, 5)))
        operaxn.plot_neutron_data(banks, {"1": bank, "3": bank},
                                  plot_config={"show_neutron_dspacing": dspacing})
        figures += [fig, fig_n]
    rows = np.array([1.0, 2.0])
    for x_label in (*NEUTRON_AXES.values(), *XRD_AXES.values()):
        figures.append(publication_figure(
            np.linspace(1.0, 5.0, 20), rows, np.ones((2, 20)), x_label=x_label,
            y_label=cfg.TIME_HOURS_LABEL, cmap="viridis", vmin=0.5, vmax=2.0,
            voltage=(np.array([3.0, 4.0]), rows)))
    fig = Figure()
    ax_time, ax_cap, ax_spec = fig.subplots(1, 3)
    plot_time_vs_voltage(ax_time, build_cycling_df())
    plot_capacity_vs_voltage(ax_cap, build_cycling_df())
    plot_capacity_vs_voltage(ax_spec, build_cycling_df(), mass_mg=10.0)
    figures.append(fig)
    texts = {text.get_text() for fig in figures for text in fig.findobj(Text)}
    in_parentheses = sorted(s for s in texts if re.search(
        r"\((V|mA|mAh|mAh/g|s|h|μs|Å|°|counts|arb\. units)\)", s))
    assert not in_parentheses, f"units must be in square brackets: {in_parentheses}"
    assert {"Voltage [V]", "Current [mA]", "Intensity [counts]", "Capacity [mAh]",
            r"2$\theta$ [°]", "$d$-spacing [Å]"} <= texts
    upright = sorted(s for s in texts if "2θ" in s)
    assert not upright, f"the θ of 2θ must be italic (mathtext): {upright}"


def _ici_texts(win):
    """Return every text drawn in an ICI window's three figures."""
    figures = (win._fig_top, win._fig_bot, win._fig_right)
    return {text.get_text() for fig in figures for text in fig.findobj(Text) if text.get_text()}


@pytest.mark.display
def test_ici_figures_put_units_in_square_brackets(tk_root, monkeypatch):
    """ICI's figure labels write [units] like every other figure, as built and on every R/k axis."""
    win = _open_ici(tk_root, monkeypatch, {"operando": _ici_df()}, fitted=True)
    try:
        texts = _ici_texts(win)
        win.rk_axis_bar.mass_var.set("10")
        for xaxis in ("capacity", "specific_capacity"):
            win.rk_axis_bar._set_xaxis(xaxis)
            texts |= _ici_texts(win)
    finally:
        win.destroy()
    empty = _open_ici(tk_root, monkeypatch, {})
    try:
        built = _ici_texts(empty)
    finally:
        empty.destroy()
    in_parentheses = sorted(s for s in texts | built if re.search(
        r"\((V|mA|mAh|mAh/g|s|h|Ω|√s|s¹ᐟ²|Ω·s⁻¹ᐟ²)\)", s))
    assert not in_parentheses, f"units must be in square brackets: {in_parentheses}"
    assert {"Time [h]", "Voltage [V]", "Current [mA]", "√Δt [s¹ᐟ²]", "ΔV [V]", "R [Ω]",
            "k [Ω·s⁻¹ᐟ²]", "Capacity [mAh]", "Specific capacity [mAh/g]"} <= texts
    assert {"√Δt [s¹ᐟ²]", "ΔV [V]", "R [Ω]", "k [Ω·s⁻¹ᐟ²]", "Voltage [V]"} <= built
    assert not [s for s in texts | built if "\u2013" in s], "an en dash used as a separator"


def test_scan_positions_absolute_and_relative():
    """Scan positions resolve against the echem start or relative clock."""
    pos = scan_positions_seconds(build_hm_scans(), "absolute",
                                 reference=pd.Timestamp("2026-06-01 10:00:00"))
    assert pos == {1: 30.0, 2: 330.0, 3: 630.0}, f"got {pos}"
    rel_pos = scan_positions_seconds([{"scan_num": 1, "timestamp": "00:05:00"}], "relative")
    assert rel_pos == {1: 300}, f"got {rel_pos}"


def test_echem_scan_curve_retains_full_resolution():
    """Scan-axis echem trace keeps full resolution and stays aligned continuously."""
    curve_v, curve_y = echem_scan_curve(
        [1, 2, 3], [1.0, 2.0, 3.0], {1: 10.0, 2: 20.0, 3: 30.0},
        np.linspace(0.0, 40.0, 81), np.linspace(3.0, 4.0, 81),
    )
    assert len(curve_v) == 41 and len(curve_y) == 41
    assert np.isclose(curve_y[0], 1.0) and np.isclose(curve_y[-1], 3.0) \
        and np.all(np.diff(curve_y) >= 0)


def test_standard_echem_accessor_returns_a_list():
    """get_standard_echem returns a list."""
    assert isinstance(get_standard_echem(), list)


@pytest.mark.display
def test_capacity_source_switch_waits_for_the_dropdown_to_close(tk_root, monkeypatch):
    """Picking a Capacity source swaps the dataset only once the dropdown has closed."""
    monkeypatch.setattr(capacity_module, "maximise_window", lambda win: win.withdraw())
    other = build_cycling_df(n_cycles=3)
    win = CapacityWindow(tk_root, {"operando": build_cycling_df(), "other": other})
    try:
        win._queue_source_change("other")
        assert win._df is not other, "the swap must wait for the dropdown to close"
        win.update()
        assert win._df is other and win._available == [1, 2, 3]
    finally:
        win.destroy()


def _ici_df(n_cycles=1, pulses=3, rest_step=1.0, spread=0.0):
    """Return +/-2 mA 30 s pulses before 20 s zero rests (every rest_step s); spread raises R, k."""
    t, i, v, now, volt = [], [], [], 0.0, 3.6
    for _ in range(n_cycles):
        for sign in (1.0, -1.0):
            for n in range(pulses):
                for _ in range(30):
                    t.append(now)
                    i.append(2.0 * sign)
                    v.append(volt)
                    now += 1
                    volt += 0.001 * sign
                for k in range(int(20 / rest_step)):
                    root_dt = np.sqrt(k * rest_step)
                    t.append(now)
                    i.append(0.0)
                    v.append(volt - sign * (0.002 * root_dt + spread * n * (1 + root_dt)))
                    now += rest_step
    # An array: pandas 3 reads a list's whole floats as nanoseconds once fractions follow
    stamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.asarray(t), unit="s")
    return pd.DataFrame({"timestamp": stamps, "echem_data": v, "current": i})


def _ici_stand_in(df, fitted=False):
    """Return a stand-in for ICIWindow's export path; fitted sets a 1-11 s window on both phases."""
    window = {"charge": (1.0, 10.0), "discharge": (1.0, 10.0)} if fitted else {}
    fake = SimpleNamespace(_df=assign_cycles(df), _export_stem="cell", _pulses={}, _ici_cache={},
                           _last_max_rest=300.0, _r1_pulse_overrides={}, _r1_phase_overrides=window,
                           reg_bar=SimpleNamespace(max_rest_var=SimpleNamespace(get=lambda: 300.0)))
    fake._max_cycle = lambda: ICIWindow._max_cycle(fake)
    fake._get_mass_mg = lambda: 10.0
    fake._get_pulses = lambda c, ph: ICIWindow._get_pulses(fake, c, ph)
    fake._pulse_key = lambda c, ph, i: ICIWindow._pulse_key(fake, c, ph, i)
    fake._get_r1 = lambda c=None, ph=None, i=None: ICIWindow._get_r1(fake, c, ph, i)
    fake._compute_ici = lambda c, ph: ICIWindow._compute_ici(fake, c, ph)
    fake._export_tables = lambda: ICIWindow._export_tables(fake)
    fake._store_pending_edit = lambda: None
    return fake


def _open_ici(tk_root, monkeypatch, sources, fitted=False):
    """Open a withdrawn ICI window past its first draw; fitted gives every pulse a 1-11 s window."""
    monkeypatch.setattr(ici_module, "maximise_window", lambda win: win.withdraw())
    win = ICIWindow(tk_root, echem_sources=sources)
    pump(win, 300)
    if fitted:
        win.reg_bar.start_var.set(1.0)
        win.reg_bar.length_var.set(10.0)
        win._on_apply_all()
        pump(win, 300)
    return win


def test_pulse_legend_moves_with_the_phase():
    """The pulse-zoom legend sits lower centre for charge and upper centre for discharge."""
    fake = _ici_stand_in(_ici_df())
    fake._fig_top = Figure()
    fake.ax_pulse, fake._ax_pulse_twin = fake._fig_top.add_subplot(), None
    locs = {}
    for phase in ("charge", "discharge"):
        ICIWindow._plot_pulse_zoom(fake, 1, 1, phase)
        legend = fake.ax_pulse.get_legend() or fake._ax_pulse_twin.get_legend()
        locs[phase] = legend._loc
    assert locs == {"charge": Legend.codes["lower center"],
                    "discharge": Legend.codes["upper center"]}, locs


def test_ici_fit_plot_adds_no_twin_axis_or_span():
    """The ICI fit plot draws on its two panels only: no top time axis, no shaded span."""
    fake = _ici_stand_in(_ici_df())
    fake._r1_phase_overrides["charge"] = (1.0, 10.0)  # a window the 1 Hz rests can fill
    fig = Figure()
    fake.ax_dv = fig.add_subplot()
    ICIWindow._plot_ici_fit(fake, 1, 1, "charge")
    assert fake.ax_dv.lines and fake.ax_dv.collections, "the fit was not drawn"
    assert len(fig.axes) == 1, "extra (twin) axes drawn"
    spans = [p for ax in fig.axes for p in ax.patches if isinstance(p, (Polygon, Rectangle))]
    assert spans == [], spans


def test_ici_export_with_a_locked_file_replaces_neither(tmp_path, monkeypatch):
    """A discharge file held open elsewhere fails the export before either file is replaced."""
    errors = []
    monkeypatch.setattr(ici_module.filedialog, "askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr(ici_module.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(ici_module.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(ici_module.messagebox, "showerror", lambda *a, **k: errors.append(a))
    for phase in ("charge", "discharge"):
        (tmp_path / f"cell_ici_{phase}.csv").write_text("old\n")
    real_open, real_to_csv = open, pd.DataFrame.to_csv

    def refuse_locked(path):
        if str(path).endswith("_discharge.csv"):
            raise PermissionError("file is open in Excel")

    def locked_open(path, *a, **k):
        refuse_locked(path)
        return real_open(path, *a, **k)

    def locked_to_csv(self, path, *a, **k):
        refuse_locked(path)
        return real_to_csv(self, path, *a, **k)

    monkeypatch.setattr(ici_module, "open", locked_open, raising=False)
    monkeypatch.setattr(pd.DataFrame, "to_csv", locked_to_csv)
    ici_module.ICIWindow._export_csv(_ici_stand_in(_ici_df(), fitted=True))
    assert errors == [("Export failed",
                       "ICI results could not be exported: file is open in Excel")], errors
    assert (tmp_path / "cell_ici_charge.csv").read_text() == "old\n", "the charge file was replaced"


def test_ici_export_refuses_when_no_pulse_is_fitted(tmp_path, monkeypatch):
    """With no fitted pulse (here no rests) Export CSV warns and writes nothing, like Capacity."""
    shown = []
    monkeypatch.setattr(ici_module.filedialog, "askdirectory",
                        lambda **k: shown.append("folder asked") or str(tmp_path))
    monkeypatch.setattr(ici_module.messagebox, "showwarning", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(ici_module.messagebox, "showinfo", lambda *a, **k: shown.append(a))
    charge_only = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(range(60), unit="s"),
        "echem_data": np.linspace(3.0, 4.0, 60), "current": [1.0] * 60})
    ici_module.ICIWindow._export_csv(_ici_stand_in(charge_only))
    assert shown == [("No data", "No fitted pulses to export")], shown
    assert list(tmp_path.iterdir()) == []


def test_ici_export_asks_before_replacing_the_pair(tmp_path, monkeypatch):
    """A second export into the same folder asks once, naming both files; No leaves them alone."""
    asked, shown = [], []
    monkeypatch.setattr(ici_module.filedialog, "askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr(ici_module.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a) and False)
    monkeypatch.setattr(ici_module.messagebox, "showinfo", lambda *a, **k: shown.append(a))
    fake = _ici_stand_in(_ici_df(), fitted=True)
    ici_module.ICIWindow._export_csv(fake)
    paths = [str(tmp_path / f"cell_ici_{phase}.csv") for phase in ("charge", "discharge")]
    assert asked == [], asked
    assert shown == [("Export complete", "ICI results exported to:\n" + "\n".join(paths))], shown
    for path in paths:
        with open(path, "w") as fh:
            fh.write("kept\n")
    ici_module.ICIWindow._export_csv(fake)
    assert asked == [("Confirm", "Replace cell_ici_charge.csv and cell_ici_discharge.csv?")], asked
    assert len(shown) == 1 and all(open(path).read() == "kept\n" for path in paths)


def _export_ici_csv(fake, tmp_path, monkeypatch):
    """Run ICIWindow's export on a stand-in into tmp_path; return the charge, discharge tables."""
    monkeypatch.setattr(ici_module.filedialog, "askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr(ici_module.messagebox, "showinfo", lambda *a, **k: None)
    ici_module.ICIWindow._export_csv(fake)
    return [pd.read_csv(tmp_path / f"cell_ici_{phase}.csv", encoding="utf-8-sig")
            for phase in ("charge", "discharge")]


def test_ici_csv_numbers_each_fitted_pulse(tmp_path, monkeypatch):
    """The ICI CSV has Pulse after Cycle and, last, the window each row was fitted with."""
    fake = _ici_stand_in(_ici_df(), fitted=True)
    fake._r1_pulse_overrides[fake._pulse_key(1, "charge", 2)] = (0.5, 0.1)  # no 1 Hz rest point
    fake._r1_pulse_overrides[fake._pulse_key(1, "discharge", 2)] = (2.0, 8.0)
    charge, discharge = _export_ici_csv(fake, tmp_path, monkeypatch)
    assert list(charge.columns) == list(discharge.columns) == [
        "Cycle", "Pulse", "Voltage (V)", "Capacity (mAh)", "Specific Capacity (mAh/g)",
        "R (Ohm)", "R Error (Ohm)", "k (Ohm s^-1/2)", "k Error (Ohm s^-1/2)", "R2",
        "Window Start (s)", "Window Length (s)"], charge.columns
    assert charge["Pulse"].tolist() == [1, 3], charge
    assert discharge["Pulse"].tolist() == [1, 2, 3], discharge
    windows = discharge[["Window Start (s)", "Window Length (s)"]].values.tolist()
    assert windows == [[1.0, 10.0], [2.0, 8.0], [1.0, 10.0]], windows
    assert np.allclose(charge["Specific Capacity (mAh/g)"], charge["Capacity (mAh)"] / 0.010)


def test_ici_and_capacity_report_the_same_specific_capacity(tmp_path, monkeypatch):
    """Each ICI pulse's (specific) capacity is the Capacity CSV's value at that pulse's V0 point."""
    df = _ici_df(n_cycles=2)
    ici_tables = _export_ici_csv(_ici_stand_in(df, fitted=True), tmp_path, monkeypatch)
    out = tmp_path / "cell_capacity.csv"
    monkeypatch.setattr(capacity_module.filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(capacity_module.messagebox, "showinfo", lambda *a, **k: None)
    CapacityWindow._export_csv(SimpleNamespace(_df=df, _export_stem="cell",
                                               all_cycles_var=SimpleNamespace(get=lambda: True),
                                               _get_cycles=lambda: [1, 2],
                                               _get_mass_mg=lambda: 10.0))
    cap = pd.read_csv(out, encoding="utf-8-sig")
    for phase, table in zip(("charge", "discharge"), ici_tables):
        assert len(table) == 2 * 3, table
        for row in table.to_dict("records"):
            leg = cap[(cap["Cycle"] == row["Cycle"]) & (cap["Phase"] == phase)]
            at_v0 = leg[np.isclose(leg["Voltage (V)"], row["Voltage (V)"])
                        & np.isclose(leg["Capacity (mAh)"], row["Capacity (mAh)"])]
            assert len(at_v0) == 1, (phase, row)
            assert np.isclose(at_v0["Specific Capacity (mAh/g)"].iloc[0],
                              row["Specific Capacity (mAh/g)"]), (phase, row)


@pytest.mark.display
def test_capacity_window_controls(tk_root, monkeypatch):
    """Capacity redraws on a mass edit and steps cycles with its arrows; it has no Save or Apply."""
    monkeypatch.setattr(capacity_module, "maximise_window", lambda win: win.withdraw())
    win = CapacityWindow(tk_root, {"operando": build_cycling_df()})
    try:
        assert not {"Save Plot", "Apply"} & set(_buttons(win)), _buttons(win)
        win.mass_var.set("10")
        pump(win, capacity_module.MASS_REPLOT_DELAY_MS + 100)
        assert win.ax_cap.get_xlabel() == "Specific capacity [mAh/g]", "a mass edit must redraw"
        win.all_cycles_var.set(False)
        win._toggle_all_cycles()
        win._next_btn.invoke()
        assert [t.get_text() for t in win.ax_time.get_legend().get_texts()] == ["Cycle 2"]
    finally:
        win.destroy()


def test_capacity_export_reports_a_locked_file(tmp_path, monkeypatch):
    """A CSV export that cannot write shows 'Export failed' instead of raising."""
    errors = []
    monkeypatch.setattr(capacity_module.filedialog, "asksaveasfilename",
                        lambda **k: str(tmp_path / "locked.csv"))
    monkeypatch.setattr(capacity_module.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(capacity_module.messagebox, "showinfo", lambda *a, **k: None)

    def locked(*a, **k):
        raise PermissionError("file is open in Excel")

    monkeypatch.setattr(pd.DataFrame, "to_csv", locked)
    t = [pd.Timestamp("2026-06-01 10:00:00") + pd.Timedelta(seconds=s) for s in range(30)]
    df = pd.DataFrame({"timestamp": t, "echem_data": np.linspace(3, 4, 30),
                       "current": [1.0] * 10 + [0.0] * 5 + [-1.0] * 10 + [0.0] * 5})
    fake = SimpleNamespace(_df=df, _export_stem="cell", _get_cycles=lambda: [1],
                           all_cycles_var=SimpleNamespace(get=lambda: True),
                           _get_mass_mg=lambda: 0.0)
    CapacityWindow._export_csv(fake)
    assert errors == [("Export failed",
                       "Capacity data could not be exported: file is open in Excel")], errors


@pytest.mark.parametrize("box, exported", [("all", [1, 2]), (7, [2]), (2.6, [2]), (None, [1])],
                         ids=["all cycles", "typed 7 of 2", "typed 2.6", "text"])
def test_capacity_csv_has_the_documented_columns(tmp_path, monkeypatch, box, exported):
    """Export CSV writes README.md's columns for the cycles shown, committing the box first."""
    out = tmp_path / "cell_capacity.csv"
    errors = []
    monkeypatch.setattr(capacity_module.filedialog, "asksaveasfilename", lambda **k: str(out))
    for kind in ("showerror", "showwarning"):
        monkeypatch.setattr(capacity_module.messagebox, kind, lambda *a, **k: errors.append(a))
    monkeypatch.setattr(capacity_module.messagebox, "showinfo", lambda *a, **k: None)
    shown = {"cycle": box}  # the cycle box: what it reads as (None: text that is no number)
    fake = SimpleNamespace(_df=build_cycling_df(), _export_stem="cell", _available=[1, 2],
                           _shown_cycle=1, all_cycles_var=SimpleNamespace(get=lambda: box == "all"),
                           cycle_var=SimpleNamespace(set=lambda c: shown.update(cycle=c)),
                           _selected_cycle=lambda: None if shown["cycle"] is None
                           else int(shown["cycle"]),  # as the IntVar reads it ('2.6' -> 2)
                           _replot=lambda: None, _get_mass_mg=lambda: 0.0)
    fake._commit_cycle = lambda: CapacityWindow._commit_cycle(fake)
    fake._get_cycles = lambda: CapacityWindow._get_cycles(fake)
    CapacityWindow._export_csv(fake)
    assert errors == [], errors
    with open(out, encoding="utf-8-sig") as fh:
        header, first = fh.readline().rstrip("\n"), fh.readline().rstrip("\n")
    assert header == ("Cycle,Phase,Time (s),Voltage (V),Capacity (mAh),"
                      "Specific Capacity (mAh/g)"), header
    assert first.startswith(f"{exported[0]},charge,") and first.endswith(","), first
    assert sorted(pd.read_csv(out, encoding="utf-8-sig")["Cycle"].unique()) == exported
    if box != "all":
        assert shown["cycle"] == exported[0], "the box must show the cycle exported"


@pytest.mark.parametrize("export, what", [("_export_figure", "Heatmap"),
                                          ("_export_all_banks", "Heatmaps")])
def test_failed_heatmap_export_says_what_could_not_be_exported(tmp_path, monkeypatch, export, what):
    """A heatmap figure or Export all banks that raises names what was not exported."""
    errors = []
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(tmp_path / "map.png"))
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **_k: errors.append(a))

    def boom(*_a):
        raise OSError("disk full")

    fake = SimpleNamespace(_matrix=np.ones((2, 2)), _export_stem=lambda: "cell",
                           _file_stem="cell", _banks=["1", "3"], save_figure=boom,
                           _rows_for=lambda _bank: (None, np.ones((2, 2)), []))
    getattr(HeatmapWindow, export)(fake)
    assert errors == [("Export failed", f"{what} could not be exported: disk full")], errors


@pytest.mark.parametrize("drawn, noun", [(["1"], "Heatmap"), (["1", "3"], "Heatmaps")])
def test_export_all_banks_counts_the_files_written(tmp_path, monkeypatch, drawn, noun):
    """Export all banks reports one written file as 'Heatmap' and more as 'Heatmaps'."""
    shown = []
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(tmp_path / "maps.png"))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **_k: shown.append(a))
    fake = SimpleNamespace(_file_stem="cell", _banks=["1", "3"],
                           save_figure=lambda path, bank: bank in drawn,
                           _rows_for=lambda bank: (None, np.ones((2, 2)) if bank in drawn else None,
                                                   []))
    HeatmapWindow._export_all_banks(fake)
    paths = [str(tmp_path / f"maps_bank{bank}.png") for bank in drawn]
    assert shown == [("Export complete", f"{noun} exported to:\n" + "\n".join(paths))], shown


@pytest.mark.parametrize("text, mass", [("12.5", 12.5), ("inf", 0.0), ("nan", 0.0), ("-5", 0.0),
                                        ("0", 0.0)])
def test_capacity_and_ici_take_only_a_finite_positive_mass(text, mass):
    """A mass that is no finite positive number counts as none in both windows; its box shows 0."""
    var = SimpleNamespace(get=lambda: text)
    assert CapacityWindow._get_mass_mg(SimpleNamespace(mass_var=var)) == mass
    ici = SimpleNamespace(rk_axis_bar=SimpleNamespace(mass_var=var))
    assert ICIWindow._get_mass_mg(ici) == mass
    for commit in (CapacityWindow._commit_mass, _RKAxisBar._commit_mass):
        box, redrawn = {"text": text}, []
        bar = SimpleNamespace(mass_var=SimpleNamespace(get=lambda: box["text"],
                                                       set=lambda t: box.update(text=t)))
        commit(bar, lambda: redrawn.append(1))
        assert box["text"] == (text if mass else "0") and redrawn == [1], (commit, box)


@pytest.mark.parametrize("label, scale", [("I/mA", 1.0), ("I/A", 1e-3)])
def test_ici_fit_reads_the_current_unit_from_the_echem_header(tmp_path, fit_df, label, scale):
    """An echem file read by the app gives the same R and k (ohm) whatever current unit it names."""
    t0 = pd.Timestamp("2026-06-01 10:00:00")
    rows = [f"{(t0 + pd.Timedelta(seconds=t)).strftime('%d/%m/%Y %H:%M:%S.%f')}"
            f"\t{float(v)!r}\t{float(i) * scale!r}"
            for t, v, i in zip(fit_df["t_s"], fit_df["echem_data"], fit_df["current"])]
    path = tmp_path / "ici.txt"
    path.write_text("\n".join([f"Date/Time\tEwe/V\t{label}"] + rows) + "\n", encoding="utf-8")
    seg = assign_cycles(core.EchemParser().parse(str(path)))
    fits = _compute_all_pulses(seg, _detect_pulses(seg, max_rest=300.0), lambda idx: (0.5, 3.0))
    assert len(fits) == 1 and np.isclose(fits[0]["I_A"], I_A), fits
    assert np.isclose(fits[0]["R"], R_TRUE) and np.isclose(fits[0]["k"], K_TRUE), fits[0]


def test_all_cycles_labels_use_primary_text_colour():
    """All-cycles selector labels use the primary text colour on both
    windows (pins the implementation source; rename-sensitive)."""
    assert "bg=BG2, fg=TEXT" in inspect.getsource(_SelectorStrip._build)
    assert "bg=BG2, fg=TEXT" in inspect.getsource(CapacityWindow._build_ui)


def test_analysis_figures_use_light_palette():
    """Capacity and Heatmap figure backgrounds stay on the light
    palette (pins the implementation source; rename-sensitive)."""
    assert "facecolor=PLOT_BG" in inspect.getsource(CapacityWindow._build_ui)
    heatmap_build_source = inspect.getsource(HeatmapWindow._build_ui)
    assert "facecolor=PLOT_BG" in heatmap_build_source
    assert "width_ratios=(4, 0.12, 0.22, 1.1)" in heatmap_build_source, \
        "heatmap must reserve a colourbar label gutter"


def test_echem_sources_offers_current_bearing_datasets(monkeypatch):
    """Analysis windows see operando plus current-bearing additional echem only."""
    operando = build_cycling_df(n_cycles=1)
    additional = build_cycling_df(n_cycles=1)
    voltage_only = additional.drop(columns=["current"])
    monkeypatch.setattr(gui_module, "get_standard_echem", lambda: [
        {"source_file": "reference_echem.txt", "data": additional},
        {"source_file": "voltage_only.txt", "data": voltage_only},
    ])
    fake_app = SimpleNamespace(
        state=SimpleNamespace(echem_df=operando),
        _has_analysis_current=OPERAXN._has_analysis_current,
    )
    analysis_sources = OPERAXN._echem_sources(fake_app)
    assert list(analysis_sources) == ["Operando echem", "reference_echem.txt"], \
        f"got {list(analysis_sources)}"
    assert analysis_sources.get("reference_echem.txt") is additional
    assert "voltage_only.txt" not in analysis_sources


def _rest_lengths_df(rests=(10, 10, 400, 10, 20)):
    """Return 1 Hz charge: a 30 s pulse before each rest length (s), then a closing 30 s run."""
    current, t, now = [], [], 0.0
    for rest in rests:
        current += [2.0] * 30 + [0.0] * rest
        t += list(np.arange(now, now + 30 + rest))
        now += 30 + rest
    current += [2.0] * 30
    t += list(np.arange(now, now + 30))
    return pd.DataFrame({"timestamp": pd.Timestamp("2026-06-01") + pd.to_timedelta(t, unit="s"),
                         "echem_data": np.linspace(3.0, 4.0, len(t)), "current": current})


@pytest.mark.display
def test_max_rest_edits_keep_windows_and_selection_on_their_pulses(tk_root, monkeypatch):
    """Max rest (300 s at first) re-detects pulses; windows, selection and range follow them."""
    win = _open_ici(tk_root, monkeypatch, {"cell": _rest_lengths_df()})
    bar, sel = win.reg_bar, win.selector

    def starts():
        return [p["pulse_start_t"] for p in win._get_pulses(1, "charge")]

    try:
        assert bar.max_rest_var.get() == 300.0 and starts() == [0.0, 40.0, 510.0, 550.0]
        assert float(sel._pulse_spin.cget("to")) == 4, "the pulse range is set from the first load"
        sel.pulse_var.set(3)  # the pulse starting at 510 s
        bar.start_var.set(2.0)
        bar.length_var.set(5.0)
        pump(win, 350)
        sel.pulse_var.set(2)
        pump(win, 350)
        bar.max_rest_var.set(310.0)  # the same four pulses
        pump(win, 350)
        assert win._get_r1(1, "charge", 2) == (0.5, 1.0), "a Max rest edit set a pulse window"
        sel.pulse_var.set(3)
        pump(win, 350)
        bar.max_rest_var.set(500.0)  # admits the 400 s rest as the new pulse 3
        pump(win, 350)
        assert starts() == [0.0, 40.0, 80.0, 510.0, 550.0], starts()
        assert win._get_r1(1, "charge", 4) == (2.0, 5.0), "the tuned pulse lost its window"
        assert win._get_r1(1, "charge", 3) == (0.5, 1.0), "the new pulse took the tuned window"
        assert sel.pulse_var.get() == 4, "the selection did not follow its pulse"
        assert float(sel._pulse_spin.cget("to")) == 5, "the pulse range was not updated"
        bar.max_rest_var.set(300.0)
        pump(win, 350)
        assert float(sel._pulse_spin.cget("to")) == 4
        assert sel.pulse_var.get() == 3 and win._get_r1(1, "charge", 3) == (2.0, 5.0)
        sel.pulse_var.set(4)  # the pulse with the 20 s rest
        pump(win, 50)
        bar.max_rest_var.set(15.0)  # drops it: the selection moves to the last pulse left
        pump(win, 350)
        assert starts() == [0.0, 40.0, 510.0] and sel.pulse_var.get() == 3
        assert float(sel._pulse_spin.cget("to")) == 3
        bar.max_rest_var.set(5.0)  # shorter than every rest
        pump(win, 350)
        assert starts() == [], "lowering Max rest below every rest must reject every pulse"
    finally:
        win.destroy()


@pytest.mark.display
def test_regression_edits_land_on_their_own_pulse(tk_root, monkeypatch):
    """A Start/Length edit stays with its pulse even when the selection moves before it settles."""
    win = _open_ici(tk_root, monkeypatch, {"cell": _ici_df()})
    try:
        win.reg_bar.length_var.set(5.0)
        win.selector.pulse_var.set(2)  # before the edit's debounce fires
        pump(win, 350)
        windows = [win._get_r1(1, "charge", pulse) for pulse in (1, 2, 3)]
        shown = (win.reg_bar.start_var.get(), win.reg_bar.length_var.get())
        win.reg_bar.length_var.set(2.0)  # on pulse 2
        win.selector.pulse_var.set(3)
        win.reg_bar.length_var.set(3.0)  # on pulse 3, before pulse 2's edit settles
        pump(win, 350)
        both = [win._get_r1(1, "charge", pulse) for pulse in (2, 3)]
    finally:
        win.destroy()
    assert windows == [(0.5, 5.0), (0.5, 1.0), (0.5, 1.0)], windows
    assert shown == (0.5, 1.0), "the bar must show pulse 2's own window"
    assert both == [(0.5, 2.0), (0.5, 3.0)], "an edit must survive a quick edit on the next pulse"


def _choose_source(win, name):
    """Pick name in the ICI window's Echem menu, as a click does."""
    menu = win.nametowidget(win.selector.source_button.cget("menu"))
    menu.invoke(menu.index(name))


def _fitted_windows(win, phase):
    """Return {pulse: (first Δt in s, points)} of cycle 1's fits in phase: the window each used."""
    return {fit["pulse_idx"]: (round(float(fit["X_fit"][0]) ** 2, 6), len(fit["X_fit"]))
            for fit in win._compute_ici(1, phase)}


def _window_samples(start, length, step=0.25, rest=20.0):
    """Return (first Δt in s, points) a regression window takes from a rest sampled every step s."""
    grid = np.arange(0.0, rest, step)
    taken = grid[(grid >= start) & (grid < start + length)]
    return float(taken[0]), len(taken)


@pytest.mark.display
def test_ici_regression_window_resolves_pulse_then_phase_then_default(tk_root, monkeypatch):
    """A pulse uses its own edit, else Apply All's window, else the default; browsing sets none."""
    df = _ici_df(rest_step=0.25)
    win = _open_ici(tk_root, monkeypatch, {"cell": df, "other": df.copy()})
    try:
        default = _window_samples(win.reg_bar.start_var.get(), win.reg_bar.length_var.get())
        edited, applied = _window_samples(1.0, 2.0), _window_samples(2.0, 4.0)
        assert len({default, edited, applied}) == 3
        everywhere = {1: default, 2: default, 3: default}
        assert _fitted_windows(win, "charge") == _fitted_windows(win, "discharge") == everywhere
        win.selector.pulse_var.set(3)
        win.selector._set_phase("discharge")
        win.selector._set_phase("charge")
        assert win.selector.pulse_var.get() == 3, "a phase toggle must keep the selected pulse"
        win.selector.pulse_var.set(2)
        pump(win, 400)
        assert _fitted_windows(win, "charge") == everywhere, "browsing must set no window"
        win.reg_bar.start_var.set(1.0)
        win.reg_bar.length_var.set(2.0)
        pump(win, 400)  # applies after the bar's debounce, with no button
        assert _fitted_windows(win, "charge") == {1: default, 2: edited, 3: default}
        assert _fitted_windows(win, "discharge") == everywhere
        win.selector.pulse_var.set(3)  # Apply All must also reach pulse 2's window
        win.reg_bar.start_var.set(2.0)
        win.reg_bar.length_var.set(4.0)
        win._on_apply_all()
        pump(win, 400)
        for phase in ("charge", "discharge"):
            assert _fitted_windows(win, phase) == {1: applied, 2: applied, 3: applied}, phase
        win.selector.pulse_var.set(1)
        win.reg_bar.start_var.set(1.0)
        win.reg_bar.length_var.set(2.0)
        pump(win, 400)
        assert _fitted_windows(win, "charge") == {1: edited, 2: applied, 3: applied}
        shown = {}
        for pulse in (2, 1):
            win.selector.pulse_var.set(pulse)
            shown[pulse] = (win.reg_bar.start_var.get(), win.reg_bar.length_var.get())
        pump(win, 400)
        assert shown == {2: (2.0, 4.0), 1: (1.0, 2.0)}, "the bar shows the selected pulse's window"
        assert _fitted_windows(win, "charge") == {1: edited, 2: applied, 3: applied}, \
            "showing a window is no edit"
        loaded = win._df
        _choose_source(win, "other")
        assert win._df is loaded, "the switch must wait until the menu has closed"
        pump(win, 100)
        assert win._df is not loaded
        for phase in ("charge", "discharge"):
            assert _fitted_windows(win, phase) == everywhere, phase
    finally:
        win.destroy()


@pytest.mark.display
def test_ici_opens_with_the_overview_panel_focused(tk_root, monkeypatch):
    """Opening the ICI window focuses its overview/pulse panel, the first plot region."""
    focused = []
    real_focus_set = tk.Misc.focus_set
    monkeypatch.setattr(tk.Misc, "focus_set",
                        lambda self: (focused.append(self), real_focus_set(self))[1])
    win = _open_ici(tk_root, monkeypatch, {"cell": _ici_df()})
    try:
        assert win._primary_canvas_frame in focused, focused
    finally:
        win.destroy()


def _rk_bands(win):
    """Draw the R/k figure; return its target band, its white bands (px) and the right overhang."""
    canvas = win._canvas_right
    canvas.draw()
    renderer = canvas.get_renderer()
    width, height = canvas.figure.bbox.width, canvas.figure.bbox.height
    axes = (win.ax_r_chg, win.ax_r_dis, win.ax_k_chg, win.ax_k_dis)
    xaxes = [ax.xaxis.get_tightbbox(renderer) for ax in axes]
    titles = [ax.title.get_window_extent(renderer) for ax in axes]
    bands = [height - titles[0].y1, xaxes[-1].y0]
    bands += [xaxes[i - 1].y0 - titles[i].y1 for i in range(1, len(axes))]
    bands.append(min(ax.yaxis.get_tightbbox(renderer).x0 for ax in axes))
    return 0.028 * height, bands, max(box.x1 for box in xaxes) - width


@pytest.mark.display
def test_ici_rk_panels_keep_even_bands_after_resize_and_redraw(tk_root, monkeypatch):
    """Each resize and R/k redraw re-spaces the four R/k plots with even bands, none cut."""
    cell = _ici_df(rest_step=0.25, spread=0.002)   # R 0-2 ohm
    wide = _ici_df(rest_step=0.25, spread=0.2)     # R 0-200 ohm: wider tick labels
    win = _open_ici(tk_root, monkeypatch, {"cell": cell, "wide": wide})
    try:
        for width, height in ((420, 520), (640, 1000), (420, 520)):
            win._canvas_right.resize(SimpleNamespace(width=width, height=height))  # as Tk does
            for redraw in ("resize", "x axis", "source"):
                if redraw == "x axis":
                    win.rk_axis_bar.mass_var.set("3.5")
                    win.rk_axis_bar._set_xaxis("specific_capacity")
                elif redraw == "source":
                    _choose_source(win, "wide" if win.selector.source_var.get() == "cell"
                                   else "cell")
                    pump(win, 100)
                band, bands, overhang = _rk_bands(win)
                assert np.allclose(bands, band, atol=1.5), (width, height, redraw, band, bands)
                assert overhang <= 0, (width, height, redraw, overhang)
            win.rk_axis_bar._set_xaxis("voltage")
    finally:
        win.destroy()


@pytest.mark.display
def test_ici_spinboxes_ignore_text_that_is_not_a_number(tk_root, monkeypatch):
    """Text or 'inf' in Cycle or Pulse raises nothing; an arrow or a mass edit redraws once."""
    errors = []
    monkeypatch.setattr(tk_root, "report_callback_exception", lambda *a: errors.append(a[1]))
    win = _open_ici(tk_root, monkeypatch, {"operando": _ici_df()})
    try:
        for text in ("a", "inf"):
            for spin in (win.selector._cycle_spin, win.selector._pulse_spin):
                spin.delete(0, "end")
                spin.insert("end", text)
            win.selector._next_cycle()
            win.selector._prev_pulse()
            win.rk_axis_bar.mass_var.set("5")
            pump(win, 300)
        win.selector.cycle_var.set(1)
        win.selector.pulse_var.set(1)
        pump(win, 50)
        full, rk = [], []
        live, right = win._update_live_plots, win._update_right_plots
        win._update_live_plots = lambda: (full.append(1), live())
        win._update_right_plots = lambda *a: (rk.append(1), right(*a))
        win.selector._pulse_spin.invoke("buttonup")
        pump(win, 50)
        mass = next(w for w in win.rk_axis_bar.winfo_children()[0].winfo_children()
                    if isinstance(w, tk.Spinbox))
        counts = []
        for edit in (lambda: mass.invoke("buttonup"),
                     lambda: [mass.insert("end", ch) for ch in "2.5"],
                     lambda: (mass.insert("end", "1"), win._on_rk_axis_changed())):  # Return
            rk.clear()
            edit()
            pump(win, 300)
            counts.append(len(rk))
    finally:
        win.destroy()
    assert errors == [], errors
    assert len(full) == 1, f"one Pulse arrow click redrew {len(full)} times"
    assert counts == [1, 1, 1], f"R/k redraws per arrow click, typed mass, Return: {counts}"


@pytest.mark.display
def test_ici_names_the_selection_one_way(tk_root, monkeypatch):
    """Titles, legends, fit colour and no-fit hint name the selection one way (Pulse 0 is 1)."""
    win = _open_ici(tk_root, monkeypatch, {"operando": _ici_df(n_cycles=2)}, fitted=True)
    try:
        win.selector.pulse_var.set(0)
        selected = win.selector.pulse_var.get()
        win.selector._set_scope("all")
        titles = [win.ax_pulse.get_title(), win.ax_dv.get_title()]
        r2_legend = [t.get_text() for t in win.ax_r2.get_legend().get_texts()]
        rk_legend = [t.get_text() for t in win.ax_r_chg.get_legend().get_texts()]
        fit_colours = set()
        for phase in ("charge", "discharge"):
            win.selector._set_phase(phase)
            fit_colours.add(win.ax_dv.lines[0].get_color())
    finally:
        win.destroy()
    plain = _open_ici(tk_root, monkeypatch, {"operando": _ici_df()})
    try:
        hint = [t.get_text() for t in plain.ax_dv.texts]
    finally:
        plain.destroy()
    assert selected == 1, "a typed Pulse 0 must select pulse 1"
    assert all(t.startswith("Cycle 1  |  Charge  |  Pulse 1") for t in titles), titles
    assert r2_legend == ["Charge C1", "Discharge C1", "Charge C2", "Discharge C2"], r2_legend
    assert rk_legend == ["Charge C1", "Charge C2"], rk_legend
    assert len(fit_colours) == 1, fit_colours
    assert hint == ["Too few rest points in the regression window\n(increase Length)"], hint


@pytest.mark.display
def test_heatmap_window_renders_and_toggles(tk_root):
    """Heatmap window draws its mesh and genuinely switches to the time axis."""
    ex = np.arange(0.0, 660.0, 30.0)
    base = pd.Timestamp("2026-06-01 10:00:00")
    hm = HeatmapWindow(tk_root, build_hm_scans(), oned_arrays=build_oned_arrays(),
                       echem_arrays={"x": ex,
                                     "y": np.linspace(3.2, 3.6, len(ex)),
                                     "timestamps": [str(base + pd.Timedelta(seconds=s))
                                                    for s in ex]})
    try:
        hm.update()
        assert hm.ax_map.collections, "mesh not drawn"
        assert hm._y_values()[1] == "Time [h]", "must start on the time axis"
        hm._set_toggle(hm.yaxis_var, hm._y_btns, "scan")
        assert hm._y_values()[1] == "Scan number"
        hm._set_toggle(hm.yaxis_var, hm._y_btns, "time")
        hm._set_toggle(hm.scale_var, hm._scale_btns, "log")
        hm.update()
        assert hm._y_values()[1] == "Time [h]", \
            "time toggle must resolve to the real time axis, not the fallback"
        assert hm.ax_map.collections, "time + log toggles must survive"
        hm.zmin_var.set("1")
        hm.zmax_var.set("15")
        hm._replot()
        minor = hm.cax.yaxis.get_minor_ticks()
        assert minor and all(t.label1.get_fontsize() == 7 for t in minor), \
            "on-screen log colour-bar minor labels must match the majors"
        assert hm.ax_v.lines, "voltage track must be plotted"
        assert "correlated" in hm._scans_btns, "X-ray window shares the scan filter"
    finally:
        hm.destroy()


def test_neutron_bank_helpers_select_bank_and_axis():
    """Bank/axis helpers list banks in order and skip scans lacking the axis."""
    arrays = build_neutron_arrays()
    assert neutron_banks(arrays) == ["1", "3"]
    assert neutron_axes(arrays, "3") == ["tof", "d"]
    picked = neutron_bank_arrays(arrays, "3", "d")
    assert sorted(picked) == [1, 3], "scan 2 has no bank 3 d data"
    _x, matrix, used = build_xrd_matrix(picked)
    assert matrix.shape == (2, 40) and used == [1, 3]
    assert np.allclose(matrix[1], 60.0)
    _x, kept, used = correlated_rows(_x, matrix, used, {3, 7})
    assert used == [3] and np.allclose(kept, 60.0)
    assert correlated_rows(_x, matrix, [1, 3], set()) == (None, None, [])
    _x, kept, used = exclude_rows(_x, matrix, [1, 3], {1, 103})
    assert used == [3] and np.allclose(kept, 60.0)
    assert exclude_rows(_x, matrix, [1, 3], {1, 3}) == (None, None, [])


def test_parse_scan_range_shared_grammar():
    """One scan-range parser serves the GIF dialog, the GIF export and heatmap exclusions."""
    assert parse_scan_range("103, 1") == [1, 103]
    assert parse_scan_range("1-3,7") == [1, 2, 3, 7]
    assert parse_scan_range("1-5,10-12") == [1, 2, 3, 4, 5, 10, 11, 12]
    assert parse_scan_range("0-3,9", max_scan=5) == [1, 2, 3], "clamped to 1..max_scan"
    assert parse_scan_range("") == [] and parse_scan_range("x,1") == []


def test_scan_range_never_joins_space_separated_numbers():
    """Spaces only pad commas and hyphens: "1 3 5" or "1-5 7" is refused, never 135 or 1-57."""
    assert parse_scan_range(" 1 - 5 , 7 ", 494) == [1, 2, 3, 4, 5, 7]
    assert parse_scan_range("1 3 5", 494) == [] and parse_scan_range("1-5 7", 494) == []


def test_heatmap_mistyped_exclude_keeps_the_previous_exclusions(caplog):
    """Text that names no scan (a typo, or scans past the last) is warned in the terminal and keeps
    the previous exclusions and their text; a blank entry clears them."""
    entry = {"text": "1-5"}
    hm = SimpleNamespace(exclude_var=SimpleNamespace(get=lambda: entry["text"],
                                                     set=lambda t: entry.update(text=t)),
                         _max_scan=494, _excluded=set(), _exclude_text="",
                         _load_matrix=lambda: None, _replot=lambda: None,
                         _forget_zoom_history=lambda: None)
    HeatmapWindow._apply_exclusions(hm)
    for typo in ("1-5;7", "1 3 5", "5-1"):
        entry["text"] = typo
        with caplog.at_level("WARNING", logger="operaxn.heatmap"):
            HeatmapWindow._apply_exclusions(hm)
        assert hm._excluded == {1, 2, 3, 4, 5} and entry["text"] == "1-5", typo
        assert typo in caplog.text
    entry["text"] = "500-520"
    HeatmapWindow._apply_exclusions(hm)
    assert hm._excluded == {1, 2, 3, 4, 5} and entry["text"] == "1-5"
    assert "Exclude '500-520' names no scan from 1 to 494" in caplog.text, caplog.text
    entry["text"] = ""
    HeatmapWindow._apply_exclusions(hm)
    assert hm._excluded == set()


@pytest.mark.parametrize("typed, boxes, shown", [
    (("zmin", "5"), {"zmin": "1", "zmax": "2"}, {"zmin": "5", "zmax": ""}),
    (("zmax", "1"), {"zmin": "1", "zmax": "2"}, {"zmin": "", "zmax": "1"}),
    (("zmax", "abc"), {"zmin": "1", "zmax": "2"}, {"zmin": "1", "zmax": ""}),
    (("xmin", "inf"), {"xmax": "20"}, {"xmin": "", "xmax": "20"}),
    (("xmin", "30"), {"xmax": "20"}, {"xmin": "30", "xmax": "20"}),
    (("ceiling", "20"), {}, {"ceiling": 50.0}),
    (("ceiling", "150"), {}, {"ceiling": 100.0}),
    (("ceiling", "abc"), {}, {"ceiling": 100.0}),
    (("scale", "log"), {"zmin": "0", "zmax": "40"}, {"zmin": "", "zmax": "40", "3": ("", "8")}),
    (("zmin", "0"), {"scale": "log", "zmax": "40"}, {"zmin": "", "zmax": "40"}),
], ids=["min-not-below-max", "max-not-above-min", "text", "inf", "x-order-free", "ceiling-low",
        "ceiling-high", "ceiling-text", "log", "typed-in-log"])
def test_heatmap_boxes_show_the_values_in_use(typed, boxes, shown):
    """A committed heatmap box shows the value the map uses; Log clears every bank's bound <= 0."""
    texts = {"xmin": "", "xmax": "", "zmin": "", "zmax": "", "ceiling": "100.0",
             "scale": "linear", **boxes}
    box = {name: SimpleNamespace(get=lambda n=name: texts[n],
                                 set=lambda value, n=name: texts.update({n: value}))
           for name in texts}
    hm = SimpleNamespace(**{f"{name}_var": var for name, var in box.items()},
                         bank_var=None, xaxis_var=None, scans_var=None,
                         _bank_limits={"3": ("-2", "8")}, _replot=lambda: None,
                         _forget_zoom_history=lambda: None)
    name, text = typed
    if name == "scale":
        HeatmapWindow._set_toggle(hm, hm.scale_var, {}, text)
    elif name == "ceiling":
        texts[name] = text
        HeatmapWindow._commit_ceiling(hm)
    else:
        texts[name] = text
        HeatmapWindow._commit_range(hm, box[name])
    observed = {**texts, "3": hm._bank_limits["3"]}
    assert {key: observed[key] for key in shown} == shown


def _refusing_range(limit):
    """Return a range stand-in that fails at once instead of building past `limit`."""
    def refusing_range(*args):
        stop = args[-1] if len(args) > 1 else args[0]
        if stop > limit:
            raise AssertionError(f"range built up to {stop}")
        return builtins.range(*args)
    return refusing_range


def test_scan_range_is_clamped_before_it_is_built(monkeypatch):
    """'1-100000000' with 494 scans returns scans 1-494 without a range past 495."""
    monkeypatch.setattr(dialog_module, "range", _refusing_range(495), raising=False)
    assert dialog_module.parse_scan_range("1-100000000", max_scan=494) == list(range(1, 495))


def test_colormap_and_intensity_helpers():
    """Reverse flips the _r suffix; limits honour entries, ceiling and log."""
    assert colormap_name("viridis", True) == "viridis_r"
    assert colormap_name("YlGnBu_r", True) == "YlGnBu"
    assert colormap_name("inferno", False) == "inferno"
    m = np.arange(1.0, 101.0).reshape(10, 10)
    assert intensity_limits(m) == (1.0, 100.0)
    assert intensity_limits(m, ceiling=90.0)[1] == pytest.approx(90.1)
    assert intensity_limits(m, vmin=5.0, vmax=50.0) == (5.0, 50.0)
    assert intensity_limits(m, vmin=50.0, vmax=50.0) == (50.0, 100.0), \
        "a typed Max not above Min returns to automatic, never Min + 1"
    lo, hi = intensity_limits(np.zeros((2, 2)), log=True)
    assert lo > 0 and hi > lo
    lo, hi = intensity_limits(np.array([[0.0, -1.0, 1.0, 100.0]]), log=True)
    assert lo == pytest.approx(1.99) and hi == 100.0, \
        "log floor must come from the positive values, not the zeros"
    lo, hi = intensity_limits(np.array([[0.0, -1.0, 1.0, 100.0]]), vmin=0.0, log=True)
    assert lo == pytest.approx(1.99), "a typed bound <= 0 takes the log floor, never 1e-10"
    assert row_edges(np.array([1.0, 2.0, 3.0])) == (0.5, 3.5)
    assert cell_edges(np.array([1.0, 2.0, 4.0])).tolist() == [0.5, 1.5, 3.0, 5.0]
    assert cell_edges(np.array([5.0])).tolist() == [4.5, 5.5], \
        "a lone row needs a real height ('nearest' shading draws it zero high)"


def test_publication_figure_layout():
    """Export figure: voltage left sharing y, rasterised map, colour bar."""
    x = np.linspace(1.0, 5.0, 20)
    y = np.array([1.0, 2.0, 3.0])
    m = np.outer(y, x)
    fig = publication_figure(x, y, m, x_label="$d$-spacing [Å]", y_label="Time [h]",
                             cmap="viridis", vmin=1.0, vmax=15.0,
                             voltage=(np.array([3.0, 3.5, 4.0]), y), title="Bank 3")
    assert len(fig.axes) == 3
    ax_v, ax, cax = fig.axes
    assert ax_v.get_xlabel() == "Voltage [V]" and ax_v.get_ylabel() == "Time [h]"
    assert ax.get_xlabel() == "$d$-spacing [Å]" and ax.get_title() == "Bank 3"
    assert ax.collections[0].get_rasterized(), "map must rasterise for PDF/SVG"
    assert ax.get_shared_y_axes().joined(ax, ax_v)
    assert ax.get_ylim() == (0.5, 3.5)
    assert cax.get_ylabel() == "Intensity [arb. units]"
    for text in (ax.xaxis.label, ax.title, ax_v.xaxis.label, ax_v.yaxis.label, cax.yaxis.label):
        assert text.get_color() == "black", "export text must be black"
    no_v = publication_figure(x, y, m, x_label=r"2$\theta$ [°]", y_label="Scan number",
                              cmap="inferno", vmin=1.0, vmax=15.0, log=True,
                              x_range=(2.0, 4.0))
    assert len(no_v.axes) == 2 and no_v.axes[0].get_xlim() == (2.0, 4.0)
    minor = no_v.axes[1].yaxis.get_minor_ticks()
    assert minor and all(t.label1.get_fontsize() == 9 for t in minor), \
        "log colour-bar minor labels must match the majors"
    assert no_v.axes[0].get_ylabel() == "Scan number"
    one = publication_figure(x, y[:1], m[:1], x_label=r"2$\theta$ [°]", y_label="Scan number",
                             cmap="viridis", vmin=1.0, vmax=15.0)
    rows = one.axes[0].collections[0].get_coordinates()[..., 1]
    assert rows.min() == 0.5 and rows.max() == 1.5, "a single row must fill its cell"


def test_heatmap_rows_are_timed_from_the_earliest_echem_point():
    """Rows share the echem curve's origin, its earliest point, even when the record is unsorted."""
    unsorted = SimpleNamespace(_echem_arrays={
        "timestamps": ["2026-06-01 10:00:20", "2026-06-01 10:00:00"]})
    assert HeatmapWindow._echem_reference(unsorted, "absolute") == \
        pd.Timestamp("2026-06-01 10:00:00")
    assert HeatmapWindow._echem_reference(unsorted, "relative") is None


@pytest.mark.display
def test_heatmap_axis_fallback_clears_x_range(tk_root):
    """A bank without d moves the window to TOF and drops the d range with it."""
    arrays = build_neutron_arrays()
    for banks in arrays.values():
        banks["0"] = {"tof": banks["1"]["tof"]}
    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=arrays,
                       dspacing=True, x_range=(1.0, 4.0))
    try:
        hm.update()
        assert hm.bank_var.get() == "0" and hm.xaxis_var.get() == "tof"
        assert hm.xmin_var.get() == "", "a d range must not seed a TOF map"
        assert str(hm._x_btns["d"]["state"]) == "disabled", "bank 0 has no d axis"
        hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
        hm._set_toggle(hm.xaxis_var, hm._x_btns, "d")
        hm.xmin_var.set("1.5")
        hm.xmax_var.set("3")
        hm._set_toggle(hm.bank_var, hm._bank_btns, "0")
        assert hm.xaxis_var.get() == "tof" and hm.xmin_var.get() == "", \
            "a forced axis change must clear the x range (units changed)"
    finally:
        hm.destroy()


@pytest.mark.display
def test_heatmap_defaults_to_scan_without_times(tk_root):
    """Without scan times the heatmap starts on scan numbers and disables the Time toggle."""
    hm = HeatmapWindow(tk_root, [{"scan_num": n} for n in (1, 2, 3)],
                       oned_arrays=build_oned_arrays())
    try:
        assert hm.yaxis_var.get() == "scan" and hm._y_values()[1] == "Scan number"
        assert str(hm._y_btns["time"]["state"]) == "disabled"
        assert list(hm._y_btns) == ["time", "scan"], "the buttons keep one order"
    finally:
        hm.destroy()


@pytest.mark.display
def test_heatmap_window_neutron_banks_and_export(tk_root, tmp_path, monkeypatch):
    """Neutron window: per-bank maps, TOF/d toggle, ranges, colour map, export."""
    scans = build_hm_scans()
    for scan, volts in zip(scans, (3.2, None, 3.6)):
        if volts is not None:
            scan["echem"] = volts
    hm = HeatmapWindow(tk_root, scans, neutron_arrays=build_neutron_arrays(),
                       dspacing=True, x_range=(1.0, 4.0))
    try:
        hm.update()
        assert hm.is_neutron and list(hm._bank_btns) == ["1", "3"]
        assert hm.yaxis_var.get() == "time" and list(hm._y_btns) == ["time", "scan"], \
            "Time is the default with scan times, listed first"
        assert hm.xaxis_var.get() == "d", "d spacing preference must seed the axis"
        assert hm.ax_map.get_xlabel() == "$d$-spacing [Å]"
        assert hm.ax_map.get_title() == "Bank 1"
        assert hm.ax_map.get_xlim() == (1.0, 4.0), "x range must seed the entries"
        assert hm.ax_v.lines, "per-scan voltage fallback must draw"

        hm._set_toggle(hm.xaxis_var, hm._x_btns, "tof")
        assert hm.ax_map.get_xlabel() == "Time of flight [μs]"
        assert hm.xmin_var.get() == "" and hm.xmax_var.get() == "", \
            "an axis change must clear the x range (units changed)"
        hm._set_toggle(hm.bank_var, hm._bank_btns, "3")
        assert hm.ax_map.get_title() == "Bank 3" and hm._scan_nums == [1, 2, 3]
        hm._set_toggle(hm.xaxis_var, hm._x_btns, "d")
        assert hm._scan_nums == [1, 3], "scan 2 has no bank 3 d pattern"

        hm.cmap_var.set("inferno")
        hm._toggle_reverse()
        assert hm.ax_map.collections[-1].get_cmap().name == "inferno_r"
        hm.zmin_var.set("25")
        hm.zmax_var.set("55")
        hm._replot()
        assert hm.ax_map.collections[-1].get_clim() == (25.0, 55.0)
        hm._reset_ranges()
        assert hm.ax_map.collections[-1].get_clim() == (20.0, 60.0)

        hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
        assert hm._scan_nums == [1, 2, 3]
        hm._set_toggle(hm.scans_var, hm._scans_btns, "correlated")
        assert hm._scan_nums == [1, 3], "scan 2 has no correlated echem"
        hm._set_toggle(hm.yaxis_var, hm._y_btns, "scan")
        assert hm.render_figure("1").axes[1].get_ylim() == (0.0, 4.0), \
            "export must use the filtered rows"
        hm._set_toggle(hm.scans_var, hm._scans_btns, "all")
        assert hm._scan_nums == [1, 2, 3]
        hm.exclude_var.set("1, 103")
        hm._apply_exclusions()
        assert hm._scan_nums == [2, 3], "excluded scans must drop out"
        hm._canvas.toolbar.push_current()
        hm.exclude_var.set("1-2, 103")
        hm._apply_exclusions()
        assert hm._scan_nums == [3]
        assert hm._canvas.toolbar._nav_stack() is None, "Home must not return to other rows"
        hm._canvas.toolbar.push_current()
        hm._replot()
        assert hm._canvas.toolbar._nav_stack() is not None, \
            "a range-entry replot (also fired by the focus loss a zoom click causes) keeps Home"
        rows = hm.ax_map.collections[-1].get_coordinates()[..., 1]
        assert (rows.min(), rows.max()) == (2.5, 3.5), "a lone row must fill its cell on screen"
        hm.exclude_var.set("1, 103")
        hm._apply_exclusions()
        hm._set_toggle(hm.scans_var, hm._scans_btns, "correlated")
        assert hm._scan_nums == [3], "filters combine"
        hm.exclude_var.set("")
        hm._apply_exclusions()
        hm._set_toggle(hm.scans_var, hm._scans_btns, "all")
        assert hm._scan_nums == [1, 2, 3]

        single = tmp_path / "bank3.pdf"
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(single))
        monkeypatch.setattr(messagebox, "showinfo", lambda *_a, **_k: None)
        hm._export_figure()
        assert single.exists() and single.stat().st_size > 0
        via_toolbar = tmp_path / "toolbar.svg"
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(via_toolbar))
        exported = []
        save = hm.save_figure
        monkeypatch.setattr(hm, "save_figure",
                            lambda path, *a, **k: exported.append(path) or save(path, *a, **k))
        # matplotlib's own save would also write the file, so check the route taken
        hm._canvas.toolbar._buttons["Save"].invoke()
        assert exported == [str(via_toolbar)] and via_toolbar.exists(), \
            "toolbar save must use the publication export"
        base = tmp_path / "maps.png"
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(base))
        hm._export_all_banks()
        assert (tmp_path / "maps_bank1.png").exists()
        assert (tmp_path / "maps_bank3.png").exists()
    finally:
        hm.destroy()


def _buttons(widget):
    """Return the text of every Tk button inside widget, depth first."""
    found = [w.cget("text") for w in widget.winfo_children() if isinstance(w, tk.Button)]
    return found + [t for w in widget.winfo_children() for t in _buttons(w)]


def _clim(hm, bank):
    """Return the colour limits a heatmap window exports bank's map with."""
    fig = hm.render_figure(bank)
    return next(c for ax in fig.axes for c in ax.collections).get_clim()


def _export_title(fig):
    """Return a rendered heatmap figure's map title (the colour bar is its last axes)."""
    return fig.axes[-2].get_title()


@pytest.mark.display
def test_heatmap_intensity_entries_follow_the_bank(tk_root):
    """Typed intensity limits stay with their bank, on screen and in Export all banks."""
    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=build_neutron_arrays())
    try:
        hm.update()
        assert hm.bank_var.get() == "1" and hm.xaxis_var.get() == "tof"
        hm.zmin_var.set("1")
        hm.zmax_var.set("2")
        hm._replot()
        assert hm.ax_map.collections[-1].get_clim() == (1.0, 2.0)
        hm._set_toggle(hm.bank_var, hm._bank_btns, "3")
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("", ""), \
            "bank 1 counts must not carry over to bank 3"
        assert hm.ax_map.collections[-1].get_clim() == (2.0, 6.0), "bank 3 auto range"
        hm.zmin_var.set("3")
        hm.zmax_var.set("4")

        assert _clim(hm, "3") == (3.0, 4.0), "the shown bank exports with its typed range"
        assert _clim(hm, "1") == (1.0, 2.0), "another bank exports with the range typed on it"
        hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("1", "2"), "bank 1's range comes back"
        assert hm.ax_map.collections[-1].get_clim() == (1.0, 2.0)
        hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("1", "2"), "re-clicking the bank keeps it"
        hm._set_toggle(hm.xaxis_var, hm._x_btns, "d")
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("1", "2"), "the x axis keeps it"
        assert _clim(hm, "3") == (3.0, 4.0)
        hm._reset_ranges()
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("", "")
        assert _clim(hm, "3") == (20.0, 60.0), "Reset returns every bank to its auto range"
    finally:
        hm.destroy()


def test_title_text_keeps_mathtext_and_shows_invalid_math_as_typed():
    """Typed titles are trimmed; valid $...$ stays mathtext, invalid $...$ is escaped and draws."""
    assert title_text("  Bank 1 ") == "Bank 1"
    assert title_text("LiNiO$_2$, cycle 1") == "LiNiO$_2$, cycle 1"
    assert title_text("Cost $5") == "Cost $5", "a lone $ is plain text already"
    shown = title_text("$x^{2$ and \\$3")
    assert shown == "\\$x^{2\\$ and \\$3"
    fig = Figure()
    FigureCanvasAgg(fig)
    fig.add_subplot().set_title(shown)
    fig.canvas.draw()  # the text as typed raises ValueError here


def test_title_text_checks_each_line_as_drawn():
    """A pasted multi-line title is checked line by line, as matplotlib parses it, and draws."""
    assert title_text("LiNiO$_2$\ncycle 1") == "LiNiO$_2$\ncycle 1"
    shown = title_text("x$$\n$ cell")
    assert shown == "x\\$\\$\n\\$ cell", "three $ in all, but the first line is invalid math"
    fig = Figure()
    FigureCanvasAgg(fig)
    fig.add_subplot().set_title(shown)
    fig.canvas.draw()


@pytest.mark.display
def test_heatmap_title_is_editable_per_bank(tk_root, tmp_path, monkeypatch, caplog):
    """Title starts on the default title, titles the map and every export, and keeps to its bank."""
    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=build_neutron_arrays())
    try:
        hm.update()
        assert hm.title_var.get() == "Bank 1" and hm.ax_map.get_title() == "Bank 1"
        assert _export_title(hm.render_figure("3")) == "Bank 3", \
            "every other bank keeps its 'Bank N' default"
        hm.title_var.set(" LiNiO$_2$, 90° bank ")
        hm._apply_title()
        assert hm.ax_map.get_title() == "LiNiO$_2$, 90° bank", "the typed title is drawn, trimmed"
        assert _export_title(hm.render_figure()) == "LiNiO$_2$, 90° bank"
        hm._set_toggle(hm.bank_var, hm._bank_btns, "3")
        assert hm.title_var.get() == "Bank 3" and hm.ax_map.get_title() == "Bank 3", \
            "bank 1's title must not carry over to bank 3"
        assert _export_title(hm.render_figure("1")) == "LiNiO$_2$, 90° bank", "bank 1 keeps its own"
        hm.title_var.set("")
        hm._apply_title()
        assert hm.ax_map.get_title() == "" and _export_title(hm.render_figure()) == "", \
            "a cleared title draws none"
        hm._set_toggle(hm.xaxis_var, hm._x_btns, "d")
        hm._reset_ranges()
        assert hm.title_var.get() == "" and hm.ax_map.get_title() == "", \
            "the x axis and Reset keep the title"

        drawn = {}
        render = hm.render_figure

        def recording(bank=None, *args):
            fig = render(bank, *args)
            drawn[bank] = _export_title(fig)
            return fig

        monkeypatch.setattr(hm, "render_figure", recording)
        base = tmp_path / "maps.png"
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(base))
        monkeypatch.setattr(messagebox, "showinfo", lambda *_a, **_k: None)
        hm._export_all_banks()
        assert drawn == {"1": "LiNiO$_2$, 90° bank", "3": ""}, "each bank exports its own title"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["maps_bank1.png", "maps_bank3.png"]
        hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
        assert hm.title_var.get() == " LiNiO$_2$, 90° bank ", "bank 1's text comes back as typed"

        hm.title_var.set("$x^{2$")
        with caplog.at_level("WARNING", logger="operaxn.heatmap"):
            hm._apply_title()
            hm._apply_title()
        assert hm.ax_map.get_title() == "\\$x^{2\\$", "text that is no mathtext is shown as typed"
        assert caplog.text.count("not valid mathtext") == 1, "warned once, in the terminal"
        hm._canvas.draw()
    finally:
        hm.destroy()
    xray = HeatmapWindow(tk_root, build_hm_scans(), oned_arrays=build_oned_arrays())
    try:
        assert xray.title_var.get() == "" and xray.ax_map.get_title() == "", "X-ray: no title"
        xray.title_var.set("Cycle 1")
        xray._apply_title()
        assert xray.ax_map.get_title() == _export_title(xray.render_figure()) == "Cycle 1"
    finally:
        xray.destroy()


@pytest.mark.display
def test_heatmap_title_apply_all_gives_every_bank_the_shown_title(tk_root, monkeypatch):
    """The Title's Apply All asks first, then every bank draws the shown title until retyped."""
    asked = []
    answer = [False]
    monkeypatch.setattr(messagebox, "askyesno",
                        lambda *a, **_k: asked.append(a) or answer[0])

    def title(bank):
        return _export_title(hm.render_figure(bank))

    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=build_neutron_arrays())
    try:
        hm.update()
        assert _buttons(hm).count("Apply All") == 2, "one for the intensity, one for the title"
        hm.title_var.set(" LiNiO$_2$ cell ")
        hm._apply_title_to_all_banks()
        assert asked == [("Confirm", "Apply title 'LiNiO$_2$ cell' to all banks?")]
        assert title("3") == "Bank 3", "No must leave every title as it was"
        answer[0] = True
        hm._apply_title_to_all_banks()
        assert hm.ax_map.get_title() == title("1") == title("3") == "LiNiO$_2$ cell"
        hm._set_toggle(hm.bank_var, hm._bank_btns, "3")
        assert hm.title_var.get() == " LiNiO$_2$ cell ", "the entry shows it"
        hm.title_var.set("Bank 3 only")
        hm._apply_title()
        assert title("3") == "Bank 3 only" and title("1") == "LiNiO$_2$ cell", \
            "a later edit changes only its own bank"
        hm.title_var.set("")
        hm._apply_title_to_all_banks()
        assert asked[-1] == ("Confirm", "Clear the title of all banks?")
        assert hm.ax_map.get_title() == title("1") == title("3") == "", "every title cleared"
    finally:
        hm.destroy()
    xray = HeatmapWindow(tk_root, build_hm_scans(), oned_arrays=build_oned_arrays())
    try:
        xray.update()
        assert "Apply All" not in _buttons(xray), "X-ray has one map"
    finally:
        xray.destroy()


@pytest.mark.display
def test_heatmap_invalid_title_is_warned_once_on_every_path(tk_root, tmp_path, monkeypatch, caplog):
    """A title that is not valid mathtext is warned once per text, however it reaches the figure."""
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: str(tmp_path / "maps.png"))
    monkeypatch.setattr(messagebox, "showinfo", lambda *_a, **_k: None)

    def warned():
        return [r.getMessage() for r in caplog.records if "not valid mathtext" in r.getMessage()]

    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=build_neutron_arrays())
    try:
        hm.update()
        with caplog.at_level("WARNING", logger="operaxn.heatmap"):
            hm.title_var.set("$x^{2$")
            hm._export_figure()  # typed, then Export: no Return or FocusOut
            assert warned() == ["Title '$x^{2$' is not valid mathtext; it is shown as typed"]
            hm._apply_title()
            hm._set_toggle(hm.bank_var, hm._bank_btns, "3")
            hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
            assert len(warned()) == 1, "the same text is never warned twice"
            hm.title_var.set("$y^{3$")
            hm._set_toggle(hm.bank_var, hm._bank_btns, "3")  # typed, then a Bank click
            hm._export_all_banks()
            hm._set_toggle(hm.bank_var, hm._bank_btns, "1")
        assert warned()[1:] == ["Title '$y^{3$' is not valid mathtext; it is shown as typed"], \
            warned()
    finally:
        hm.destroy()


@pytest.mark.display
def test_heatmap_apply_all_shares_the_shown_banks_range(tk_root, monkeypatch):
    """Apply All asks first, then every bank uses the shown map's range until edited or Reset."""
    asked = []
    answer = [False]
    monkeypatch.setattr(messagebox, "askyesno",
                        lambda *a, **_k: asked.append(a) or answer[0])

    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=build_neutron_arrays())
    try:
        hm.update()
        assert "Apply All" in _buttons(hm)
        hm.zmax_var.set("2")
        hm._apply_to_all_banks()
        assert asked == [("Confirm", "Apply intensity range 1 - 2 to all banks?")], \
            "the shown map's range (typed Max, auto Min) is offered"
        assert _clim(hm, "3") == (2.0, 6.0), "No must leave every range as it was"
        answer[0] = True
        hm._apply_to_all_banks()
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("1", "2"), "the entries show it"
        assert _clim(hm, "1") == _clim(hm, "3") == (1.0, 2.0), "every bank shares one scale"
        hm._set_toggle(hm.bank_var, hm._bank_btns, "3")
        assert (hm.zmin_var.get(), hm.zmax_var.get()) == ("1", "2")
        hm.zmax_var.set("5")
        assert _clim(hm, "3") == (1.0, 5.0) and _clim(hm, "1") == (1.0, 2.0), \
            "a later edit changes only its own bank"
        hm._reset_ranges()
        assert _clim(hm, "1") == (1.0, 3.0) and _clim(hm, "3") == (2.0, 6.0), \
            "Reset is the way back"
    finally:
        hm.destroy()
    xray = HeatmapWindow(tk_root, build_hm_scans(), oned_arrays=build_oned_arrays())
    try:
        assert "Apply All" not in _buttons(xray), "X-ray maps have no banks to share a range"
    finally:
        xray.destroy()


@pytest.mark.display
def test_heatmap_redraws_free_the_maps_they_replace(tk_root, monkeypatch):
    """Every kind of heatmap redraw frees the map, and with it the colour bar, that it replaces."""
    monkeypatch.setattr(messagebox, "askyesno", lambda *_a, **_k: True)
    hm = HeatmapWindow(tk_root, build_hm_scans(), neutron_arrays=build_neutron_arrays())
    # A colour bar holds its map, so a freed map means a freed colour bar
    maps = []
    try:
        hm.update()
        maps.append(weakref.ref(hm.ax_map.collections[-1]))
        for _ in range(3):
            for redraw in (lambda: hm._set_toggle(hm.bank_var, hm._bank_btns, "3"),
                           lambda: hm._set_toggle(hm.xaxis_var, hm._x_btns, "d"),
                           lambda: hm._set_toggle(hm.scale_var, hm._scale_btns, "log"),
                           hm._toggle_reverse, hm._apply_to_all_banks, hm._reset_ranges,
                           lambda: hm._set_toggle(hm.scale_var, hm._scale_btns, "linear"),
                           lambda: hm._set_toggle(hm.bank_var, hm._bank_btns, "1"),
                           lambda: hm._set_toggle(hm.xaxis_var, hm._x_btns, "tof")):
                redraw()
                maps.append(weakref.ref(hm.ax_map.collections[-1]))
        gc.collect()
        alive = [i for i, ref in enumerate(maps) if ref() is not None]
        assert alive == [len(maps) - 1], \
            f"{len(alive)} of {len(maps)} maps still in memory; only the shown one may be"
    finally:
        hm.destroy()


@pytest.mark.display
def test_heatmap_exclude_box_is_clamped(tk_root, monkeypatch):
    """The heatmap's Exclude entry is clamped to the highest scan, so a typo returns at once."""
    top = max(s["scan_num"] for s in build_hm_scans())
    hm = HeatmapWindow(tk_root, build_hm_scans(), oned_arrays=build_oned_arrays())
    try:
        monkeypatch.setattr(dialog_module, "range", _refusing_range(top + 1), raising=False)
        hm.exclude_var.set("2-100000000")
        hm._apply_exclusions()
        assert hm._excluded == set(range(2, top + 1))
    finally:
        hm.destroy()


@pytest.mark.display
def test_heatmap_replot_timers_never_outlive_reset_or_close(tk_root, monkeypatch):
    """Reset redraws at once and cancels the replot its ceiling reset queued; a replot still queued
    when the window closes is cancelled with it, as its callback goes with the window."""
    jobs = []
    queue = HeatmapWindow._queue_replot

    def recording_queue(self, *args):
        queue(self, *args)
        jobs.append(self._replot_job)

    monkeypatch.setattr(HeatmapWindow, "_queue_replot", recording_queue)
    hm = HeatmapWindow(tk_root, build_hm_scans(), oned_arrays=build_oned_arrays())
    try:
        hm.ceiling_var.set(50.0)
        hm._reset_ranges()  # sets the ceiling back to 100, which queues a replot
        assert len(jobs) == 2 and not set(jobs) & pending_after_ids(tk_root), \
            "Reset left a queued replot"
        hm.ceiling_var.set(99.0)
        hm.update_idletasks()  # the redraws' own idle draws run; the queued replot (250 ms) waits
        assert jobs[-1] in pending_after_ids(tk_root)
    finally:
        errors = background_errors(tk_root, hm.destroy)
    assert jobs[-1] not in pending_after_ids(tk_root), "a queued replot outlived the window"
    assert not errors, errors


@pytest.mark.display
def test_capacity_timers_never_outlive_the_window(tk_root, monkeypatch):
    """A mass redraw and a source swap still queued when Capacity closes are cancelled with it."""
    monkeypatch.setattr(capacity_module, "maximise_window", lambda win: win.withdraw())
    win = CapacityWindow(tk_root, {"operando": build_cycling_df()})
    try:
        win._queue_mass_replot()
        win._queue_source_change("operando")
        jobs = {win._mass_job, win._source_job}
        assert None not in jobs and jobs <= pending_after_ids(tk_root)
    finally:
        errors = background_errors(tk_root, win.destroy)
    assert not jobs & pending_after_ids(tk_root), "a queued Capacity job outlived the window"
    assert not errors, errors


def _plot_controls_jobs(root, monkeypatch):
    """Queue the plot controls' debounced updates; return their ids and the panel."""
    controls = gui_module.PlotControls(root, gui_module.VisualiserConfig(), lambda *a: None,
                                       lambda *a: None, lambda *a: None)
    controls.intensity_debouncer.schedule(lambda: None)
    controls.echem_debouncer.schedule(lambda: None)
    return {controls.intensity_debouncer._after_id, controls.echem_debouncer._after_id}, controls


def _ici_jobs(root, monkeypatch):
    """Queue every edit the ICI window debounces; return their ids and the window."""
    win = _open_ici(root, monkeypatch, {"cell": _ici_df(), "other": _ici_df()})
    before = pending_after_ids(root)
    win.reg_bar.start_var.set(1.0)
    win.reg_bar.max_rest_var.set(299.0)
    win.rk_axis_bar.mass_var.set("3")
    win.update_idletasks()  # matplotlib's own idle draws go first
    _choose_source(win, "other")
    return pending_after_ids(root) - before, win


def _figure_sync_jobs(root, monkeypatch):
    """Queue embed_figure's re-sync and watch_figure_sync's watchdog; return ids and window."""
    before = pending_after_ids(root)
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.geometry("200x150+20000+20000")  # mapped off-screen: an unmapped frame gets no <Configure>
    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True)
    canvas = embed_figure(Figure(), frame)
    watch_figure_sync(win, (canvas,))
    win.update()  # maps the window: the frame's <Configure> queues a re-sync
    jobs = pending_after_ids(root) - before
    assert canvas._operaxn_sync_job in jobs and len(jobs) == 2, jobs
    return jobs, win


@pytest.mark.display
@pytest.mark.parametrize("queue", [_plot_controls_jobs, _ici_jobs, _figure_sync_jobs],
                         ids=["plot controls", "ICI window", "shared figure timers"])
def test_queued_jobs_never_outlive_their_window(tk_root, monkeypatch, queue):
    """Debounced updates and timers still queued when their window goes are cancelled with it."""
    jobs, owner = queue(tk_root, monkeypatch)
    assert jobs and None not in jobs and jobs <= pending_after_ids(tk_root), jobs
    errors = background_errors(tk_root, owner.destroy)
    assert not jobs & pending_after_ids(tk_root), "a queued job outlived its window"
    assert not errors, errors


@pytest.mark.display
@pytest.mark.parametrize("close", ["close button", "Clear All"])
@pytest.mark.parametrize("which", ["heatmap", "ici", "capacity"])
def test_closing_an_analysis_window_frees_it(tk_root, monkeypatch, which, close):
    """A Heatmap, ICI or Capacity window closed by its close button or by Clear All is freed at once;
    while it is open, a second click lifts it."""
    for module in (gui_module, capacity_module, heatmap_module, ici_module):
        monkeypatch.setattr(module, "maximise_window", lambda win: win.withdraw())
    monkeypatch.setattr(messagebox, "askyesno", lambda *_a, **_k: True)
    master = tk.Toplevel(tk_root)
    master.withdraw()
    try:
        app = OPERAXN(master=master)
        app.state.scans = build_hm_scans()
        app.state.oned_arrays = build_oned_arrays()
        app.state.echem_df = build_cycling_df()
        open_window = {"heatmap": app._open_heatmap, "ici": app._open_ici_window,
                       "capacity": app._open_capacity_window}[which]
        open_window()
        win = getattr(app, f"_{which}_window")
        pump(win, 300)
        open_window()
        assert getattr(app, f"_{which}_window") is win, "a second click must lift the open window"
        ref = weakref.ref(win)
        if close == "close button":
            win.tk.eval(win.protocol("WM_DELETE_WINDOW"))
        else:
            app._clear_all()
        del win
        gc.collect()
        assert ref() is None, f"{which} window closed by {close} is still referenced"
    finally:
        master.destroy()
