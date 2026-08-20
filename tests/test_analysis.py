"""Echem analysis checks: capacity/ICI/heatmap maths and GUI wiring."""
import inspect
import os
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

pytestmark = pytest.mark.gui

tk = pytest.importorskip("tkinter")

import operaxn
import operaxn.gui as gui_module
import operaxn.output as output_module
from operaxn.capacity import (
    _assign_time, classify_phases, assign_cycles, compute_capacity,
    plot_capacity_vs_voltage, plot_time_vs_voltage,
)
from operaxn.gui import ButtonPanel, FileListPanel, OPERAXN
from operaxn.heatmap import (
    HeatmapWindow, build_xrd_matrix, echem_scan_curve, scan_positions_seconds,
)
from operaxn.ici import (
    ICIWindow, PLOT_BG, _SelectorStrip, _compute_all_pulses, _compute_fit,
    _detect_pulses, _style_ax,
)
from operaxn.input import get_standard_echem

R_TRUE, K_TRUE, I_MA = 50.0, 5.0, 2.0
I_A = I_MA / 1000.0
V0 = 3.8


def build_cycling_df(n_cycles=2, active_s=10, rest_s=5):
    """Constant-current blocks: +1 mA charge, rest, -1 mA discharge, rest."""
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
    """Exact dV = -I(R + k*sqrt(dt)) relaxation: one 5 s pulse then 40 rest points."""
    active_t = np.arange(5, dtype=float)
    rest_t = 4.0 + 0.25 * np.arange(1, 41)
    rest_v = V0 - I_A * (R_TRUE + K_TRUE * np.sqrt(rest_t - rest_t[0]))
    return pd.DataFrame({
        "t_s": np.concatenate([active_t, rest_t]),
        "current": np.concatenate([np.full(5, I_MA), np.zeros(40)]),
        "echem_data": np.concatenate([np.full(5, V0), rest_v]),
    })


def build_oned_arrays():
    """Three flat scans on a 50-point grid; scan 3 uses a coarser grid."""
    hm_x = np.linspace(10, 80, 50)
    return {
        1: {"x": hm_x, "y": np.full(50, 1.0), "timestamp": "2026-06-01 10:00:00", "echem": 3.2},
        2: {"x": hm_x, "y": np.full(50, 2.0), "timestamp": "2026-06-01 10:05:00", "echem": 3.4},
        3: {"x": np.linspace(10, 80, 25), "y": np.full(25, 3.0),
            "timestamp": "2026-06-01 10:10:00", "echem": 3.6},
    }


def build_hm_scans():
    """Three scans with display timestamps and correlation midpoints 30 s later."""
    return [
        {"scan_num": 1, "timestamp": "2026-06-01 10:00:00",
         "timestamp_for_correlation": pd.Timestamp("2026-06-01 10:00:30")},
        {"scan_num": 2, "timestamp": "2026-06-01 10:05:00",
         "timestamp_for_correlation": pd.Timestamp("2026-06-01 10:05:30")},
        {"scan_num": 3, "timestamp": "2026-06-01 10:10:00",
         "timestamp_for_correlation": pd.Timestamp("2026-06-01 10:10:30")},
    ]


def test_classify_phases_rests_inherit_preceding_phase():
    """Rests are folded into the preceding charge/discharge phase."""
    charge_df, discharge_df = classify_phases(build_cycling_df())
    assert len(charge_df) == 2 * 15, f"got {len(charge_df)}"
    assert len(discharge_df) == 2 * 15, f"got {len(discharge_df)}"


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
    assert ax_cap.get_xlabel() == "Specific Capacity (mAh/g)"


def test_no_legend_warning_when_nothing_plotted():
    """Discharge-only data never starts a cycle; no legend warning is emitted."""
    discharge_only = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-06-01 10:00:00") + pd.Timedelta(seconds=s)
                      for s in range(6)],
        "echem_data": [3.5] * 6,
        "current": [-1.0] * 6,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig = Figure()
        ax1, ax2 = fig.subplots(1, 2)
        plot_time_vs_voltage(ax1, discharge_only)
        plot_capacity_vs_voltage(ax2, discharge_only)


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
            seg_current.append(2.0); seg_t.append(t); t += 1.0
        for _ in range(rest_len):
            seg_current.append(0.0); seg_t.append(t); t += 1.0
    pulse_df = pd.DataFrame({
        "t_s": seg_t, "current": seg_current, "echem_data": [3.8] * len(seg_t),
    })
    pulses = _detect_pulses(pulse_df, max_rest=300.0)
    assert len(pulses) == 3, f"got {len(pulses)}"
    assert bool(pulses[0]["pulse_mask"].sum() == 5 and pulses[0]["relax_mask"].sum() == 10), \
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
    """A regression window with fewer than 2 points yields no fit."""
    fit_pulses = _detect_pulses(fit_df, max_rest=300.0)
    assert _compute_fit(fit_df, fit_pulses[0], r1_start=0.5, r1_length=0.1) is None


def test_compute_all_pulses_returns_one_indexed_fit_per_pulse(fit_df):
    """_compute_all_pulses yields one fit per detected pulse, tagged pulse_idx."""
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


def test_button_panel_configuration():
    """Analysis buttons exist, precede export, and use single-codepoint emoji."""
    button_keys = [key for key, _, _, _ in ButtonPanel.BUTTON_CONFIGS]
    assert "capacity_plot" in button_keys
    assert "ici_analysis" in button_keys
    assert "heatmap" in button_keys
    assert button_keys.index("capacity_plot") < button_keys.index("export_plots") \
        and button_keys.index("ici_analysis") < button_keys.index("export_plots")
    assert all("️" not in text for _, text, _, _ in ButtonPanel.BUTTON_CONFIGS)


@pytest.mark.parametrize("attr", [
    "_open_capacity_window",
    "_open_ici_window",
    "_append_standard_echem_async",
    "_echem_sources",
    "_has_analysis_current",
    "_open_heatmap",
    "_choose_additional_echem",
])
def test_app_handler_exists(attr):
    """OPERAXN exposes the analysis and echem handlers the buttons rely on."""
    assert hasattr(OPERAXN, attr)


def test_ici_window_api():
    """ICIWindow takes source selection/upload hooks; regression edits
    auto-apply, with apply-all as the only manual action."""
    params = inspect.signature(ICIWindow.__init__).parameters
    assert "echem_sources" in params
    assert "on_add_source" in params
    assert hasattr(ICIWindow, "_on_apply_all")
    assert hasattr(ICIWindow, "_apply_regression_change") \
        and not hasattr(ICIWindow, "_on_apply"), "regression edits must auto-apply"
    assert hasattr(ICIWindow, "_focus_primary_panel")


def test_standard_echem_accessor_and_package_export():
    """get_standard_echem returns a list and ICIWindow is exported from the package."""
    assert isinstance(get_standard_echem(), list)
    assert hasattr(operaxn, "ICIWindow")


def test_source_switches_wait_for_dropdown_close():
    """ICI and Capacity source switches defer through after_idle (pins
    the implementation source; rename-sensitive)."""
    assert "after_idle" in inspect.getsource(ICIWindow._on_source_change)
    assert "after_idle" in inspect.getsource(OPERAXN._open_capacity_window)


def test_ici_fit_plot_has_no_redundant_overlays():
    """The ICI fit plot draws neither a top time axis nor a regression
    highlight overlay (pins the _plot_ici_fit source; rename-sensitive)."""
    ici_fit_source = inspect.getsource(ICIWindow._plot_ici_fit)
    assert "twiny" not in ici_fit_source
    assert "axvspan" not in ici_fit_source


def test_pulse_legend_position_follows_phase():
    """Pulse zoom legend switches position between charge and discharge
    (pins the _plot_pulse_zoom source; rename-sensitive)."""
    phase_toggle_source = inspect.getsource(ICIWindow._plot_pulse_zoom)
    assert "phase == 'charge'" in phase_toggle_source
    assert "upper center" in phase_toggle_source
    assert "lower center" in phase_toggle_source


def test_phase_change_preserves_pulse_selection():
    """Phase toggles keep the selected pulse (pins the
    _on_selection_change source; rename-sensitive)."""
    assert "pulse_var.set(1)" not in inspect.getsource(ICIWindow._on_selection_change)


def test_capacity_window_controls():
    """The capacity window auto-applies mass and steps cycles with no
    Save/Apply buttons (pins the window source; rename-sensitive)."""
    capacity_window_source = inspect.getsource(OPERAXN._open_capacity_window)
    assert "Save Plot" not in capacity_window_source
    assert 'text="Apply"' not in capacity_window_source
    assert "_queue_mass_replot" in capacity_window_source
    assert "_step_cycle" in capacity_window_source \
        and "All cycles" in capacity_window_source


def test_all_cycles_labels_use_primary_text_colour():
    """All-cycles selector labels use the primary text colour on both
    windows (pins the implementation source; rename-sensitive)."""
    assert "bg=BG2, fg=TEXT" in inspect.getsource(_SelectorStrip._build)
    assert "fg=OPERAXNTheme.COLORS['text_primary']" in \
        inspect.getsource(OPERAXN._open_capacity_window)


def test_analysis_figures_use_light_palette():
    """Capacity and Heatmap figure backgrounds stay on the light
    palette (pins the implementation source; rename-sensitive)."""
    assert "plot_bg = '#ffffff'" in inspect.getsource(OPERAXN._open_capacity_window)
    heatmap_build_source = inspect.getsource(HeatmapWindow._build_ui)
    assert "facecolor=PLOT_BG" in heatmap_build_source
    assert "width_ratios=(4, 0.12, 0.22, 1.1)" in heatmap_build_source, \
        "heatmap must reserve a colourbar label gutter"


class _FakeListbox:
    """Minimal Tk Listbox stand-in whose delete/insert mutate a plain
    items list (insert appends regardless of index; see is a no-op)."""

    def __init__(self, items):
        self.items = list(items)

    def size(self):
        return len(self.items)

    def delete(self, index):
        del self.items[index]

    def insert(self, _index, item):
        self.items.append(item)

    def see(self, _index):
        pass


def test_progress_cannot_replace_loaded_scan_rows():
    """show_progress leaves loaded scan rows untouched."""
    fake_listbox = _FakeListbox(["Scan 1"])
    fake_panel = SimpleNamespace(
        _items_cache=["Scan 1"], listbox=fake_listbox,
        update_idletasks=lambda: None,
    )
    FileListPanel.show_progress(fake_panel, "Added echem")
    assert fake_listbox.items == ["Scan 1"]


def test_progress_remains_available_before_scans_load():
    """show_progress replaces stale progress rows before any scans load."""
    fake_listbox = _FakeListbox(["Old progress"])
    fake_panel = SimpleNamespace(
        _items_cache=[], listbox=fake_listbox,
        update_idletasks=lambda: None,
    )
    FileListPanel.show_progress(fake_panel, "Loading scans")
    assert fake_listbox.items == ["Loading scans"]


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


@pytest.fixture
def tk_root():
    """Withdrawn Tk root for windowed checks; Tk() startup can fail
    transiently on busy displays, so it retries before skipping."""
    root = None
    for attempt in range(3):
        try:
            root = tk.Tk()
            root.withdraw()
            break
        except Exception:
            time.sleep(0.5)
    if root is None:
        pytest.skip("no display available")
    yield root
    root.destroy()


def _pump(win, ms):
    """Blocking pump of the Tk event loop for ms milliseconds."""
    flag = tk.BooleanVar(win)
    win.after(ms, lambda: flag.set(True))
    win.wait_variable(flag)


@pytest.mark.display
def test_max_rest_edits_redetect_pulses(tk_root):
    """max_rest bar edits re-detect pulses through the live ICI window."""
    wiring_current, wiring_t = [], []
    now = 0.0
    for rest in (90, 90, 400):
        wiring_current += [2.0] * 60 + [0.0] * rest
        wiring_t += list(np.arange(now, now + 60 + rest))
        now += 60 + rest
    wiring_current += [-2.0] * 60 + [0.0] * 90
    wiring_t += list(np.arange(now, now + 150))
    wiring_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-06-01 10:00:00") + pd.Timedelta(seconds=s)
                      for s in wiring_t],
        "echem_data": np.linspace(3.0, 4.0, len(wiring_t)),
        "current": wiring_current,
    })
    win = ICIWindow(tk_root, wiring_df)
    try:
        _pump(win, 350)
        n_default = len(win._get_pulses(1, "charge"))
        win.reg_bar.max_rest_var.set(500.0)
        _pump(win, 350)
        n_raised = len(win._get_pulses(1, "charge"))
        win.reg_bar.max_rest_var.set(50.0)
        _pump(win, 350)
        n_lowered = len(win._get_pulses(1, "charge"))
    finally:
        win.destroy()
    assert n_default == 2, f"max_rest must exclude long rests by default, got {n_default}"
    assert n_raised == 3, f"raising max_rest must admit the long-rest pulse, got {n_raised}"
    assert n_lowered == 0, f"lowering max_rest must reject all pulses, got {n_lowered}"


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
        assert bool(hm.ax_map.collections), "mesh not drawn"
        assert hm._y_values()[1] == "Scan number", "must start in scan mode"
        hm._set_toggle(hm.yaxis_var, hm._y_btns, "time")
        hm._set_toggle(hm.scale_var, hm._scale_btns, "log")
        hm.update()
        assert hm._y_values()[1] == "Time (h)", \
            "time toggle must resolve to the real time axis, not the fallback"
        assert bool(hm.ax_map.collections), "time + log toggles must survive"
        assert bool(hm.ax_v.lines), "voltage track must be plotted"
    finally:
        hm.destroy()


# ============================================================================
# Plot-cache staleness (placeholder branches, GIF frames, clear-all)
# ============================================================================

@pytest.fixture(autouse=True)
def _clean_plot_cache():
    """Plot cache is keyed on id(ax) and test axes are freed constantly."""
    operaxn.clear_plot_cache()
    yield
    operaxn.clear_plot_cache()


def test_oned_guard_heals_raw_clear():
    """A raw ax.clear() must not leave the next plot updating a detached line."""
    ax = Figure().subplots()
    x = np.linspace(10, 80, 50)
    operaxn.plot_oned_data(ax, x, np.arange(50.0), 1)
    ax.clear()
    operaxn.plot_oned_data(ax, x, np.arange(50.0) + 5, 2)
    assert len(ax.lines) == 1, "detached cached line took the update path"
    assert np.allclose(ax.lines[0].get_ydata(), np.arange(50.0) + 5)


def test_twod_guard_evicts_foreign_artist():
    """A cache entry pointing at another axes' image is evicted, not updated."""
    img = np.arange(1.0, 65.0).reshape(8, 8)
    ax_a = Figure().subplots()
    im_a = operaxn.plot_twod_data(ax_a, img, 1)
    ax_b = Figure().subplots()
    output_module._plot_cache.set(f"twod_{id(ax_b)}", im_a)
    operaxn.plot_twod_data(ax_b, img, 1)
    assert ax_b.images, "no image drawn on the new axes"
    assert ax_b.images[0] is not im_a, "foreign artist reused"


def test_echem_recreates_after_raw_clear():
    """The echem update path must not trust a line detached by a raw clear."""
    ax = Figure().subplots()
    t = np.arange(10.0)
    v = 3.7 + 0.01 * np.arange(10)
    operaxn.plot_echem_data(ax, t, v)
    ax.clear()
    operaxn.plot_echem_data(ax, t, v)
    assert ax.lines, "voltage trace not recreated after a raw clear"


def test_oned_plot_recovers_after_placeholder():
    """A no-1D scan must not blank every later 1D scan (19 Aug field report)."""
    ax = Figure().subplots()
    x = np.linspace(10, 80, 50)
    y1, y3 = np.arange(50.0), np.arange(50.0) + 7
    fake = SimpleNamespace(
        axes={"oned": ax},
        state=SimpleNamespace(oned_arrays={1: {"x": x, "y": y1},
                                           3: {"x": x, "y": y3}}),
        config=operaxn.VisualiserConfig(),
    )
    OPERAXN._update_oned_plot(fake, 1, {})
    assert len(ax.lines) == 1
    OPERAXN._update_oned_plot(fake, 2, {})
    assert not ax.lines
    assert any("No 1D data" in t.get_text() for t in ax.texts)
    OPERAXN._update_oned_plot(fake, 3, {})
    assert len(ax.lines) == 1, "scan after placeholder rendered blank"
    assert np.allclose(ax.lines[0].get_ydata(), y3)
    assert not any("No 1D data" in t.get_text() for t in ax.texts), "stale placeholder"


def test_twod_plot_recovers_and_colorbar_singular():
    """The 2D pane recovers after a 2D-less scan with exactly one live colorbar."""
    fig = Figure()
    ax = fig.subplots()
    img = np.arange(1.0, 65.0).reshape(8, 8)
    fake = SimpleNamespace(
        axes={"twod": ax},
        state=SimpleNamespace(twod_arrays={1: {"image": img}, 3: {"image": img * 2}},
                              current_scan_idx=0, intensity_limits={}),
        controls=SimpleNamespace(update_intensity_range=lambda *a: None),
        config=operaxn.VisualiserConfig(),
        twod_image=None,
    )
    fake._calculate_intensity_limits = OPERAXN._calculate_intensity_limits.__get__(fake)
    OPERAXN._update_twod_plot(fake, 1, {})
    assert len(ax.images) == 1 and len(fig.axes) == 2, "image + colorbar expected"
    OPERAXN._update_twod_plot(fake, 2, {})
    assert not ax.images
    assert len(fig.axes) == 1, "orphaned colorbar axes left behind"
    OPERAXN._update_twod_plot(fake, 3, {})
    assert len(ax.images) == 1, "scan after placeholder rendered blank"
    assert len(fig.axes) == 2, "colorbar missing or stacked"
    assert not any("No 2D data" in t.get_text() for t in ax.texts)


def test_echem_plot_recovers_after_placeholder():
    """The echem pane recovers after an echem-less dataset was shown."""
    ax = Figure().subplots()
    with_data = {"x": np.arange(10.0) * 60.0, "y": 3.7 + 0.01 * np.arange(10)}
    fake = SimpleNamespace(
        axes={"echem": ax},
        state=SimpleNamespace(echem_arrays=with_data, scans=[],
                              echem_df=pd.DataFrame(), time_method="absolute",
                              current_scan_idx=0),
        config=operaxn.VisualiserConfig(),
    )
    OPERAXN._update_echem_plot(fake)
    assert ax.lines
    fake.state.echem_arrays = {"x": np.array([])}
    OPERAXN._update_echem_plot(fake)
    assert any("No echem data" in t.get_text() for t in ax.texts)
    fake.state.echem_arrays = with_data
    OPERAXN._update_echem_plot(fake)
    assert ax.lines, "echem trace not recreated after placeholder"
    assert not any("No echem data" in t.get_text() for t in ax.texts)


def test_gif_frame_leaves_cache_clean(monkeypatch, tmp_path):
    """GIF frames must not cache artists keyed on their throwaway axes."""
    x = np.linspace(10, 80, 50)
    monkeypatch.setattr(gui_module, "get_correlated_data",
                        lambda *a, **k: {"scan_num": 1,
                                         "oned": {"x": x, "y": np.arange(50.0)},
                                         "twod": None})
    fake = SimpleNamespace(
        state=SimpleNamespace(data_source=gui_module.DataSourceType.INHOUSE,
                              scans=[], echem_df=pd.DataFrame(),
                              echem_arrays={"x": np.array([])},
                              intensity_limits={}, time_method="absolute"),
        config=operaxn.VisualiserConfig(),
    )
    frame = OPERAXN._create_gif_frame(fake, str(tmp_path), 0, {"scan_num": 1}, dpi=50)
    assert frame is not None and os.path.isfile(frame)
    assert output_module._plot_cache.cache == {}, \
        f"GIF frame cached artists on freed axes: {list(output_module._plot_cache.cache)}"


def test_clear_all_clears_plot_cache():
    """Clear-all evicts the cache entries keyed on the axes it destroys."""
    output_module._plot_cache.set("sentinel", object())
    fake = SimpleNamespace(
        state=SimpleNamespace(scans=[], dialog_windows=[], reset=lambda: None),
        plot_container=SimpleNamespace(winfo_children=lambda: []),
        _show_empty_state=lambda: None,
        file_list=SimpleNamespace(update_items=lambda items: None),
        scan_selector=SimpleNamespace(set_range=lambda a, b: None,
                                      set_value=lambda v: None),
        button_panel=SimpleNamespace(update_states=lambda states: None),
        controls=SimpleNamespace(grid_remove=lambda: None),
        fig=None, axes={}, canvas=None, twod_image=None,
    )
    OPERAXN._clear_all(fake)
    assert output_module._plot_cache.cache == {}, "clear-all left cache entries"


def test_needs_recreate_current_rebuild_matrix():
    """A wanted current trace whose cache entry is gone or whose line is
    detached always forces a rebuild, never a set_data no-op."""
    plotter = output_module._echem_plotter
    t = np.arange(10.0)

    # Current-only layout, cache wiped mid-session
    ax = Figure().subplots()
    plotter._recreate_plot(ax, t, np.array([]), 0.1 + 0.0 * t, False, True)
    operaxn.clear_plot_cache()
    assert plotter._needs_recreate(ax, False, True), \
        "cleared cache with wanted current must force a rebuild"

    # Both-traces layout, current line detached from its twin
    ax = Figure().subplots()
    operaxn.plot_echem_data(ax, t, 3.7 + 0.01 * t, 0.1 + 0.0 * t)
    current_line = output_module._plot_cache.get(f"echem_c_{id(ax)}")
    assert current_line is not None
    current_line.remove()
    assert plotter._needs_recreate(ax, True, True), \
        "detached current line must force a rebuild"


def test_scan_time_positions_align_with_scans():
    """The positions list is 1:1 with scans, None for untimestamped ones."""
    echem = pd.DataFrame({"timestamp": pd.date_range("2026-06-01 10:00", periods=5,
                                                     freq="60s"),
                          "echem_data": np.linspace(3.0, 3.4, 5)})
    scans = [
        {"scan_num": 1, "timestamp": "2026-06-01 10:00:00"},
        {"scan_num": 2, "timestamp": None},
        {"scan_num": 3, "timestamp": "2026-06-01 10:02:00"},
    ]
    positions = operaxn.get_scan_time_positions(scans, echem, "absolute")
    assert len(positions) == 3, "must stay aligned with the scan list"
    assert positions[0] == 0.0 and positions[1] is None and positions[2] == 120.0


def test_scan_marker_skips_none_position():
    """A None position hides the marker instead of crashing or misplacing it."""
    ax = Figure().subplots()
    t = np.arange(10.0) * 3600.0
    v = 3.7 + 0.01 * np.arange(10)
    operaxn.plot_echem_data(ax, t, v, scan_times=[0.0, None, 7200.0],
                            current_scan_idx=0)
    marker = output_module._plot_cache.get(f"echem_marker_{id(ax)}")
    assert marker is not None and marker.get_visible()
    operaxn.plot_echem_data(ax, t, v, scan_times=[0.0, None, 7200.0],
                            current_scan_idx=1)
    assert not marker.get_visible(), "None position must hide the marker"
    operaxn.plot_echem_data(ax, t, v, scan_times=[0.0, None, 7200.0],
                            current_scan_idx=2)
    assert marker.get_visible(), "marker must reappear for a timed scan"
    assert marker.get_xdata()[0] == 2.0, f"got {marker.get_xdata()[0]}"


def test_gif_frame_marks_scan_position_not_frame_index(monkeypatch, tmp_path):
    """A sub-range GIF must mark each frame at its own scan's time."""
    x = np.linspace(10, 80, 50)
    captured = {}

    def spy_plot_echem(ax, tx, ty, current, scan_times, idx, plot_config=None):
        captured["idx"] = idx

    monkeypatch.setattr(gui_module, "get_correlated_data",
                        lambda *a, **k: {"scan_num": 5,
                                         "oned": {"x": x, "y": np.arange(50.0)},
                                         "twod": None})
    monkeypatch.setattr(gui_module, "plot_echem_data", spy_plot_echem)
    monkeypatch.setattr(gui_module, "get_scan_time_positions",
                        lambda *a, **k: [float(i) for i in range(10)])
    fake = SimpleNamespace(
        state=SimpleNamespace(data_source=gui_module.DataSourceType.INHOUSE,
                              scans=[{"scan_num": n} for n in range(1, 11)],
                              echem_df=pd.DataFrame({"timestamp": [1]}),
                              echem_arrays={"x": np.arange(5.0),
                                            "y": 3.7 + np.arange(5.0)},
                              intensity_limits={}, time_method="absolute"),
        config=operaxn.VisualiserConfig(),
    )
    frame = OPERAXN._create_gif_frame(fake, str(tmp_path), 0, {"scan_num": 5}, dpi=50)
    assert frame is not None
    assert captured["idx"] == 4, \
        f"frame 0 of a range starting at scan 5 must mark index 4, got {captured['idx']}"


def test_current_only_refresh_never_spawns_empty_twin():
    """Repeated current-only refreshes update in place with no phantom twin."""
    fig = Figure()
    ax = fig.subplots()
    t = np.arange(10.0)
    c = 0.1 + 0.01 * np.arange(10)
    cfg = {"show_voltage": False, "show_current": True}
    for _ in range(3):
        operaxn.plot_echem_data(ax, t, np.array([]), c, plot_config=cfg)
    assert not hasattr(ax, "_ax2"), "empty twin axes spawned on refresh"
    assert len(fig.axes) == 1, f"got {len(fig.axes)} axes"
    data_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 10]
    assert len(data_lines) == 1, "current trace duplicated or lost"
