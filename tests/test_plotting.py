"""Main-window plotting: the plot cache, GIF frames and export, Plot Settings and image exports."""
import os
import shutil
import sys
import warnings
import weakref
from collections.abc import Mapping
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

pytestmark = pytest.mark.gui

tk = pytest.importorskip("tkinter")

import operaxn
import operaxn.dialog as dialog_module
import operaxn.gui as gui_module
import operaxn.output as output_module
from operaxn.dialog import GIFSettings, PlotSettingsDialog
from operaxn.gui import OPERAXN, VisualiserConfig
from operaxn.input import DataSourceType
from operaxn.output import OneDPlotter, PlotConfig, export_single_scan

from _helpers import export_failed


@pytest.fixture(autouse=True)
def _clean_plot_cache():
    """Clear the id(ax)-keyed plot cache around each test, as test axes are freed constantly."""
    operaxn.clear_plot_cache()
    yield
    operaxn.clear_plot_cache()


# ============================================================================
# Plot cache (placeholder branches, guards, clear-all)
# ============================================================================

@pytest.mark.parametrize("pane", ["oned", "echem"])
def test_oned_guard_heals_raw_clear(pane):
    """After a raw ax.clear() the next 1D or echem plot draws a new line, not a detached one."""
    ax = Figure().subplots()
    x = np.linspace(10, 80, 50)
    plot = operaxn.plot_oned_data if pane == "oned" else operaxn.plot_echem_data
    plot(ax, x, np.arange(50.0))
    ax.clear()
    plot(ax, x, np.arange(50.0) + 5)
    assert len(ax.lines) == 1, "detached cached line took the update path"
    assert np.allclose(ax.lines[0].get_ydata(), np.arange(50.0) + 5)


def test_twod_guard_evicts_foreign_artist():
    """A cache entry pointing at another axes' image is evicted, not updated."""
    img = np.arange(1.0, 65.0).reshape(8, 8)
    ax_a = Figure().subplots()
    im_a = operaxn.plot_twod_data(ax_a, img)
    ax_b = Figure().subplots()
    output_module._plot_cache.set(f"twod_{id(ax_b)}", im_a)
    operaxn.plot_twod_data(ax_b, img)
    assert ax_b.images, "no image drawn on the new axes"
    assert ax_b.images[0] is not im_a, "foreign artist reused"


def test_oned_plot_recovers_after_placeholder():
    """A no-1D scan must not blank every later 1D scan."""
    ax = Figure().subplots()
    x = np.linspace(10, 80, 50)
    y1, y3 = np.arange(50.0), np.arange(50.0) + 7
    fake = SimpleNamespace(
        axes={"oned": ax},
        state=SimpleNamespace(oned_arrays={1: {"x": x, "y": y1},
                                           3: {"x": x, "y": y3}}),
        config=VisualiserConfig(),
    )
    OPERAXN._update_oned_plot(fake, 1)
    assert len(ax.lines) == 1
    OPERAXN._update_oned_plot(fake, 2)
    assert not ax.lines
    assert any("No 1D data" in t.get_text() for t in ax.texts)
    OPERAXN._update_oned_plot(fake, 3)
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
        config=VisualiserConfig(),
        twod_image=None,
    )
    fake._calculate_intensity_limits = OPERAXN._calculate_intensity_limits.__get__(fake)
    OPERAXN._update_twod_plot(fake, 1)
    assert len(ax.images) == 1 and len(fig.axes) == 2, "image + colorbar expected"
    OPERAXN._update_twod_plot(fake, 2)
    assert not ax.images
    assert len(fig.axes) == 1, "orphaned colorbar axes left behind"
    OPERAXN._update_twod_plot(fake, 3)
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
        config=VisualiserConfig(),
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
        state=SimpleNamespace(data_source=DataSourceType.INHOUSE,
                              scans=[], echem_df=pd.DataFrame(),
                              echem_arrays={"x": np.array([])},
                              intensity_limits={}, time_method="absolute"),
        config=VisualiserConfig(),
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
        _is_busy=lambda: False,
    )
    OPERAXN._clear_all(fake)
    assert output_module._plot_cache.cache == {}, "clear-all left cache entries"


def test_data_check_reads_no_2d_image_when_1d_data_exist():
    """The loaded-data check reads no 2D image while a 1D pattern exists, and one when none does."""
    reads = []

    class Lazy(Mapping):
        """A make_twod_arrays-like mapping that records each image read."""

        def __getitem__(self, scan_num):
            reads.append(scan_num)
            return {"image": np.ones((2, 2))}

        def __iter__(self):
            return iter([1, 2])

        def __len__(self):
            return 2

    fake = SimpleNamespace(state=SimpleNamespace(data_source=DataSourceType.INHOUSE,
                                                 oned_arrays={1: {"x": [1.0], "y": [1.0]}},
                                                 twod_arrays=Lazy()))
    assert OPERAXN._check_data_availability(fake) and reads == []
    fake.state.oned_arrays = {}
    assert OPERAXN._check_data_availability(fake) and reads == [1]


def test_needs_recreate_current_rebuild_matrix():
    """A wanted current trace with a wiped cache entry or a detached line forces a rebuild."""
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


IMG = np.outer(np.linspace(1, 50, 120), np.linspace(1, 30, 100)) + 1.0


def _render(fig):
    """Draw fig on an Agg canvas and return a copy of its RGBA pixels."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def test_twod_update_keeps_norm_and_pixels():
    """A 2D update re-limits the image's own LogNorm and renders exactly like a fresh plot."""
    fig = Figure(figsize=(5, 4), dpi=95)
    ax = fig.subplots()
    im = operaxn.plot_twod_data(ax, IMG, intensity_limits=(2.0, 900.0))
    norm = im.norm
    im2 = operaxn.plot_twod_data(ax, IMG * 1.5, intensity_limits=(5.0, 400.0))
    assert im2 is im and im.norm is norm, "the update must not swap the norm"
    assert (norm.vmin, norm.vmax) == (5.0, 400.0)
    updated = _render(fig)
    operaxn.clear_plot_cache()
    fresh_fig = Figure(figsize=(5, 4), dpi=95)
    operaxn.plot_twod_data(fresh_fig.subplots(), IMG * 1.5, intensity_limits=(5.0, 400.0))
    assert np.array_equal(updated, _render(fresh_fig))


def test_intensity_handler_keeps_norm():
    """The 2D intensity handler re-limits the image in place without replacing its norm."""
    fig = Figure()
    ax = fig.subplots()
    im = operaxn.plot_twod_data(ax, IMG, intensity_limits=(2.0, 900.0))
    norm = im.norm
    fake = SimpleNamespace(
        state=SimpleNamespace(twod_arrays={1: {"image": IMG}}, intensity_limits={},
                              current_scan_idx=0),
        axes={"twod": ax}, twod_image=im, canvas=None,
        controls=SimpleNamespace(get_intensity_limits=lambda: (3.0, 300.0)))
    OPERAXN._update_plots_with_intensity(fake)
    assert im.norm is norm and (norm.vmin, norm.vmax) == (3.0, 300.0)


def test_settings_change_keeps_1d_2d_artists_and_rebuilds_echem():
    """Plot Settings changes keep the cached 1D line and 2D image; only the echem pane rebuilds."""
    fig = Figure()
    a1, a2, a3 = fig.subplots(3)
    operaxn.plot_oned_data(a1, np.linspace(10, 80, 50), np.arange(50.0))
    line = a1.lines[0]
    im = operaxn.plot_twod_data(a2, IMG, intensity_limits=(2.0, 900.0))
    t = np.arange(10.0) * 600
    operaxn.plot_echem_data(a3, t, 3.7 + 0.01 * t / 600, 0.1 + 0 * t)
    calls = []
    fake = SimpleNamespace(axes={"oned": a1, "twod": a2, "echem": a3},
                           _update_plots=lambda: calls.append(1))
    OPERAXN._on_plot_settings_update(fake)
    assert calls == [1]
    assert output_module._plot_cache.get(f"oned_{id(a1)}") is line
    assert output_module._plot_cache.get(f"twod_{id(a2)}") is im
    assert output_module._plot_cache.get(f"echem_v_{id(a3)}") is None and not a3.lines, \
        "the echem pane is rebuilt"


def test_log_limits_floors_and_orders():
    """log_limits floors vmin at VMIN_FLOOR and lifts a vmax at or below it to ten times vmin."""
    assert output_module.log_limits(0, 0) == (1e-10, 1e-9)
    assert output_module.log_limits(5, 2) == (5, 50)
    assert output_module.log_limits(-1, 3) == (1e-10, 3)
    assert output_module.log_limits(2, 9) == (2, 9)


# ============================================================================
# GIF export
# ============================================================================

# The GIF progress page: never stopped
DIALOG = SimpleNamespace(begin_progress=lambda *a: None, update_progress=lambda *a: True)


def _gif_fake(data_source, **arrays):
    """Return a stand-in app for the GIF frame methods: no echem, the given plot arrays."""
    plot_arrays = {"oned_arrays": {}, "twod_arrays": {}, "neutron_arrays": {}, **arrays}
    return SimpleNamespace(
        state=SimpleNamespace(data_source=data_source, scans=[], echem_df=pd.DataFrame(),
                              echem_arrays={"x": np.array([])}, intensity_limits={},
                              time_method="absolute", **plot_arrays),
        config=VisualiserConfig(data_source=data_source),
        _gif_placeholder=OPERAXN._gif_placeholder,
    )


def _gif_writer(fake):
    """Wire a _gif_fake for the real _generate_gif: its own layout and frames, messages logged."""
    fake._gif_layout = lambda nums: OPERAXN._gif_layout(fake, nums)
    fake._create_gif_frame = lambda *a: OPERAXN._create_gif_frame(fake, *a)
    fake.messages = []
    fake._show_message = lambda *a: fake.messages.append(a)
    fake._close_dialog = lambda dialog: None
    return fake


def _frame_size(path):
    """Return a PNG frame's (width, height)."""
    with Image.open(path) as image:
        return image.size


def _mixed_2d_fake(monkeypatch):
    """Return a _gif_fake of scans 1 and 2 with distinct 1D data and a 2D image for scan 2 only."""
    x = np.linspace(10, 80, 50)
    image = (np.arange(64.0).reshape(8, 8) + 1) * 10

    def correlated(scans, scan_num, state=None):
        return {"scan_num": scan_num, "oned": {"x": x, "y": np.arange(50.0) * scan_num},
                "twod": image if scan_num == 2 else None}

    monkeypatch.setattr(gui_module, "get_correlated_data", correlated)
    return _gif_fake(DataSourceType.INHOUSE,
                     **{"oned_arrays": {1: {}, 2: {}}, "twod_arrays": {2: {}}})


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
        state=SimpleNamespace(data_source=DataSourceType.INHOUSE,
                              scans=[{"scan_num": n} for n in range(1, 11)],
                              echem_df=pd.DataFrame({"timestamp": [1]}),
                              echem_arrays={"x": np.arange(5.0),
                                            "y": 3.7 + np.arange(5.0)},
                              intensity_limits={}, time_method="absolute"),
        config=VisualiserConfig(),
    )
    frame = OPERAXN._create_gif_frame(fake, str(tmp_path), 0, {"scan_num": 5}, dpi=50)
    assert frame is not None
    assert captured["idx"] == 4, \
        f"frame 0 of a range starting at scan 5 must mark index 4, got {captured['idx']}"


def test_gif_marks_an_unreadable_2d_image_as_the_main_window_does(monkeypatch, tmp_path):
    """A scan whose 2D image failed to load gets the main window's red error text."""
    x = np.linspace(10, 80, 50)
    monkeypatch.setattr(gui_module, "get_correlated_data",
                        lambda *a, **k: {"scan_num": 1,
                                         "oned": {"x": x, "y": np.arange(50.0)},
                                         "twod": None})
    marks = []
    fake = _gif_fake(DataSourceType.INHOUSE,
                     **{"oned_arrays": {1: {}}, "twod_arrays": {1: {"error": True}}})
    fake._gif_placeholder = lambda ax, text, color="grey": marks.append((text, color))
    layout = OPERAXN._gif_layout(fake, [1])
    assert OPERAXN._create_gif_frame(fake, str(tmp_path), 0, {"scan_num": 1}, dpi=40,
                                     layout=layout)
    main = SimpleNamespace(axes={"twod": Figure().subplots()},
                           state=SimpleNamespace(twod_arrays={1: {"error": True}}))
    OPERAXN._update_twod_plot(main, 1)
    drawn = [(text.get_text(), text.get_color()) for text in main.axes["twod"].texts]
    assert len(drawn) == 1 and marks == drawn, (marks, drawn)


def test_gif_neutron_frames_keep_size_as_intensity_grows(monkeypatch, tmp_path):
    """Neutron banks autoscale per scan; their frames must still all be one size."""
    x = np.linspace(1000, 20000, 200)

    def correlated(scans, scan_num, state=None):
        scale = 1.0 if scan_num == 1 else 30000.0
        return {"scan_num": scan_num,
                "neutron": {str(b): {"tof": {"x": x, "y": scale * (1 + np.sin(x / 900))}}
                            for b in range(1, 6)}}

    monkeypatch.setattr(gui_module, "get_correlated_data", correlated)
    fake = _gif_fake(DataSourceType.NEUTRON,
                     **{"neutron_arrays": {1: {}, 2: {}}})
    layout = OPERAXN._gif_layout(fake, [1, 2])
    sizes = [_frame_size(OPERAXN._create_gif_frame(fake, str(tmp_path), i, {"scan_num": n},
                                                   dpi=50, layout=layout))
             for i, n in enumerate((1, 2))]
    assert sizes[0] == sizes[1], sizes


def test_generate_gif_writes_one_frame_size_for_mixed_2d(monkeypatch, tmp_path):
    """The GIF writer lays a mixed-2D range out alike: one frame size, a placeholder without 2D."""
    fake = _gif_writer(_mixed_2d_fake(monkeypatch))
    fake.state.scans = [{"scan_num": 1}, {"scan_num": 2}]
    sizes, texts, layouts = [], [], []
    real, layout, placeholder = fake._create_gif_frame, fake._gif_layout, fake._gif_placeholder

    def render(*args):
        frame = real(*args)
        sizes.append(_frame_size(frame))  # measured as rendered: a GIF has one canvas size
        return frame

    fake._create_gif_frame = render
    fake._gif_layout = lambda nums: layouts.append(layout(nums)) or layouts[-1]
    fake._gif_placeholder = lambda ax, text, *a: texts.append(text) or placeholder(ax, text, *a)
    out = str(tmp_path / "mixed.gif")
    OPERAXN._generate_gif(fake, out, GIFSettings(fps=2, dpi=40, loop=0, scan_list=[1, 2]), DIALOG)
    assert fake.messages and fake.messages[-1][0] == "Export complete", fake.messages
    assert layouts and set(layouts) == {(True, True, False, False)}, layouts
    assert len(sizes) == 2 and sizes[0] == sizes[1], sizes
    assert texts == ["No 2D data for this scan"], texts


def _gif_app(monkeypatch, n):
    """Return a _gif_writer stand-in with n scans of distinct 1D data and a message log."""
    x = np.linspace(10, 80, 50)

    def correlated(scans, scan_num, state=None):
        return {"scan_num": scan_num, "oned": {"x": x, "y": np.arange(50.0) * scan_num}}

    monkeypatch.setattr(gui_module, "get_correlated_data", correlated)
    fake = _gif_fake(DataSourceType.INHOUSE,
                     oned_arrays={i: {} for i in range(1, n + 1)})
    fake.state.scans = [{"scan_num": i} for i in range(1, n + 1)]
    return _gif_writer(fake)


def test_generate_gif_keeps_frame_timing_without_fps_keyword(monkeypatch, tmp_path):
    """Each frame lasts 1000 / fps ms and the loop count is kept, with no fps deprecation."""
    fake = _gif_app(monkeypatch, 3)
    out = str(tmp_path / "timed.gif")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OPERAXN._generate_gif(fake, out, GIFSettings(fps=4, dpi=40, loop=2, scan_list=[1, 2, 3]),
                              DIALOG)
    assert fake.messages and fake.messages[-1][0] == "Export complete", fake.messages
    assert not [w for w in caught if "fps" in str(w.message)], [str(w.message) for w in caught]
    with Image.open(out) as gif:
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
        assert gif.info["loop"] == 2
    assert durations == [250, 250, 250], durations


def test_generate_gif_streams_frames_to_the_writer(monkeypatch, tmp_path):
    """The writer holds a fixed few full-colour frames (first, previous, next), not one per scan."""
    refs = []
    peak = [0]
    original = Image.Image.save

    def counting(self, fp, format=None, **params):
        # Count the frames handed to the GIF writer that are still alive as each new one
        # arrives; PIL images are unhashable, so by weak reference
        if not params.get("save_all"):  # the PNG frames matplotlib writes
            return original(self, fp, format, **params)
        refs.append(weakref.ref(self))

        def frames(source):
            for image in source:
                refs.append(weakref.ref(image))
                peak[0] = max(peak[0], sum(ref() is not None for ref in refs))
                yield image
        params["append_images"] = frames(params.get("append_images", []))
        return original(self, fp, format, **params)

    monkeypatch.setattr(Image.Image, "save", counting)
    fake = _gif_app(monkeypatch, 8)
    out = str(tmp_path / "long.gif")
    OPERAXN._generate_gif(fake, out, GIFSettings(fps=5, dpi=40, loop=0,
                                                 scan_list=list(range(1, 9))), DIALOG)
    assert fake.messages and fake.messages[-1][0] == "Export complete", fake.messages
    with Image.open(out) as gif:
        assert gif.n_frames == 8
    assert peak[0] <= 3, peak[0]


def _frames(tmp_path, n=3):
    """Write n distinct opaque PNG frames and return their paths."""
    paths = []
    for i in range(n):
        p = str(tmp_path / f"f{i}.png")
        Image.fromarray(np.full((40, 60, 4), (40 * i, 80, 120, 255), np.uint8)).save(p)
        paths.append(p)
    return paths


def _gif(out, files, monkeypatch=None, fail_on=None, fps=2, loop=0):
    """Run the real _generate_gif on prepared frames; return the last message."""
    if fail_on is not None:
        real, calls = Image.open, [0]

        def flaky(path, *a, **k):
            calls[0] += 1
            if calls[0] == fail_on:
                raise OSError("disk full")
            return real(path, *a, **k)
        monkeypatch.setattr(Image, "open", flaky)
    msgs = []

    def frame(temp_dir, idx, scan, dpi, layout=None):
        dst = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        shutil.copyfile(files[idx], dst)
        return dst
    fake = SimpleNamespace(state=SimpleNamespace(scans=[{"scan_num": i + 1}
                                                        for i in range(len(files))]),
                           _gif_layout=lambda nums: (True, False, False, False),
                           _create_gif_frame=frame, _show_message=lambda *a: msgs.append(a),
                           _close_dialog=lambda d: None)
    OPERAXN._generate_gif(fake, out, GIFSettings(fps=fps, dpi=100, loop=loop,
                                                 scan_list=list(range(1, len(files) + 1))), DIALOG)
    assert msgs, "the GIF export showed no message"
    return msgs[-1]


def test_gif_export_needs_no_imageio_and_keeps_its_bytes(tmp_path, monkeypatch):
    """With imageio unavailable a GIF still exports, byte for byte the GIF Pillow writes from the
    frames' RGB pixels at 1000 / fps ms per frame and the chosen loop count."""
    rng = np.random.default_rng(0)
    files = []
    for i in range(4):
        rgba = rng.integers(0, 256, (30, 40, 4), dtype=np.uint8)
        rgba[..., 3] = 255 if i % 2 else rng.integers(0, 256, (30, 40), dtype=np.uint8)
        files.append(str(tmp_path / f"f{i}.png"))
        Image.fromarray(rgba).save(files[-1])
    files.append(files[3])  # the writer merges a repeated frame
    expected = str(tmp_path / "expected.gif")
    pixels = [Image.fromarray(np.asarray(Image.open(f))[..., :3]) for f in files]
    pixels[0].save(expected, format="GIF", save_all=True, append_images=pixels[1:],
                   duration=1000 / 3, loop=2)

    monkeypatch.setitem(sys.modules, "imageio", None)  # any import of imageio now fails
    assert not hasattr(gui_module, "imageio"), "the GIF writer must not hold imageio"
    out = str(tmp_path / "anim.gif")
    assert _gif(out, files, fps=3, loop=2)[0] == "Export complete"
    with open(out, "rb") as written, open(expected, "rb") as reference:
        assert written.read() == reference.read()


def test_failed_gif_write_keeps_the_previous_gif(tmp_path, monkeypatch):
    """A write failing part-way shows the error, leaves the old GIF byte for byte and no .part."""
    files = _frames(tmp_path)
    out = str(tmp_path / "anim.gif")
    assert _gif(out, files)[0] == "Export complete"
    before = open(out, "rb").read()
    title, text, _ = _gif(out, files[::-1], monkeypatch, fail_on=2)
    assert (title, text) == export_failed("GIF", "disk full")
    assert open(out, "rb").read() == before
    assert not os.path.exists(out + ".part")


# ============================================================================
# Plot Settings and image exports
# ============================================================================

@pytest.mark.display
def test_plot_settings_apply_redraws_once(tk_root):
    """Applying plot settings redraws through the update callback exactly once."""
    calls = []
    dlg = PlotSettingsDialog(tk_root, VisualiserConfig(), lambda: calls.append(1))
    try:
        dlg._apply_settings()
    finally:
        dlg.destroy()
    assert calls == [1]


@pytest.mark.display
def test_plot_settings_axis_choices_stay_plain_text(tk_root):
    """Plot Settings keeps plain Tk text for TOF, d-spacing and 2θ (figure labels use mathtext)."""
    def texts(widget, kinds=(tk.Label, tk.Checkbutton, tk.Radiobutton)):
        own = [w.cget("text") for w in widget.winfo_children() if isinstance(w, kinds)]
        return own + [t for w in widget.winfo_children() for t in texts(w, kinds)]

    dlg = PlotSettingsDialog(tk_root, VisualiserConfig(data_source=DataSourceType.NEUTRON), None)
    try:
        assert texts(dlg, tk.Radiobutton) == ["Time of flight (μs)", "d-spacing (Å)"]
        assert not [t for t in texts(dlg) if "$" in t]
    finally:
        dlg.destroy()
    dlg = PlotSettingsDialog(tk_root, VisualiserConfig(), None)
    try:
        shown = texts(dlg)
        assert {"X-axis range (2θ [°]):", "Show d-spacing instead of 2θ"} <= set(shown)
        assert not [t for t in shown if "$" in t]
    finally:
        dlg.destroy()


@pytest.mark.display
def test_crop_sliders_never_cross(tk_root):
    """A crop slider dragged past its partner stops 1 % short of it, in the config and its label."""
    def widgets(widget):
        return [w for child in widget.winfo_children() for w in [child] + widgets(child)]

    config = VisualiserConfig()
    dlg = PlotSettingsDialog(tk_root, config, None)
    try:
        dlg.ymax_2d_var.set(40)
        dlg.ymin_2d_var.set(60)  # a drag sets the variable, then Tk runs the row's command
        scale = next(w for w in widgets(dlg) if isinstance(w, tk.Scale)
                     and str(w.cget("variable")) == str(dlg.ymin_2d_var))
        dlg.tk.call(scale.cget("command"), "60")
        label = scale.master.grid_slaves(row=int(scale.grid_info()["row"]), column=2)[0]
        assert (config.twod_ymin_percent, config.twod_ymax_percent) == (39.0, 40.0)
        assert label.cget("text") == "39%"
    finally:
        dlg.destroy()


@pytest.mark.display
def test_intensity_entries_and_sliders_keep_valid_limits(tk_root, monkeypatch):
    """2D Min/Max: bad typed bounds go back, far ones widen the sliders; Apply All asks first."""
    updates, applied, asked, answers = [], [], [], [False, True]
    monkeypatch.setattr(gui_module.messagebox, "askyesno",
                        lambda *a, **_k: asked.append(a) or answers.pop(0))
    controls = gui_module.PlotControls(tk_root, VisualiserConfig(), lambda: updates.append(1),
                                       lambda *a: None, lambda lo, hi: applied.append((lo, hi)))

    def typed(entry, text, is_min):
        entry.delete(0, "end")
        entry.insert(0, text)
        controls._commit_entry(is_min)
        return controls.get_intensity_limits(), entry.get()

    def span():
        return [(float(s.cget("from")), float(s.cget("to")))
                for s in (controls.min_slider, controls.max_slider)]

    try:
        controls.update_intensity_range(1.0, 1000.0, 10.0, 100.0)
        assert controls.get_intensity_limits() == (10.0, 100.0) and span() == [(0.0, 3.0)] * 2
        for text in ("0", "-5", "nan", "abc", "100", "250"):  # not positive, or not below Max
            assert typed(controls.min_entry, text, True) == ((10.0, 100.0), "10"), text
        for text in ("0", "inf", "10", "5"):  # not finite, or not above Min
            assert typed(controls.max_entry, text, False) == ((10.0, 100.0), "100"), text
        assert updates == [], "a bound put back draws nothing"
        assert typed(controls.max_entry, "5000", False) == ((10.0, 5000.0), "5000")
        assert span() == [(0.0, 3.7)] * 2 and updates == [1], "a far bound widens both sliders"
        controls._min_slider_log.set(1.5)
        controls._on_slider(True)  # as a drag does: 10**position, Max left exact
        assert controls.get_intensity_limits() == (31.62, 5000.0)
        controls._max_slider_log.set(0.5)
        controls._on_slider(False)  # past Min: stops one step above it
        assert controls.get_intensity_limits() == (31.62, 32.36)
        controls._apply_to_all()
        assert asked == [("Confirm", "Apply intensity range 31.62 - 32.36 to all scans?")]
        assert applied == [], "No applies nothing"
        controls._apply_to_all()
        assert applied == [(31.62, 32.36)]
    finally:
        controls.destroy()


def _limit_stub():
    """Return (stand-in dialog, its config, a text box, its variable) for _validate_entry."""
    config = VisualiserConfig()
    stub = SimpleNamespace(config=config)
    stub._set_config_value = lambda k, v: PlotSettingsDialog._set_config_value(stub, k, v)
    box = {"text": ""}
    var = SimpleNamespace(get=lambda: box["text"], set=lambda v: box.update(text=v))
    return stub, config, box, var


@pytest.mark.parametrize("text", ["inf", "nan"])
def test_plot_settings_clear_a_non_finite_limit(text):
    """A non-finite Plot Settings limit is cleared to auto, as unparseable text already is."""
    stub, config, box, var = _limit_stub()
    box["text"] = text
    PlotSettingsDialog._validate_entry(stub, "xmax", var)
    assert config.oned_xmax is None and box["text"] == ""


def test_plot_settings_keep_a_decreasing_dspacing_range():
    """Finite limits are stored as typed, including X-ray d-spacing Min above Max."""
    stub, config, box, var = _limit_stub()
    config.show_dspacing = True
    for key, text in (("xmin", "5.2"), ("xmax", "2.07"), ("neutron_xmax", "15000")):
        box["text"] = text
        PlotSettingsDialog._validate_entry(stub, key, var)
        assert box["text"] == text
    assert (config.oned_xmin, config.oned_xmax, config.neutron_xmax) == (5.2, 2.07, 15000.0)


def test_dspacing_blank_limit_is_filled_decreasing():
    """With one X limit blank the X-ray d view keeps the chosen part and runs decreasing."""
    tth = np.linspace(17.05, 43.59, 300)
    plotter = OneDPlotter()
    d = plotter.transformer.theta_to_d_spacing(tth)
    # A scalar input gives a 0-d value, so take it through atleast_1d
    d40, d25 = (float(np.atleast_1d(plotter.transformer.theta_to_d_spacing(v))[0])
                for v in (40.0, 25.0))
    for xmin, xmax, expected in [(None, d40, (d.max(), d40)), (d25, None, (d25, d.min())),
                                 (None, None, (d.max(), d.min()))]:
        ax = Figure().subplots()
        plotter._create_plot(ax, tth, np.arange(300.0),
                             PlotConfig(show_dspacing=True, oned_xmin=xmin, oned_xmax=xmax,
                                        use_cache=False), "k")
        assert np.allclose(ax.get_xlim(), expected)
    ax = Figure().subplots()
    plotter._create_plot(ax, tth, np.arange(300.0),
                         PlotConfig(oned_xmax=40.0, use_cache=False), "k")
    assert np.allclose(ax.get_xlim(), (17.05, 40.0)), "2-theta view must not change"


@pytest.mark.parametrize("source, expected", [
    (DataSourceType.INHOUSE, (0.04, 0.36, 0.05, 0.6)),
    (DataSourceType.SYNCHROTRON, (0.1, 0.9, 0.05, 0.6)),
])
def test_cropped_image_extent_follows_the_crop(source, expected):
    """The 2D image spans the cropped part of the source's base extent (exports rely on it)."""
    config = VisualiserConfig(data_source=source)
    config.twod_xmin_percent, config.twod_xmax_percent = 10, 90
    config.twod_ymin_percent, config.twod_ymax_percent = 5, 60
    image = np.arange(1.0, 5401.0).reshape(60, 90)
    im = operaxn.plot_twod_data(Figure().subplots(), image,
                                plot_config={**config.to_dict(), "use_cache": False})
    assert np.allclose(im.get_extent(), expected), im.get_extent()


def test_failed_plot_update_leaves_drawing_switched_on():
    """A plot error in the update block restores draw_idle without drawing; the next one draws."""
    calls = []
    canvas = SimpleNamespace(draw_idle=lambda: calls.append("draw"))
    original = canvas.draw_idle
    frame = SimpleNamespace(canvas=canvas)
    with pytest.raises(ValueError):
        with OPERAXN._plot_update_context(frame):
            canvas.draw_idle()
            raise ValueError("Axis limits cannot be NaN or Inf")
    assert canvas.draw_idle is original, "drawing stays switched off for the session"
    assert calls == [], "a half-updated figure must not be drawn"
    with OPERAXN._plot_update_context(frame):
        canvas.draw_idle()
    assert calls == ["draw"], "one draw per successful update"


@pytest.mark.display
@pytest.mark.parametrize("neutron", [False, True], ids=["xray", "neutron"])
def test_main_figure_x_labels_stay_inside_a_short_window(tk_root, neutron):
    """At the plot height a small screen leaves (300 px), the bottom row's x labels are not cut."""
    fig, axes = operaxn.create_figure_layout(not neutron, not neutron, True, neutron)
    win = tk.Toplevel(tk_root)
    win.withdraw()
    try:
        x = np.linspace(0.5, 4.0, 200)
        if neutron:
            bank = {"d": {"x": x, "y": np.arange(200.0)}}
            operaxn.plot_neutron_data(axes, {str(n): bank for n in range(1, 6)},
                                      {"show_neutron_dspacing": True, "use_cache": False})
        else:
            operaxn.plot_oned_data(axes["oned"], x, np.arange(200.0),
                                   {"show_dspacing": True, "use_cache": False})
        canvas = gui_module.FigureCanvasTkAgg(fig, master=win)  # as the main window embeds it
        canvas.resize(SimpleNamespace(width=1060, height=300))  # as Tk does
        canvas.draw()
        renderer = canvas.get_renderer()
        lowest = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
        assert lowest >= 2, f"x labels cut off: {lowest:.1f} px above the figure's edge"
    finally:
        plt.close(fig)
        win.destroy()


@pytest.mark.parametrize("twod, written", [(np.ones(50), False), (np.ones((8, 8)), True)],
                         ids=["drawing-error", "written"])
def test_failed_single_scan_export_closes_its_figure(tmp_path, twod, written):
    """export_single_scan closes its figure whether the drawing fails or the PNG is written."""
    scan = {"scan_num": 1, "oned": {"x": np.linspace(10, 80, 100), "y": np.arange(100.0)},
            "twod": twod}
    out = tmp_path / "scan.png"
    before = len(plt.get_fignums())
    for _ in range(3):
        if written:
            export_single_scan(scan, str(out))
        else:
            with pytest.raises(ValueError):
                export_single_scan(scan, str(out))
    assert len(plt.get_fignums()) == before
    assert out.is_file() == written and (not written or out.read_bytes()[:4] == b"\x89PNG")


def test_echem_only_export_is_refused(tmp_path):
    """A scan with no drawable panel raises instead of saving a title-only image."""
    out = tmp_path / "title_only.png"
    with pytest.raises(ValueError, match="no 1D, 2D or neutron data"):
        export_single_scan({"scan_num": 3, "oned": None, "twod": None, "echem_value": 3.7},
                           str(out), plot_types={"oned": True, "twod": True, "echem": True})
    assert not out.exists()


@pytest.mark.display
def test_export_dialog_needs_a_pattern_for_xray(tk_root, monkeypatch):
    """Only "Echem values" ticked is refused for X-ray data; a blank scan range means every scan."""
    shown, got = [], []
    monkeypatch.setattr(dialog_module.messagebox, "showerror",
                        lambda *a, **_k: shown.append(a))
    dlg = dialog_module.ExportOptionsDialog(tk_root, is_neutron=False, num_scans=5,
                                            on_export=lambda opts, dialog: got.append(opts))
    try:
        dlg.plot_vars["oned"].set(False)
        dlg.plot_vars["twod"].set(False)
        dlg._do_export()
        assert shown and not got
        dlg.plot_vars["twod"].set(True)
        dlg._do_export()
        assert got, "a 2D selection must still export"
        dlg.export_var.set("multiple")
        dlg.range_var.set("")
        dlg._do_export()
        assert got[-1].scan_list == [1, 2, 3, 4, 5], "a blank range means every scan"
    finally:
        if dlg.winfo_exists():
            dlg.destroy()
