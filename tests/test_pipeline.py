"""End-to-end pipeline: echem parsing, correlation, .nxs generation and reload."""
import glob
import io
import logging
import os
import tempfile
import uuid

import h5py
import numpy as np
import pandas as pd
import pytest

import core

import _builders
from _helpers import assert_pynxtools_valid, need_dataset, needs_fabio


# ============================================================================
# Correlation
# ============================================================================

@pytest.fixture
def pipeline(inhouse_dir):
    """Return core.process_raw output over the synthetic in-house directory."""
    return core.process_raw([inhouse_dir], core.DataSourceType.INHOUSE)


def test_process_raw_scan_count(pipeline):
    """Three XRD .dat files yield three scans."""
    scans, _ = pipeline
    assert len(scans) == 3, f"got {len(scans)}"


def test_process_raw_echem_rows(pipeline):
    """The 40-row echem file arrives intact in the combined frame."""
    _, echem = pipeline
    assert len(echem) == 40, f"got {len(echem)}"


def test_process_raw_scan1_voltage(pipeline):
    """Scan at 10:05 with 120 s exposure correlates at its 10:06 midpoint to V = 3.76."""
    scans, _ = pipeline
    assert abs(scans[0].echem - 3.76) < 1e-9, f"got {scans[0].echem}"


def test_tolerance_bounds_the_match_past_the_record_end(tmp_path):
    """A midpoint 200 s past the last echem point matches at the 300 s default but not at 100 s,
    in process_raw and generate alike; the file stores the tolerance with the echem record."""
    src = str(tmp_path / "late_scan")
    os.makedirs(src)
    # Echem covers 10:00-10:39 (last point 4.09 V); the 120 s scan starting
    # at 10:41:20 has its midpoint at 10:42:20, 200 s after that point
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    _builders.write_xrd_dat(os.path.join(src, "scan_001.dat"), "2024-02-05T10:41:20")

    scans, _ = core.process_raw([src], core.DataSourceType.INHOUSE)
    assert abs(scans[0].echem - 4.09) < 1e-9, scans[0].echem
    scans, _ = core.process_raw([src], core.DataSourceType.INHOUSE,
                                echem_time_tolerance=100)
    assert scans[0].echem is None and scans[0].current is None

    for tolerance, expected in ((300, 4.09), (100, None)):
        out = str(tmp_path / f"late_{tolerance}.nxs")
        ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE,
                                 echem_time_tolerance=tolerance)
        assert ok, str(msgs)
        with h5py.File(out) as f:
            stored = f["entry/operando_electrochemistry/echem_time_tolerance"]
            assert stored[()] == float(tolerance) and stored.attrs.get("units") == "s"
            env = f["entry/scan_000001/environment"]
            if expected is None:
                assert "voltage" not in env and "current" not in env
            else:
                assert abs(env["voltage"][()] - expected) < 1e-9
        loaded = core.load(out)
        assert loaded.global_metadata.get("echem_time_tolerance") == float(tolerance)
        got = loaded.scans[0].echem
        assert got is None if expected is None else abs(got - expected) < 1e-9, got


# ============================================================================
# Echem parsing
# ============================================================================

def test_echem_parse_builder_file(tmp_path):
    """A builder echem file parses to 40 dayfirst-timestamped voltage/current rows."""
    path = tmp_path / "echem.txt"
    _builders.write_echem_txt(str(path))
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 40, "expected 40 parsed rows"
    assert str(df["timestamp"].dtype).startswith("datetime64"), f"got {df['timestamp'].dtype}"
    first = df["timestamp"].iloc[0]
    assert first == pd.Timestamp("2024-02-05 10:00:00"), \
        f"dayfirst violated: 05/02/2024 must be 5 February, got {first}"
    assert np.allclose(df["echem_data"].values, 3.7 + 0.01 * np.arange(40))
    assert np.allclose(df["current"].values, 0.1 + 0.001 * np.arange(40))


@pytest.mark.parametrize("header", [True, False],
                         ids=["keyword-header", "headerless-positional"])
def test_echem_header_detection(tmp_path, header):
    """Keyword headers are skipped; headerless files use positional columns."""
    rows = ["Time\tEwe/V\tI/mA"] if header else []
    rows += [f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\t{0.1 + m * 0.001:.4f}"
             for m in range(5)]
    path = tmp_path / "echem.txt"
    path.write_text("\n".join(rows) + "\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 5, "header row must not count as data"
    assert np.allclose(df["echem_data"].values, 3.7 + 0.01 * np.arange(5))
    assert np.allclose(df["current"].values, 0.1 + 0.001 * np.arange(5))


def test_echem_missing_current_column(tmp_path):
    """A two-column file parses to None: rows shorter than the current index skip."""
    rows = ["Time\tEwe/V"]
    rows += [f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}" for m in range(5)]
    path = tmp_path / "nocurrent.txt"
    path.write_text("\n".join(rows) + "\n")
    assert core.EchemParser().parse(str(path)) is None


def test_echem_unparseable_current_is_none(tmp_path):
    """Rows with a non-numeric current field keep their voltage and get current=None."""
    rows = ["Time\tEwe/V\tI/mA"]
    rows += [f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\tN/A" for m in range(5)]
    path = tmp_path / "badcurrent.txt"
    path.write_text("\n".join(rows) + "\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 5
    assert df["current"].isna().all()
    assert np.allclose(df["echem_data"].values, 3.7 + 0.01 * np.arange(5))


ARBIN_HEADER = "Data_Point\tDate_Time\tTest_Time(s)\tCurrent(A)\tVoltage(V)"

# Full MITS-style export: derived columns (dV/dt carries the 'v/' substring)
# must never steal a slot from the real Voltage/Current columns
ARBIN_FULL_HEADER = ("Data_Point\tTest_Time(s)\tDate_Time\tCurrent(A)\t"
                     "Voltage(V)\tCharge_Capacity(Ah)\tDischarge_Capacity(Ah)\t"
                     "Charge_Energy(Wh)\tDischarge_Energy(Wh)\tdV/dt(V/s)\t"
                     "Internal_Resistance(Ohm)\tStep_Time(s)\tStep_Index\t"
                     "Cycle_Index")


@pytest.mark.parametrize("header,sample,expected", [
    (ARBIN_HEADER, None, {"time": 1, "voltage": 4, "current": 3}),
    ("time/s\tEwe/V\tI/mA", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Absolute Time\tVoltage\tCurrent", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Date\tTime\tVoltage\tCurrent", "05/02/2024\t10:00:00\t3.7\t0.1", {"time": 1}),
    (ARBIN_FULL_HEADER, None, {"time": 2, "voltage": 4, "current": 3}),
    ("dV/dt(V/s)\tDate_Time\tVoltage(V)\tCurrent(A)", None,
     {"time": 1, "voltage": 2, "current": 3}),
    ("Date_Time\tVoltage(V)\tCurrent(A)\tAux_Voltage_1(V)\tVoltage_Range"
     "\tCurrent_Range", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Date_Time\tAux_Voltage_1(V)\tCurrent_Range\tVoltage(V)\tCurrent(A)",
     None, {"time": 0, "voltage": 3, "current": 4}),
], ids=["arbin-date-time", "biologic", "simple", "split-date-clock",
        "arbin-full-export", "derived-column-first", "aux-and-range-after",
        "aux-and-range-before"])
def test_detect_columns_time_slot(header, sample, expected):
    """Column detection matrix: Arbin's Date_Time beats elapsed seconds, a bare Date loses to a
    clock column, and derived, aux and range columns never take the voltage or current slot."""
    columns = core.EchemParser()._detect_columns(header, sample)
    assert {k: columns[k] for k in expected} == expected, f"got {columns}"


def test_arbin_header_txt_parses(tmp_path):
    """An Arbin-style table parses to real 2024 timestamps, not elapsed seconds."""
    rows = [ARBIN_HEADER]
    rows += [f"{m + 1}\t2024-02-05 10:{m:02d}:00\t{m * 60.0:.1f}"
             f"\t{0.1 + m * 0.001:.4f}\t{3.7 + m * 0.01:.4f}" for m in range(10)]
    path = tmp_path / "arbin.txt"
    path.write_text("\n".join(rows) + "\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 10, "all rows must parse"
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-02-05 10:00:00"), \
        f"ISO timestamp mangled (elapsed-seconds column or day/month swap): " \
        f"{df['timestamp'].iloc[0]}"
    assert np.allclose(df["echem_data"].values, 3.7 + 0.01 * np.arange(10))
    assert np.allclose(df["current"].values, 1000 * (0.1 + 0.001 * np.arange(10))), \
        "Current(A) values must be scaled to the stored milliamp convention"


def test_arbin_full_export_parses_real_voltage(tmp_path):
    """A full MITS-style export parses its real Voltage(V) column, never the derived dV/dt."""
    rows = [ARBIN_FULL_HEADER]
    rows += [f"{m + 1}\t{m * 60.0:.1f}\t2024-02-05 10:{m:02d}:00"
             f"\t{0.1 + m * 0.001:.4f}\t{3.7 + m * 0.01:.4f}\t0.001\t0\t0.004"
             f"\t0\t-0.0001\t0.05\t0\t2\t1" for m in range(5)]
    path = tmp_path / "arbin_full.txt"
    path.write_text("\n".join(rows) + "\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 5
    assert np.allclose(df["echem_data"].values, 3.7 + 0.01 * np.arange(5)), \
        f"dV/dt hijacked the voltage slot: {list(df['echem_data'])}"
    assert np.allclose(df["current"].values, 1000 * (0.1 + 0.001 * np.arange(5)))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-02-05 10:00:00")


_TS = [f"2024-02-05 10:{m:02d}:00" for m in range(5)]


def _arbin_rows(ts_list):
    """Return Arbin-shaped data rows for the given (possibly blank) timestamps."""
    return [f"{i + 1}\t{ts}\t{i * 60.0:.1f}\t0.1\t{3.7 + i * 0.01:.4f}"
            for i, ts in enumerate(ts_list)]


@pytest.mark.parametrize("rows,n,first_ts", [
    ([ARBIN_HEADER, ""] + _arbin_rows(_TS[:3]), 3, _TS[0]),
    ([ARBIN_HEADER] + _arbin_rows(_TS[:2] + [""] + _TS[3:]), 4, _TS[0]),
    (["Time\tVoltage\tCurrent"] +
     [f"2024/02/05 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\t0.1" for m in range(3)],
     3, _TS[0]),
    (["Start_Date\tDate_Time\tTest_Time(s)\tCurrent(A)\tVoltage(V)"] +
     [f"2024-02-05 09:00:00\t{ts}\t{m * 60.0:.1f}\t0.1\t{3.7 + m * 0.01:.4f}"
      for m, ts in enumerate(_TS)], 5, _TS[0]),
    (["Date_Time\tTest_Time(s)\tVoltage\tCurrent", "\t0.0\t3.70\t0.1"] +
     [f"{ts}\t{m * 60.0:.1f}\t{3.7 + m * 0.01:.4f}\t0.1"
      for m, ts in enumerate(_TS[1:], start=1)], 4, _TS[1]),
], ids=["blank-line-after-header", "blank-timestamp-cell", "slash-year-first",
        "constant-metadata-datetime", "malformed-first-cell"])
def test_parse_timestamp_edge_cases(tmp_path, rows, n, first_ts):
    """Blank lines and cells drop, slash year-first dates are not day/month-swapped, a constant
    metadata datetime loses to the per-row column, and a malformed first cell recovers;
    every surviving timestamp is real and unique."""
    path = tmp_path / "echem.txt"
    path.write_text("\n".join(rows) + "\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == n, \
        f"got {None if df is None else len(df)} rows"
    assert df["timestamp"].notna().all(), "NaT must never reach the frame"
    assert df["timestamp"].nunique() == n, "timestamps collapsed onto one value"
    assert df["timestamp"].iloc[0] == pd.Timestamp(first_ts), \
        f"got {df['timestamp'].iloc[0]}"


@pytest.mark.parametrize("header,make_row", [
    ("Date_Time\tTest_Time(s)\tPotential(V)\tCurrent(A)",
     lambda m: f"2024-02-05 10:{m:02d}:00\t{m * 60.0:.1f}\t{3.7 + m * 0.01:.4f}\t0.1"),
    ("Data_Point\tTest_Time(s)\tStart_Date\tVoltage(V)\tCurrent(A)",
     lambda m: f"{m + 1}\t{m * 60.0:.1f}\t2024-02-05\t{3.7 + m * 0.01:.4f}\t0.1"),
    ("Start_Date\tTest_Time(s)\tVoltage\tCurrent",
     lambda m: f"2024-02-05\t{m * 60.0:.1f}\t{3.7 + m * 0.01:.4f}\t0.1"),
], ids=["vacated-index-default", "bare-date-last-match", "bare-date-first-match"])
def test_unusable_headers_fail_safe(tmp_path, header, make_row):
    """Unusable headers parse to None rather than mis-correlate: a vacated index never feeds the
    positional defaults and a bare date never wins in either match order."""
    rows = [header] + [make_row(m) for m in range(5)]
    path = tmp_path / "unusable.txt"
    path.write_text("\n".join(rows) + "\n")
    assert core.EchemParser().parse(str(path)) is None


@pytest.mark.parametrize("header,scale", [
    ("Data_Point\tDate_Time\tTest_Time(s)\tCurrent(A)\tVoltage(V)", 1000.0),
    ("Time\tVoltage\tCurrent(mA)", 1.0),
    ("time/s\tEwe/V\tI/mA", 1.0),
    ("Time\tVoltage\tCurrent", 1.0),
    ("Time\tVoltage\tArea (A)", 1.0),
], ids=["arbin-amps", "milliamps", "biologic", "unitless", "non-current-amps"])
def test_current_unit_scale(header, scale):
    """Only a parenthesised amp unit on the current column triggers scaling."""
    parser = core.EchemParser()
    columns = parser._detect_columns(header)
    assert parser._current_unit_scale(header, columns) == scale


def test_arbin_xlsx_pipeline_correlation(tmp_path):
    """An Arbin .xlsx correlates end to end like the .txt echem it mirrors."""
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    _builders.write_inhouse_scans(str(data_dir))
    _builders.write_arbin_xlsx(str(data_dir / "cellA.xlsx"))
    scans, echem = core.process_raw([str(data_dir)], core.DataSourceType.INHOUSE)
    assert len(scans) == 3 and len(echem) == 40, f"got {len(scans)} scans, {len(echem)} rows"
    assert abs(scans[0].echem - 3.76) < 1e-9, f"got {scans[0].echem}"
    assert abs(scans[0].current - 106.0) < 1e-6, \
        f"0.106 A at the 10:06 midpoint must correlate as 106 mA, got {scans[0].current}"


def test_convert_tabular_dest_dir(tmp_path):
    """dest_dir places the converted file where the caller manages cleanup."""
    xlsx = tmp_path / "cellB.xlsx"
    _builders.write_arbin_xlsx(str(xlsx))
    out_dir = tmp_path / "converted"
    os.makedirs(out_dir)
    txt = core.convert_tabular_to_txt(str(xlsx), dest_dir=str(out_dir))
    assert os.path.dirname(txt) == str(out_dir) and os.path.isfile(txt)
    df = core.EchemParser().parse(txt)
    assert df is not None and len(df) == 40


@needs_fabio
def test_zip_twod_references_survive(tmp_path):
    """ZIP-extracted 2D sources persist beside the .nxs, not in the deleted extraction tempdir."""
    zpath = _builders.write_twod_zip(str(tmp_path / "cell.zip"), tmp_path)
    out = str(tmp_path / "zipped.nxs")
    ok, msgs = core.generate([zpath], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    m = core.load(out)
    source = str(m.scans[0].twod_source)
    assert os.path.isfile(source), f"2D reference is a dead path: {source}"
    assert source.startswith(str(tmp_path / "zipped_images")), source
    assert any("2D images copied" in msg for msg in msgs), msgs


def test_missing_echem_warning_is_logged(tmp_path, caplog):
    """A run without echem still generates and logs the no-echem warning to the terminal."""
    src = tmp_path / "xrd_only"
    src.mkdir()
    _builders.write_inhouse_scans(str(src))
    out = str(tmp_path / "xrd_only.nxs")
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([str(src)], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    assert "Warning: No echem data available" in msgs
    assert any(r.levelno == logging.WARNING and r.getMessage() == "No echem data available"
               for r in caplog.records), [r.getMessage() for r in caplog.records]


@needs_fabio
def test_folder_twod_references_untouched(tmp_path):
    """Folder inputs keep referencing the user's own files; no images dir."""
    src = str(tmp_path / "src")
    _builders.write_twod_dir(src)
    out = str(tmp_path / "plain.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    m = core.load(out)
    assert os.path.samefile(str(m.scans[0].twod_source),
                            os.path.join(src, "image_001.edf"))
    assert not os.path.isdir(str(tmp_path / "plain_images"))


def test_xlsx_conversion_leaves_no_temp_files(tmp_path):
    """The pipeline's converted copies live in the session tempdir, not %TEMP%."""
    stem = f"oxleak_{uuid.uuid4().hex}"  # per-run stem: immune to old residue
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    _builders.write_xrd_dat(str(data_dir / "scan_001.dat"), _builders.SCAN_TIMES[0])
    _builders.write_arbin_xlsx(str(data_dir / f"{stem}.xlsx"))
    core.process_raw([str(data_dir)], core.DataSourceType.INHOUSE)
    pattern = os.path.join(tempfile.gettempdir(), f"{stem}_*.txt")
    assert glob.glob(pattern) == [], "conversion leaked a temp file"


# ============================================================================
# Round-trip
# ============================================================================

@pytest.fixture
def inhouse_rt(inhouse_dir, tmp_path):
    """Return pipeline scans/echem plus the generated-and-reloaded in-house .nxs and model."""
    scans, echem = core.process_raw([inhouse_dir], core.DataSourceType.INHOUSE)
    nxs = str(tmp_path / "inhouse.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE)
    assert ok, f"generate in-house nxs failed: {msgs}"
    return scans, echem, nxs, core.load(nxs)


@pytest.fixture
def neutron_rt(neutron_dir, tmp_path):
    """Return the generated-and-reloaded neutron model."""
    nxs = str(tmp_path / "neutron.nxs")
    ok, msgs = core.generate([neutron_dir], nxs, core.DataSourceType.NEUTRON)
    assert ok, f"generate neutron nxs failed: {msgs}"
    return core.load(nxs)


def test_inhouse_nxs_is_canonical(inhouse_rt):
    """A generated in-house file passes the canonical sniff and keeps its source."""
    _, _, nxs, model = inhouse_rt
    assert core.is_canonical_nxs(nxs)
    assert model.data_source == "inhouse", f"got {model.data_source}"


def test_inhouse_scan_fields_roundtrip(inhouse_rt):
    """Voltage, current, timestamp and exposure survive write -> load per scan."""
    scans, _, _, model = inhouse_rt
    assert len(model.scans) == 3, f"got {len(model.scans)}"
    for pipeline_scan, loaded in zip(scans, model.scans):
        assert loaded.echem == pipeline_scan.echem, \
            f"scan {loaded.scan_num}: file={loaded.echem} pipeline={pipeline_scan.echem}"
        assert loaded.current == pipeline_scan.current, f"scan {loaded.scan_num}: current"
        assert loaded.timestamp == pipeline_scan.timestamp, \
            f"scan {loaded.scan_num}: file={loaded.timestamp} pipeline={pipeline_scan.timestamp}"
        assert loaded.exposure_time == pipeline_scan.exposure_time, \
            f"scan {loaded.scan_num}: exposure"


def test_inhouse_oned_data_exact(inhouse_rt):
    """1D x/y arrays and the source filename round-trip exactly."""
    _, _, _, model = inhouse_rt
    s1 = model.scans[0]
    assert s1.oned is not None and np.allclose(s1.oned["x"], _builders.ONED_X)
    assert np.allclose(s1.oned["y"], _builders.inhouse_scan_y(1))
    assert s1.oned["source"] == "scan_001.dat", f"got {s1.oned['source']}"


def test_inhouse_echem_df_roundtrip(inhouse_rt):
    """Operando echem rows, voltages and absolute timestamps survive reload."""
    _, echem, _, model = inhouse_rt
    assert model.echem_df is not None and len(model.echem_df) == 40
    assert np.allclose(model.echem_df["echem_data"].values, echem["echem_data"].values)
    assert model.echem_df["timestamp"].iloc[0] == pd.Timestamp("2024-02-05 10:00:00"), \
        f"got {model.echem_df['timestamp'].iloc[0]}"


def test_neutron_scan_count(neutron_rt):
    """The single logbook entry with data files yields one loaded scan."""
    assert len(neutron_rt.scans) == 1, f"got {len(neutron_rt.scans)}"


def test_neutron_times_roundtrip(neutron_rt):
    """Logbook start/end and the derived midpoint timestamp survive reload."""
    ns = neutron_rt.scans[0]
    assert ns.neutron_start == "2024-02-05 10:00:00" \
        and ns.neutron_end == "2024-02-05 10:30:00", \
        f"got {ns.neutron_start} / {ns.neutron_end}"
    assert ns.timestamp == "2024-02-05 10:15:00", f"got {ns.timestamp}"


def test_neutron_correlated_voltage(neutron_rt):
    """Midpoint 10:15 correlates to echem V = 3.85."""
    ns = neutron_rt.scans[0]
    assert ns.echem is not None and abs(ns.echem - 3.85) < 1e-9, f"got {ns.echem}"


def test_neutron_banks_roundtrip(neutron_rt):
    """Both banks reload with tof+d entries and exact per-bank data arrays."""
    ns = neutron_rt.scans[0]
    assert ns.neutron is not None and set(ns.neutron.keys()) == {"1", "2"}, \
        f"got {None if ns.neutron is None else set(ns.neutron.keys())}"
    for bank in (1, 2):
        b = ns.neutron[str(bank)]
        assert set(b.keys()) == {"tof", "d"}, f"bank {bank}: {set(b.keys())}"
        assert np.allclose(b["tof"]["x"], _builders.TOF_X), f"bank {bank} tof x"
        assert np.allclose(b["tof"]["y"], 50.0 + bank + np.arange(40, dtype=float)), \
            f"bank {bank} tof y"
        assert np.allclose(b["d"]["x"], _builders.D_X), f"bank {bank} d x"
        assert np.allclose(b["d"]["y"], 70.0 + bank + np.arange(40, dtype=float)), \
            f"bank {bank} d y"


def test_default_cache_path_outside_data_dir(inhouse_dir):
    """The auto-cache .nxs path never lands inside the raw data directory."""
    cache = core.default_cache_path([inhouse_dir])
    assert not cache.startswith(inhouse_dir) and cache.endswith(".nxs"), cache


def test_no_undefined_names_in_packages():
    """Pyflakes finds no undefined names in core or operaxn, so a definition removed while still
    referenced fails loudly (the dialog.py logger regression)."""
    pyflakes_api = pytest.importorskip("pyflakes.api")
    from pyflakes.reporter import Reporter
    root = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
    issues = []
    for pkg in ("core", "operaxn"):
        for path in glob.glob(os.path.join(root, pkg, "*.py")):
            out, err = io.StringIO(), io.StringIO()
            pyflakes_api.checkPath(path, Reporter(out, err))
            issues += [line for line in out.getvalue().splitlines()
                       if "undefined name" in line]
    assert issues == [], "\n".join(issues)


# ============================================================================
# Real datasets
# ============================================================================
# Smoke tests over local instrument datasets, run only when
# OPERAXN_TEST_DATA points at them. Expected layout under $OPERAXN_TEST_DATA:
#     neutron_tof.zip     time-of-flight neutron export (zip): logbook +
#                         per-bank TOF/d files + echem
#     synchrotron_i11/    Diamond i11-1 synchrotron dataset folder (.nxs +
#                         .hdf + integrated .xy files)
#     lab_xrd_full/       laboratory XRD full dataset (1D .dat + 2D EDF +
#                         echem: the complete workflow)
#     polaris/            Mantid-processed POLARIS file for the instrument
#                         profile check (POLARIS164798-164802.nxs;
#                         test_polaris_profile_matches_mantid_file)
#
# The entries are placeholders for public example datasets: the assertion
# floors are calibrated to the authors' local reference data and will be
# recalibrated as public datasets are released. Each test skips when the
# env var is unset or its dataset is absent.

@pytest.mark.realdata
def test_neutron_zip(tmp_path):
    """The neutron zip yields >50 scans and >1000 echem rows; every scan carries error-bearing
    banks, some correlate to echem, and the file validates."""
    src = need_dataset("neutron_tof.zip")
    nxs = str(tmp_path / "neutron_real.nxs")
    ok, msgs = core.generate([src], nxs, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    rm = core.load(nxs)
    assert len(rm.scans) > 50, f"got {len(rm.scans)}"
    assert rm.echem_df is not None and len(rm.echem_df) > 1000, \
        f"got {None if rm.echem_df is None else len(rm.echem_df)}"
    matched = [x for x in rm.scans if x.echem is not None]
    assert len(matched) > 0, f"{len(matched)}/{len(rm.scans)} matched"
    with_window = [x for x in matched if x.voltage_min is not None]
    assert len(with_window) > 0, f"{len(with_window)}/{len(matched)}"
    with_banks = [x for x in rm.scans if x.neutron]
    with_errors = [x for x in with_banks
                   if any("e" in t for bk in x.neutron.values() for t in bk.values())]
    assert len(with_banks) == len(rm.scans)
    assert len(with_errors) == len(with_banks)
    assert_pynxtools_valid(nxs, "neutron zip")


@pytest.mark.realdata
def test_synchrotron_folder(tmp_path):
    """The synchrotron folder generates loadable scans and harvests the instrument name i11-1."""
    src = need_dataset("synchrotron_i11")
    nxs = str(tmp_path / "synchrotron_real.nxs")
    ok, msgs = core.generate([src], nxs, core.DataSourceType.SYNCHROTRON)
    assert ok, str(msgs)
    dm = core.load(nxs)
    assert len(dm.scans) > 0, f"got {len(dm.scans)}"
    inst = dm.global_metadata.get("instrument", {})
    assert inst.get("name") == "i11-1", str(inst.get("name"))
    assert_pynxtools_valid(nxs, "synchrotron folder")


@pytest.mark.realdata
def test_lab_full_folder(tmp_path):
    """The lab XRD folder yields >400 scans with 1D patterns and >400 with correlated echem."""
    src = need_dataset("lab_xrd_full")
    nxs = str(tmp_path / "lab_full_real.nxs")
    ok, msgs = core.generate([src], nxs, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    fm = core.load(nxs)
    with_1d = [x for x in fm.scans if x.oned is not None]
    matched = [x for x in fm.scans if x.echem is not None]
    assert len(with_1d) > 400, f"{len(with_1d)}/{len(fm.scans)}"
    assert len(matched) > 400, f"{len(matched)}/{len(fm.scans)}"
    assert_pynxtools_valid(nxs, "lab full")


@pytest.mark.realdata
def test_polaris_profile_matches_mantid_file():
    """The shipped POLARIS constants agree with a Mantid-processed POLARIS
    file's focussed banks (IDF-wide vs focussed means: 2 mm, 0.06 deg)."""
    derived = core.profile_from_mantid_nexus(
        need_dataset("polaris/POLARIS164798-164802.nxs"))
    shipped = core.config.INSTRUMENT_PROFILES["polaris"]
    assert derived["instrument_name"] == "POLARIS"
    assert derived["pre_sample_flightpath_m"] == shipped["pre_sample_flightpath_m"]
    assert sorted(derived["detector_banks"]) == sorted(shipped["detector_banks"])
    for bank, spec in shipped["detector_banks"].items():
        got = derived["detector_banks"][bank]
        assert abs(got["distance_m"] - spec["distance_m"]) <= 0.0025, (bank, got)
        assert abs(got["polar_angle_deg"] - spec["polar_angle_deg"]) <= 0.06, (bank, got)
        assert got["azimuthal_angle_deg"] is None, (bank, got)
    assert derived["info"]["idf_name"] == "Polaris_upgrade"


def test_neutron_midpoint_not_floored(tmp_path):
    """An odd-length run correlates at its exact half-second midpoint, not the floored one
    (flooring picked the wrong echem point for 75 of 108 POLARIS runs)."""
    src = str(tmp_path / "odd_run")
    os.makedirs(src)
    # 7 s run: exact midpoint 10:00:03.5, floored 10:00:03
    with open(os.path.join(src, "logbook.txt"), "w") as f:
        f.write("123456\tuser\tsample title\tMon Feb 05 10:00:00 2024\t"
                "Mon Feb 05 10:00:07 2024\t7\tok\textra\n")
    _builders.write_neutron_banks(src)
    # Points at 03.0 s and 03.9 s: the exact midpoint is nearer 03.9
    with open(os.path.join(src, "echem.txt"), "w") as f:
        f.write("Time\tEwe/V\tI/mA\n"
                "05/02/2024 09:59:00.0\t3.0000\t0.1000\n"
                "05/02/2024 10:00:03.0\t3.0000\t0.1000\n"
                "05/02/2024 10:00:03.9\t3.9000\t0.1000\n"
                "05/02/2024 10:01:00.0\t3.0000\t0.1000\n")
    scans, _ = core.process_raw([src], core.DataSourceType.NEUTRON)
    assert len(scans) == 1
    assert scans[0].timestamp_for_correlation == pd.Timestamp("2024-02-05 10:00:03.5")
    assert scans[0].echem == 3.9, f"correlated at the floored midpoint: {scans[0].echem}"
    out = str(tmp_path / "odd_run.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        env = f["entry/scan_000001/environment"]
        stored = {k: pd.to_datetime(env[k][()].decode()).tz_localize(None)
                  for k in ("scan_timestamp", "midpoint_adjusted_timestamp")}
        assert stored["scan_timestamp"] == stored["midpoint_adjusted_timestamp"]
        assert stored["scan_timestamp"] == pd.Timestamp("2024-02-05 10:00:03.5")
        assert abs(env["voltage"][()] - 3.9) < 1e-9


def test_window_extrema_span_the_window(pipeline):
    """Voltage/current extrema cover every echem point in the window, both bounds inclusive:
    scan 1 (10:05, 120 s) sees 10:05 to 10:07 and its matched value lies within the range."""
    scans, _ = pipeline
    s1 = scans[0]
    assert (s1.voltage_min, s1.voltage_max) == (3.75, 3.77)
    assert (s1.current_min, s1.current_max) == (0.105, 0.107)
    assert s1.voltage_min <= s1.echem <= s1.voltage_max


def test_window_extrema_absent_without_exposure(tmp_path):
    """A scan with no header exposure has no extrema but keeps its nearest-point voltage."""
    src = str(tmp_path / "no_exposure")
    os.makedirs(src)
    lines = ["# Date 2024-02-05T10:05:00", "# tth Intensity"]
    lines += [f"{x:.4f} {v:.4f}"
              for x, v in zip(_builders.ONED_X, _builders.inhouse_scan_y(1))]
    with open(os.path.join(src, "scan_001.dat"), "w") as f:
        f.write("\n".join(lines) + "\n")
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    scans, _ = core.process_raw([src], core.DataSourceType.INHOUSE)
    assert len(scans) == 1 and scans[0].exposure_time is None
    assert scans[0].echem == 3.75
    assert scans[0].voltage_min is None and scans[0].voltage_max is None


def test_window_extrema_follow_relative_alignment(tmp_path, inhouse_dir):
    """Relative-mode extrema use the correlation's clock shift: the earliest midpoint maps onto
    the first echem point, so scan 1 matches 3.70 V inside a 3.70 to 3.71 V window."""
    out = str(tmp_path / "relative.nxs")
    ok, msgs = core.generate([inhouse_dir], out, core.DataSourceType.INHOUSE,
                             time_method=core.TimeMethod.RELATIVE)
    assert ok, str(msgs)
    s1 = core.load(out).scans[0]
    assert s1.echem == 3.70
    assert (s1.voltage_min, s1.voltage_max) == (3.70, 3.71)
    assert s1.voltage_min <= s1.echem <= s1.voltage_max


def test_synchrotron_exposure_is_the_scan_window(tmp_path):
    """A synchrotron scan's exposure is its recorded end minus start (readout and processing
    included), never a device-named NXdetector's count_time (i11-1: pixium_hdf), so the
    correlation midpoint stays centred on the recorded window."""
    src = str(tmp_path / "synchrotron")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00",
                                     "2024-02-05T10:05:39", count_time=30.0)
    _builders.write_synchrotron_scan(src, "000002", "2024-02-05T10:10:00",
                                     "2024-02-05T10:10:39")
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    scans, _ = core.process_raw([src], core.DataSourceType.SYNCHROTRON)
    assert [sc.exposure_time for sc in scans] == [39.0, 39.0], \
        [sc.exposure_time for sc in scans]


def test_file_cache_replacing_a_key_counts_it_once():
    """Re-putting a key (two threads reading one file) keeps the byte count to what is held."""
    import sys
    import threading
    from core.readers import FileCache
    cache = FileCache(max_size_mb=1)
    for _ in range(3):
        cache.put("same", np.zeros(50_000))
    assert cache.current_size == sys.getsizeof(cache.cache["same"])
    cache.put("other", np.zeros(80_000))
    assert sorted(cache.cache) == ["other", "same"], "phantom bytes must not force an eviction"
    # A larger replacement evicts by count; the replaced key's stale count must not stall it
    cache.get("other")
    cache.get("other")
    worker = threading.Thread(target=cache.put, args=("same", np.zeros(100_000)), daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "put must not spin on a stale access count"
    assert sorted(cache.cache) == ["same"], "the least-accessed entry gives way"


def test_generate_rejects_an_unusable_tolerance(tmp_path, caplog):
    """Direct API callers get a clear error, not a logged traceback, for a tolerance the
    correlator cannot use."""
    src = str(tmp_path / "inhouse")
    _builders.write_inhouse_dir(src)
    for bad in (None, "300", 0, -5, float("nan"), float("inf"), True, 300j, 1e10, 10**400):
        caplog.clear()
        ok, msgs = core.generate([src], str(tmp_path / "bad.nxs"),
                                 core.DataSourceType.INHOUSE, echem_time_tolerance=bad)
        assert not ok and "echem_time_tolerance" in msgs[0], (bad, msgs)
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], bad
        with pytest.raises(ValueError, match="echem_time_tolerance"):
            core.process_raw([src], core.DataSourceType.INHOUSE, echem_time_tolerance=bad)
    assert not os.path.exists(tmp_path / "bad.nxs")
    ok, msgs = core.generate([src], str(tmp_path / "over.nxs"), core.DataSourceType.INHOUSE,
                             echem_time_tolerance=1e10)
    assert "up to 1,000,000,000" in msgs[0], "the message names the limit"
    ok, msgs = core.generate([src], str(tmp_path / "limit.nxs"), core.DataSourceType.INHOUSE,
                             echem_time_tolerance=1e9)
    assert ok, "the limit itself is usable"


def test_regenerating_keeps_the_embedded_image_shape(tmp_path):
    """A second generate of one synchrotron image in a session still records original_shape
    (a cached read skips the step that measures it)."""
    src = str(tmp_path / "synchrotron")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00",
                                     "2024-02-05T10:05:39")
    with h5py.File(os.path.join(src, "pixium-000001.hdf"), "w") as f:
        f["entry/instrument/detector/data"] = np.ones((16, 16), dtype=np.float32)
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    for run in range(2):
        out = str(tmp_path / f"run{run}.nxs")
        ok, msgs = core.generate([src], out, core.DataSourceType.SYNCHROTRON,
                                 include_2d_images=True)
        assert ok, str(msgs)
        with h5py.File(out) as f:
            note = f["entry/scan_000001/image_source"]
            assert "original_shape" in note, f"run {run} lost original_shape"
            assert list(note["original_shape"][()]) == [16, 16]
