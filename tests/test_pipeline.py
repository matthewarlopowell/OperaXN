"""End-to-end pipeline suite: raw instrument directories through echem
correlation and generation to canonical .nxs, and back out again via load.
"""
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

import core

import _builders
from _helpers import assert_pynxtools_valid


# ============================================================================
# Correlation
# ============================================================================

@pytest.fixture
def pipeline(inhouse_dir):
    """core.process_raw output over the synthetic in-house directory."""
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


@pytest.mark.parametrize("header,sample,expected", [
    (ARBIN_HEADER, None, {"time": 1, "voltage": 4, "current": 3}),
    ("time/s\tEwe/V\tI/mA", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Absolute Time\tVoltage\tCurrent", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Date\tTime\tVoltage\tCurrent", "05/02/2024\t10:00:00\t3.7\t0.1", {"time": 1}),
], ids=["arbin-date-time", "biologic", "simple", "split-date-clock"])
def test_detect_columns_time_slot(header, sample, expected):
    """One matrix for the time-slot rules: Arbin's Date_Time beats the
    elapsed-seconds column, single-time headers are untouched, and a bare
    Date column loses the slot to a clock column."""
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


_TS = [f"2024-02-05 10:{m:02d}:00" for m in range(5)]


def _arbin_rows(ts_list):
    """Arbin-shaped data rows for the given (possibly blank) timestamps."""
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
    """Timestamp-edge matrix: blank lines and blank cells drop cleanly, slash
    year-first dates are not day/month-swapped, a constant metadata datetime
    loses to the varying per-row column, and a malformed first cell recovers
    via later rows. Every surviving timestamp is real, unique, and ordered."""
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
    """Headers with no usable timestamp column parse to None (safe reject)
    rather than silently mis-correlating: a vacated index must not feed the
    positional defaults, and a bare date column never wins the slot from
    either side of the last-match-wins ordering."""
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
    for i, ts in enumerate(_builders.SCAN_TIMES, start=1):
        _builders.write_xrd_dat(str(data_dir / f"scan_{i:03d}.dat"), ts,
                                y=_builders.inhouse_scan_y(i))
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


needs_fabio = pytest.mark.skipif(not core.FABIO_AVAILABLE,
                                 reason="fabio not installed")


@needs_fabio
def test_zip_twod_references_survive(tmp_path):
    """ZIP-extracted 2D sources are persisted beside the .nxs; references
    into the deleted extraction tempdir would never be viewable."""
    src = str(tmp_path / "src")
    _builders.write_twod_dir(src)
    zpath = str(tmp_path / "cell.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for name in os.listdir(src):
            z.write(os.path.join(src, name), name)
    out = str(tmp_path / "zipped.nxs")
    ok, msgs = core.generate([zpath], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    m = core.load(out)
    source = str(m.scans[0].twod_source)
    assert os.path.isfile(source), f"2D reference is a dead path: {source}"
    assert source.startswith(str(tmp_path / "zipped_images")), source
    assert any("2D images copied" in msg for msg in msgs), msgs


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
    import glob
    import tempfile
    import uuid
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
    """Pipeline scans/echem plus the generated-and-reloaded in-house model."""
    scans, echem = core.process_raw([inhouse_dir], core.DataSourceType.INHOUSE)
    nxs = str(tmp_path / "inhouse.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE)
    assert ok, f"generate in-house nxs failed: {msgs}"
    return scans, echem, nxs, core.load(nxs)


@pytest.fixture
def neutron_rt(neutron_dir, tmp_path):
    """Generated-and-reloaded neutron model."""
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
    """Voltage, current, timestamp, and exposure survive write -> load per scan."""
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
    """Operando echem rows, voltages, and absolute timestamps survive reload."""
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
    """Pyflakes undefined-name sweep over core and operaxn: a definition
    removed as unused while later code still references it must fail loudly
    (the dialog.py logger regression class)."""
    pyflakes_api = pytest.importorskip("pyflakes.api")
    from pyflakes.reporter import Reporter
    import glob
    import io
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
#
# The entries are placeholders for public example datasets: the assertion
# floors are calibrated to the authors' local reference data and will be
# recalibrated as public datasets are released. Each test skips when the
# env var is unset or its dataset is absent.

def _need(relpath):
    """Path to a dataset under OPERAXN_TEST_DATA, or skip when absent."""
    root = os.environ.get("OPERAXN_TEST_DATA")
    if not root:
        pytest.skip("OPERAXN_TEST_DATA not set")
    path = os.path.join(root, relpath)
    if not os.path.exists(path):
        pytest.skip(f"{relpath} not present under {root}")
    return path


@pytest.mark.realdata
def test_neutron_zip(tmp_path):
    """The neutron zip yields >50 scans and >1000 echem rows; every
    scan carries error-bearing banks and some correlate to echem."""
    src = _need("neutron_tof.zip")
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
    """The synchrotron folder generates loadable scans and harvests
    the instrument name as i11-1."""
    src = _need("synchrotron_i11")
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
    """The lab XRD folder yields >400 scans with 1D patterns and >400
    with correlated echem."""
    src = _need("lab_xrd_full")
    nxs = str(tmp_path / "lab_full_real.nxs")
    ok, msgs = core.generate([src], nxs, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    fm = core.load(nxs)
    with_1d = [x for x in fm.scans if x.oned is not None]
    matched = [x for x in fm.scans if x.echem is not None]
    assert len(with_1d) > 400, f"{len(with_1d)}/{len(fm.scans)}"
    assert len(matched) > 400, f"{len(matched)}/{len(fm.scans)}"
    assert_pynxtools_valid(nxs, "lab full")
