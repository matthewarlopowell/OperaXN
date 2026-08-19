"""End-to-end pipeline suite: raw instrument directories through echem
correlation and generation to canonical .nxs, and back out again via load.
"""
import os

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
