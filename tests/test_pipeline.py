"""End-to-end pipeline: ingest, timestamps, echem, correlation, readers, round-trip, examples."""
import ctypes
import filecmp
import glob
import importlib
import importlib.util
import io
import logging
import os
import shutil
import tempfile
import threading
import uuid
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import core
import core.readers as readers
from core import nxs_writer

import _builders
from _helpers import assert_pynxtools_valid, need_dataset, needs_fabio


# ============================================================================
# Correlation
# ============================================================================

@pytest.fixture
def pipeline(inhouse_dir):
    """Return core.process_raw output over the synthetic in-house directory."""
    return core.process_raw([inhouse_dir], core.DataSourceType.INHOUSE)


def test_process_raw_correlates_the_inhouse_dir(pipeline):
    """Three scans and 40 echem rows load; scan 1 matches its 10:06 midpoint at 3.76 V."""
    scans, echem = pipeline
    assert len(scans) == 3, f"got {len(scans)}"
    assert len(echem) == 40, f"got {len(echem)}"
    assert abs(scans[0].echem - 3.76) < 1e-9, f"got {scans[0].echem}"


def test_partial_echem_match_is_counted(tmp_path, caplog):
    """When only some scans match within the tolerance, one INFO line counts them; else none."""
    src = str(tmp_path / "partial")
    os.makedirs(src)
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    _builders.write_xrd_dat(os.path.join(src, "scan_001.dat"), "2024-02-05T10:05:00")
    # Midpoint 200 s past the last echem point (as in the test below)
    _builders.write_xrd_dat(os.path.join(src, "scan_002.dat"), "2024-02-05T10:41:20")

    def counted(**kwargs):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="core"):
            core.process_raw([src], core.DataSourceType.INHOUSE, **kwargs)
        return [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("Echem matched")]

    assert counted(echem_time_tolerance=100) == ["Echem matched 1 of 2 scans within 100 s"]
    assert counted() == [], "every scan matched: nothing to count"


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


def test_echem_that_matches_no_scan_is_reported_once(tmp_path, caplog):
    """No match in absolute mode names both ranges once; a match, relative or no echem is silent."""
    src = tmp_path / "lab"
    _builders.write_inhouse_dir(str(src))
    # The same log exported month-first: 02/05/2024 reads day-first as 2 May
    _builders.write_echem_txt(str(src / "echem.txt"), date="02/05/2024")
    runs = {}
    for method in (core.TimeMethod.ABSOLUTE, core.TimeMethod.RELATIVE):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="core"):
            scans, _ = core.process_raw([str(src)], core.DataSourceType.INHOUSE, time_method=method)
        found = [r.getMessage() for r in caplog.records if "No scan matched" in r.getMessage()]
        runs[method] = ([s.echem is not None for s in scans], found)
    assert runs[core.TimeMethod.ABSOLUTE] == ([False] * 3, [
        "No scan matched the echem data: 3 scans from 2024-02-05 10:06 to 2024-02-05 10:16, "
        "echem from 2024-05-02 10:00 to 2024-05-02 10:39, tolerance 300 s; check the echem date "
        "format and the time method"])
    assert runs[core.TimeMethod.RELATIVE] == ([True] * 3, [])
    for prepare in (lambda: _builders.write_echem_txt(str(src / "echem.txt")),
                    lambda: os.remove(src / "echem.txt")):
        prepare()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="core"):
            core.process_raw([str(src)], core.DataSourceType.INHOUSE)
        assert not [r for r in caplog.records if "No scan matched" in r.getMessage()]


def test_scan_processor_keeps_absolute_timestamps_in_relative_mode(inhouse_dir):
    """A direct ScanProcessor caller in RELATIVE mode gets absolute scan and echem timestamps;
    relative display is a GUI view."""
    with core.FileProcessor(core.DataSourceType.INHOUSE) as processor:
        records = processor.process_paths([inhouse_dir])
        classified = core.FileClassificationManager(core.DataSourceType.INHOUSE).classify_files(
            pd.DataFrame([r.__dict__ for r in records]))
        scans, echem = core.ScanProcessor(core.TimeMethod.RELATIVE).process_scans(classified)
    assert scans and all(pd.Timestamp(s.timestamp).year == 2024 for s in scans), \
        [s.timestamp for s in scans]
    assert pd.api.types.is_datetime64_any_dtype(echem["timestamp"]), echem["timestamp"].dtype
    assert scans[0].echem is not None, "relative matching still correlates"


def test_offset_echem_rows_correlate_by_wall_clock(tmp_path):
    """ISO echem rows carrying +00:00 parse to naive wall clock and match naive scans."""
    src = str(tmp_path / "offset_echem")
    os.makedirs(src)
    _builders.write_inhouse_scans(src)
    rows = ["Time\tEwe/V\tI/mA"] + [
        f"2024-02-05T10:{m:02d}:00+00:00\t{3.7 + m * 0.01:.4f}\t0.1" for m in range(40)]
    with open(os.path.join(src, "echem.txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    scans, echem = core.process_raw([src], core.DataSourceType.INHOUSE)
    assert echem["timestamp"].dt.tz is None
    assert [round(s.echem, 4) for s in scans] == [3.76, 3.81, 3.86]


@pytest.mark.parametrize("stamp", ["2024-02-05T10:05:00+00:00", "2024-02-05T10:05:00Z",
                                   "2024-02-05T10:05:00-05:00", "2024-02-05T10:05:00+01"],
                         ids=["plus-offset", "zulu", "minus-offset", "short-offset"])
def test_scan_time_with_a_utc_offset_correlates(tmp_path, stamp):
    """A scan time carrying an offset keeps its wall clock and correlates."""
    src = tmp_path / "lab"
    os.makedirs(src)
    _builders.write_lab_dat(str(src / "scan_001.dat"), stamp)
    _builders.write_eclab_echem(str(src / "echem.txt"))
    out = str(tmp_path / "tz.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert abs(core.load(out).scans[0].echem - 3.76) < 1e-9


def test_negative_offset_synchrotron_times_correlate(tmp_path):
    """A '-05:00' synchrotron start time (UTC-minus beamline) generates and correlates."""
    src = tmp_path / "syn"
    os.makedirs(src)
    _builders.write_synchrotron_scan(str(src), "000001", "2024-02-05T10:05:00-05:00",
                                     "2024-02-05T10:05:39-05:00")
    _builders.write_eclab_echem(str(src / "echem.txt"))
    out = str(tmp_path / "syn.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.SYNCHROTRON)
    assert ok, msgs
    assert core.load(out).scans[0].echem is not None


def _warning_messages(caplog):
    """Return the WARNING and higher messages caplog holds."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_synchrotron_scan_without_start_time_is_dropped(tmp_path, caplog):
    """A scan .nxs with no start_time is left out with a warning naming it, and the file validates."""
    src = tmp_path / "syn"
    os.makedirs(src)
    _builders.write_synchrotron_scan(str(src), "000001", "2024-02-05T10:05:00",
                                     "2024-02-05T10:05:39")
    _builders.write_synchrotron_scan(str(src), "000002", "2024-02-05T10:10:00",
                                     "2024-02-05T10:10:39")
    with h5py.File(src / "i11-1-000002.nxs", "a") as f:
        # end_time alone would still time the scan (it is a timestamp fallback), so both go
        del f["entry1/start_time"]
        del f["entry1/end_time"]
    _builders.write_eclab_echem(str(src / "echem.txt"))
    out = str(tmp_path / "syn.nxs")
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([str(src)], out, core.DataSourceType.SYNCHROTRON)
    assert ok, msgs
    assert len(core.load(out).scans) == 1
    messages = _warning_messages(caplog)
    assert "1 scan has no start time and was skipped: i11-1-000002.nxs" in messages, messages
    assert_pynxtools_valid(out, "synchrotron without start_time")


@pytest.mark.parametrize("change", ["unreadable-end-time", "start-time-as-array"])
def test_synchrotron_scan_with_a_readable_start_time_is_kept(tmp_path, change):
    """An unreadable end_time costs only the exposure; a 1-element start_time array is decoded."""
    src = str(tmp_path / "syn")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    _builders.write_synchrotron_scan(src, "000002", "2024-02-05T10:10:00", "2024-02-05T10:10:39")
    with h5py.File(os.path.join(src, "i11-1-000002.nxs"), "a") as f:
        if change == "unreadable-end-time":
            del f["entry1/end_time"]
            f["entry1/end_time"] = b"not recorded"
        else:
            del f["entry1/start_time"]
            f["entry1/start_time"] = np.array([b"2024-02-05T10:10:00"])
    scans, _ = core.process_raw([src], core.DataSourceType.SYNCHROTRON)
    assert [pd.Timestamp(s.timestamp) for s in scans] == [
        pd.Timestamp("2024-02-05 10:05:00"), pd.Timestamp("2024-02-05 10:10:00")]
    assert scans[1].exposure_time == (None if change == "unreadable-end-time" else 39.0)


def test_synchrotron_inf_exposure_dataset_counts_as_none(tmp_path):
    """An exposure_time dataset of inf is no exposure: end_time minus start_time gives it."""
    src = str(tmp_path / "syn")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    with h5py.File(os.path.join(src, "i11-1-000001.nxs"), "a") as f:
        f["entry1/instrument/detector/exposure_time"] = np.inf
    scans, _ = core.process_raw([src], core.DataSourceType.SYNCHROTRON)
    assert [s.exposure_time for s in scans] == [39.0]


def test_synchrotron_scan_with_an_unreadable_start_time_is_timed_from_its_end_time(tmp_path):
    """A start_time that does not read as a date falls back to end_time; the load goes on."""
    src = str(tmp_path / "syn")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    _builders.write_synchrotron_scan(src, "000002", "2024-02-05T10:10:00", "2024-02-05T10:10:39")
    with h5py.File(os.path.join(src, "i11-1-000002.nxs"), "a") as f:
        del f["entry1/start_time"]
        f["entry1/start_time"] = b"not recorded"
    scans, _ = core.process_raw([src], core.DataSourceType.SYNCHROTRON)
    assert [pd.Timestamp(s.timestamp) for s in scans] == [
        pd.Timestamp("2024-02-05 10:05:00"), pd.Timestamp("2024-02-05 10:10:39")]
    assert scans[1].exposure_time is None


@pytest.mark.parametrize("dat_exposure, edf_exposure", [
    pytest.param("inf", None, id="dat-inf"), pytest.param("-120", None, id="dat-negative"),
    pytest.param("120.0", "inf", id="edf-inf", marks=needs_fabio)])
def test_unusable_header_exposure_counts_as_none(tmp_path, dat_exposure, edf_exposure):
    """A non-finite or non-positive header exposure is no exposure: the scan correlates at its start."""
    src = tmp_path / "lab"
    src.mkdir()
    _builders.write_xrd_dat(str(src / "scan_001.dat"), _builders.SCAN_TIMES[0], exposure=dat_exposure)
    if edf_exposure is not None:
        _builders.write_edf_image(str(src / "image_001.edf"), _builders.SCAN_TIMES[0],
                                  ExposureTime=edf_exposure)
    _builders.write_echem_txt(str(src / "echem.txt"))
    out = str(tmp_path / "lab.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert abs(core.load(out).scans[0].echem - 3.75) < 1e-9, core.load(out).scans[0].echem


# ============================================================================
# Scan timestamps
# ============================================================================

def _stored_timestamps(paths):
    """Return the timestamp classify_files stores per path, None for a skipped file."""
    df = pd.DataFrame({"path": [str(p) for p in paths]})
    out = core.FileClassificationManager(core.DataSourceType.INHOUSE).classify_files(df)
    return [ts if isinstance(ts, str) else None for ts in out["timestamp"]]


@needs_fabio
@pytest.mark.parametrize("value, expected", [
    pytest.param("2024-02-05 10:05:00 GMT", "2024-02-05 10:05:00", id="GMT"),
    pytest.param("2024-02-05T10:05:00 UTC", "2024-02-05 10:05:00", id="UTC"),
    pytest.param("Tue Feb  6 10:05:00 2024", "2024-02-06 10:05:00", id="ctime-Tue"),
    pytest.param("Thu Feb  8 10:05:00 2024", "2024-02-08 10:05:00", id="ctime-Thu"),
    pytest.param("20240205T100500", "2024-02-05 10:05:00", id="iso-basic"),
    pytest.param("20240205T1005", "2024-02-05 10:05:00", id="iso-basic-no-seconds"),
    pytest.param("20240205", "2024-02-05 00:00:00", id="iso-basic-date"),
    pytest.param("20240205 100500", "2024-02-05 10:05:00", id="basic-with-space"),
    pytest.param("2024.02.05 10:05:00", "2024-02-05 10:05:00", id="dotted"),
    pytest.param("20240205100500", "2024-02-05 10:05:00", id="compact-14-digits"),
    pytest.param("202402051005", "2024-02-05 10:05:00", id="compact-12-digits"),
    pytest.param("2024 02 05 10:05:00", "2024-02-05 10:05:00", id="space-separated"),
])
def test_header_dates_read_alike_in_dat_and_edf(tmp_path, value, expected):
    """A .dat and an EDF read a header date alike: 'T' is a separator only between digits, year-first never swaps."""
    dat, edf = tmp_path / "scan.dat", tmp_path / "scan.edf"
    _builders.write_xrd_dat(str(dat), value)
    _builders.write_edf_image(str(edf), value)
    for ts in _stored_timestamps([dat, edf]):
        assert ts is not None and pd.Timestamp(ts) == pd.Timestamp(expected), ts


@needs_fabio
@pytest.mark.parametrize("value", ["2024-06-05T10:05:00z", "2024-06-05T10:05:00Z"])
def test_lower_case_z_is_dropped_like_upper_case(tmp_path, value):
    """A 'z' suffix is dropped like 'Z', so the writer and the correlation read one wall clock."""
    dat, edf = tmp_path / "scan.dat", tmp_path / "scan.edf"
    _builders.write_xrd_dat(str(dat), value)
    _builders.write_edf_image(str(edf), value)
    for ts in _stored_timestamps([dat, edf]):
        assert ts == "2024-06-05 10:05:00", ts
        assert pd.Timestamp(ts).tzinfo is None


@needs_fabio
@pytest.mark.parametrize("value", [
    "2024-02-05T10:05:00", "05/02/2024 10:05:00", "02/13/2024 10:05:00", "5-Feb-2024 10:05:00",
    "Tue Feb  6 10:05:00 2024", "05/02/2024 10:05:00 PM", "2024-02-05 10:05:00 +0100", "10:05:00",
])
def test_dat_and_edf_twins_store_one_timestamp(tmp_path, value):
    """A .dat and an EDF with the same Date text store the same timestamp, so they pair."""
    dat, edf = tmp_path / "scan.dat", tmp_path / "scan.edf"
    _builders.write_xrd_dat(str(dat), value)
    _builders.write_edf_image(str(edf), value)
    dat_ts, edf_ts = _stored_timestamps([dat, edf])
    assert dat_ts is not None and dat_ts == edf_ts, (dat_ts, edf_ts)


@needs_fabio
@pytest.mark.parametrize("template, echem_date", [
    pytest.param("05/02/2024 10:{m}:00", "05/02/2024", id="day-first"),
    pytest.param("Tue Feb  6 10:{m}:00 2024", "06/02/2024", id="ctime-Tue"),
    pytest.param("5-Feb-2024 10:{m}:00", "05/02/2024", id="month-name"),
    pytest.param("2024-02-05 10:{m}:00 UTC", "05/02/2024", id="UTC"),
])
def test_non_iso_lab_pairs_keep_pairing(tmp_path, template, echem_date):
    """A .dat and its EDF with the same non-ISO Date pair into one scan and correlate."""
    dates = [template.format(m=m) for m in ("05", "10", "15")]
    src = str(tmp_path / "pairs")
    _builders.write_dated_pairs(src, dates, dates, echem_date)
    scans, _ = core.process_raw([src], core.DataSourceType.INHOUSE)
    assert [(bool(s.oned), bool(s.twod)) for s in scans] == [(True, True)] * 3, \
        [s.timestamp for s in scans]
    assert [round(s.echem, 2) for s in scans] == [3.76, 3.81, 3.86]


def test_single_timestamps_skip_format_guessing(monkeypatch):
    """Reader and writer timestamp helpers parse single values without pd.to_datetime."""
    def boom(*a, **k):
        raise AssertionError("pd.to_datetime called for a single value")
    monkeypatch.setattr(pd, "to_datetime", boom)
    assert core.nxs_reader._display_ts("2024-02-05T10:05:00+00:00") == "2024-02-05 10:05:00"
    assert core.nxs_reader._display_ts("01:02:03") == "01:02:03"
    assert core.nxs_reader._display_ts("Date") == "Date"
    assert core.nxs_writer._iso("2024-02-05 10:05:00").startswith("2024-02-05T10:05:00")
    assert core.nxs_writer._iso("Date") is None
    assert core.nxs_writer._ts_sort_key("Date") == (1, pd.Timestamp.max)
    scan = core.Scan(scan_num=1, timestamp="2024-02-05 10:05:00", exposure_time=120.0)
    assert core.nxs_writer._scan_window(scan)[1] == pd.Timestamp("2024-02-05 10:07:00")


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


@pytest.mark.parametrize("header", [True, False], ids=["headed", "header-less"])
def test_echem_missing_current_column(tmp_path, header):
    """A two-column (voltage-only) file parses to None, with or without a header."""
    rows = ["Time\tEwe/V"] if header else []
    rows += [f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}" for m in range(5)]
    path = tmp_path / "nocurrent.txt"
    path.write_text("\n".join(rows) + "\n")
    assert core.EchemParser().parse(str(path)) is None


@pytest.mark.parametrize("tail", ["\t", ""], ids=["blank-current", "missing-current"])
def test_echem_row_with_a_blank_last_current_keeps_its_time_and_voltage(tmp_path, tail):
    """A row whose current, its last cell, is blank or missing keeps its time and voltage."""
    path = tmp_path / "echem.txt"
    path.write_text("time/s\tEwe/V\tI/mA\n05/02/2024 10:00:00\t3.70\t1.0\n"
                    f"05/02/2024 10:01:00\t3.71{tail}\n05/02/2024 10:02:00\t3.72\t1.0\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 3, None if df is None else len(df)
    assert df["timestamp"].iloc[1] == pd.Timestamp("2024-02-05 10:01:00")
    assert df["echem_data"].iloc[1] == 3.71 and np.isnan(df["current"].iloc[1])
    assert list(df["current"].iloc[[0, 2]]) == [1.0, 1.0]


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
    ("time/s\tEwe/V\tI/mA", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Absolute Time\tVoltage\tCurrent", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Date\tTime\tVoltage\tCurrent", "05/02/2024\t10:00:00\t3.7\t0.1", {"time": 1}),
    ("dV/dt(V/s)\tDate_Time\tVoltage(V)\tCurrent(A)", None,
     {"time": 1, "voltage": 2, "current": 3}),
    ("Date_Time\tVoltage(V)\tCurrent(A)\tAux_Voltage_1(V)\tVoltage_Range"
     "\tCurrent_Range", None, {"time": 0, "voltage": 1, "current": 2}),
    ("Date_Time\tAux_Voltage_1(V)\tCurrent_Range\tVoltage(V)\tCurrent(A)",
     None, {"time": 0, "voltage": 3, "current": 4}),
], ids=["biologic", "simple", "split-date-clock",
        "derived-column-first", "aux-and-range-after",
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


@pytest.mark.parametrize("kind", ["eclab-realistic", "arbin-month-first"])
def test_echem_time_column_reads_the_real_layouts(tmp_path, kind):
    """The EC-Lab and Arbin month-first echem layouts read every row's time, first to last, in order."""
    path = tmp_path / "echem.txt"
    if kind == "eclab-realistic":
        _builders.write_eclab_echem(str(path))
        first, last = pd.Timestamp("2024-02-05 10:00:00"), pd.Timestamp("2024-02-05 10:39:00")
    else:
        rows = ["Date_Time\tCycle_Index\tCurrent(A)\tVoltage(V)"]
        rows += [f"07/15/2024 19:{m:02d}:54\t1\t0\t{3.47 + m * 0.001:.9f}" for m in range(11, 41)]
        path.write_bytes(("\r\n".join(rows) + "\r\n").encode())
        first, last = pd.Timestamp("2024-07-15 19:11:54"), pd.Timestamp("2024-07-15 19:40:54")
    df = core.EchemParser().parse(str(path))
    assert df["timestamp"].iloc[0] == first and df["timestamp"].iloc[-1] == last
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].notna().all()


def _write_offset_echem(path, stamps):
    """Write an Arbin-style echem .txt with the given ISO Date_Time values."""
    rows = ["Date_Time\tVoltage(V)\tCurrent(A)"]
    rows += [f"{s}\t{3.7 + 0.01 * i:.4f}\t0.0001" for i, s in enumerate(stamps)]
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


@pytest.mark.parametrize("stamps, warned", [
    pytest.param([f"2024-10-27T01:{m:02d}:00+01:00" for m in range(0, 60, 5)]
                 + [f"2024-10-27T01:{m:02d}:00+00:00" for m in range(0, 60, 5)], True, id="clock-back"),
    pytest.param([f"2024-03-31T00:{m:02d}:00+00:00" for m in range(30, 60, 5)]
                 + [f"2024-03-31T02:{m:02d}:00+01:00" for m in range(0, 30, 5)], True, id="clock-forward"),
    pytest.param([f"2024-06-05T10:{m:02d}:00+01:00" for m in range(0, 40, 5)], False, id="one-offset"),
    pytest.param([f"2024-06-05T10:{m:02d}:00Z" for m in range(0, 40, 5)], False, id="utc"),
])
def test_echem_offset_change_is_warned_once(tmp_path, caplog, stamps, warned):
    """An echem file whose times change UTC offset logs one warning naming it; rows stay wall clock."""
    path = tmp_path / "cycler.txt"
    _write_offset_echem(str(path), stamps)
    with caplog.at_level(logging.WARNING, logger="core"):
        df = core.EchemParser().parse(str(path))
    assert len(df) == len(stamps) and df["timestamp"].iloc[0].tzinfo is None
    hits = [r.getMessage() for r in caplog.records if "change UTC offset" in r.getMessage()]
    assert len(hits) == (1 if warned else 0), hits
    assert all("cycler.txt" in h for h in hits)


@pytest.mark.parametrize("zone", ["UTC", "GMT"])
def test_year_first_echem_times_with_a_zone_word_keep_their_wall_clock(tmp_path, zone):
    """Year-first echem times ending in ' UTC' or ' GMT' load at their wall clock, not dropped."""
    path = tmp_path / "cycler.txt"
    _write_offset_echem(str(path), [f"2024-02-05 10:{m:02d}:00 {zone}" for m in range(0, 40, 5)])
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 8, df
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-02-05 10:00:00")
    assert df["timestamp"].iloc[0].tzinfo is None


@pytest.mark.parametrize("first_cell, step", [
    pytest.param("0.000000000000000E+000", "{:.15E}", id="ec-lab-e-notation"),
    pytest.param("0.00", "{:.2f}", id="two-decimals"),
])
def test_echem_elapsed_seconds_file_is_skipped(tmp_path, caplog, first_cell, step):
    """A time column of plain numbers (elapsed seconds) skips the file with a warning naming it."""
    path = tmp_path / "elapsed.txt"
    rows = ["time/s\tEcell/V\tI/mA", f"{first_cell}\t3.7\t0.1"]
    rows += [f"{step.format(1990.5 + 0.5 * i)}\t{3.7 + 0.001 * i:.4f}\t0.1" for i in range(200)]
    path.write_text("\n".join(rows) + "\n")
    with caplog.at_level(logging.WARNING, logger="core"):
        assert core.EchemParser().parse(str(path)) is None
    assert any("elapsed.txt" in r.getMessage() and "plain numbers" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("placeholder", ["30/12/1899 00:00:00", "01/01/0001 00:00:00"])
def test_echem_rows_before_1900_are_dropped(tmp_path, caplog, placeholder):
    """A placeholder row dated before 1900 is dropped (with a count where pandas can read it);
    the rest of the file is kept."""
    path = tmp_path / "echem.txt"
    _builders.write_echem_txt(str(path))
    lines = path.read_text().splitlines()
    lines.insert(1, f"{placeholder}\t3.0\t0.0")
    path.write_text("\n".join(lines) + "\n")
    with caplog.at_level(logging.WARNING, logger="core"):
        df = core.EchemParser().parse(str(path))
    assert len(df) == 40 and df["timestamp"].min() == pd.Timestamp("2024-02-05 10:00:00")
    if placeholder.endswith("1899 00:00:00"):
        assert any("1 echem row dated before 1900 (placeholders?) was skipped" in r.getMessage()
                   for r in caplog.records)


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


@pytest.mark.parametrize("header, cells, reason", [
    ("Time\tTemperature 1 (C)\tTemperature 2 (C)", ["25.0", "26.0"],
     "column 'Temperature 1 (C)' is not a voltage column"),
    ("Time\tVoltage(V)\tTemperature(C)", ["3.7", "25.0"],
     "column 'Temperature(C)' is not a current column"),
    ("Time\tTemperature(C)\tCurrent(mA)", ["25.0", "0.1"],
     "column 'Temperature(C)' is not a voltage column"),
    ("time/s\tEwe/V\tcycle number\t<I>/mA", ["3.7", "1", "0.1"],
     "column 'cycle number' is not a current column"),
    ("Time\tEwe/V", ["3.7", "0.1"], "no current column"),
], ids=["temperature-log", "temperature-as-current", "temperature-as-voltage",
        "cycle-number-as-current", "unnamed-third-value"])
def test_positional_column_without_its_unit_skips_the_file(tmp_path, caplog, header, cells, reason):
    """A voltage or current taken by position without its unit skips the file with one warning."""
    rows = [header] + ["\t".join([f"05/02/2024 10:{m:02d}:00"] + cells) for m in range(5)]
    path = tmp_path / "log.txt"
    path.write_text("\n".join(rows) + "\n")
    with caplog.at_level(logging.WARNING, logger="core"):
        assert core.EchemParser().parse(str(path)) is None
    assert [r.getMessage() for r in caplog.records] == [f"Skipping log.txt: {reason}"]


@pytest.mark.parametrize("header", [
    ["Time", "Potential (V)", "I (mA)"], ["time/s", "E/V", "<I>/mA"], ["Time", "U [V]", "I [mA]"],
], ids=["potential-volts", "e-slash-volts", "bracketed-units"])
def test_positional_columns_with_their_unit_still_load(tmp_path, caplog, header):
    """Voltage and current columns taken by position still load when their labels carry the unit."""
    path = tmp_path / "echem.txt"
    _write_echem(path, header)
    with caplog.at_level(logging.WARNING, logger="core"):
        df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 5 and np.allclose(df["current"], 2.0)
    assert np.allclose(df["echem_data"], 3.7 + 0.01 * np.arange(5)) and not caplog.records


def test_temperature_log_beside_the_echem_is_left_out(tmp_path, caplog):
    """A 'Time'-headed temperature log beside the echem is skipped; the echem alone correlates."""
    src = tmp_path / "data"
    _builders.write_inhouse_dir(str(src))
    rows = ["Time\tTemperature 1 (C)\tTemperature 2 (C)"]
    rows += [f"05/02/2024 10:{m:02d}:30\t25.{m:02d}\t26.0" for m in range(40)]
    (src / "cell_temperature.txt").write_text("\n".join(rows) + "\n")
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, echem = core.process_raw([str(src)], core.DataSourceType.INHOUSE)
    assert len(echem) == 40 and {os.path.basename(f) for f in echem["source_file"]} == {"echem.txt"}
    assert scans[0].voltage_max < 4 and abs(scans[0].echem - 3.76) < 1e-9, scans[0].voltage_max
    assert ("Skipping cell_temperature.txt: column 'Temperature 1 (C)' is not a voltage column"
            in [r.getMessage() for r in caplog.records])


def test_combined_echem_files_are_named(tmp_path, caplog):
    """Echem from two files is combined into one record and one info line names both."""
    src = tmp_path / "data"
    src.mkdir()
    _builders.write_inhouse_scans(str(src))
    _builders.write_echem_txt(str(src / "cell_b.txt"), n_rows=20)
    _builders.write_echem_txt(str(src / "cell_a.txt"), n_rows=20, hour=11)
    with caplog.at_level(logging.INFO, logger="core"):
        _, echem = core.process_raw([str(src)], core.DataSourceType.INHOUSE)
    assert len(echem) == 40
    lines = [r.getMessage() for r in caplog.records if "files:" in r.getMessage()]
    assert lines == ["Echem data from 2 files: cell_a.txt, cell_b.txt"], lines


@pytest.mark.parametrize("header,scale", [
    ("Time\tVoltage\tCurrent(mA)", 1.0),
    ("time/s\tEwe/V\tI/mA", 1.0),
    ("Time\tVoltage\tCurrent", 1.0),
    ("Time\tVoltage\tArea (A)", 1.0),
], ids=["milliamps", "biologic", "unitless", "non-current-amps"])
def test_current_unit_scale(header, scale):
    """Milliamp, unlabelled and non-current columns keep a milliamp scale of 1.0."""
    # Columns given, not detected: detection skips the 'Area (A)' file before any scale is read
    assert core.EchemParser()._current_unit_scale(header, {"current": 2}) == scale


@pytest.mark.parametrize("header,scale", [
    ("time/s\tEwe/V\tI/A", 1000.0), ("time/s\tEwe/V\t<I>/A", 1000.0),
    ("Time\tVoltage\tCurrent/A", 1000.0), ("time/s\tEwe/V\tI/uA", 0.001),
    ("time/s\tEwe/V\t<I>/mA", 1.0), ("Time\tVoltage\tCurrent(A)", 1000.0),
    ("Time\tVoltage\tArea (A)", 1.0), ("Time\tVoltage\tCurrent [A]", 1000.0),
    ("Time\tVoltage\tCurrent [mA]", 1.0),
], ids=["slash-amps", "averaged-amps", "current-slash-amps", "microamps", "averaged-milliamps",
        "parenthesised-amps", "non-current-amps", "bracketed-amps", "bracketed-milliamps"])
def test_current_unit_scale_reads_the_unit(header, scale):
    """The unit after '/' or '(' of a current label sets the milliamp scale."""
    assert core.EchemParser()._current_unit_scale(header, {"current": 2}) == scale


def _write_echem(path, header, sep="\t", encoding="utf-8", current="2.0", eol="\r\n", tail=b"", n=5,
                 volt=lambda m: f"{3.7 + m * 0.01:.4f}", time=lambda m: f"05/02/2024 10:{m:02d}:00"):
    """Write an echem table (absolute day-first times) in `encoding`, then append raw `tail` bytes."""
    rows = [sep.join(header)]
    for m in range(n):
        rows.append(sep.join([time(m), volt(m), current] + [""] * (len(header) - 3)))
    path.write_bytes((eol.join(rows) + eol).encode(encoding) + tail)


@pytest.mark.parametrize("unit,stored", [("A", 2000.0), ("mA", 2.0), ("µA", 0.002)],
                         ids=["amps", "milliamps", "microamps"])
def test_current_unit_after_slash_sets_milliamps(tmp_path, unit, stored):
    """An EC-Lab 'I/A' or 'I/µA' current is stored in mA; 'I/mA' is stored as written."""
    path = tmp_path / "echem.txt"
    _write_echem(path, ["time/s", "Ewe/V", f"I/{unit}"])
    df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["current"], stored), list(df["current"])


def test_quoted_current_header_keeps_its_amp_unit(tmp_path):
    """A header written with quoted cells ('"Current(A)"') still scales amps to mA."""
    path = tmp_path / "echem.txt"
    _write_echem(path, ['"Date_Time"', '"Voltage(V)"', '"Current(A)"'])
    df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["current"], 2000.0), list(df["current"])


@pytest.mark.parametrize("header,level,texts", [
    ("I/A", logging.INFO, ("'I/A'", "converted to mA (x1000)")),
    ("Current (Amps)", logging.WARNING, ("'Current (Amps)'", "has no recognised unit")),
    ("I/mA", None, ()),
], ids=["scaled", "unrecognised", "milliamps-silent"])
def test_current_unit_decision_is_logged(tmp_path, caplog, header, level, texts):
    """A scaled current logs one info line, an unrecognised unit one warning, a mA column nothing."""
    path = tmp_path / "echem.txt"
    _write_echem(path, ["time/s", "Ewe/V", header])
    with caplog.at_level(logging.INFO, logger="core"):
        assert core.EchemParser().parse(str(path)) is not None
    lines = [r for r in caplog.records if "current column" in r.getMessage()]
    if level is None:
        assert not lines, [r.getMessage() for r in lines]
    else:
        assert len(lines) == 1 and lines[0].levelno == level, [r.getMessage() for r in lines]
        assert all(text in lines[0].getMessage() for text in texts), lines[0].getMessage()


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_stray_byte_after_a_utf8_header_keeps_microamps(tmp_path, suffix):
    """One cp1252 byte in a late row of a UTF-8 'I/µA' file leaves 100 µA stored as 0.1 mA."""
    sep = "\t" if suffix == ".txt" else ","
    path = tmp_path / f"echem{suffix}"
    _write_echem(path, ["time/s", "Ewe/V", "I/µA", "note"], sep=sep, current="100.0",
                 tail=sep.join(["05/02/2024 10:05:00", "3.7500", "100.0", ""]).encode() + b"\x96\r\n")
    txt = core.convert_tabular_to_txt(str(path), dest_dir=str(tmp_path))
    df = core.EchemParser().parse(txt)
    assert df is not None and len(df) == 6 and np.allclose(df["current"], 0.1), \
        None if df is None else list(df["current"])


def test_amp_unit_followed_by_more_text_still_scales():
    """'Current(A) avg' is still scaled by its '(A)', so amps convert to mA."""
    parser = core.EchemParser()
    header = "Time\tVoltage\tCurrent(A) avg"
    assert parser._current_unit_scale(header, parser._detect_columns(header)) == 1000.0


@pytest.mark.parametrize("encoding,extra", [("utf-8", []), ("cp1252", ["C/µF"])],
                         ids=["utf8-header", "ansi-header"])
def test_stray_byte_glued_to_a_value_is_dropped(tmp_path, encoding, extra):
    """A non-UTF-8 byte glued to a current value is dropped, so the row keeps its value."""
    path = tmp_path / "echem.txt"
    _write_echem(path, ["time/s", "Ewe/V", "I/mA"] + extra, encoding=encoding,
                 tail=b"\t".join([b"05/02/2024 10:05:00", b"3.7500", b"2.0\xb0"] + [b""] * len(extra))
                 + b"\r\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 6 and np.allclose(df["current"], 2.0), \
        None if df is None else list(df["current"])


def test_stray_byte_glued_to_the_first_row_of_a_headerless_file_is_dropped(tmp_path):
    """A stray byte glued to a header-less file's first row is dropped, so the row keeps its value."""
    path = tmp_path / "echem.txt"
    rows = [f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\t2.0".encode() for m in range(5)]
    path.write_bytes(b"\r\n".join([rows[0] + b"\xb0"] + rows[1:]) + b"\r\n")
    df = core.EchemParser().parse(str(path))
    assert df is not None and len(df) == 5 and np.allclose(df["current"], 2.0), \
        None if df is None else list(df["current"])


def test_cp1252_microamp_header_is_not_read_as_amps(tmp_path):
    """An ANSI 'I/µA' header keeps its micro sign, so 100 µA is stored as 0.1 mA."""
    path = tmp_path / "echem.txt"
    with open(path, "w", encoding="cp1252") as f:
        f.write("time/s\tEwe/V\tI/µA\n" + "".join(
            f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\t100.0\n" for m in range(5)))
    df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["current"], 0.1)


@pytest.mark.parametrize("header, row", [
    (["I/A", "time/s", "Ewe/V"],
     lambda m: ["0.002", f"05/02/2024 10:{m:02d}:00", f"{3.7 + m * 0.01:.4f}"]),
    (["time/s", "Ewe/V", "I/mA", "cycle time/s"],
     lambda m: [f"05/02/2024 10:{m:02d}:00", f"{3.7 + m * 0.01:.4f}", "2.0", f"{m * 60.0:.1f}"]),
], ids=["current-first", "time-first"])
def test_echem_file_with_a_bom_reads_as_without(tmp_path, header, row):
    """A UTF-8 BOM changes nothing: 'I/A' first still converts x1000 and time/s keeps the time slot."""
    rows = [header] + [row(m) for m in range(5)]
    text = ("\r\n".join("\t".join(r) for r in rows) + "\r\n").encode()
    (tmp_path / "plain.txt").write_bytes(text)
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbf" + text)
    df = core.EchemParser().parse(str(tmp_path / "bom.txt"))
    assert df is not None and np.allclose(df["current"], 2.0), \
        None if df is None else list(df["current"])
    pd.testing.assert_frame_equal(df, core.EchemParser().parse(str(tmp_path / "plain.txt")))


@pytest.mark.parametrize("unit,stored", [("nA", 2e-6), ("pA", 2e-9)], ids=["nanoamps", "picoamps"])
def test_nanoamp_and_picoamp_currents_are_stored_in_milliamps(tmp_path, unit, stored):
    """An 'I/nA' or 'Current (pA)' column is converted to mA instead of being read as mA."""
    path = tmp_path / "echem.txt"
    _write_echem(path, ["time/s", "Ewe/V", f"I/{unit}"])
    df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["current"], stored, rtol=1e-12, atol=0), list(df["current"])


def test_unit_line_names_the_file(tmp_path, caplog):
    """The current-unit line names the echem file as well as the column."""
    path = tmp_path / "cell7.txt"
    _write_echem(path, ["time/s", "Ewe/V", "I/A"])
    with caplog.at_level(logging.INFO, logger="core"):
        assert core.EchemParser().parse(str(path)) is not None
    lines = [r.getMessage() for r in caplog.records if "current column" in r.getMessage()]
    assert lines == ["Echem current column 'I/A' in cell7.txt converted to mA (x1000)"], lines


@pytest.mark.parametrize("header", [
    ["Time", "Voltage(mV)", "Current(mA)"], ["time/s", "Ewe/mV", "I/mA"], ["Time", "E(mV)", "I(mA)"],
], ids=["arbin-style", "ec-lab-style", "positional"])
def test_millivolt_voltage_is_stored_in_volts(tmp_path, caplog, header):
    """An mV voltage column, named or taken by position, is stored in volts with one info line."""
    path = tmp_path / "cell7.txt"
    _write_echem(path, header, volt=lambda m: f"{3700 + m * 10}")
    with caplog.at_level(logging.INFO, logger="core"):
        df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["echem_data"], 3.7 + 0.01 * np.arange(5)), \
        None if df is None else list(df["echem_data"])
    lines = [r.getMessage() for r in caplog.records if "voltage column" in r.getMessage()]
    assert lines == [f"Echem voltage column '{header[1]}' in cell7.txt converted to V (x0.001)"]


@pytest.mark.parametrize("label,warned", [("Ewe/V", False), ("Voltage", False), ("Voltage(kV)", True)],
                         ids=["volts", "unlabelled", "other-unit"])
def test_voltage_not_in_millivolts_is_read_as_written(tmp_path, caplog, label, warned):
    """A V or unlabelled voltage is read as written silently; any other unit too, with a warning."""
    path = tmp_path / "cell7.txt"
    _write_echem(path, ["Time", label, "I/mA"])
    with caplog.at_level(logging.INFO, logger="core"):
        df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["echem_data"], 3.7 + 0.01 * np.arange(5))
    lines = [(r.levelno, r.getMessage()) for r in caplog.records if "voltage column" in r.getMessage()]
    expected = f"Echem voltage column '{label}' in cell7.txt is not in V; values read as V"
    assert lines == ([(logging.WARNING, expected)] if warned else []), lines


@pytest.mark.parametrize("label,millivolts,line", [
    ("Voltage(V) avg", False, None), ("Ewe/V vs. Li", False, None),
    ("E(mV) avg", True, "converted to V (x0.001)"), ("Ewe/Vref", False, "is not in V; values read as V"),
], ids=["volts-avg", "volts-vs-li", "millivolts-avg", "unit-glued-to-text"])
def test_voltage_unit_followed_by_more_text_is_read(tmp_path, caplog, label, millivolts, line):
    """A V or mV right after '/', '(' or '[' and followed by more text is that unit; 'Ewe/Vref' is not."""
    path = tmp_path / "cell7.txt"
    _write_echem(path, ["Time", label, "I/mA"],
                 volt=(lambda m: f"{3700 + m * 10}") if millivolts else (lambda m: f"{3.7 + m * 0.01:.4f}"))
    with caplog.at_level(logging.INFO, logger="core"):
        df = core.EchemParser().parse(str(path))
    assert df is not None and np.allclose(df["echem_data"], 3.7 + 0.01 * np.arange(5)), \
        None if df is None else list(df["echem_data"])
    lines = [r.getMessage() for r in caplog.records if "voltage column" in r.getMessage()]
    assert lines == ([f"Echem voltage column '{label}' in cell7.txt {line}"] if line else []), lines


_ROWS = [(f"05/02/2024 10:{m:02d}:00", 3.7 + m * 0.01, 0.1 + m * 0.001) for m in range(40)]


@pytest.mark.parametrize("encoding, text", [
    ("utf-8", "Time,Ewe/V,I/mA\n" + "".join(f"{t},{v:.4f},{i:.4f}\n" for t, v, i in _ROWS)),
    ("utf-8", "Time;Ewe/V;I/mA\n" + "".join(f"{t};{v:.4f};{i:.4f}\n".replace(".", ",")
                                             for t, v, i in _ROWS)),
    ("utf-8", "Time\tEwe/V\tI/mA\n" + "".join(f"{t}\t{v:.4f}\t{i:.4f}\n" for t, v, i in _ROWS)),
    ("cp1252", "Time,Ewe/V,I/µA\n" + "".join(f"{t},{v:.4f},{i * 1000:.4f}\n"
                                                  for t, v, i in _ROWS)),
], ids=["comma", "semicolon-decimal-comma", "tab", "cp1252-microamps"])
def test_csv_dialects_load(tmp_path, encoding, text):
    """Comma, semicolon (decimal comma), tab and ANSI CSV echem all load and correlate."""
    src = tmp_path / "data"
    src.mkdir()
    _builders.write_inhouse_scans(str(src))
    with open(src / "echem.csv", "w", encoding=encoding) as f:
        f.write(text)
    scans, echem = core.process_raw([str(src)], core.DataSourceType.INHOUSE)
    assert len(echem) == 40
    assert abs(scans[0].echem - 3.76) < 1e-9 and abs(scans[0].current - 0.106) < 1e-9


def test_unit_lines_name_the_users_csv_not_its_converted_copy(tmp_path, caplog):
    """The current-unit lines of an operando and a standard .csv name those files, not the
    temporary .txt copies they are parsed from."""
    src = tmp_path / "data"
    src.mkdir()
    _builders.write_inhouse_scans(str(src))
    text = "Time,Ewe/V,I/A\n" + "".join(f"{t},{v:.4f},{i / 1000:.7f}\n" for t, v, i in _ROWS)
    (src / "cell7.csv").write_text(text)
    (tmp_path / "std7.csv").write_text(text)
    with caplog.at_level(logging.INFO, logger="core"):
        ok, msgs = core.generate([str(src)], str(tmp_path / "units.nxs"), core.DataSourceType.INHOUSE,
                                 standard_echem_files=[str(tmp_path / "std7.csv")])
    assert ok, msgs
    lines = sorted(r.getMessage() for r in caplog.records if "current column" in r.getMessage())
    assert lines == [f"Echem current column 'I/A' in {name} converted to mA (x1000)"
                     for name in ("cell7.csv", "std7.csv")], lines


def test_skipped_standard_echem_files_are_named(tmp_path, caplog):
    """A chosen standard echem file that is missing or holds no echem is named in a warning."""
    src = str(tmp_path / "inhouse")
    _builders.write_inhouse_dir(src)
    (tmp_path / "notes.txt").write_text("Time\tVoltage\tCurrent\n")
    out = str(tmp_path / "std.nxs")
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE,
                                 standard_echem_files=[str(tmp_path / "gone.txt"),
                                                       str(tmp_path / "notes.txt")])
    assert ok, msgs
    messages = [r.getMessage() for r in caplog.records]
    assert "Standard electrochemistry file gone.txt not found; skipped" in messages, messages
    assert "No electrochemistry data found in notes.txt; not stored" in messages, messages


def test_unconverted_standard_table_is_named_once(tmp_path, caplog):
    """A missing or unreadable standard .csv/.xlsx is named once, by its conversion error."""
    src = str(tmp_path / "inhouse")
    _builders.write_inhouse_dir(src)
    (tmp_path / "broken.xlsx").write_bytes(b"not a workbook")
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([src], str(tmp_path / "std.nxs"), core.DataSourceType.INHOUSE,
                                 standard_echem_files=[str(tmp_path / "gone.csv"),
                                                       str(tmp_path / "broken.xlsx")])
    assert ok, msgs
    for name in ("gone.csv", "broken.xlsx"):
        named = [r.getMessage() for r in caplog.records if name in r.getMessage()]
        assert len(named) == 1 and named[0].startswith("Error converting tabular file "), named


def _parse(path, tmp_path):
    """Convert a .csv/.xlsx echem table into tmp_path and parse the copy."""
    txt = core.convert_tabular_to_txt(str(path), dest_dir=str(tmp_path))
    return core.EchemParser().parse(txt)


def test_tab_csv_with_decimal_commas_loads(tmp_path):
    """A tab-separated .csv with decimal commas (EC-Lab, French or German locale) loads its values."""
    path = tmp_path / "echem.csv"
    _write_echem(path, ["time/s", "Ewe/V", "I/mA"], current="0,1", volt=lambda m: f"3,{70 + m}")
    df = _parse(path, tmp_path)
    assert df is not None and len(df) == 5, df
    assert np.allclose(df["echem_data"], [3.70, 3.71, 3.72, 3.73, 3.74]) and np.allclose(df["current"], 0.1)


@pytest.mark.parametrize("text,current", [
    ("Datum;Ewe/V;I/mA\r" + "".join(f"05.02.2024 10:{m:02d}:00,936;3,{70 + m};0,1\r" for m in range(5)), 0.1),
    (("Time,Ewe/V,I/µA,note\r" + "".join(f"05/02/2024 10:{m:02d}:00,3.7,100.0,x\r" for m in range(5)))[:-2]
     + "\x00\r", 0.1),
], ids=["semicolon-decimal-comma", "utf8-micro-stray-byte"])
def test_cr_only_csv_is_read_by_its_first_line(tmp_path, text, current):
    """A CR-only CSV (old Mac Excel) takes its separator and encoding from the first line, not the file."""
    path = tmp_path / "echem.csv"
    path.write_bytes(text.encode("utf-8").replace(b"\x00", b"\x96"))
    df = _parse(path, tmp_path)
    assert df is not None and len(df) == 5 and np.allclose(df["current"], current), \
        None if df is None else list(df["current"])


def test_stray_byte_glued_to_a_csv_value_is_dropped(tmp_path):
    """A non-UTF-8 byte glued to a value of a UTF-8 CSV is dropped, so the value is read as for .txt."""
    path = tmp_path / "echem.csv"
    _write_echem(path, ["time/s", "Ewe/V", "I/µA"], sep=",", current="100.0",
                 tail=b"05/02/2024 10:05:00,3.75,100.0\xb0\r\n")
    df = _parse(path, tmp_path)
    assert df is not None and len(df) == 6 and np.allclose(df["current"], 0.1), \
        None if df is None else list(df["current"])


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
    # The sidecar is the normal outcome, so it is no warning
    assert os.path.isfile(tmp_path / "zipped_images" / ".operaxn_images")
    assert not any("2D images copied" in msg for msg in msgs), msgs


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


def test_collection_warms_every_file_up_front(neutron_dir, monkeypatch):
    """Every collected file is opened once across the worker pool before classification."""
    import builtins
    monkeypatch.setattr(importlib.import_module("core.generate"),
                        "PARALLEL_PROCESSING_THRESHOLD", 0)
    opened = []
    real_open = builtins.open

    def recording_open(file, mode="r", *args, **kwargs):
        if mode == "rb":
            opened.append(os.path.normpath(str(file)))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    with core.FileProcessor(core.DataSourceType.NEUTRON) as processor:
        records = processor.process_paths([neutron_dir])
    collected = {os.path.normpath(r.path) for r in records}
    assert collected and collected <= set(opened), sorted(collected - set(opened))


def test_warm_up_skips_unreadable_paths(tmp_path, monkeypatch):
    """A file gone since collection is skipped; warming never raises."""
    monkeypatch.setattr(importlib.import_module("core.generate"),
                        "PARALLEL_PROCESSING_THRESHOLD", 0)
    present = tmp_path / "scan.dat"
    present.write_text("1 2\n")
    core.FileProcessor._warm_up([str(tmp_path / "gone.dat"), str(present)])


def test_inhouse_dat_without_a_header_is_named(tmp_path, caplog):
    """A header-stripped .dat is named in the one unrecognised-file line, never as an orphan."""
    src = tmp_path / "lab"
    _builders.write_inhouse_dir(str(src))
    # As the real stripped copies of scan 00008: the data rows only
    rows = [line for line in (src / "scan_002.dat").read_text().splitlines()
            if not line.startswith("#")]
    (src / "scan_004.dat").write_text("\n".join(rows) + "\n")
    (src / "scan_005.xy").write_text("10.0 1.0\n")

    def lines():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="core"):
            scans, _ = core.process_raw([str(src)], core.DataSourceType.INHOUSE)
        assert len(scans) == 3
        return [r.getMessage() for r in caplog.records if "ignored" in r.getMessage()]

    assert sorted(lines()) == [
        "1 .dat/.edf file was not recognised as scan data and was ignored: scan_004.dat "
        "(scans need a readable Date in their header)",
        "1 file did not belong to any scan and was ignored (1 .xy)"]
    (src / "notes.txt").write_text("calibration\t25.3\t40.1\n")
    named = [m for m in lines() if "not recognised" in m]
    assert len(named) == 1 and named[0].startswith(
        "2 files were not recognised as scan, echem or logbook data and were ignored: "), named
    assert "notes.txt" in named[0] and "scan_004.dat" in named[0]
    assert named[0].endswith("(scans need a readable Date in their header; echem files need a "
                             "header line naming time, voltage or current)")


@needs_fabio
def test_damaged_edf_is_named_once(tmp_path, caplog):
    """An EDF fabio cannot read is named in its one ERROR line, not again as unrecognised."""
    src = tmp_path / "lab"
    _builders.write_inhouse_dir(str(src))
    (src / "image_009.edf").write_bytes(b"not an edf image")
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, _ = core.process_raw([str(src)], core.DataSourceType.INHOUSE)
    assert len(scans) == 3
    named = [(r.levelname, r.getMessage()) for r in caplog.records
             if "image_009.edf" in r.getMessage()]
    assert len(named) == 1 and named[0][0] == "ERROR" and \
        named[0][1].startswith("Error classifying EDF file "), named


# ============================================================================
# Input files
# ============================================================================

def test_text_file_that_is_neither_echem_nor_logbook_is_named(tmp_path, caplog):
    """A headerless temperature log is not echem; one warning names it, echem and logbook stay silent."""
    src = tmp_path / "wish"
    _builders.write_wish_dir(str(src))
    with open(src / "cell_temperature_log.txt", "w") as f:
        f.write("".join(f"05/02/2024 10:{m:02d}:00\t25.3\t40.1\n" for m in range(30)))
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, echem = core.process_raw([str(src)], core.DataSourceType.NEUTRON)
    assert len(scans) == 1 and len(echem) == 40
    named = [m for m in _warning_messages(caplog) if "not recognised as echem or logbook" in m]
    assert named == ["1 text file was not recognised as echem or logbook data and was ignored: "
                     "cell_temperature_log.txt (echem files need a header line naming time, "
                     "voltage or current)"], _warning_messages(caplog)


def test_echem_and_logbook_files_raise_no_unrecognised_warning(tmp_path, caplog):
    """The real shapes (EC-Lab echem, WISH logbook) never trigger the unrecognised-text warning."""
    src = tmp_path / "wish"
    _builders.write_wish_dir(str(src))
    with caplog.at_level(logging.WARNING, logger="core"):
        core.process_raw([str(src)], core.DataSourceType.NEUTRON)
    assert not [m for m in _warning_messages(caplog) if "not recognised as echem" in m]


def test_one_unmatched_file_reads_singular(tmp_path, caplog):
    """One orphan .xy is reported as '1 file did not belong to any scan and was ignored (1 .xy)'
    and the scan with its .nxs still generates."""
    src = tmp_path / "syn"
    os.makedirs(src)
    _builders.write_synchrotron_scan(str(src), "000001", "2024-02-05T10:05:00",
                                     "2024-02-05T10:05:39")
    shutil.copy(src / "i11-1-000001_integration_tth_0000.xy",
                src / "i11-1-000003_integration_tth_0000.xy")
    _builders.write_eclab_echem(str(src / "echem.txt"))
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([str(src)], str(tmp_path / "syn.nxs"),
                                 core.DataSourceType.SYNCHROTRON)
    assert ok, msgs
    messages = _warning_messages(caplog)
    assert "1 file did not belong to any scan and was ignored (1 .xy)" in messages, messages


def test_zip_with_dot_slash_members_loads(tmp_path):
    """A ZIP made by Windows 'tar -a -cf x.zip .' (members './scan_001.dat') loads like any other."""
    src = tmp_path / "lab"
    _builders.write_lab_dir(str(src))
    zp = tmp_path / "lab.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("./", b"")
        for name in sorted(os.listdir(src)):
            # writestr keeps the name as given; write() would normalise "./" away
            z.writestr(f"./{name}", (src / name).read_bytes())
    out = str(tmp_path / "lab.nxs")
    ok, msgs = core.generate([str(zp)], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert len(core.load(out).scans) == 3


def test_two_zips_with_the_same_member_names_keep_both_cells(tmp_path):
    """Two cells zipped with identical member names keep their own scans and echem."""
    zips = []
    for cell, hour in (("A", 10), ("B", 14)):
        src = tmp_path / f"src{cell}"
        os.makedirs(src)
        _builders.write_lab_dat(str(src / "scan_001.dat"), f"2024-02-05T{hour}:05:00")
        _builders.write_eclab_echem(str(src / "echem.txt"), hour=hour)
        zips.append(_builders.zip_tree(str(src), tmp_path / f"cell{cell}.zip"))
    scans, echem = core.process_raw(zips, core.DataSourceType.INHOUSE)
    assert sorted(pd.Timestamp(s.timestamp) for s in scans) == [
        pd.Timestamp("2024-02-05 10:05:00"), pd.Timestamp("2024-02-05 14:05:00")]
    assert echem["timestamp"].nunique() == 80


def test_zip_skips_nested_macos_forks(tmp_path):
    """Nested AppleDouble '._' members never replace the real bank files."""
    src = tmp_path / "wish"
    _builders.write_wish_dir(str(src))
    zp = _builders.zip_tree(str(src), tmp_path / "run.zip", prefix="run/")
    with zipfile.ZipFile(zp, "a") as z:
        for name in os.listdir(src / "diffraction-data"):
            z.writestr(f"run/diffraction-data/._{name}", b"\x00\x05\x16\x07" + b"\x00" * 76)
    out = str(tmp_path / "run.nxs")
    ok, msgs = core.generate([zp], out, core.DataSourceType.NEUTRON)
    assert ok, msgs
    banks = core.load(out).scans[0].neutron
    assert banks and set(banks) == {"1", "2"}, banks


def test_unextractable_zip_member_costs_only_that_file(tmp_path, caplog):
    """A member that cannot be extracted (bad CRC) is skipped with a warning; later members still load."""
    src = tmp_path / "lab"
    _builders.write_lab_dir(str(src))
    zp = tmp_path / "lab.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for name in ("scan_001.dat", "scan_002.dat", "scan_003.dat", "echem.txt"):
            z.write(src / name, f"cell/{name}")
    with zipfile.ZipFile(zp) as z:
        info = z.getinfo("cell/scan_002.dat")
    raw = bytearray(zp.read_bytes())
    pos = info.header_offset + 30 + len(info.filename) + len(info.extra) + info.file_size // 2
    raw[pos] ^= 0x01
    zp.write_bytes(bytes(raw))
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, echem = core.process_raw([str(zp)], core.DataSourceType.INHOUSE)
    assert len(scans) == 2 and len(echem) == 40, (len(scans), len(echem))
    assert any(m.startswith("Could not extract cell/scan_002.dat from lab.zip: Bad CRC-32")
               for m in _warning_messages(caplog)), _warning_messages(caplog)


@needs_fabio
def test_zip_images_none_copied_reports_no_sidecar(tmp_path, monkeypatch, caplog):
    """If every ZIP image move fails: an 'N of N' warning, no 'Persisted' line, no sidecar left."""
    src = tmp_path / "cell"
    src.mkdir()
    for i, ts in enumerate(_builders.SCAN_TIMES[:2], start=1):
        _builders.write_xrd_dat(str(src / f"scan_{i:03d}.dat"), ts)
        _builders.write_edf_image(str(src / f"image_{i:03d}.edf"), ts)
    _builders.write_echem_txt(str(src / "echem.txt"))
    zip_path = _builders.zip_tree(str(src), str(tmp_path / "cell.zip"))
    generate_module = importlib.import_module("core.generate")

    def move(source, dest, *args, **kwargs):
        raise OSError(3, "The system cannot find the path specified", dest)

    monkeypatch.setattr(generate_module.shutil, "move", move)
    out = str(tmp_path / "out" / "cell.nxs")
    os.makedirs(os.path.dirname(out))
    with caplog.at_level(logging.INFO, logger="core"):
        ok, msgs = core.generate([zip_path], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert [m for m in msgs if "2D images" in m] == [
        "Warning: 2 of 2 2D images from the ZIP could not be copied and will not be viewable"], msgs
    assert os.listdir(tmp_path / "out") == ["cell.nxs"]
    assert not [r.getMessage() for r in caplog.records if r.getMessage().startswith("Persisted")]


@needs_fabio
def test_one_failed_zip_image_copy_costs_one_image(tmp_path, monkeypatch):
    """A ZIP image whose sidecar copy fails is reported by count; the other images stay viewable."""
    src = tmp_path / "cell"
    src.mkdir()
    for i, ts in enumerate(_builders.SCAN_TIMES[:2], start=1):
        _builders.write_xrd_dat(str(src / f"scan_{i:03d}.dat"), ts)
        _builders.write_edf_image(str(src / f"image_{i:03d}.edf"), ts)
    _builders.write_echem_txt(str(src / "echem.txt"))
    zip_path = _builders.zip_tree(str(src), str(tmp_path / "cell.zip"))

    generate_module = importlib.import_module("core.generate")
    real_move = generate_module.shutil.move
    failed = []

    def move(source, dest, *args, **kwargs):
        if not failed:
            failed.append(source)
            raise OSError(3, "The system cannot find the path specified", dest)
        return real_move(source, dest, *args, **kwargs)

    monkeypatch.setattr(generate_module.shutil, "move", move)
    out = str(tmp_path / "out" / "cell.nxs")
    os.makedirs(os.path.dirname(out))
    ok, msgs = core.generate([zip_path], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert any(m.startswith("Warning: 1 of 2 2D images from the ZIP could not be copied") for m in msgs), msgs
    sidecar = tmp_path / "out" / "cell_images"
    assert len(list(sidecar.rglob("*.edf"))) == 1
    assert len(failed) == 1
    sources = [str(s.twod_source) for s in core.load(out).scans if s.twod_source]
    assert sum(os.path.isfile(p) and str(sidecar) in p for p in sources) == 1, sources


@needs_fabio
def test_zip_image_held_open_elsewhere_keeps_its_copy(tmp_path, monkeypatch):
    """A move whose copy landed but whose source another program holds open (the unlink is
    refused) keeps the copy: every scan shows its persisted image and no failure is counted."""
    src = tmp_path / "cell"
    src.mkdir()
    for i, ts in enumerate(_builders.SCAN_TIMES[:2], start=1):
        _builders.write_xrd_dat(str(src / f"scan_{i:03d}.dat"), ts)
        _builders.write_edf_image(str(src / f"image_{i:03d}.edf"), ts)
    _builders.write_echem_txt(str(src / "echem.txt"))
    zip_path = _builders.zip_tree(str(src), str(tmp_path / "cell.zip"))
    generate_module = importlib.import_module("core.generate")

    def move(source, dest, *args, **kwargs):
        # shutil.move's copy fallback, then the unlink Windows refuses (WinError 32)
        shutil.copy2(source, dest)
        raise PermissionError(13, "in use", source)

    monkeypatch.setattr(generate_module.shutil, "move", move)
    out = str(tmp_path / "out" / "cell.nxs")
    os.makedirs(os.path.dirname(out))
    ok, msgs = core.generate([zip_path], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert not [m for m in msgs if "could not be copied" in m], msgs
    sidecar = tmp_path / "out" / "cell_images"
    sources = [str(s.twod_source) for s in core.load(out).scans if s.twod_source]
    assert len(sources) == 2 and all(os.path.isfile(p) and str(sidecar) in p for p in sources), sources


def test_run_in_two_logbooks_is_one_scan(tmp_path):
    """A run listed in a cumulative second logbook is written once."""
    src = tmp_path / "wish"
    _builders.write_wish_dir(str(src))
    shutil.copy(src / "Cell_log_book.txt", src / "Cell_log_book_day2.txt")
    scans, _ = core.process_raw([str(src)], core.DataSourceType.NEUTRON)
    assert len(scans) == 1, [s.scan_num for s in scans]


def test_one_repeated_logbook_run_reads_singular(tmp_path, caplog):
    """A run listed twice in one logbook gives one scan and a one-entry warning."""
    src = str(tmp_path / "wish")
    line = _builders.wish_logbook_line("123456", "2024-02-05 10:00:00", "2024-02-05 10:30:00")
    _builders.write_wish_dir(src, lines=[line, line])
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, _ = core.process_raw([src], core.DataSourceType.NEUTRON)
    assert len(scans) == 1
    messages = _warning_messages(caplog)
    assert "1 logbook entry repeats a run already listed and was skipped" in messages, messages


def test_logbook_selected_with_its_folder_is_read_once(tmp_path, caplog):
    """Selecting a folder and its logbook again gives each run once and no repeated-run warning."""
    src = str(tmp_path / "wish")
    _builders.write_wish_dir(src)
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, _ = core.process_raw([src, os.path.join(src, "Cell_log_book.txt")],
                                    core.DataSourceType.NEUTRON)
    assert len(scans) == 1, [s.scan_num for s in scans]
    assert not [m for m in _warning_messages(caplog) if "logbook" in m], _warning_messages(caplog)


def _short_path(path):
    """Return the 8.3 form of an existing path; skip the test where the volume keeps none."""
    buf = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf)) or buf.value == path:
        pytest.skip("no 8.3 name on this volume")
    return buf.value


_windows_only = pytest.mark.skipif(os.name != "nt", reason="case-insensitive and 8.3 file names")


@pytest.mark.parametrize("select", [
    pytest.param(lambda d: [d, os.path.join(d, "cell", "scan_001.dat"),
                            os.path.join(d, "cell", "echem.txt")], id="files-in-it"),
    pytest.param(lambda d: [d, os.path.join(d, "cell")], id="subfolder"),
    pytest.param(lambda d: [d, os.path.join(d, "cell", os.pardir)], id="dot-segments"),
    pytest.param(lambda d: [d, d.upper().replace("\\", "/")], id="other-case", marks=_windows_only),
    pytest.param(lambda d: [d, _short_path(d)], id="short-name", marks=_windows_only),
    pytest.param(lambda d: [_builders.zip_tree(d, d + ".zip")] * 2, id="zip-twice"),
])
def test_a_file_reached_twice_is_read_once(tmp_path, select):
    """A folder with files in it, its subfolder, another spelling or one ZIP twice loads once."""
    d = str(tmp_path / "cell data folder")
    _builders.write_lab_dir(os.path.join(d, "cell"))
    scans, echem = core.process_raw(select(d), core.DataSourceType.INHOUSE)
    assert len(scans) == 3 and len(echem) == 40, (len(scans), len(echem))


def test_table_selected_with_its_folder_is_converted_once(tmp_path, caplog):
    """An Excel echem selected with its folder is converted and read once; the repeat is logged."""
    src = tmp_path / "cell"
    src.mkdir()
    _builders.write_inhouse_scans(str(src))
    _builders.write_arbin_xlsx(str(src / "echem.xlsx"))
    with caplog.at_level(logging.INFO, logger="core"):
        _, echem = core.process_raw([str(src), str(src / "echem.xlsx")],
                                    core.DataSourceType.INHOUSE)
    messages = [r.getMessage() for r in caplog.records]
    assert len(echem) == 40, len(echem)
    assert messages.count("Converted echem.xlsx to txt format") == 1, messages
    assert "1 file was found more than once and is read once: echem.xlsx" in messages, messages


def test_a_file_reached_three_times_is_counted_and_named_once(tmp_path, caplog):
    """The read-once line counts and names each repeated file once, however often it repeats."""
    d = str(tmp_path / "cell")
    _builders.write_lab_dir(d)
    echem, dat = os.path.join(d, "echem.txt"), os.path.join(d, "scan_001.dat")

    def repeats(paths):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="core"):
            scans, rows = core.process_raw(paths, core.DataSourceType.INHOUSE)
        assert len(scans) == 3 and len(rows) == 40, (len(scans), len(rows))
        return [r.getMessage() for r in caplog.records if "more than once" in r.getMessage()]

    assert repeats([d, echem, echem]) == [
        "1 file was found more than once and is read once: echem.txt"]
    assert repeats([d, echem, dat, echem, dat]) == [
        "2 files were found more than once and are read once: echem.txt, scan_001.dat"]


def test_one_logbook_entry_without_data_reads_singular(tmp_path, caplog):
    """One logbook run without bank files is reported as one entry, by its run number."""
    src = str(tmp_path / "wish")
    lines = [_builders.wish_logbook_line("123456", "2024-02-05 10:00:00", "2024-02-05 10:30:00"),
             _builders.wish_logbook_line("123999", "2024-02-05 11:00:00", "2024-02-05 11:30:00")]
    _builders.write_wish_dir(src, lines=lines)
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, _ = core.process_raw([src], core.DataSourceType.NEUTRON)
    assert len(scans) == 1
    messages = _warning_messages(caplog)
    assert "1 logbook entry had no matching data files and was skipped: 123999" in messages, \
        messages


def test_logbook_entries_without_data_are_counted_and_listed_by_run(tmp_path, caplog):
    """Logbook runs without bank files are counted in one warning and listed by run number."""
    src = str(tmp_path / "wish")
    runs = ("123999", "123800", "123700", "123600", "123456")  # newest first, as WISH logs them
    lines = [_builders.wish_logbook_line(run, f"2024-02-05 {h}:00:00", f"2024-02-05 {h}:30:00")
             for h, run in zip(range(14, 9, -1), runs)]
    _builders.write_wish_dir(src, lines=lines)
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, _ = core.process_raw([src], core.DataSourceType.NEUTRON)
    assert len(scans) == 1
    messages = _warning_messages(caplog)
    assert ("4 logbook entries had no matching data files and were skipped: "
            "123600, 123700, 123800 and 1 more") in messages, messages


def test_run_with_bank_files_but_no_logbook_entry_is_named(tmp_path, caplog):
    """A run with bank files but no logbook row is named in one warning; the scans are unchanged."""
    src, extra = str(tmp_path / "wish"), str(tmp_path / "extra")
    _builders.write_wish_dir(src)
    expected, _ = core.process_raw([src], core.DataSourceType.NEUTRON)
    _builders.write_wish_dir(extra, run="153999")
    banks = os.path.join(extra, "diffraction-data")
    for name in os.listdir(banks):
        shutil.copy(os.path.join(banks, name), os.path.join(src, "diffraction-data"))
    with caplog.at_level(logging.WARNING, logger="core"):
        scans, _ = core.process_raw([src], core.DataSourceType.NEUTRON)
    hits = [m for m in _warning_messages(caplog) if "no logbook entry" in m]
    assert hits == ["1 run has bank files but no logbook entry and was skipped: 153999"], hits
    assert [(s.scan_num, s.timestamp, s.echem, s.neutron_files) for s in scans] == \
        [(s.scan_num, s.timestamp, s.echem, s.neutron_files) for s in expected]


def test_neutron_bank_files_without_a_logbook_fail_naming_it(tmp_path, caplog):
    """Neutron bank files selected without their logbook fail naming it and log no traceback."""
    src = str(tmp_path / "wish")
    _builders.write_wish_dir(src)
    os.remove(os.path.join(src, "Cell_log_book.txt"))
    with caplog.at_level(logging.INFO, logger="core"):
        ok, msgs = core.generate([src], str(tmp_path / "x.nxs"), core.DataSourceType.NEUTRON)
    assert not ok and msgs == ["Error: No neutron logbook found in the selection: "
                               "runs are timed from the logbook"], msgs
    assert not [r for r in caplog.records if r.exc_info], "a refusal logs no traceback"


def test_neutron_set_whose_only_logbook_has_a_li_title_generates(tmp_path):
    """End to end: the set's own logbook with a 'Li/NMC811' title is read as a logbook."""
    src = str(tmp_path / "n")
    _builders.write_detailed_neutron_dir(src)
    path = os.path.join(src, "SC1_log_book.txt")
    with open(path) as f:
        text = f.read()
    assert "Op-Run1-SC1-Diag" in text  # the builder's title, so the edit cannot silently miss
    with open(path, "w") as f:
        f.write(text.replace("Op-Run1-SC1-Diag", "Li/NMC811 cell"))
    ok, msgs = core.generate([src], str(tmp_path / "n.nxs"), core.DataSourceType.NEUTRON)
    assert ok, msgs
    assert len(core.load(str(tmp_path / "n.nxs")).scans) == 1


def test_neutron_run_whose_logbook_line_lost_a_tab_generates(tmp_path, caplog):
    """End to end: a duration that fills its column (real SC1 line 540) still gives the scan."""
    src = str(tmp_path / "n")
    _builders.write_detailed_neutron_dir(src)
    path = os.path.join(src, "SC1_log_book.txt")
    with open(path, newline="") as f:
        text = f.read()
    assert '"2m 6s"\t"5.44"' in text  # the builder's fields, so the edit cannot silently miss
    with open(path, "w", newline="") as f:
        f.write(text.replace('"2m 6s"\t"5.44"', '"1m 27s" "5.44"'))
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([src], str(tmp_path / "n.nxs"), core.DataSourceType.NEUTRON)
    assert ok, msgs
    assert not [m for m in _warning_messages(caplog) if "logbook" in m], _warning_messages(caplog)
    assert len(core.load(str(tmp_path / "n.nxs")).scans) == 1


def _lab_pairs(src):
    """Three azimAvg .dat files with their EDF images (the 4225 shape) plus echem."""
    _builders.write_lab_dir(str(src))
    for i, ts in enumerate(_builders.SCAN_TIMES, start=1):
        _builders.write_edf_image(str(src / f"image_{i:03d}.edf"), ts)


@needs_fabio
def test_laboratory_pairs_loaded_as_synchrotron_are_refused(tmp_path, caplog):
    """Lab .dat + .edf pairs as Synchrotron fail with 'check the data source' and no traceback."""
    src = tmp_path / "lab"
    _lab_pairs(src)
    with caplog.at_level(logging.INFO, logger="core"):
        ok, msgs = core.generate([str(src)], str(tmp_path / "x.nxs"),
                                 core.DataSourceType.SYNCHROTRON)
    assert not ok and any("check the data source" in m for m in msgs), msgs
    assert not [r for r in caplog.records if r.exc_info], "a refusal logs no traceback"


@needs_fabio
def test_laboratory_pairs_beside_a_synchrotron_scan_are_refused(tmp_path):
    """One scan .nxs in the selection does not switch the guard off (mixed 4225 + HM28 folder)."""
    src = tmp_path / "mixed"
    _lab_pairs(src)
    _builders.write_synchrotron_scan(str(src), "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    ok, msgs = core.generate([str(src)], str(tmp_path / "x.nxs"), core.DataSourceType.SYNCHROTRON)
    assert not ok and any("check the data source" in m for m in msgs), msgs


def test_dat_only_set_still_loads_as_synchrotron(tmp_path):
    """.dat files alone (no pairs to lose) keep loading as synchrotron, one scan per file."""
    src = tmp_path / "lab"
    _builders.write_lab_dir(str(src))
    out = str(tmp_path / "x.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.SYNCHROTRON)
    assert ok, msgs
    assert len(core.load(out).scans) == 3


def test_synchrotron_scans_still_load(tmp_path):
    """A GDA scan .nxs with its .xy is untouched by the guard."""
    src = tmp_path / "syn"
    os.makedirs(src)
    _builders.write_synchrotron_scan(str(src), "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    _builders.write_eclab_echem(str(src / "echem.txt"))
    ok, msgs = core.generate([str(src)], str(tmp_path / "x.nxs"), core.DataSourceType.SYNCHROTRON)
    assert ok, msgs


# ============================================================================
# Readers
# ============================================================================

@pytest.mark.parametrize("name, text", [
    ("scan.dat", "# Date 2024-02-05T10:05:00\n# tth Intensity\n"),
    ("scan.dat", ""),
    ("scan.xy", ""),
    ("scan.xy", "# only a comment\n"),
])
def test_one_d_file_without_data_rows_is_unreadable(tmp_path, name, text):
    """An empty or header-only 1D file raises one IOError naming the path once, with no numpy warning."""
    path = tmp_path / name
    path.write_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(IOError, match="no data rows$") as err:
            core.DataReaderFactory.read_file(str(path), use_cache=False)
    assert str(err.value).count(str(path)) == 1, str(err.value)
    assert not caught, [str(w.message) for w in caught]


def test_header_only_dat_is_unreadable_through_the_single_pass_reader(tmp_path):
    """read_with_lines (the writer's neutron read) also refuses a file with no data rows."""
    path = tmp_path / "123456-1_10raw-0.dat"
    path.write_text("# Mantid-like comment\n# X Y E\n")
    reader = core.DataReaderFactory.get_reader(str(path), is_neutron=True)
    with pytest.raises(IOError):
        reader.read_with_lines(str(path))


@pytest.mark.parametrize("text", [
    "1.0 2.0#c\n3.0 4.0\n",
    "# Date 2024-02-05T10:05:00\n# tth Intensity\n1.0 2.0#first point\n3.0 4.0\n",
])
def test_dat_first_row_with_a_glued_comment_is_kept(tmp_path, text):
    """A '#' glued to the first data row does not drop that row; numpy strips the comment."""
    path = tmp_path / "glued.dat"
    path.write_text(text)
    data = core.DataReaderFactory.read_file(str(path), use_cache=False)
    assert data.tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("text,expected", [
    ("XYDATA\n1 2 3 4\n5 6 7 8\n", [[1, 2, 3], [5, 6, 7]]),          # X, Y, E only
    ("XYDATA\n1 2 3\n4 5\n", [[1, 2], [4, 5]]),                         # ragged: error column dropped
    ("XYDATA\n1 2 abc\n4 5 6\n", [[1, 2], [4, 5]]),                     # non-numeric E
    ("XYDATA\n1 2 3\nfoo bar\n4 5 6\n", [[1, 2, 3], [4, 5, 6]]),        # stray text line skipped
    ("XYDATA\n1 2 3#c\n4 5 6\n", [[1, 2], [4, 5]]),                     # glued comment as the line parse reads it
    ("1 2 3 4\n5 6 7 8\n", [[1, 2, 3, 4], [5, 6, 7, 8]]),               # clean file: numpy reads every column
    ("# h\n1 2\n3 4\n", [[1, 2], [3, 4]]),
], ids=["four-cols", "ragged", "text-error", "stray-line", "glued-comment", "clean-four", "clean-two"])
def test_dat_reader_columns_per_layout(tmp_path, text, expected):
    """An XYDATA file keeps X, Y and a readable E column; a clean table keeps every column."""
    path = tmp_path / "x.dat"
    path.write_text(text)
    assert np.array_equal(core.DATReader()._read_impl(str(path)), np.array(expected, dtype=float))


@pytest.mark.parametrize("shape", ["lab", "polaris", "wish"])
def test_real_dat_shapes_read_in_one_pass(tmp_path, monkeypatch, shape):
    """Real .dat layouts load through numpy (no failed attempt, no line-by-line fallback) with X, Y and E as written."""
    path = str(tmp_path / f"{shape}.dat")
    x = np.linspace(1000.0, 20000.0, 40)
    y, e = 50.0 + np.arange(40.0), np.full(40, 0.25)
    if shape == "lab":
        _builders.write_lab_dat(path, "2024-02-05T10:05:00", x=x, y=y)
        e = np.sqrt(y)
    else:
        _builders.write_mantid_bank(path, shape.upper(), "tof", 1, x, y, e)
    raised = []
    real = np.loadtxt

    def spy(*a, **k):
        try:
            return real(*a, **k)
        except ValueError as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(readers.np, "loadtxt", spy)
    data = core.DATReader(data_type="neutron").read(path, use_cache=False)
    assert np.allclose(data[:, 0], x) and np.allclose(data[:, 1], y) and np.allclose(data[:, 2], e, atol=1e-6)
    assert raised == [], "fast path failed and fell back to the line parser"


def test_bare_table_dat_is_opened_once(tmp_path, monkeypatch):
    """A header-less bank table (Run1's layout) is opened once: numpy parses the bytes already read."""
    path = str(tmp_path / "123456-1_10raw-0.dat")
    with open(path, "w") as f:
        f.write("0.94279 75.47654295 2.26047823\n0.94379 76.1 2.3\n")
    opened = []
    real_open, real_loadtxt = open, np.loadtxt

    def counting_open(file, *args, **kwargs):
        opened.append(file)
        return real_open(file, *args, **kwargs)

    def counting_loadtxt(source, *args, **kwargs):
        if isinstance(source, (str, os.PathLike)):
            opened.append(source)  # numpy opens a path itself
        return real_loadtxt(source, *args, **kwargs)

    monkeypatch.setattr(readers, "open", counting_open, raising=False)
    monkeypatch.setattr(readers.np, "loadtxt", counting_loadtxt)
    data, _ = core.DataReaderFactory.get_reader(path, is_neutron=True).read_with_lines(path)
    assert data.tolist() == [[0.94279, 75.47654295, 2.26047823], [0.94379, 76.1, 2.3]]
    assert opened == [path], opened


@pytest.mark.parametrize("text", [
    "0.94279 75.47654295 2.26047823\n0.94379 76.1 2.3\n",
    "1 2 3 4\n5 6 7 8\n",
    "# h\n1 2\n3 4\n",
    "XYDATA\n# X Y E\n1 2 3\n4 5 6\n",
], ids=["bank", "clean-four", "comment-header", "xydata"])
def test_dat_with_a_bom_reads_as_without(tmp_path, text):
    """A UTF-8 BOM changes neither rows, columns nor lines (it cost a bare table its first row)."""
    (tmp_path / "plain.dat").write_bytes(text.encode())
    (tmp_path / "bom.dat").write_bytes(b"\xef\xbb\xbf" + text.encode())
    reader = core.DATReader(data_type="neutron")
    data, lines = reader.read_with_lines(str(tmp_path / "bom.dat"))
    expected, expected_lines = reader.read_with_lines(str(tmp_path / "plain.dat"))
    assert data.shape == expected.shape and np.array_equal(data, expected), data
    assert lines == expected_lines, lines[:2]


def test_lab_dat_with_a_bom_loads_with_its_date_and_data(tmp_path):
    """A UTF-8 BOM before '# Date' changes nothing: the scan keeps its time, pattern and voltage."""
    models = []
    for name in ("plain", "bom"):
        src = str(tmp_path / name)
        _builders.write_inhouse_dir(src)
        if name == "bom":
            path = os.path.join(src, "scan_001.dat")
            with open(path, "rb") as f:
                raw = f.read()
            with open(path, "wb") as f:
                f.write(b"\xef\xbb\xbf" + raw)
        out = str(tmp_path / f"{name}.nxs")
        ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE)
        assert ok, msgs
        models.append(core.load(out))
    plain, bom = models
    assert len(bom.scans) == 3, [s.timestamp for s in bom.scans]
    for a, b in zip(plain.scans, bom.scans):
        assert (b.timestamp, b.echem) == (a.timestamp, a.echem), b.scan_num
        assert b.oned.keys() == a.oned.keys(), b.scan_num
        assert all(np.array_equal(b.oned[k], a.oned[k]) for k in a.oned), b.scan_num


def test_hdf_group_at_a_known_path_is_refused_not_guessed(tmp_path):
    """A group at a known path is refused; a Pixium frame and an unknown layout still read."""
    frame = np.arange(1, 4097, dtype=np.uint32).reshape(1, 64, 64)
    reader = core.HDFReader(max_display_size=0)
    eiger = str(tmp_path / "eiger.hdf")
    with h5py.File(eiger, "w") as f:
        grp = f.create_group("entry/data")
        grp.attrs["NX_class"], grp.attrs["signal"] = "NXdata", "image"
        grp["image"] = frame
    with pytest.raises(IOError, match="No suitable data found"):
        reader.read(eiger, use_cache=False)
    # HM28 layout: the frame at the first known path and /entry/data/data; /entry/data is a group
    pixium = str(tmp_path / "pixium.hdf")
    with h5py.File(pixium, "w") as f:
        f["entry/instrument/detector/data"] = frame
        f["entry/data/data"] = f["entry/instrument/detector/data"]
    assert np.array_equal(reader.read(pixium, use_cache=False), frame[0])
    # No known path: the largest 2D/3D dataset
    other = str(tmp_path / "other.hdf")
    with h5py.File(other, "w") as f:
        f["scan/thumbnail"] = frame[0, :16, :16]
        f["scan/image"] = frame[0]
    assert np.array_equal(reader.read(other, use_cache=False), frame[0])


def _write_frame(path, shape=(400, 400)):
    """Write an HDF image of positive uint32 counts at /entry/data/data."""
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/data/data",
                         data=(np.arange(np.prod(shape), dtype=np.uint32) % 997 + 1).reshape(shape))


def test_downsampled_hdf_frame_owns_its_pixels_and_is_counted(tmp_path):
    """A display-size read keeps only the strided pixels, and the cache counts those bytes."""
    path = str(tmp_path / "frame.hdf")
    _write_frame(path)
    full = core.HDFReader(max_display_size=0).read(path, use_cache=False)
    img = core.HDFReader(max_display_size=100).read(path)
    assert np.array_equal(img, full[::4, ::4]) and img.dtype == full.dtype
    assert img.base is None, "the downsampled frame must not pin the full-size array"
    assert readers._file_cache.current_size == img.nbytes


def test_file_cache_counts_the_array_a_view_keeps_alive():
    """A view is sized by its base array, so the byte budget evicts."""
    cache = readers.FileCache(max_size_mb=1)
    for key in "abc":
        cache.put(key, np.zeros(60_000)[::10])  # 480 KB base each, 48 KB views
    assert cache.current_size <= cache.max_size_bytes
    assert len(cache.cache) == 2, sorted(cache.cache)


def test_file_cache_evicts_least_recently_used():
    """Under budget pressure the entry used longest ago goes, not the one used least often."""
    cache = readers.FileCache(max_size_mb=1)
    cache.put("a", np.zeros(50_000))  # 400 KB each, two fit
    cache.get("a")
    cache.get("a")
    cache.put("b", np.zeros(50_000))
    cache.put("c", np.zeros(50_000))
    assert sorted(cache.cache) == ["b", "c"]
    cache.get("b")
    cache.put("d", np.zeros(50_000))
    assert sorted(cache.cache) == ["b", "d"]


@needs_fabio
def test_generate_leaves_the_reader_cache_empty(tmp_path, polaris_src):
    """1D, embedded EDF and neutron bank reads by the writer bypass the cache."""
    src = str(tmp_path / "inhouse")
    _builders.write_detailed_inhouse_dir(src)
    ok, msgs = core.generate([src], str(tmp_path / "inhouse.nxs"),
                             core.DataSourceType.INHOUSE, include_2d_images=True)
    assert ok, msgs
    ok, msgs = core.generate([polaris_src], str(tmp_path / "polaris.nxs"),
                             core.DataSourceType.NEUTRON)
    assert ok, msgs
    assert readers._file_cache.cache == {}, sorted(readers._file_cache.cache)[:3]


# ============================================================================
# Round-trip
# ============================================================================

@pytest.fixture
def inhouse_rt(inhouse_dir, tmp_path):
    """Return pipeline scans plus the model reloaded from the generated in-house .nxs."""
    scans, _ = core.process_raw([inhouse_dir], core.DataSourceType.INHOUSE)
    nxs = str(tmp_path / "inhouse.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE)
    assert ok, f"generate in-house nxs failed: {msgs}"
    return scans, core.load(nxs)


@pytest.fixture
def neutron_rt(neutron_dir, tmp_path):
    """Return the generated-and-reloaded neutron model."""
    nxs = str(tmp_path / "neutron.nxs")
    ok, msgs = core.generate([neutron_dir], nxs, core.DataSourceType.NEUTRON)
    assert ok, f"generate neutron nxs failed: {msgs}"
    return core.load(nxs)


def test_inhouse_scan_fields_roundtrip(inhouse_rt):
    """Voltage, current, timestamp and exposure survive write -> load per scan."""
    scans, model = inhouse_rt
    assert len(model.scans) == 3, f"got {len(model.scans)}"
    for pipeline_scan, loaded in zip(scans, model.scans):
        assert loaded.echem == pipeline_scan.echem, \
            f"scan {loaded.scan_num}: file={loaded.echem} pipeline={pipeline_scan.echem}"
        assert loaded.current == pipeline_scan.current, f"scan {loaded.scan_num}: current"
        assert loaded.timestamp == pipeline_scan.timestamp, \
            f"scan {loaded.scan_num}: file={loaded.timestamp} pipeline={pipeline_scan.timestamp}"
        assert loaded.exposure_time == pipeline_scan.exposure_time, \
            f"scan {loaded.scan_num}: exposure"


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


def test_mantid_split_set_writes_banks_1_to_5_from_their_own_files(tmp_path):
    """A WISH run saved as Mantid split files -0 to -13 writes banks 1-5, each from its own file."""
    src = tmp_path / "split"
    _builders.write_wish_dir(str(src), banks=())
    for i in range(14):
        _builders.write_mantid_bank(str(src / "diffraction-data" / f"123456-{i}.dat"), "WISH",
                                    "tof", i, _builders.TOF_X, 50.0 + i + np.arange(40.0),
                                    np.ones(40))
    out = str(tmp_path / "split.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.NEUTRON)
    assert ok, msgs
    banks = core.load(out).scans[0].neutron
    assert sorted(banks, key=int) == ["1", "2", "3", "4", "5"], sorted(banks)
    for bank in range(1, 6):
        assert banks[str(bank)]["tof"]["source"] == f"123456-{bank}.dat", bank
        assert banks[str(bank)]["tof"]["y"][0] == 50.0 + bank, bank
    assert_pynxtools_valid(out, "mantid split")


@pytest.mark.parametrize("rename", [lambda name: name[:-4] + ".DAT", str.upper],
                         ids=["upper-case-extension", "all-upper-case"])
def test_upper_case_wish_set_writes_tof_and_d(tmp_path, rename):
    """A WISH set whose bank files end in .DAT, or are all upper case, writes every bank's TOF and d."""
    src = tmp_path / "upper"
    _builders.write_wish_dir(str(src))
    data = src / "diffraction-data"
    for name in os.listdir(data):
        os.replace(data / name, data / rename(name))
    out = str(tmp_path / "upper.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.NEUTRON)
    assert ok, msgs
    banks = core.load(out).scans[0].neutron
    assert {b: sorted(v) for b, v in banks.items()} == {"1": ["d", "tof"], "2": ["d", "tof"]}, banks
    expected = rename(_builders.wish_bank_names("123456", 1)[0])
    assert banks["1"]["tof"]["source"] == expected, banks["1"]["tof"]["source"]
    assert_pynxtools_valid(out, "upper-case WISH")


def test_neutron_writer_parses_each_bank_header_from_its_read(tmp_path, monkeypatch):
    """The writer takes each bank's Mantid header from the lines it read, not a second open."""
    src = str(tmp_path / "neutron")
    _builders.write_detailed_neutron_dir(src)
    calls = []
    real = core.nxs_writer.parse_mantid_header
    monkeypatch.setattr(core.nxs_writer, "parse_mantid_header",
                        lambda path: calls.append(path) or real(path))
    out = str(tmp_path / "n.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.NEUTRON)
    assert ok, msgs
    assert len(calls) == 1, f"only the entry-level instrument harvest may reopen a bank, got {len(calls)}"
    with h5py.File(out) as f:
        b = f["entry/scan_000001/bank_1"]
        assert b.attrs["spectrum"] == 1 and b.attrs["x_label"] == "Time-of-flight"
        assert b.attrs["y_unit"] == "per microsecond"


def test_neutron_scan_with_no_readable_bank_is_written_with_a_warning(tmp_path, caplog):
    """A run whose every bank file is unreadable is written without data and named in a warning."""
    src = str(tmp_path / "neutron")
    lines = [_builders.wish_logbook_line("123456", "2024-02-05 10:00:00", "2024-02-05 10:30:00"),
             _builders.wish_logbook_line("123457", "2024-02-05 10:32:00", "2024-02-05 10:36:00")]
    _builders.write_wish_dir(src, run="123456", banks=(1, 2), lines=lines)
    for bank in (1, 2):
        for name in _builders.wish_bank_names("123457", bank):
            with open(os.path.join(src, "diffraction-data", name), "w") as f:
                f.write("XYDATA\nnot a number\n")
    out = str(tmp_path / "out.nxs")
    with caplog.at_level(logging.WARNING, logger="core"):
        ok, msgs = core.generate([src], out, core.DataSourceType.NEUTRON)
    assert ok, msgs
    assert ("Scan 2 (run 123457): no bank file could be read; written without data"
            in [r.getMessage() for r in caplog.records]), [r.getMessage() for r in caplog.records]
    with h5py.File(out) as f:
        assert "bank_1" in f["entry/scan_000001"]
        assert not any(name.startswith("bank_") for name in f["entry/scan_000002"])
    assert_pynxtools_valid(out)


def test_scan_whose_dat_has_no_rows_is_kept_without_a_pattern(tmp_path, caplog):
    """The scan, its time and its echem stay; it has no data group and an ERROR names it."""
    src = str(tmp_path / "inhouse")
    _builders.write_inhouse_dir(src)
    with open(os.path.join(src, "scan_002.dat")) as f:
        header = [ln for ln in f if ln.startswith("#")]
    with open(os.path.join(src, "scan_002.dat"), "w") as f:
        f.writelines(header)
    out = str(tmp_path / "no_rows.nxs")
    with caplog.at_level(logging.ERROR):
        ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    with h5py.File(out, "r") as f:
        assert "data" in f["entry/scan_000001"] and "data" in f["entry/scan_000003"]
        assert "data" not in f["entry/scan_000002"]
        assert "environment/voltage" in f["entry/scan_000002"]
    assert any("Error reading 1D data for scan 2" in r.getMessage() for r in caplog.records)
    assert_pynxtools_valid(out, "no rows")


def test_embedded_hdf_frame_keeps_its_dtype(tmp_path):
    """'Include 2D images' stores a uint32 frame as uint32 with the floor applied, and it validates."""
    src = str(tmp_path / "synchrotron")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    frame = (np.arange(256, dtype=np.uint32) % 50).reshape(16, 16)  # zeros raised to the floor 1
    with h5py.File(os.path.join(src, "pixium-000001.hdf"), "w") as f:
        f["entry/instrument/detector/data"] = frame
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    out = str(tmp_path / "embedded.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.SYNCHROTRON, include_2d_images=True)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        stored = f["entry/scan_000001/image_data/data"]
        assert stored.dtype == np.uint32, f"stored as {stored.dtype}"
        assert np.array_equal(stored[()], np.maximum(frame, 1))
    assert_pynxtools_valid(out)
    model = core.load(out)
    assert np.array_equal(model.scans[0].twod, np.maximum(frame, 1).astype(float))


@needs_fabio
def test_embedded_edf_frame_keeps_its_dtype(tmp_path):
    """'Include 2D images' stores a float32 EDF frame as float32 with the floor applied, and it validates."""
    src = str(tmp_path / "inhouse")
    os.makedirs(src)
    _builders.write_xrd_dat(os.path.join(src, "scan_001.dat"), _builders.SCAN_TIMES[0])
    # Values -2 to 61: the mask -2, -1 and the gap 0 are raised to the floor 1
    _builders.write_edf_image(os.path.join(src, "image_001.edf"), _builders.SCAN_TIMES[0], offset=-2.0)
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    expected = np.maximum(np.arange(64, dtype=np.float32).reshape(8, 8) - 2, 1)
    out = str(tmp_path / "embedded.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE, include_2d_images=True)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        stored = f["entry/scan_000001/image_data/data"]
        assert stored.dtype == np.float32, f"stored as {stored.dtype}"
        assert np.array_equal(stored[()], expected)
    assert_pynxtools_valid(out)
    model = core.load(out)
    assert np.array_equal(model.scans[0].twod, expected.astype(float))


@needs_fabio
@pytest.mark.parametrize("counter", ["nan", "inf"])
def test_non_finite_monitor_counter_writes_no_integral(tmp_path, counter):
    """An EDF Monitor of nan or inf writes no scan monitor, as a zero counter does."""
    src = str(tmp_path / "lab")
    os.makedirs(src)
    _builders.write_edf_image(os.path.join(src, "image_001.edf"), _builders.SCAN_TIMES[0],
                              monitor=counter)
    out = str(tmp_path / "lab.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    with h5py.File(out) as f:
        assert "monitor" not in f["entry/scan_000001"], list(f["entry/scan_000001"])
    assert_pynxtools_valid(out, f"Monitor {counter}")


@pytest.mark.parametrize("value, written", [(0.495, 0.495), ("0.495", 0.495), ("n/a", None),
                                            ("nan", None), (float("inf"), None)])
def test_profile_wavelength_is_written_when_numeric_and_skipped_with_a_warning_otherwise(
        tmp_path, monkeypatch, caplog, value, written):
    """A finite profile 'wavelength' (angstrom) is written; text, nan or inf warns and is skipped."""
    monkeypatch.setitem(core.config.DEFAULT_PROFILES["synchrotron"], "wavelength", value)
    src = str(tmp_path / "synchrotron")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00", "2024-02-05T10:05:39")
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    out = str(tmp_path / "syn.nxs")
    with caplog.at_level(logging.WARNING):
        ok, msgs = core.generate([src], out, core.DataSourceType.SYNCHROTRON)
    assert ok, str(msgs)
    with h5py.File(out, "r") as f:
        inst = f["entry/instrument"]
        if written is None:
            assert "crystal" not in inst
            text = f"Skipping profile wavelength {value!r}: not a finite number"
            assert any(r.getMessage() == text for r in caplog.records)
        else:
            assert inst["crystal/wavelength"][()] == pytest.approx(written)
            assert inst["crystal/wavelength"].attrs["units"] == "angstrom"
    assert_pynxtools_valid(out, f"wavelength {value!r}")


def _failing_scan_writer(monkeypatch, fail_on=2):
    """Make NXSWriter._write_scan raise on scan `fail_on`, as a full disk would."""
    original = nxs_writer.NXSWriter._write_scan

    def write_scan(self, entry, scan):
        if scan.scan_num == fail_on:
            raise OSError("simulated disk full")
        return original(self, entry, scan)

    monkeypatch.setattr(nxs_writer.NXSWriter, "_write_scan", write_scan)


def test_failed_regenerate_keeps_previous_file(inhouse_dir, tmp_path, monkeypatch):
    """A write that fails part way leaves the earlier file at that path whole, and no .part."""
    out = str(tmp_path / "run.nxs")
    ok, msgs = core.generate([inhouse_dir], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    _failing_scan_writer(monkeypatch)
    ok, msgs = core.generate([inhouse_dir], out, core.DataSourceType.INHOUSE)
    assert not ok and "simulated disk full" in msgs[0], msgs
    assert len(core.load(out).scans) == 3, "the previous 3-scan file must survive"
    assert not os.path.exists(out + ".part")


def test_failed_first_write_leaves_no_file(inhouse_dir, tmp_path, monkeypatch):
    """A failed write to a new path leaves nothing that is_canonical_nxs could accept."""
    out = str(tmp_path / "new.nxs")
    _failing_scan_writer(monkeypatch)
    ok, _ = core.generate([inhouse_dir], out, core.DataSourceType.INHOUSE)
    assert not ok
    assert not os.path.exists(out) and not os.path.exists(out + ".part")


def _pairs_zip(tmp_path, name, prefix=""):
    """Zip three scan_00i .dat/.edf pairs with echem under prefix and return the ZIP's path."""
    src = tmp_path / "pairs"
    if not src.is_dir():
        _builders.write_dated_pairs(str(src), _builders.SCAN_TIMES, _builders.SCAN_TIMES)
    return _builders.zip_tree(str(src), str(tmp_path / name), prefix=prefix)


def _interrupt(monkeypatch, stage):
    """Return (generate keyword arguments, the expected message) for a 3-scan run stopped at `stage`:
    "failure" (scan 2's write fails), "sidecar" (before the 2D moves) or "write" (after scan 1)."""
    cancel = threading.Event()
    if stage == "failure":
        _failing_scan_writer(monkeypatch)
        return {}, "Error: simulated disk full"
    if stage == "sidecar":
        return ({"cancel": cancel,
                 "progress_callback": lambda msg: msg == "Writing NeXus file..." and cancel.set()},
                "Error: Generation cancelled")
    original = nxs_writer.NXSWriter._write_scan

    def write_scan(self, entry, scan):
        cancel.set()
        return original(self, entry, scan)

    monkeypatch.setattr(nxs_writer.NXSWriter, "_write_scan", write_scan)
    return {"cancel": cancel}, "Error: Generation cancelled"


def _live_twod(out):
    """Return {scan number: image bytes} for the scans of out whose 2D source file exists."""
    return {s.scan_num: Path(s.twod_source).read_bytes() for s in core.load(out).scans
            if s.twod_source and os.path.isfile(s.twod_source)}


@needs_fabio
@pytest.mark.parametrize("stage", ["failure", "sidecar", "write"])
def test_interrupted_zip_regenerate_keeps_previous_images(tmp_path, monkeypatch, stage):
    """An interrupted ZIP regenerate keeps the old .nxs and its 2D images; a later success replaces both."""
    out = str(tmp_path / "run.nxs")
    ok, msgs = core.generate([_pairs_zip(tmp_path, "a.zip")], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    before, images = Path(out).read_bytes(), _live_twod(out)
    assert len(images) == 3, images.keys()
    moved = _pairs_zip(tmp_path, "b.zip", prefix="cell/")
    with monkeypatch.context() as m:
        kwargs, message = _interrupt(m, stage)
        ok, msgs = core.generate([moved], out, core.DataSourceType.INHOUSE, **kwargs)
    assert not ok and msgs == [message], msgs
    assert Path(out).read_bytes() == before
    assert _live_twod(out) == images, "previous 2D images lost"
    ok, msgs = core.generate([moved], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    assert len(_live_twod(out)) == 3
    assert sorted(n for n in os.listdir(tmp_path) if n.startswith("run")) == ["run.nxs", "run_images"]


@needs_fabio
@pytest.mark.parametrize("stage", ["failure", "sidecar", "write"])
def test_interrupted_first_zip_write_leaves_nothing(tmp_path, monkeypatch, stage):
    """A ZIP run to a new path that fails or is cancelled leaves no .nxs, .part or images folder."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    kwargs, message = _interrupt(monkeypatch, stage)
    ok, msgs = core.generate([_pairs_zip(tmp_path, "a.zip")], str(out_dir / "run.nxs"),
                             core.DataSourceType.INHOUSE, **kwargs)
    assert not ok and msgs == [message], msgs
    assert os.listdir(out_dir) == []


def test_file_name_attribute_names_the_destination(inhouse_dir, tmp_path):
    """NXroot@file_name is the final name, never the temporary part name."""
    out = str(tmp_path / "named.nxs")
    ok, msgs = core.generate([inhouse_dir], out, core.DataSourceType.INHOUSE)
    assert ok, msgs
    with h5py.File(out, "r") as f:
        name = f.attrs["file_name"]
    assert (name.decode() if isinstance(name, bytes) else name) == "named.nxs"


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
# Example data
# ============================================================================

GENERATOR = Path(__file__).resolve().parents[1] / "examples" / "make_example_data.py"


def _generator():
    """Import examples/make_example_data.py as a module."""
    spec = importlib.util.spec_from_file_location("make_example_data", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_matches_builders(tmp_path):
    """The walkthrough's laboratory and neutron files are byte-identical to tests/_builders.py."""
    gen = _generator()
    gen.make_laboratory(str(tmp_path / "ex" / "laboratory"))
    gen.make_neutron(str(tmp_path / "ex" / "neutron"))
    _builders.write_inhouse_dir(str(tmp_path / "b" / "laboratory"))
    _builders.write_neutron_dir(str(tmp_path / "b" / "neutron"))
    for sub in ("laboratory", "neutron"):
        cmp = filecmp.dircmp(tmp_path / "ex" / sub, tmp_path / "b" / sub)
        assert not (cmp.left_only or cmp.right_only), sub
        _, differ, errors = filecmp.cmpfiles(tmp_path / "ex" / sub, tmp_path / "b" / sub,
                                             cmp.common_files, shallow=False)
        assert not differ and not errors, (sub, differ, errors)


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
    cache = readers.FileCache(max_size_mb=1)
    for _ in range(3):
        cache.put("same", np.zeros(50_000))
    assert cache.current_size == cache.cache["same"].nbytes
    cache.put("other", np.zeros(80_000))
    assert sorted(cache.cache) == ["other", "same"], "phantom bytes must not force an eviction"
    # A larger replacement evicts the least recently used entry; a stale tick must not stall it
    cache.get("other")
    cache.get("other")
    worker = threading.Thread(target=cache.put, args=("same", np.zeros(100_000)), daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "put must not spin on a stale tick"
    assert sorted(cache.cache) == ["same"], "the least recently used entry gives way"


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
