"""Classification: file classifiers, groupers and metadata parsers."""
import logging
import os

import pandas as pd
import pytest

from operaxn import core

import _builders


# ============================================================================
# Synchrotron grouping
# ============================================================================

SYN_FILES = {
    "/data/sample_123456.nxs": "o1",
    "/data/sample_123456.hdf": "o2",
    "/data/sample_123456_integration_00.xy": "o3",
    "/data/other_777777.nxs": "o4",
    "/data/orphan_999999.hdf": "o5",
    "/data/sample_123456_extra.xy": "o6",
}


@pytest.mark.parametrize("files, xy", [
    pytest.param(list(SYN_FILES), "/data/sample_123456_extra.xy", id="fixed-mix"),
    pytest.param([f for f in SYN_FILES if f != "/data/sample_123456_extra.xy"],
                 "/data/sample_123456_integration_00.xy", id="without-extra-xy"),
])
def test_synchrotron_grouping(files, xy):
    """A .nxs anchors a group (a lone one too), an orphan .hdf none; the last .xy grouped wins."""
    assert core.SynchrotronFileGrouper().group_files(files) == {
        "123456": {"nxs": "/data/sample_123456.nxs", "hdf": "/data/sample_123456.hdf", "xy": xy},
        "777777": {"nxs": "/data/other_777777.nxs"}}


@pytest.mark.parametrize("prefix", ["i11-1-", "cell_0.5C_"])
def test_synchrotron_dotted_names_keep_their_xy(prefix):
    """A dot inside the stem (e.g. '0.5C') neither drops the .xy nor changes the scan ID."""
    files = [f"{prefix}12345.nxs", f"pixium_{prefix}12345.hdf",
             f"{prefix}12345_integration_tth_0000_cell.xy"]
    groups = core.SynchrotronFileGrouper().group_files(files)
    assert groups == {"12345": {"nxs": files[0], "hdf": files[1], "xy": files[2]}}
    assert {core.SynchrotronFileGrouper.extract_scan_id(f) for f in files} == {"12345"}


# ============================================================================
# File classifiers
# ============================================================================

def test_txt_classifier_echem_vs_logbook(tmp_path):
    """.txt with echem keywords classifies as echem; a logbook line as neutron_meta."""
    echem = tmp_path / "echem.txt"
    _builders.write_echem_txt(str(echem))
    logbook = tmp_path / "logbook.txt"
    logbook.write_text(_builders.LOGBOOK_LINE + "\n")
    clf = core.TXTClassifier()
    assert clf.classify(str(echem)) == ("echem", None, None)
    assert clf.classify(str(logbook)) == ("neutron_meta", None, None)


def test_echem_logger_with_one_ctime_column_stays_echem(tmp_path):
    """An echem table with a record counter above 10,000 and one ctime column is not a logbook."""
    path = tmp_path / "cell_logger.txt"
    rows = ["Record\tTime\tEwe/V\tI/mA\tT1\tT2\tT3\tT4"]
    rows += [
        f"{100001 + i}\tWed Oct 15 09:{i:02d}:00 2025\t{3.7 + i / 100:.2f}\t0.1\t25\t25\t25\t25"
        for i in range(6)]
    path.write_text("\n".join(rows) + "\n")
    assert core.TXTClassifier().classify(str(path))[0] == "echem"


@pytest.mark.parametrize("value, expected", [
    pytest.param("2023-12-07T10:11:25", "2023-12-07 10:11:25", id="iso-T"),
    pytest.param("2024-02-05 10:05:00", "2024-02-05 10:05:00", id="iso-space"),
    pytest.param("05/02/2024 10:05:00", "2024-02-05 10:05:00", id="day-first"),
    pytest.param("Mon Feb  5 10:05:00 2024", "2024-02-05 10:05:00", id="ctime"),
])
def test_dat_date_value_is_read_whole(tmp_path, value, expected):
    """A .dat is oned with its exposure; the whole '# Date' is read, slash dates day-first."""
    path = tmp_path / "scan.dat"
    _builders.write_xrd_dat(str(path), value)
    assert core.DATClassifier().classify(str(path)) == ("oned", expected, 120.0)


@pytest.mark.parametrize("value, expected", [
    pytest.param("of scan 2024-02-05T10:05:00", "2024-02-05 10:05:00", id="run-on-label-iso"),
    pytest.param("not set 2024", None, id="last-token-a-year"),
    pytest.param("of acquisition 05/02/2024 10:05", None, id="last-token-a-time"),
])
def test_dat_date_falls_back_only_to_a_year_first_token(tmp_path, value, expected):
    """A run-on label keeps its ISO last token; a run-on value ending in a bare year or time
    skips the file instead of landing on 1 January or today's date."""
    path = tmp_path / "scan.dat"
    _builders.write_xrd_dat(str(path), value)
    result = core.DATClassifier().classify(str(path))[:2]
    assert result == (("oned", expected) if expected else (None, None)), result


@pytest.mark.parametrize("value, expected", [
    ("2024-07-15T18:52:17.306+01", "2024-07-15T18:52:17.306"),
    ("2024-02-05T10:05:00Z", "2024-02-05T10:05:00"),
    ("2024-02-05 10:05:00-05:00", "2024-02-05 10:05:00"),
    ("2024-02-05 10:05:00 +0100", "2024-02-05 10:05:00"),
    ("2024-02-05", "2024-02-05"),
    ("05/02/2024 10:05:00", "05/02/2024 10:05:00"),
    ("Mon Feb 05 10:05:00 2024", "Mon Feb 05 10:05:00 2024"),
])
def test_strip_utc_offset_keeps_the_wall_clock(value, expected):
    """Offsets and Z go; dates, day-first and ctime strings are left exactly as they were."""
    assert core.classify.strip_utc_offset(value) == expected


def test_classification_manager_inhouse(inhouse_dir):
    """classify_files fills oned/echem columns over the laboratory dir and returns a copy."""
    paths = [os.path.join(inhouse_dir, name) for name in sorted(os.listdir(inhouse_dir))]
    df = pd.DataFrame({"path": paths})
    out = core.FileClassificationManager(core.DataSourceType.INHOUSE).classify_files(df)
    assert out is not df and list(df.columns) == ["path"], "input frame must be untouched"
    oned = out[out["oned"].notna()]
    assert sorted(os.path.basename(p) for p in oned["oned"]) == \
        ["scan_001.dat", "scan_002.dat", "scan_003.dat"]
    assert list(oned["timestamp"]) == ["2024-02-05 10:05:00", "2024-02-05 10:10:00",
                                       "2024-02-05 10:15:00"]
    assert list(oned["exposure_time"]) == [120.0, 120.0, 120.0]
    echem_rows = out[out["echem"].notna()]
    assert len(echem_rows) == 1, str(out["echem"].tolist())
    assert os.path.basename(echem_rows["echem"].iloc[0]) == "echem.txt"


# ============================================================================
# Neutron logbooks
# ============================================================================

def test_neutron_metadata_parse(tmp_path):
    """The builder logbook line parses to one row with scan_id and datetime start/end."""
    path = tmp_path / "logbook.txt"
    path.write_text(_builders.LOGBOOK_LINE + "\n")
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and len(df) == 1
    assert {"scan_id", "start_time", "end_time"} <= set(df.columns), str(df.columns)
    row = df.iloc[0]
    assert row["scan_id"] == "123456", f"got {row['scan_id']}"
    assert row["start_time"] == pd.Timestamp("2024-02-05 10:00:00"), f"got {row['start_time']}"
    assert row["end_time"] == pd.Timestamp("2024-02-05 10:30:00"), f"got {row['end_time']}"


def test_neutron_logbook_single_surname_user(tmp_path):
    """WISH single-surname lines yield users and proposal; POLARIS lines are unchanged.
    Only a lone capitalised word (apostrophes and hyphens allowed) counts as the surname."""
    def wish_line(run, user, start_min):
        return (f'{run}\tcell55 event # 1\t"5m 0s"\t"3.0059"\t{user}\t2510713\t'
                f'Fri Feb 28 16:{start_min:02d}:29 2025\t'
                f'Fri Feb 28 16:{start_min + 5:02d}:36 2025\t24_5')

    lines = [wish_line("59268", "Perez", 17), wish_line("59269", "no beam", 27),
             wish_line("59270", "O'Neil-Smith", 37), wish_line("59271", "ok", 47),
             _builders.DETAILED_LOGBOOK_LINE]
    path = tmp_path / "logbook.txt"
    path.write_text("\n".join(lines) + "\n")
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and len(df) == 5, str(df)
    rows = {r["scan_id"]: r for _, r in df.iterrows()}
    assert rows["59268"]["users"] == "Perez", str(rows["59268"])
    assert rows["59268"]["proposal"] == "2510713", str(rows["59268"])
    assert pd.isna(rows["59269"]["users"]), "free text taken as the user"
    assert rows["59269"]["proposal"] == "2510713", str(rows["59269"])
    assert rows["59270"]["users"] == "O'Neil-Smith", str(rows["59270"])
    assert pd.isna(rows["59271"]["users"]), "lowercase word taken as the user"
    assert rows["123456"]["users"] == "Smith,Jones", str(rows["123456"])
    assert rows["123456"]["proposal"] == "7654321", str(rows["123456"])


@pytest.mark.parametrize("users, title", [
    ("Sato,Menon", "Op-Run1-SC1-Diag"),
    ("Perez,Montgomery", "Op-Run1-SC1-Diag"),
    ("Smith,Jones", "Monitor check"),
    ("Smith,Jones", "Saturated LiPF6 cell"),
    ("Friedrich", "Op-Run1-SC1-Diag"),
    ("Smith,Jones", "NMC Thu run"),
], ids=["sato", "montgomery", "monitor-title", "saturated-title", "friedrich", "thu-title"])
def test_logbook_fields_naming_a_weekday_still_parse(tmp_path, users, title):
    """A user or title containing a weekday abbreviation never takes the start-time slot."""
    path = tmp_path / "SC1_log_book.txt"
    _builders.write_logbook(str(path), [_builders.polaris_logbook_line(
        "123456", "2024-02-05 10:00:00", "2024-02-05 10:30:00", title=title, users=users)])
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and len(df) == 1, "logbook line dropped"
    row = df.iloc[0]
    assert row["start_time"] == pd.Timestamp("2024-02-05 10:00:00")
    assert row["end_time"] == pd.Timestamp("2024-02-05 10:30:00")
    assert row["users"] == users and row["run_title"] == title, row.to_dict()


@pytest.mark.parametrize("title", ["Li/NMC811 operando", "Si/Gr pouch", "Rest time 2h",
                                   "Voltage hold 4.3V"])
def test_logbook_with_an_echem_word_in_its_title_is_a_logbook(tmp_path, title):
    """A run title holding an echem keyword ('i/', 'time', 'voltage') stays a logbook."""
    path = tmp_path / "Cell_log_book.txt"
    _builders.write_logbook(str(path), [_builders.wish_logbook_line(
        "123456", "2024-02-05 10:00:00", "2024-02-05 10:30:00", title=title)], "WISH")
    assert core.TXTClassifier().classify(str(path)) == ("neutron_meta", None, None)


def test_bom_logbook_keeps_its_first_run(tmp_path):
    """A logbook saved as UTF-8 with BOM parses every line, the first included."""
    path = tmp_path / "Cell_log_book.txt"
    lines = [_builders.wish_logbook_line(run, f"2024-02-05 1{i}:00:00", f"2024-02-05 1{i}:30:00")
             for i, run in enumerate(("123456", "123457", "123458"))]
    path.write_bytes(b"\xef\xbb\xbf" + ("\n".join(lines) + "\n").encode("utf-8"))
    assert core.TXTClassifier().classify(str(path))[0] == "neutron_meta"
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and df["scan_id"].tolist() == ["123456", "123457", "123458"]


def _polaris_line(run, start, end, middle):
    """Return a POLARIS logbook line whose fields from title to proposal are `middle` as written."""
    return (f"{run}\t{middle}\t{_builders.journal_time(start)}\t"
            f"{_builders.journal_time(end)}\t")


@pytest.mark.parametrize("duration", ['"1m 27s"', '"10m 5s"', '"17m 13s"'],
                         ids=["eight-characters", "ten-minute-run", "longer-than-its-column"])
def test_logbook_duration_that_fills_its_column_parses(tmp_path, duration):
    """The POLARIS export aligns its columns with tabs, so an 8+ character duration is followed
    by one space (7 fields); it parses like its 8- and 9-field neighbours."""
    lines = [
        _polaris_line("153973", "2025-11-19 07:46:07", "2025-11-19 07:48:14",
                      'LDE-Run1-SC1-Op-Diag\t"2m 7s"\t"5.4427"\tPerez,Menon\t2520660'),
        _polaris_line("153974", "2025-11-19 07:48:35", "2025-11-19 07:50:41",
                      'LDE-Run1-SC1-Op-Diag\t"2m 6s"\t"5.439"\t\tPerez,Menon\t2520660'),
        _polaris_line("153975", "2025-11-19 07:52:01", "2025-11-19 07:53:28",
                      f'LDE-Run1-SC1-Op-Diag\t{duration} "3.7223"\tPerez,Menon\t2520660'),
    ]
    path = tmp_path / "SC1_log_book.txt"
    _builders.write_logbook(str(path), lines)
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and df["scan_id"].tolist() == ["153973", "153974", "153975"], str(df)
    for _, row in df.iterrows():
        assert (row["run_title"], row["users"], row["proposal"]) == \
            ("LDE-Run1-SC1-Op-Diag", "Perez,Menon", "2520660"), row.to_dict()
    assert (df.iloc[2]["start_time"], df.iloc[2]["end_time"]) == \
        (pd.Timestamp("2025-11-19 07:52:01"), pd.Timestamp("2025-11-19 07:53:28"))


@pytest.mark.parametrize("title, duration", [
    ("LDE-Run1-SC1-Op-Diag-24c", '"2m 6s"'),
    ("cell55 Without Vacuum HR", '"4h 17m 47s"'),
], ids=["dashed-title", "spaced-title"])
def test_logbook_title_that_fills_its_column_keeps_its_own_text(tmp_path, title, duration):
    """A 24-character title runs into the duration and charge after single spaces; the run
    title stops before the duration and the users and proposal are still found."""
    path = tmp_path / "SC1_log_book.txt"
    _builders.write_logbook(str(path), [_polaris_line(
        "153975", "2025-11-19 07:52:01", "2025-11-19 07:53:28",
        f'{title} {duration} "5.4427"\tPerez,Menon\t2520660')])
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and len(df) == 1, "logbook line dropped"
    row = df.iloc[0]
    assert (row["run_title"], row["users"], row["proposal"]) == \
        (title, "Perez,Menon", "2520660"), row.to_dict()


@pytest.mark.parametrize("runs", [["123457"], ["123457", "123458"]], ids=["one", "two"])
def test_unreadable_logbook_run_line_is_named(tmp_path, caplog, runs):
    """A line that starts with a run number but has no separate start and end time is named
    in one warning, never skipped silently; a line without a run number stays silent."""
    unreadable = {
        # Users that fill their column run into the proposal and the start time
        "123457": ('123457\tOp-Run1-SC1-Diag\t"2m 6s"\t"5.44"\tSmith,Jones,Wang 7654321 '
                   'Mon Feb 5 11:00:00 2024\tMon Feb 5 11:30:00 2024\t'),
        # A run still in progress has no end time yet
        "123458": ('123458\tOp-Run1-SC1-Diag\t""\t""\tSmith,Jones\t7654321\t'
                   'Mon Feb 5 12:00:00 2024\t'),
    }
    lines = ["Run\tTitle\tDuration",
             _builders.polaris_logbook_line("123456", "2024-02-05 10:00:00", "2024-02-05 10:30:00")]
    lines += [unreadable[run] for run in runs]
    path = tmp_path / "SC1_log_book.txt"
    _builders.write_logbook(str(path), lines)
    with caplog.at_level(logging.WARNING, logger="operaxn.core"):
        df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and df["scan_id"].tolist() == ["123456"], str(df)
    expected = ("1 logbook line has no readable start and end time and was skipped: 123457"
                if len(runs) == 1 else
                "2 logbook lines have no readable start and end time and were skipped: "
                "123457, 123458")
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [expected]


def test_logbook_whose_first_lines_lost_a_tab_is_a_logbook(tmp_path):
    """A logbook of 10-minute runs (every duration fills its column) is still a logbook, even
    with an echem word in its title, and every line parses."""
    lines = [_polaris_line(str(153975 + i), f"2025-11-19 0{i}:00:00", f"2025-11-19 0{i}:10:0{i}",
                           f'Li/NMC811-SC1-Op-Diag\t"10m {i}s" "3.7223"\tPerez,Menon\t2520660')
             for i in range(6)]
    path = tmp_path / "SC1_log_book.txt"
    _builders.write_logbook(str(path), lines)
    assert core.TXTClassifier().classify(str(path)) == ("neutron_meta", None, None)
    df = core.NeutronMetadataParser.parse(str(path))
    assert df is not None and len(df) == 6, str(df)


# ============================================================================
# Neutron bank files
# ============================================================================

def test_neutron_file_grouping(tmp_path):
    """Builder bank files group as scan -> measurement -> tof/d paths."""
    _builders.write_neutron_banks(str(tmp_path))
    files = [str(tmp_path / name) for name in sorted(os.listdir(tmp_path))]
    groups = core.NeutronFileGrouper().group_files(files)
    assert set(groups.keys()) == {"123456"}, str(groups)
    assert set(groups["123456"].keys()) == {"1", "2"}, str(groups["123456"])
    for meas in ("1", "2"):
        assert set(groups["123456"][meas].keys()) == {"tof", "d"}, str(groups["123456"][meas])
        assert groups["123456"][meas]["tof"].endswith(f"123456-{meas}-0.dat")
        assert groups["123456"][meas]["d"].endswith(f"123456-{meas}-d-0.dat")


def test_parse_mantid_header(tmp_path):
    """Instrument, X-axis unit and spectrum number harvest from a Mantid header."""
    path = tmp_path / "mantid.dat"
    path.write_text(
        "# File generated by Mantid, Instrument POLARIS\n"
        "# The X-axis unit is: Time-of-flight, The Y-axis unit is:  per microsecond\n"
        "# Spectrum 1\n"
        "1000.0 51.0 1.0\n")
    info = core.parse_mantid_header(str(path))
    assert info.get("instrument") == "POLARIS", str(info)
    assert info.get("x_unit") == "Time-of-flight", str(info)
    assert info.get("spectrum") == 1, str(info)


@pytest.mark.parametrize("filename, expected", [
    pytest.param(_builders.polaris_bank_names("153868", 1)[0], ("153868", "1", "tof"),
                 id="polaris-tof"),
    pytest.param(_builders.polaris_bank_names("153868", 5)[1], ("153868", "5", "d"),
                 id="polaris-d"),
    pytest.param(_builders.wish_bank_names("59268", 1)[0], ("59268", "1", "tof"), id="wish-tof"),
    pytest.param(_builders.wish_bank_names("59268", 5)[1], ("59268", "5", "d"), id="wish-d"),
    pytest.param("POL153868-b_13-d.dat", None, id="polaris-bank-13"),
    pytest.param("98765-13.dat", None, id="split-bank-13"),
    pytest.param("98765-23.dat", None, id="split-bank-23"),
    pytest.param("59268-10raw-0.dat", None, id="wish-panel-10"),
    pytest.param("POL153868-b_6-TOF.dat", None, id="polaris-bank-6"),
    pytest.param("98765-9.dat", None, id="split-bank-9"),
    pytest.param("cell_d_59268-1_10raw-0.dat", ("59268", "1", "tof"), id="cell_d-tof"),
    pytest.param("cell_d_59268-1_10raw-d-0.dat", ("59268", "1", "d"), id="cell_d-d"),
    pytest.param("LiNi_d_POL153868-b_1-TOF.dat", ("153868", "1", "tof"), id="d-before-run"),
])
def test_neutron_bank_names(filename, expected):
    """Real names hold, banks above 5 are skipped and a '_d_' before the run is no d marker."""
    info = core.NeutronFileGrouper._extract_neutron_file_info(filename)
    got = None if info is None else (info["scan_id"], info["measurement"], info["type"])
    assert got == expected


def _bank_warnings(caplog):
    """Return the WARNING messages the neutron grouper logged."""
    return [r.getMessage() for r in caplog.records
            if r.name == "operaxn.core.classify" and r.levelno == logging.WARNING]


def test_mantid_split_banks_keep_their_own_slots(caplog):
    """Split files 98765-0 to -13 fill banks 1-5 from their own files; one line names the rest."""
    files = sorted(f"/x/98765-{i}.dat" for i in range(14))
    with caplog.at_level(logging.WARNING, logger="operaxn.core"):
        groups = core.NeutronFileGrouper().group_files(files)
    assert {b: slot["tof"] for b, slot in groups["98765"].items()} == \
        {str(b): f"/x/98765-{b}.dat" for b in range(1, 6)}, groups
    assert _bank_warnings(caplog) == [
        "9 .dat files were not recognised as neutron bank 1-5 files and were ignored: "
        "98765-0.dat, 98765-10.dat, 98765-11.dat and 6 more"]


@pytest.mark.parametrize("banks, extra, line", [
    pytest.param(range(1, 10), [], "8 .dat files were not recognised as neutron bank 1-5 files and "
                 "were ignored: POL153868-b_6-TOF.dat, POL153868-b_6-d.dat, POL153868-b_7-TOF.dat "
                 "and 5 more", id="banks-6-to-9"),
    pytest.param((1, 2), ["/x/POL153868-b_13-d.dat"], "1 .dat file was not recognised as a neutron "
                 "bank 1-5 file and was ignored: POL153868-b_13-d.dat", id="one-bank-13-file"),
])
def test_banks_6_to_9_are_skipped_and_named(caplog, banks, extra, line):
    """Banks past 5 are not loaded nor fill bank 1; one line names them, in the singular for one."""
    files = [f"/x/{name}" for bank in banks
             for name in _builders.polaris_bank_names("153868", bank)] + extra
    with caplog.at_level(logging.WARNING, logger="operaxn.core"):
        groups = core.NeutronFileGrouper().group_files(files)
    expected = {}
    for bank in (range(1, 6) if len(banks) > 5 else banks):
        tof, d = _builders.polaris_bank_names("153868", bank)
        expected[str(bank)] = {"tof": f"/x/{tof}", "d": f"/x/{d}"}
    assert groups["153868"] == expected, groups
    assert _bank_warnings(caplog) == [line]


@pytest.mark.parametrize("case", ["upper-case-extension", "cell_d-prefix", "all-upper-case"])
def test_upper_case_dat_bank_files_group(caplog, case):
    """Bank files in .DAT, all upper case or after 'cell_d_' keep TOF and d apart, unwarned."""
    if case == "upper-case-extension":
        pairs = {run: {b: [f"{folder}{n[:-4]}.DAT" for n in names(run, b)] for b in (1, 2)}
                 for run, folder, names in (("153868", "/p/", _builders.polaris_bank_names),
                                            ("59268", "/w/", _builders.wish_bank_names))}
    elif case == "cell_d-prefix":
        pairs = {"59268": {1: [f"/x/cell_d_{n}" for n in _builders.wish_bank_names("59268", 1)]}}
    else:
        pairs = {"59268": {1: [f"/x/{n.upper()}" for n in _builders.wish_bank_names("59268", 1)]}}
    files = [path for banks in pairs.values() for pair in banks.values() for path in pair]
    with caplog.at_level(logging.WARNING, logger="operaxn.core"):
        groups = core.NeutronFileGrouper().group_files(files)
    assert groups == {run: {str(b): {"tof": tof, "d": d} for b, (tof, d) in banks.items()}
                      for run, banks in pairs.items()}, groups
    assert _bank_warnings(caplog) == []
