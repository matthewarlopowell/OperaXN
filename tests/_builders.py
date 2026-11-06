"""Synthetic dataset builders shared by the test suite; no data files are committed.

Defaults make correlation exact: echem runs at 60 s cadence from 10:00
with V = 3.7 + 0.01/step, so an XRD scan at 10:05 with 120 s exposure
correlates at its 10:06 midpoint to V = 3.76, and a neutron scan logged
10:00-10:30 at its 10:15 midpoint to V = 3.85. Three families share these
values: the plain family (50/40-point grids, 3 scans, simple logbook), the
real-file shapes (instrument export layouts; they back the
inhouse_dir and neutron_dir fixtures) and the DETAILED_* family (30-point
grids, 2 scans, richer logbook), which covers the Sigma column, POLARIS
Mantid headers, EDF harvest and trailing-timestamp logbook layout, so keep
it apart. examples/make_example_data.py duplicates the plain writers
standalone: keep the values in sync.
"""
import os
import zipfile

import h5py
import numpy as np
import pandas as pd

import core

ONED_X = np.linspace(10, 80, 50)
TOF_X = np.linspace(1000, 20000, 40)
D_X = np.linspace(0.5, 3.0, 40)

SCAN_TIMES = ["2024-02-05T10:05:00", "2024-02-05T10:10:00", "2024-02-05T10:15:00"]
LOGBOOK_LINE = ("123456\tuser\tsample title\tMon Feb 05 10:00:00 2024\t"
                "Mon Feb 05 10:30:00 2024\t1800\tok\textra")


def inhouse_scan_y(i):
    """Return the intensity array written for 1-based in-house scan number ``i``."""
    return 100 + 10 * i + np.arange(50, dtype=float)


def write_xrd_dat(path, timestamp, exposure="120.0", y=None):
    """Write an XRD .dat: '# Date <timestamp>', '# ExposureTime', a column header, x/y rows."""
    if y is None:
        y = inhouse_scan_y(1)
    lines = [f"# Date {timestamp}", f"# ExposureTime {exposure}", "# tth Intensity"]
    lines += [f"{x:.4f} {v:.4f}" for x, v in zip(ONED_X, y)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_echem_txt(path, n_rows=40, v0=3.7, dv=0.01, i0=0.1, di=0.001,
                    date="05/02/2024", hour=10):
    """Write an echem .txt: tab-delimited header and dayfirst timestamps at 60 s cadence."""
    rows = ["Time\tEwe/V\tI/mA"]
    for m in range(n_rows):
        rows.append(f"{date} {hour}:{m:02d}:00\t{v0 + m * dv:.4f}\t{i0 + m * di:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def write_arbin_xlsx(path, n_rows=40, v0=3.7, dv=0.01, i0=0.1, di=0.001):
    """Write an Arbin-style echem .xlsx matching write_echem_txt's anchors: absolute
    Date_Time plus an elapsed-seconds Test_Time(s) trap column."""
    times = pd.date_range("2024-02-05 10:00:00", periods=n_rows, freq="60s")
    df = pd.DataFrame({
        "Data_Point": np.arange(1, n_rows + 1),
        "Date_Time": times,
        "Test_Time(s)": np.arange(n_rows, dtype=float) * 60.0,
        "Current(A)": i0 + di * np.arange(n_rows),
        "Voltage(V)": v0 + dv * np.arange(n_rows),
    })
    df.to_excel(path, index=False)


def write_inhouse_scans(dirpath):
    """Write the 3 plain XRD scans (10:05/10:10/10:15) into an existing dirpath."""
    for i, ts in enumerate(SCAN_TIMES, start=1):
        write_xrd_dat(os.path.join(dirpath, f"scan_{i:03d}.dat"), ts,
                      y=inhouse_scan_y(i))


def write_inhouse_dir(dirpath):
    """Write 3 XRD scans at 10:05/10:10/10:15 and echem covering 10:00-10:39."""
    os.makedirs(dirpath, exist_ok=True)
    write_inhouse_scans(dirpath)
    write_echem_txt(os.path.join(dirpath, "echem.txt"))


def write_neutron_banks(dirpath, scan_id="123456", banks=(1, 2)):
    """Write <id>-<bank>-0.dat (TOF) and <id>-<bank>-d-0.dat (d-spacing) for each bank."""
    for bank in banks:
        tof_y = 50.0 + bank + np.arange(40, dtype=float)
        d_y = 70.0 + bank + np.arange(40, dtype=float)
        with open(os.path.join(dirpath, f"{scan_id}-{bank}-0.dat"), "w") as f:
            f.write("# Time-of-flight Y E\n")
            f.write("\n".join(f"{x:.10f} {y:.10f} 1.0"
                              for x, y in zip(TOF_X, tof_y)) + "\n")
        with open(os.path.join(dirpath, f"{scan_id}-{bank}-d-0.dat"), "w") as f:
            f.write("# d-Spacing Y E\n")
            f.write("\n".join(f"{x:.10f} {y:.10f} 1.0"
                              for x, y in zip(D_X, d_y)) + "\n")


def write_neutron_dir(dirpath):
    """Write a 10:00-10:30 logbook, 2 banks x (TOF, d) and echem covering the window."""
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "logbook.txt"), "w") as f:
        f.write(LOGBOOK_LINE + "\n")
    write_neutron_banks(dirpath)
    write_echem_txt(os.path.join(dirpath, "echem.txt"))


# The detailed family: see the module docstring for why these values differ.
DETAILED_ONED_X = np.linspace(10, 80, 30)
DETAILED_TOF_X = np.linspace(1000, 20000, 30)
DETAILED_D_X = np.linspace(0.5, 3.0, 30)

DETAILED_SCAN_TIMES = ["2024-02-05T10:05:00", "2024-02-05T10:10:00"]
DETAILED_LOGBOOK_LINE = (
    '123456\tOp-Run1-SC1-Diag\t"2m 6s"\t"5.44"\tSmith,Jones\t'
    "7654321\tMon Feb 05 10:00:00 2024\tMon Feb 05 10:30:00 2024")

CYCLING_PROTOCOL = {
    "technique": "GCPL C/10 with 1 h rest",
    "voltage_window_lower": 2.8,
    "voltage_window_upper": 4.3,
    "c_rate": "C/10",
    "instrument": "Biologic VMP3",
    "software": "EC-Lab 11.50",
    "raw_data_file": "doi:10.5281/zenodo.0000000",
}


def write_edf_image(path, timestamp, offset=1.0, monitor="0", **header):
    """Write an 8x8 EDF image with the harvested header fields (``header`` overrides any)."""
    import fabio
    img = fabio.edfimage.EdfImage(
        data=np.arange(64, dtype=np.float32).reshape(8, 8) + offset,
        header={"Date": timestamp, "ExposureTime": "120.0",
                "WaveLength": "1.541891e-10", "SampleDistance": "0.177",
                "DetectorModel": "PILATUS3 100K", "Monitor": monitor,
                "pilct1": "155717", "Comment": "Cell_TEST", **header})
    img.write(path)


def write_dated_pairs(dirpath, dat_dates, edf_dates, echem_date="05/02/2024"):
    """Write scan_00i .dat/.edf pairs with the given Date texts and echem from 10:00 to 10:39."""
    os.makedirs(dirpath, exist_ok=True)
    for i, (dat_date, edf_date) in enumerate(zip(dat_dates, edf_dates), start=1):
        write_xrd_dat(os.path.join(dirpath, f"scan_{i:03d}.dat"), dat_date, y=inhouse_scan_y(i))
        write_edf_image(os.path.join(dirpath, f"scan_{i:03d}.edf"), edf_date, offset=float(i))
    write_echem_txt(os.path.join(dirpath, "echem.txt"), date=echem_date)


def write_twod_dir(dirpath):
    """Write one XRD .dat and one EDF image at the same timestamp (2D fixtures)."""
    os.makedirs(dirpath, exist_ok=True)
    write_xrd_dat(os.path.join(dirpath, "scan_001.dat"), SCAN_TIMES[0])
    write_edf_image(os.path.join(dirpath, "image_001.edf"), SCAN_TIMES[0])


def write_twod_zip(zip_path, workdir):
    """Write a ZIP of write_twod_dir's files, staged under workdir, and return its path."""
    src = os.path.join(str(workdir), "twod_src")
    write_twod_dir(src)
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in os.listdir(src):
            z.write(os.path.join(src, name), name)
    return str(zip_path)


def write_detailed_inhouse_dir(dirpath):
    """Write two 30-point azimAvg scans, EC-Lab echem, and an EDF image when fabio is present."""
    os.makedirs(dirpath)
    for i, ts in enumerate(DETAILED_SCAN_TIMES, start=1):
        write_lab_dat(os.path.join(dirpath, f"scan_{i:03d}.dat"), ts,
                      y=100.0 * i + np.arange(30), x=DETAILED_ONED_X)
    write_eclab_echem(os.path.join(dirpath, "echem.txt"))
    if core.FABIO_AVAILABLE:
        write_edf_image(os.path.join(dirpath, "image_001.edf"),
                        DETAILED_SCAN_TIMES[0])


def write_detailed_neutron_dir(dirpath):
    """Write a POLARIS-shaped run: logbook, 2 banks x (TOF, d) in dat_files/, EC-Lab echem."""
    data = os.path.join(dirpath, "dat_files")
    os.makedirs(data)
    write_logbook(os.path.join(dirpath, "SC1_log_book.txt"), [DETAILED_LOGBOOK_LINE + "\t"])
    n = len(DETAILED_TOF_X)
    for bank in (1, 2):
        tof_name, d_name = polaris_bank_names("123456", bank)
        write_mantid_bank(os.path.join(data, tof_name), "POLARIS", "tof", bank, DETAILED_TOF_X,
                          50.0 + bank + np.arange(n), 0.3 + 0.001 * np.arange(n))
        write_mantid_bank(os.path.join(data, d_name), "POLARIS", "d", bank, DETAILED_D_X,
                          70.0 + bank + np.arange(n), np.full(n, 0.2))
    write_eclab_echem(os.path.join(dirpath, "LDE_run1_SC1_C01_Absolute.txt"))


# ============================================================================
# Real-file shapes
# ============================================================================
# Tiny synthetic files laid out like the instrument exports in the reference
# sets, so tests take the code paths real data takes: the in-house azimAvg .dat
# (a '#' key block, then an uncommented column line), EC-Lab absolute-time echem
# (header ending in a tab, CRLF, E+000 numbers), Mantid focussed banks (leading
# XYDATA line; POLARIS CRLF, WISH LF with "Spectrum N "), POLARIS and WISH bank
# names, and ISIS logbook lines (trailing tab, unpadded day). Values reuse the
# grids above, so the 3.76 V / 3.85 V correlation anchors still hold.

LAB_DAT_RULE = "#" * 80
LAB_DAT_COLUMNS = ("    tth(°)                Intensity(a.u.)            "
                   "Sigma_I(a.u.)            ")
MANTID_AXES = {"tof": ("Time-of-flight", "per microsecond"),
               "d": ("d-Spacing", "per Angstrom")}


def write_lab_dat(path, timestamp, exposure="120", y=None, x=None):
    """Write an azimAvg .dat: '#' key block, the uncommented 'tth(°) ...' line, X Y Sigma rows."""
    x = ONED_X if x is None else x
    y = inhouse_scan_y(1) if y is None else y
    keys = [("EDF_DataBlockID", "0.Image.Psd"), ("Comment", "Cell_TEST -"),
            ("Date", timestamp), ("ExposureTime", exposure), ("Humidity-xhg", "NA"),
            ("Wavelength", "1.541891e-10"), ("DetectorModel", "PILATUS3 100K-20Hz"),
            ("Monitor", "0"), ("Operation", "Azimuthal average I(q)"),
            ("SampleDistance", "0.176"), ("Center_1", "122.907"), ("Center_2", "354.688")]
    lines = [LAB_DAT_RULE] + [f"# {k:<33}{v}" for k, v in keys]
    lines += [LAB_DAT_RULE, LAB_DAT_COLUMNS]
    lines += [f"{a:15.6f}{b:26.4f}{np.sqrt(abs(b)):26.6f}           " for a, b in zip(x, y)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


def _eclab_number(value):
    """Format a number as EC-Lab does: 7-decimal mantissa and a three-digit exponent."""
    mantissa, exponent = f"{value:.7E}".split("E")
    return f"{mantissa}E{int(exponent):+04d}"


def write_eclab_echem(path, n_rows=40, v0=3.7, dv=0.01, i0=0.1, di=0.001,
                      date="05/02/2024", hour=10):
    """Write write_echem_txt's values as an EC-Lab export (trailing-tab header, E+000, CRLF)."""
    rows = ["time/s\tEwe/V\tI/mA\t"]
    rows += [f"{date} {hour}:{m:02d}:00.0000\t{_eclab_number(v0 + m * dv)}\t"
             f"{_eclab_number(i0 + m * di)}" for m in range(n_rows)]
    with open(path, "w", newline="") as f:
        f.write("\r\n".join(rows) + "\r\n")


def write_mantid_bank(path, instrument, axis, spectrum, x, y, e):
    """Write a Mantid focussed bank (XYDATA, '#' header, X Y E) as POLARIS (CRLF) or WISH (LF)."""
    label, y_unit = MANTID_AXES[axis]
    newline = "\r\n" if instrument == "POLARIS" else "\n"
    pad = " " if instrument == "WISH" else ""
    lines = ["XYDATA", f"# File generated by Mantid, Instrument {instrument}",
             f"# The X-axis unit is: {label}, The Y-axis unit is:  {y_unit}",
             "# Data for spectra :0", f"# Spectrum {spectrum}{pad}",
             f"# {label}              Y                 E"]
    lines += [f"{a:15.8f}{b:18.8f}{c:18.8f}" for a, b, c in zip(x, y, e)]
    with open(path, "w", newline="") as f:
        f.write(newline.join(lines) + newline)


def polaris_bank_names(run, bank):
    """Return POLARIS's (TOF, d) names for a run and bank: POL<run>-b_<bank>-TOF.dat, -d.dat."""
    return f"POL{run}-b_{bank}-TOF.dat", f"POL{run}-b_{bank}-d.dat"


def wish_bank_names(run, bank):
    """Return WISH's (TOF, d) names: <run>-<bank>_<11-bank>raw-0.dat and its -d-0 twin."""
    return f"{run}-{bank}_{11 - bank}raw-0.dat", f"{run}-{bank}_{11 - bank}raw-d-0.dat"


def journal_time(ts):
    """Format a time as the ISIS logbook writes it, day unpadded: 'Mon Feb 5 10:00:00 2024'."""
    t = pd.Timestamp(ts)
    return f"{t:%a %b} {t.day} {t:%H:%M:%S %Y}"


def polaris_logbook_line(run, start, end, title="Op-Run1-SC1-Diag", users="Smith,Jones",
                         proposal="7654321", duration='"2m 6s"', charge='"5.44"'):
    """Return a POLARIS logbook line: run, title, duration, charge, users, proposal, start, end."""
    return "\t".join([str(run), title, duration, charge, users, proposal,
                      journal_time(start), journal_time(end), ""])


def wish_logbook_line(run, start, end, title="cell55 event # 1", user="Perez",
                      proposal="2510713", duration='"30m 0s"', charge='"3.0059"'):
    """Return a WISH logbook line: the POLARIS fields, then the '24_5' cycle and a trailing tab."""
    return "\t".join([str(run), title, duration, charge, user, proposal,
                      journal_time(start), journal_time(end), "24_5", ""])


def write_logbook(path, lines, instrument="POLARIS"):
    """Write logbook lines with the instrument's line ending (POLARIS CRLF, WISH LF)."""
    newline = "\r\n" if instrument == "POLARIS" else "\n"
    with open(path, "w", newline="") as f:
        f.write(newline.join(lines) + newline)


def zip_tree(src, zip_path, prefix=""):
    """Zip every file under src by its forward-slash relative path under prefix; return the path."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(src):
            for name in sorted(files):
                full = os.path.join(root, name)
                z.write(full, prefix + os.path.relpath(full, src).replace(os.sep, "/"))
    return str(zip_path)


def write_lab_dir(dirpath, times=SCAN_TIMES):
    """Write write_inhouse_dir's scans and echem (same names and values) in the real shapes."""
    os.makedirs(dirpath, exist_ok=True)
    for i, ts in enumerate(times, start=1):
        write_lab_dat(os.path.join(dirpath, f"scan_{i:03d}.dat"), ts, y=inhouse_scan_y(i))
    write_eclab_echem(os.path.join(dirpath, "echem.txt"))


def write_wish_dir(dirpath, run="123456", banks=(1, 2), lines=None):
    """Write write_neutron_dir's run (same window and values) in the real WISH shapes."""
    data = os.path.join(dirpath, "diffraction-data")
    os.makedirs(data, exist_ok=True)
    if lines is None:
        lines = [wish_logbook_line(run, "2024-02-05 10:00:00", "2024-02-05 10:30:00")]
    write_logbook(os.path.join(dirpath, "Cell_log_book.txt"), lines, "WISH")
    for bank in banks:
        tof_name, d_name = wish_bank_names(run, bank)
        write_mantid_bank(os.path.join(data, tof_name), "WISH", "tof", bank, TOF_X,
                          50.0 + bank + np.arange(40.0), np.ones(40))
        write_mantid_bank(os.path.join(data, d_name), "WISH", "d", bank, D_X,
                          70.0 + bank + np.arange(40.0), np.ones(40))
    write_eclab_echem(os.path.join(dirpath, "Cell_C01_Absolute.txt"))


# Minimal Mantid IDF: source 14 m upstream of a sample at the origin
MANTID_IDF = """<?xml version="1.0" encoding="UTF-8" ?>
<instrument xmlns="http://www.mantidproject.org/IDF/1.0" name="Test_upgrade"
            valid-from="2020-01-01 00:00:00" valid-to="2099-12-31 23:59:59">
  <component type="H2O_moderator">
    <location z="-14.0" />
  </component>
  <component type="sample_holder">
    <location x="0.0" y="0.0" z="0.0"/>
  </component>
  <type name="H2O_moderator" is="Source"></type>
  <type name="sample_holder" is="SamplePos"></type>
  <type name="bank1" />
</instrument>
"""


def _mantid_string(group, name, text):
    """Write ``text`` as Mantid stores strings: a shape-(1,) fixed-length bytes dataset."""
    raw = text.encode()
    group.create_dataset(name, data=np.array([raw], dtype=f"S{len(raw)}"))


def write_mantid_processed_file(path):
    """Write a Mantid SaveNexusProcessed lookalike for instrument TESTINST with MANTID_IDF.
    Bank 1: 8-element ring at 2theta 10 deg; bank 2: 4-element panel at 2theta 90, phi 90 deg."""
    ring = np.array([[2.25, 10.0, phi] for phi in range(0, 360, 45)], dtype=float)
    panel = np.array([[1.0, 88.0, 80.0], [1.0, 92.0, 80.0],
                      [1.0, 88.0, 100.0], [1.0, 92.0, 100.0]], dtype=float)
    with h5py.File(path, "w") as f:
        for number, positions in ((1, ring), (2, panel)):
            ws = f.create_group(f"mantid_workspace_{number}")
            ws.attrs["NX_class"] = "NXentry"
            inst = ws.create_group("instrument")
            _mantid_string(inst, "name", "TESTINST")
            inst.create_dataset("detector/detector_positions", data=positions)
            _mantid_string(inst.create_group("instrument_xml"), "data", MANTID_IDF)
            mon = inst.create_group("physical_monitors")
            mon.create_dataset("detector_number", data=np.array([11, 12, 21], dtype=np.int32))
            mon.create_dataset("distance", data=np.array([7.6, 7.6, 3.3]))
            mon.create_dataset("polar_angle", data=np.array([180.0, 180.0, 0.0]))
            mon.create_dataset("azimuthal_angle", data=np.array([0.0, 0.0, 0.0]))
            _mantid_string(ws.create_group("process/MantidEnvironment"), "data",
                           "Framework Version: 6.16.1\nOS name: Windows NT\n")


def write_synchrotron_scan(dirpath, scan_id, start, end, count_time=None):
    """Write a GDA-style i11 scan .nxs and its integrated .xy, grouped by trailing scan ID.
    entry1 has start/end times and a device-named NXdetector, pixium_hdf (count_time if given)."""
    with h5py.File(os.path.join(dirpath, f"i11-1-{scan_id}.nxs"), "w") as f:
        f["/entry1/start_time"] = start.encode()
        f["/entry1/end_time"] = end.encode()
        det = f.create_group("/entry1/instrument/pixium_hdf")
        det.attrs["NX_class"] = "NXdetector"
        if count_time is not None:
            det["count_time"] = np.array([count_time])
    xy = os.path.join(dirpath, f"i11-1-{scan_id}_integration_tth_0000.xy")
    with open(xy, "w") as f:
        f.write("\n".join(f"{x:.4f} {v:.4f}"
                          for x, v in zip(ONED_X, inhouse_scan_y(1))) + "\n")
