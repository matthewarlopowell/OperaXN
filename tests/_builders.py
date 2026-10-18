"""Synthetic dataset builders shared by the test suite; no data files are committed.

Defaults make correlation exact: echem runs at 60 s cadence from 10:00
with V = 3.7 + 0.01/step, so an XRD scan at 10:05 with 120 s exposure
correlates at its 10:06 midpoint to V = 3.76, and a neutron scan logged
10:00-10:30 at its 10:15 midpoint to V = 3.85. The plain family
(50/40-point grids, 3 scans, simple logbook) serves most tests; the
DETAILED_* family (30-point grids, 2 scans, richer logbook) covers the
Sigma column, POLARIS Mantid headers, EDF harvest and trailing-timestamp
logbook layout, so keep the two apart. examples/make_example_data.py
duplicates these writers standalone: keep the values in sync.
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
    """Write an XRD .dat: '# Date <timestamp>', '# ExposureTime', a column header, x/y rows.
    The timestamp must be the last token of the '# Date' line."""
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


def write_edf_image(path, timestamp, offset=1.0, monitor="0"):
    """Write an 8x8 EDF image with the harvested header fields; ``monitor`` "0" means none."""
    import fabio
    img = fabio.edfimage.EdfImage(
        data=np.arange(64, dtype=np.float32).reshape(8, 8) + offset,
        header={"Date": timestamp, "ExposureTime": "120.0",
                "WaveLength": "1.541891e-10", "SampleDistance": "0.177",
                "DetectorModel": "PILATUS3 100K", "Monitor": monitor,
                "pilct1": "155717", "Comment": "Cell_TEST"})
    img.write(path)


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
    """Write two 30-point XRD scans with Sigma, echem, and an EDF image when fabio is present."""
    os.makedirs(dirpath)
    for i, ts in enumerate(DETAILED_SCAN_TIMES, start=1):
        y = 100.0 * i + np.arange(30)
        e = np.sqrt(y)
        lines = [f"# Date {ts}", "# ExposureTime 120.0", "# tth Intensity Sigma"]
        lines += [f"{x:.10f} {v:.10f} {sig:.10f}"
                  for x, v, sig in zip(DETAILED_ONED_X, y, e)]
        with open(os.path.join(dirpath, f"scan_{i:03d}.dat"), "w") as f:
            f.write("\n".join(lines) + "\n")
    write_echem_txt(os.path.join(dirpath, "echem.txt"))
    if core.FABIO_AVAILABLE:
        write_edf_image(os.path.join(dirpath, "image_001.edf"),
                        DETAILED_SCAN_TIMES[0])


def write_detailed_neutron_dir(dirpath):
    """Write a trailing-timestamp logbook, 2 Mantid POLARIS banks x (TOF, d) and 40-row echem."""
    os.makedirs(dirpath)
    with open(os.path.join(dirpath, "logbook.txt"), "w") as f:
        f.write(DETAILED_LOGBOOK_LINE + "\n")
    for bank in (1, 2):
        with open(os.path.join(dirpath, f"123456-{bank}-0.dat"), "w") as f:
            f.write("XYDATA\n# File generated by Mantid, Instrument POLARIS\n"
                    "# The X-axis unit is: Time-of-flight, The Y-axis unit is:  per microsecond\n"
                    f"# Spectrum {bank}\n# Time-of-flight  Y  E\n")
            f.write("\n".join(f"{x:.6f} {50.0 + bank + i:.6f} {0.3 + 0.001 * i:.6f}"
                              for i, x in enumerate(DETAILED_TOF_X)) + "\n")
        with open(os.path.join(dirpath, f"123456-{bank}-d-0.dat"), "w") as f:
            f.write("XYDATA\n# File generated by Mantid, Instrument POLARIS\n"
                    "# The X-axis unit is: d-Spacing, The Y-axis unit is:  per Angstrom\n"
                    f"# Spectrum {bank}\n# d-Spacing  Y  E\n")
            f.write("\n".join(f"{x:.6f} {70.0 + bank + i:.6f} {0.2:.6f}"
                              for i, x in enumerate(DETAILED_D_X)) + "\n")
    write_echem_txt(os.path.join(dirpath, "echem.txt"))


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
