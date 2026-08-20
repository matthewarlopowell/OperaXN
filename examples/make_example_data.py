"""Generate a small synthetic operando dataset to try OperaXN with.

Creates example_data/laboratory (3 laboratory XRD scans + electrochemistry)
and example_data/neutron (one logbook entry, 2 detector banks in TOF and
d-spacing + electrochemistry) next to this script. See README.md for the
walkthrough.

Deliberately standalone: duplicates the writers in tests/_builders.py so
the example reads without the test suite -- keep the values in sync.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "example_data")

SCAN_TIMES = ["2024-02-05T10:05:00", "2024-02-05T10:10:00", "2024-02-05T10:15:00"]


def write_echem(path):
    """Voltage ramp 3.70-4.09 V at 60 s cadence from 10:00 (dayfirst dates)."""
    rows = ["Time\tEwe/V\tI/mA"]
    for m in range(40):
        rows.append(f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\t{0.1 + m * 0.001:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def make_laboratory(dirpath):
    """3 XRD scans (10:05/10:10/10:15, 120 s exposure) + the echem log."""
    os.makedirs(dirpath, exist_ok=True)
    x = np.linspace(10, 80, 50)
    for i, ts in enumerate(SCAN_TIMES, start=1):
        y = 100 + 10 * i + np.arange(50, dtype=float)
        lines = [f"# Date {ts}", "# ExposureTime 120.0", "# tth Intensity"]
        lines += [f"{xi:.4f} {yi:.4f}" for xi, yi in zip(x, y)]
        with open(os.path.join(dirpath, f"scan_{i:03d}.dat"), "w") as f:
            f.write("\n".join(lines) + "\n")
    write_echem(os.path.join(dirpath, "echem.txt"))


def make_neutron(dirpath):
    """Logbook scan 123456 (10:00-10:30), 2 banks x (TOF, d) + the echem log."""
    os.makedirs(dirpath, exist_ok=True)
    logbook = ("123456\tuser\tsample title\tMon Feb 05 10:00:00 2024\t"
               "Mon Feb 05 10:30:00 2024\t1800\tok\textra")
    with open(os.path.join(dirpath, "logbook.txt"), "w") as f:
        f.write(logbook + "\n")
    tof_x = np.linspace(1000, 20000, 40)
    d_x = np.linspace(0.5, 3.0, 40)
    for bank in (1, 2):
        tof_y = 50.0 + bank + np.arange(40, dtype=float)
        d_y = 70.0 + bank + np.arange(40, dtype=float)
        with open(os.path.join(dirpath, f"123456-{bank}-0.dat"), "w") as f:
            f.write("# Time-of-flight Y E\n")
            f.write("\n".join(f"{x:.10f} {y:.10f} 1.0"
                              for x, y in zip(tof_x, tof_y)) + "\n")
        with open(os.path.join(dirpath, f"123456-{bank}-d-0.dat"), "w") as f:
            f.write("# d-Spacing Y E\n")
            f.write("\n".join(f"{x:.10f} {y:.10f} 1.0"
                              for x, y in zip(d_x, d_y)) + "\n")
    write_echem(os.path.join(dirpath, "echem.txt"))


def main():
    """Generate both datasets and print what to expect when loading them."""
    make_laboratory(os.path.join(OUT, "laboratory"))
    make_neutron(os.path.join(OUT, "neutron"))
    print(f"Example data written to {OUT}")
    print("  laboratory/  3 XRD scans (10:05, 10:10, 10:15) + echem")
    print("               expected correlation: scan 1 -> 3.76 V")
    print("  neutron/     scan 123456 (10:00-10:30), banks 1-2, TOF + d + echem")
    print("               expected correlation: midpoint 10:15 -> 3.85 V")
    print("Launch `operaxn` and load either folder from the upload dialog.")


if __name__ == "__main__":
    main()
