"""Generate a small synthetic operando dataset to try OperaXN with.

Writes example_data/laboratory (3 laboratory XRD scans + electrochemistry),
example_data/neutron (one logbook entry, 2 detector banks in TOF and
d-spacing + electrochemistry) and example_data/analysis (an ICI cycling log
for the analysis windows) next to this script; README.md has the
walkthrough. Deliberately standalone (readable without the test suite), the
laboratory and neutron writers duplicate tests/_builders.py: keep the values
in sync. The analysis log has no counterpart there.
"""
import os
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "example_data")

SCAN_TIMES = ["2024-02-05T10:05:00", "2024-02-05T10:10:00", "2024-02-05T10:15:00"]


def write_echem(path):
    """Write the echem log: a 3.70-4.09 V ramp at 60 s cadence from 10:00, day-first dates."""
    rows = ["Time\tEwe/V\tI/mA"]
    for m in range(40):
        rows.append(f"05/02/2024 10:{m:02d}:00\t{3.7 + m * 0.01:.4f}\t{0.1 + m * 0.001:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def make_laboratory(dirpath):
    """Write 3 XRD scans (10:05/10:10/10:15, 120 s exposure) and the echem log."""
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
    """Write logbook run 123456 (10:00-10:30), 2 banks x (TOF, d) and the echem log."""
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


ICI_PULSES = 10      # pulses in each charge and each discharge
ICI_CURRENT = 0.5    # mA
ICI_PULSE_S = 360    # s of current per pulse, logged every 30 s
ICI_REST_STEPS = 50  # rest samples per pulse, 0.1 s apart (a 5 s rest)


def ici_parameters(soc, cycle):
    """Return the (R in ohm, k in ohm s^-1/2) the ICI log encodes at a state of charge."""
    return (40.0 + 80.0 * (soc - 0.5) ** 2) * (1.0 + 0.1 * cycle), 8.0 + 6.0 * (1.0 - soc)


def write_ici_echem(path):
    """Write an ICI log: 2 cycles of 10 charge then 10 discharge pulses, each with a 5 s rest.
    Rests relax as dV = -I (R + k sqrt(dt)) from the pulse's last voltage, dt from rest start."""
    start = datetime(2024, 2, 5, 12, 0, 0)
    rows = ["Time\tEwe/V\tI/mA"]
    tenths = 0  # elapsed time in 0.1 s steps, kept integral so the stamps are exact

    def add(volts, milliamps):
        stamp = start + timedelta(seconds=tenths / 10)
        rows.append(f"{stamp:%d/%m/%Y %H:%M:%S}.{tenths % 10}\t{volts:.5f}\t{milliamps:.4f}")

    for cycle in range(2):
        for sign in (1, -1):  # charge, then discharge
            for pulse in range(ICI_PULSES):
                done = (pulse + 1) / ICI_PULSES
                soc_end = done if sign > 0 else 1.0 - done
                r, k = ici_parameters(soc_end, cycle)
                amps = sign * ICI_CURRENT / 1000.0
                for step in range(0, ICI_PULSE_S + 1, 30):
                    soc = soc_end - sign * (ICI_PULSE_S - step) / ICI_PULSE_S / ICI_PULSES
                    v0 = 3.6 + 0.5 * soc + amps * (r + 10.0 * k)  # steady polarisation
                    add(v0, sign * ICI_CURRENT)
                    tenths += 300 if step < ICI_PULSE_S else 1
                for n in range(ICI_REST_STEPS):
                    add(v0 - amps * (r + k * np.sqrt(n / 10)), 0.0)
                    tenths += 1
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def make_analysis(dirpath):
    """Write the ICI log used to try the Capacity and ICI windows."""
    os.makedirs(dirpath, exist_ok=True)
    write_ici_echem(os.path.join(dirpath, "ici_echem.txt"))


def main():
    """Generate the datasets and print what to expect when loading them."""
    make_laboratory(os.path.join(OUT, "laboratory"))
    make_neutron(os.path.join(OUT, "neutron"))
    make_analysis(os.path.join(OUT, "analysis"))
    print(f"Example data written to {OUT}")
    print("  laboratory/  3 XRD scans (10:05, 10:10, 10:15) + echem")
    print("               expected correlation: scan 1 -> 3.76 V")
    print("  neutron/     run 123456 (10:00-10:30), banks 1-2, TOF + d + echem")
    print("               expected correlation: midpoint 10:15 -> 3.85 V")
    print("  analysis/    ici_echem.txt: 2 cycles x 10 charge + 10 discharge ICI pulses")
    print("               add it from the Echem menu of the Capacity or ICI window")
    print("Launch `operaxn` and load either folder from the upload dialog.")


if __name__ == "__main__":
    main()
