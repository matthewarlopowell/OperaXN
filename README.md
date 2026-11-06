# OperaXN

**OPERAndo X-ray and Neutron diffraction data visualisation tool**

[![CI](https://github.com/matthewarlopowell/OperaXN/actions/workflows/ci.yml/badge.svg)](https://github.com/matthewarlopowell/OperaXN/actions/workflows/ci.yml)

**OperaXN** is a Python desktop application for correlating, visualising and
sharing *operando* diffraction data collected at laboratory XRD, synchrotron
XRD and neutron diffraction sources. Raw diffraction and electrochemistry files
are time-correlated and consolidated into a single standardised NeXus (`.nxs`)
file which OperaXN then visualises. The same file can be shared, re-opened
directly and read by any HDF5/NeXus tool.

## Statement of need

*Operando* diffraction experiments, where a battery or other electrochemical
cell is cycled while diffraction patterns are collected, produce two streams
of data in unrelated formats: diffraction files whose layout depends on the
instrument (laboratory diffractometer, synchrotron beamline or neutron
spallation source), and electrochemistry logs from a separate potentiostat.
Relating a structural change to the electrochemical state in which it occurred
requires aligning these streams in time, which is usually done ad hoc with
per-experiment scripts and is rarely preserved with the data. OperaXN gives
experimentalists a single tool that classifies the raw files, performs the
time correlation, stores the combined experiment in a standardised,
self-describing NeXus file, and visualises it, so the correlated dataset,
not just the raw files, is what gets analysed, shared and archived.

## Features

- Automated time-correlation of electrochemical (voltage/current) and
  diffraction datasets by absolute timestamp or elapsed time
- Generation of NeXus files conforming to the
  `NXoperando_monopd` / `NXoperando_tofnpd` application definitions
  (see [definitions/](definitions)), including per-scan electrochemical
  state alongside the full cycling record and, for profiled instruments
  (POLARIS), the nominal instrument geometry (flight path and per-bank
  distance/scattering angle)
- Simultaneous visualisation of X-ray (1D and 2D) and neutron (per-bank
  TOF/d-spacing) diffraction data with electrochemical cycling
- Operando heatmap (stacked patterns vs scan/time with the voltage track;
  one map per bank on TOF or d-spacing for neutron data, a filter to the
  scans with correlated echem or minus listed scans, x and intensity
  ranges, a choice of colour maps and a publication-figure export to PNG,
  PDF or SVG), capacity analysis and ICI (intermittent current
  interruption) analysis
- Export the diffraction plots as publication-quality figures (PNG, PDF, SVG)
  or animated GIFs, and the correlated data and analysis results as Excel
  summaries and CSV
- Supports `.nxs`, `.dat`, `.xy`, `.edf`, `.hdf`, `.txt`, `.xlsx`, `.csv`
  and `.zip` inputs

## Installation

### macOS prerequisite

The system Python bundled with macOS does not include a compatible version of
Tcl/Tk and the GUI will not render. Before installing, download and install
Python from **[python.org](https://www.python.org/downloads/)** (3.10 or
later). The official installer bundles Tcl/Tk 8.6+.

### Linux prerequisite

The GUI is built with Tkinter, which many Linux distributions package
separately from Python. If `operaxn` stops with
`ModuleNotFoundError: No module named 'tkinter'` (or `'_tkinter'`), install it
for the Python you use and run `operaxn` again (no reinstall of OperaXN is
needed):

```bash
sudo apt install python3-tk        # Debian, Ubuntu
sudo dnf install python3-tkinter   # Fedora
```

A conda Python already includes Tkinter.

### From source

```bash
git clone https://github.com/matthewarlopowell/OperaXN.git
cd OperaXN
python3 -m venv .venv       # Windows: py -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
```

### For development

```bash
pip install -e .
```

## Usage

```bash
operaxn
```

Upload raw data (or an existing `.nxs`) via the GUI; every load and
generation option is collected in a single upload dialog. Once the scans are
listed, press **Plot** to draw them. Export the generated NeXus file at any
point to share the experiment.

2D images are read from their source files when a scan is first shown, so
keep the data folder reachable while the experiment is open; a scan whose
image cannot be found shows "Error reading 2D data" until the folder is
back. A `.zip` upload reads from its own extracted copy, and images embedded
with "Include 2D images in the NeXus file" need no source file. 2D images
stored by reference stay outside the `.nxs`: to share an experiment with its
2D, tick "Include 2D images in the NeXus file", or send the image files with
it (for a ZIP upload, the `<name>_images` folder written beside the export).

No data to hand? [examples/](examples) generates a small synthetic operando
dataset and walks through loading it.

### Command line options

```bash
operaxn --help          # Show all options
operaxn --version       # Print the version
operaxn --debug         # Enable debug logging
operaxn --profile FILE  # Print an instrument-profile block derived from a Mantid-processed .nxs
```

## Input data

Select a folder (searched with its subfolders), files, or a `.zip` holding
them, and the data source the files come from. Hidden files and macOS
`__MACOSX` entries are skipped, and a file reached twice (a folder plus a file
inside it) is read once.

- **Laboratory X-ray diffraction**: 1D patterns as `.dat` (2θ, intensity and
  optionally its uncertainty) with a `# Date` line in the first 30 lines and
  optionally `# ExposureTime`, and 2D images as `.edf` with a `Date` header
  and optionally an `ExposureTime` (or `Exposure_Time`, `ExpTime`, `Exposure`)
  field. A `.dat` and an `.edf` with the same date are one scan; a file
  without a readable date is skipped and named in the terminal.
- **Synchrotron X-ray diffraction** (Diamond i11-1): one scan `.nxs` per
  acquisition, timed by its `start_time` (the exposure is the detector's
  `exposure_time`, `count_time` or `preset`, else `end_time` minus
  `start_time`; a window of an hour or more is ignored and the scan correlates
  at its start time), with the integrated pattern `.xy` and the detector image
  `.hdf` that carry the same scan number: the last run of digits in the name,
  for an `.xy` named `..._integration_...` the digits before `_integration_`.
  An `.xy` or `.hdf` without its `.nxs` is not read.
- **Time-of-flight neutron diffraction** (ISIS): one `.dat` per detector bank
  (banks 1 to 5) and axis, as Mantid exports it, named `<run>-<bank>...` or
  `POL<run>-b_<bank>...` with a run number of 5 to 7 digits; `-d-`, `_d_` or a
  final `_d` or `-d` after the bank marks the d-spacing file, the other is
  time of flight. Runs are timed from the instrument logbook, a tab-separated
  `.txt` whose run lines start with the run number and hold the start and end
  times as the logbook writes them (`Mon Feb 05 10:00:00 2024`), so a
  selection without a logbook is refused. A run with bank files but no
  logbook line is named in the terminal.
- **Electrochemistry** (any source): a `.txt` (tab-separated), `.csv`
  (comma, semicolon or tab; a semicolon or tab table uses decimal commas) or
  `.xlsx` (first sheet) whose first line names its columns: a time column of
  dates with clock times (not elapsed seconds), a voltage column (`Ewe`,
  `Ecell` or `Voltage` in its name) and a current column (`Current` or `I/`).
  Columns are found by name; a voltage or current column found only by its
  position must carry its unit (a voltage label in V or mV; a current label
  such as `I (mA)` or `<I>/mA` in A, mA, µA, nA or pA), or the file is skipped
  with a warning naming the column. Voltage is stored in V and current in mA.
  Several files are combined into one record, and the terminal lists them.
  Standard electrochemistry files picked in the upload dialog may also have no
  header line: they are read as time, voltage and current in their first three
  columns.

## Time correlation

OperaXN correlates diffraction scans with electrochemistry data by timestamp.
Two modes are available (selected at load time):

- **Absolute**: scan timestamps are matched directly to echem timestamps via
  nearest-neighbour lookup at the exposure midpoint.
- **Relative**: both streams are correlated by elapsed time: the first echem
  point is taken as simultaneous with the earliest scan exposure midpoint (the
  correlation timestamp, normally the first scan's), and the stored
  `voltage_timestamp` values stay in the echem clock. Useful when diffraction
  and echem clocks are not synchronised.

A saved `.nxs` file opens with the mode it was generated with; the upload
dialog shows that mode and locks the choice until raw data is selected again.
Times are compared as the local wall-clock times the instruments logged: a
UTC offset (`+01:00`, `Z`, `UTC`, `GMT`) after a scan or echem time is
ignored. An echem log whose offset changes part-way (a daylight-saving
change) is matched by wall clock as well, so the hour around the change can
pair with the wrong pass; the terminal warns when a file does this.

In both modes a scan is matched to the nearest echem point when the two are
less than a tolerance apart, 300 s by default and set at generation in the
upload dialog ("Echem time tolerance"; left blank, the default applies).
The tolerance applies on both sides of the record: a scan whose
correlation timestamp lies less than it before the first echem point or
after the last is matched to that end point, and a scan at least the
tolerance from every point is left uncorrelated. The value used is recorded
in the file as `operando_electrochemistry/echem_time_tolerance`; files
written before that field was added used 300 s.

Echem timestamps, and slash dates in a laboratory `.dat` file's `# Date`
line or an EDF `Date` header, are read day-first (the UK convention).
Year-first dates are never swapped: `2024-02-05` and `2024/02/05` everywhere,
and in those headers also `20240205` and `2024.02.05`. Arbin exports use US
month/day order: a date whose day is 12 or below is then read with month and
day swapped, so an Arbin file correlates correctly only when every day number
in it is above 12.

Correlation runs once at generation; the results are stored in the `.nxs`
file alongside the full cycling record.

## Analysis windows

The **Heatmap** and **Capacity** buttons open analysis windows on the
loaded experiment. `<name>` below is the experiment name: the opened `.nxs`
file's name, or the name of the first uploaded folder or file.

**Heatmap** stacks the 1D patterns as an intensity map, against elapsed time
or scan number, beside the correlated voltage. X-ray data is shown on 2θ or
d-spacing, and neutron data one detector bank at a time on time of flight or
d-spacing. The map can be limited to the scans with a correlated voltage, or
can leave out listed scans. For neutron data the **Intensity** limits belong to
the bank shown (blank is automatic) and are kept when you switch bank;
**Apply All** gives every bank the shown bank's range after a confirmation,
and **Reset** clears the ranges of every bank. **Title** sets the map's title
on screen and in the exports (by default the bank for neutron data, none for
X-ray; clear it for no title); like the limits, each bank keeps its own, and
the **Apply All** beside it gives every bank the shown title. Text between
`$` signs is set as matplotlib mathtext, for example `LiNiO$_2$`.
**Export figure...** saves a publication figure (PNG, PDF or SVG), by default
`<name>_heatmap.png`; for neutron data the bank number is added
(`<name>_heatmap_bank1.png`), and **Export all banks...** saves one figure per
bank, each with its own limits and title.

**Capacity** needs electrochemistry with a current column. It offers the
operando record and every standard electrochemistry file that has current;
the **Echem** menu switches between them and **Add echem...** adds a file.
It applies these rules:

- Current is in mA. A current column labelled in another unit, such as
  `Current(A)`, `I/A`, `Current [A]`, `I/uA` or `I/nA`, is converted when the
  file is read; a column with no recognised unit is read as mA, with a
  terminal warning.
- Positive current is charge and negative current is discharge. If your
  potentiostat uses the opposite sign, charge and discharge are swapped.
- A rest (zero current, or a blank current cell) belongs to the phase before
  it, so the rest after a charge pulse is part of that charge.
- The first charge starts cycle 1, and each charge that follows a discharge
  starts the next cycle.
- Capacity is the integral of |current| over time (trapezoidal rule), in mAh.
  It starts again from 0 for the charge and for the discharge of every
  cycle. With a mass in mg entered, specific capacity is capacity divided by
  that mass, in mAh/g.

**Capacity** plots voltage against time (hours from the first point of the
record) and against capacity, one colour per cycle (above ten cycles the
legend names only the first and last), for all cycles or one at a time.
**Export CSV** saves the cycles shown (by default as
`<name>_capacity.csv`), one row per point, with the columns `Cycle`, `Phase`,
`Time (s)` (seconds from the first point of the record), `Voltage (V)`,
`Capacity (mAh)` and `Specific Capacity (mAh/g)` (empty without a mass).

## NeXus files

Generated files conform to the custom application definitions in
[definitions/](definitions): a single `NXentry` holding the instrument and
sample description plus the full electrochemical cycling record, and one
`NXsubentry` per diffraction acquisition carrying the electrochemical state
of the cell during that scan. Files also capture the cycling protocol
(technique, voltage window and C-rate from the upload dialog, plus the
electrochemistry files used) and the cell preparation date; deviations from
the paper's Figure 4 are tabulated in
[definitions/README.md](definitions/README.md). Files can be inspected with any HDF5
tool (e.g. [DAWN](https://dawnsci.org/), NeXpy, h5web) and validated with
FAIRmat's [pynxtools](https://github.com/FAIRmat-NFDI/pynxtools).

## Dependencies

- Python >= 3.10
- Tkinter (part of most Python installs; see Installation for Linux and macOS)
- NumPy >= 1.21.3
- pandas >= 2.0.0
- Matplotlib >= 3.5.0
- h5py >= 3.6.0
- FabIO >= 0.14.0
- openpyxl >= 3.1.5

## Testing

The automated test suite runs on synthetic data generated at test time; no
data files are required, apart from the opt-in `realdata` tests:

```bash
pip install -e ".[test]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on the test markers.
CI runs the suite on Python 3.10–3.13 on Linux and Python 3.12 on Windows,
on every push to main and every pull request, including NeXus schema
validation of the generated files.

## Project structure

```
OperaXN/
  bin/
    core/          # Pipeline: classify, correlate, NeXus read/write
    operaxn/       # Tk GUI: visualisation and analysis
  definitions/     # NXoperando_monopd / NXoperando_tofnpd NXDL definitions
  examples/        # Synthetic example dataset generator and walkthrough
  schema/          # Vendored NXDL schema used to validate the definitions
  tests/           # Pytest suite (synthetic fixtures, no data files needed)
  .github/         # CI workflow
  pyproject.toml
  requirements.txt
```

## Contributing

Bug reports, feature requests and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If you use OperaXN in your research, please cite it: citation metadata is in
[CITATION.cff](CITATION.cff), and GitHub's *"Cite this repository"* button
generates BibTeX/APA from it directly. The NeXus application definitions and
the standardisation rationale are described in the accompanying perspective:
Tan et al., *Standardizing Operando Diffraction Studies for Battery Systems*,
ACS Energy Lett.,
[doi:10.1021/acsenergylett.6c01791](https://doi.org/10.1021/acsenergylett.6c01791).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The NXDL meta-schema vendored in [schema/](schema) (`nxdl.xsd`, `nxdlTypes.xsd`)
is not covered by that licence: it is NIAC/NeXus upstream material distributed
under the LGPL-3.0-or-later, whose text is in [schema/LGPL.txt](schema/LGPL.txt).
See [schema/README.md](schema/README.md) for its provenance and terms.

## AI usage

This project was developed with the assistance of Claude (Anthropic).
