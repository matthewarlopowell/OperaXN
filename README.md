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
  state alongside the full cycling record
- Simultaneous visualisation of X-ray (1D and 2D) and neutron (per-bank
  TOF/d-spacing) diffraction data with electrochemical cycling
- Operando heatmap (stacked patterns vs scan/time with the voltage track),
  capacity analysis and ICI (intermittent current interruption) analysis
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
generation option is collected in a single upload dialog. Export the
generated NeXus file at any point to share the experiment.

No data to hand? [examples/](examples) generates a small synthetic operando
dataset and walks through loading it.

### Command line options

```bash
operaxn --help          # Show all options
operaxn --version       # Print the version
operaxn --debug         # Enable debug logging
operaxn --check-deps    # Verify dependencies
```

## Time correlation

OperaXN correlates diffraction scans with electrochemistry data by timestamp.
Two modes are available (selected at load time):

- **Absolute** -> scan timestamps are matched directly to echem timestamps via
  nearest-neighbour lookup at the exposure midpoint.
- **Relative** -> both datasets are zeroed to their respective first timestamps
  and correlated by elapsed time. Useful when diffraction and echem clocks are
  not synchronised.

Correlation runs once at generation; the results are stored in the `.nxs`
file alongside the full cycling record.

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
- NumPy >= 1.20.0
- Pandas >= 2.0.0
- Matplotlib >= 3.4.0
- h5py >= 3.0.0
- Fabio >= 0.14.0
- OpenPyXL >= 3.0.0
- PSutil >= 5.8.0
- ImageIO >= 2.9.0

## Testing

The automated test suite runs on synthetic data generated at test time; no
data files are required, apart from the opt-in `realdata` tests:

```bash
pip install -e '.[test]'
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
under the LGPL-3.0. See [schema/README.md](schema/README.md) for its provenance
and terms.

## AI usage

This project was developed with the assistance of Claude (Anthropic).
