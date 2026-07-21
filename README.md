# OperaXN

**OPERAndo X-ray and Neutron diffraction data visualisation tool**

**OperaXN** is a Python desktop application for correlating, visualising and
sharing *operando* diffraction data collected at laboratory XRD, synchrotron
XRD and neutron diffraction sources. Raw diffraction and electrochemistry files
are time-correlated and consolidated into a single standardised NeXus (`.nxs`)
file which OperaXN then visualises. The same file can be shared, re-opened
directly and read by any HDF5/NeXus tool.

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
- Export publication-quality figures (PNG, PDF, SVG), animated GIFs, and
  Excel summaries
- Supports `.nxs`, `.dat`, `.xy`, `.edf`, `.hdf`, `.txt`, and `.zip` inputs

## Installation

### macOS prerequisite

The system Python bundled with macOS does not include a compatible version of
Tcl/Tk and the GUI will not render. Before installing, download and install
Python from **[python.org](https://www.python.org/downloads/)** (3.11 or
later). The official installer bundles Tcl/Tk 8.6+.

### From source

```bash
git clone https://github.com/matthewarlopowell/OperaXN.git
cd OperaXN
python3 -m venv .venv
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

### Command line options

```bash
operaxn --help          # Show all options
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
of the cell during that scan. Files can be inspected with any HDF5
tool (e.g. [DAWN](https://dawnsci.org/), NeXpy, h5web) and validated with
FAIRmat's [pynxtools](https://github.com/FAIRmat-NFDI/pynxtools).

## Dependencies

- Python >= 3.9
- NumPy >= 1.20.0
- Pandas >= 2.0.0
- Matplotlib >= 3.4.0
- h5py >= 3.0.0
- Fabio >= 0.14.0
- OpenPyXL >= 3.0.0
- PSutil >= 5.8.0
- ImageIO >= 2.9.0

## Project structure

```
OperaXN/
  bin/
    core/          # Pipeline: classify, correlate, NeXus read/write
    operaxn/       # Tk GUI: visualisation and analysis
  definitions/     # NXoperando_monopd / NXoperando_tofnpd NXDL definitions
  pyproject.toml
  requirements.txt
```

## Contributing

Bug reports, feature requests and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## AI usage

This project was developed with the assistance of Claude (Anthropic).
