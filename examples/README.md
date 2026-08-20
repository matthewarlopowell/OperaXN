# Examples

## Try OperaXN in two minutes

Generate a small synthetic operando dataset (no instrument data needed):

```bash
python examples/make_example_data.py
```

This writes two raw datasets to `examples/example_data/`:

- **`laboratory/`**: three laboratory XRD scans (`.dat`, collected 10:05–10:15
  with 120 s exposures) plus an electrochemistry log (`echem.txt`, a voltage
  ramp from 3.70 V at 60 s cadence starting 10:00)
- **`neutron/`**: one neutron scan (logbook entry 10:00–10:30, two detector
  banks each in time-of-flight and d-spacing) plus the same electrochemistry
  log

Then:

1. Launch the GUI: `operaxn`
2. In the upload dialog, select the data source (Laboratory X-ray diffraction
   for `laboratory/`, neutron for `neutron/`), point it at the generated folder
   and load with **absolute** time correlation.
3. Each diffraction scan is now annotated with the electrochemical state of
   the cell at acquisition time: the laboratory scan at 10:05 correlates to
   3.76 V (its exposure midpoint 10:06), the neutron scan to 3.85 V (its
   10:15 logbook midpoint).
4. Export the NeXus file to see the standardised `.nxs` output; it conforms
   to the application definitions in [../definitions/](../definitions) and
   can be re-opened directly by OperaXN or any HDF5 tool.

The automated test suite builds identical synthetic datasets (via
`tests/_builders.py`) and verifies these exact correlation values.

## Real instrument datasets (`OPERAXN_TEST_DATA`)

The test suite also contains end-to-end smoke tests against real instrument
data (the real-data section of `tests/test_pipeline.py`, marker `realdata`).
These are skipped unless the `OPERAXN_TEST_DATA` environment variable points
at a folder containing one or more of the following, by exact name:

| Name | Contents |
|---|---|
| `neutron_tof.zip` | Neutron time-of-flight: logbook + per-bank TOF/d files + echem |
| `synchrotron_i11/` | Synchrotron (Diamond i11-1): `.nxs` + `.hdf` + integrated `.xy` files |
| `lab_xrd_full/` | Laboratory XRD, full: 1D `.dat` + 2D `.edf` + echem |

Public datasets in this layout are being prepared for release. Until they
land, the tests' expected values are calibrated to the authors' local
reference data, so the layout above is the slot they will drop into rather
than a set of files you can download today. Any subset may be present;
missing datasets are reported as skips.

The same mechanism can be pointed at your own instrument data to exercise
the full pipeline, but note that the tests' assertion values (scan counts,
correlation floors) are calibrated to the reference datasets, so expect
failures rather than a clean pass with other data. To contribute a public
example dataset, add it under this layout and add a test to the real-data
section of `tests/test_pipeline.py` describing what it should produce.
See [../CONTRIBUTING.md](../CONTRIBUTING.md).
