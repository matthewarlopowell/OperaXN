# NeXus application definitions

Custom NeXus application definitions for operando powder diffraction of
electrochemical cells, as described in the OperaXN perspective paper
(Tan et al., *Standardizing Operando Diffraction Studies for Battery
Systems*, ACS Energy Lett.,
[doi:10.1021/acsenergylett.6c01791](https://doi.org/10.1021/acsenergylett.6c01791)):

- **`NXoperando_monopd.nxdl.xml`**: monochromatic X-ray/neutron diffraction
  (modelled on NXmonopd)
- **`NXoperando_tofnpd.nxdl.xml`**: time-of-flight neutron diffraction
  (modelled on NXtofnpd)

Both hold a single `NXentry` with experiment-level instrument/sample metadata
and the full electrochemical cycling record, plus one `NXsubentry` per
diffraction acquisition (`scan_000001` ...) holding only that acquisition's
data and its environment (acquisition window and electrochemical state of
the cell). Status: proposed definitions, not
yet submitted upstream to nexusformat/definitions or FAIRmat-NFDI.

OperaXN writes files conforming to these definitions; see below for how to
validate a generated file independently.

## Version

**1.0.0**: first release, shipped with OperaXN 2.0.0. Follows the published
specification (Figure 4 of the perspective paper). The version is stamped on
`entry/definition@version` in every generated file.

## Deviations from the published specification (Figure 4)

The paper's Figure 4 is the target. Where the shipped definitions differ,
the difference is either *forced* by what the source data can supply (the
writer must never emit a non-conforming file), or a *deferred* tightening of
a human-entered field that will be made required at NIAC submission.

| Item | Figure 4 | Here | Reason | Status |
|---|---|---|---|---|
| `extends` | NXmonopd / NXtofnpd | `NXobject` | The parents require entry-level detector `data`/`polar_angle` that cannot coexist with the per-subentry layout. | forced |
| `sample/description` | required | recommended | Human-entered; cannot be harvested from source data. | tighten at NIAC |
| `cycling_protocol` (NXnote) | required | recommended; `technique` required within the group | Human-entered; the writer omits the whole group when `technique` is empty so no non-conforming file can be produced. | tighten at NIAC |
| `operando_electrochemistry` | required | recommended | Absent from diffraction-only files (no electrochemistry supplied). | forced |
| `environment/voltage`, `environment/current` | required | recommended | Present when the scan was correlated within the correlation tolerance (300 s); absent for uncorrelated scans and diffraction-only files. | forced |
| `environment/voltage_min`, `voltage_max`, `current_min`, `current_max` | required | optional | Extrema over every electrochemistry point inside [`start_time`, `end_time`], both bounds inclusive, in the same clock alignment as the correlation; absent when the window holds no point (short exposures against a sparse logging cadence) or `end_time` is unknown. | forced |
| `environment/scan_timestamp` | required | required | Always written (presence guarantee equals `start_time`'s). | matches |
| `end_time` (entry) | optional | optional | | matches |
| `process`, `process/correlation_method` | required | required | Always written. | matches |
| `program_name` | required | absent | Duplicate of `process/program` and `process/version`; `process` is the single provenance record. | structural choice |
| `MONITOR` (inherited from NXtofnpd: `mode`, `preset`, `distance`, `data`, `time_of_flight`) | required | tofnpd: one optional entry-level `monitor` holding `mode` (required within the group) and the `detector_number` / `distance` / `polar_angle` / `azimuthal_angle` arrays `[nMon]` of every beam-monitor element; monopd: optional entry-level `monitor` holding `mode` and the shared `preset` of in-house acquisitions, plus a per-scan `monitor/integral` only for acquisitions whose EDF header carries a positive beam-monitor counter | tofnpd: the monitor record is invariant over the experiment, so it is held once like `instrument`. Reduced files record no monitor; `mode` and the monitor geometry come from the per-instrument profile (POLARIS: `timer`, its runs end on the clock; 14 elements at seven positions from the instrument definition), so an unprofiled instrument writes no group. The parent's single scalar `distance` presumes one monitor; the reduction of focussed data used none in particular, so all are described. `preset` is not recorded (the realised duration of each acquisition is `environment/exposure_time`) and reduced data has no monitor spectrum. monopd: in-house acquisitions count to a time preset shared by all scans, so mode and preset are held once; the Pilatus counters in the EDF header are the detector's own totals, not a monitor reading, and are not recorded; synchrotron sources record none. | forced |
| `environment/echem_index_start`, `_end` | required | absent | Window indices into the `operando_electrochemistry` arrays are a pure function of the acquisition window (searchsorted), and nothing consumed them. | structural choice |
| Electrochemistry array length symbol | `nP` | `nE` | Figure 4 uses `nP` for the electrochemistry series; monopd needs `nP` for the diffraction pattern length, so the electrochemistry arrays use `nE`. | naming |
| `SCAN/data` (monopd) | required | optional | Image-only acquisitions carry `image_source` instead; the writer enforces at least one of `data` / `image_source`. | forced |
| `MONITOR` / `DATA` placement | printed at subentry indent in Fig. 4 (the parents define them at entry level) | one `data` group (monopd) or one NXdata per bank (tofnpd) per acquisition subentry; `monitor` at entry level for tofnpd and per subentry for in-house monopd acquisitions | Each acquisition's pattern, and for in-house data its beam-monitor reading, lives in its own subentry; the neutron monitor record is invariant and lives with the instrument. | structural choice |
| `SUBENTRY/instrument` | link | HDF5 soft link with `@target` (documented, not declared) | NXDL links cannot be optional (XSD); pynxtools traverses the soft link and checks `@target` but does not match a `<link>` against a linked group, so declaring one fails every conforming file. Linking avoids N copies of the invariant instrument. | forced |
| `SUBENTRY/definition`, `title`, `start_time`, `end_time` | subentry-level fields | absent; the acquisition window is `environment/start_time` / `end_time` | `definition` and `title` repeat entry-level facts (the neutron run title is kept verbatim in `environment/logbook_entry`); the window sits in the environment group it bounds. | structural choice |
| `environment/capacity` | optional | absent | Never computed; removed rather than reserved. | structural choice |
| `pre_sample_flightpath` (tofnpd) | required | optional | Not in reduced files; supplied from the per-instrument profile (POLARIS: 14.0 m, see [Instrument profiles](#instrument-profiles)). Optional because an unprofiled instrument leaves it unset, and the writer then skips `None`. | forced for unprofiled instruments |

Additions beyond Figure 4 (all optional or recommended, so a file carrying
only the Figure 4 content still validates): `definition@version` and
`definition@URL`; extra `process` provenance (`program`, `version`, `date`,
`data_source`, `total_scans`);
extra `instrument` fields and the verbatim
`edf_metadata` / `synchrotron_metadata` NXcollection harvests; `user`;
`standard_electrochemistry` (non-correlated electrochemistry files);
`environment/voltage_timestamp` and `logbook_entry`; `data/data_errors`;
`image_source` and `image_data`; per-bank NXdata attributes (`spectrum`,
`y_unit`, `x_label`,
`source_file`, `measurement_number`); tofnpd per-bank detector geometry
arrays.

## Deviations from the ratified NXmonopd / NXtofnpd parents

| Item | Parent | Here | Forced/Choice | Reason |
|---|---|---|---|---|
| Intensities (`data`) | NX_INT | NX_NUMBER | forced | Azimuthally averaged / focussed, normalised: processed data, not raw counts. |
| `data_errors` field | absent | optional NX_NUMBER, `[nP]` in monopd (tofnpd leaves it undimensioned: bank lengths vary) | choice (addition) | Carries the propagated uncertainties (Sigma_I) that reduction software exports, named per the NXdata `FIELDNAME_errors` convention (the bare `errors` name is deprecated since NIAC 2018). |
| Per-bank axes (tofnpd) | single `time_of_flight` `[nTimeChan]` | one NXdata per bank, `bank_N` (time-of-flight) and optional sibling `bank_N_d` (d-spacing) | choice | Focussed banks have different native axes and lengths. |
| `data` XOR `image_source` (monopd) | `data` required | at least one of `data` / `image_source` (writer-enforced) | forced | Image-only acquisitions have no 1-D pattern yet must be recorded. |
| `instrument` in each subentry | n/a (single entry) | HDF5 soft link + `@target`, not an NXDL `<link>` | forced | See Figure 4 table: `<link>` cannot be optional and pynxtools does not match it against a linked group. |
| `echem_index_first`/`_last` | n/a | `_first`/`_last` naming | forced | `_end` is a reserved suffix; parents have no equivalent. |
| `twod_*` process fields | n/a | absent for neutron | choice | 2-D images do not arise in reduced neutron data. |
| `rotation_angle` (monopd) | required NX_FLOAT | absent | choice | The operando cell is stationary / the angle is not recorded. |
| Detector `polar_angle` / `data` | instrument-level, `[nDet]` under NXdetector | per-scan `SCAN/data` NXdata `[nP]` | structural | Each acquisition is its own subentry with its own pattern. |
| Entry-level NXdata links | `data` NXdata linking to detector fields | direct datasets in each subentry's NXdata | structural | No detector-level arrays to link to; per-subentry storage. |
| `monitor` group (tofnpd) | required at entry level: `mode`, `preset`, `distance` (scalar), `data`, `time_of_flight` | optional entry-level group with `mode` (required within it) and `detector_number` / `distance` / `polar_angle` / `azimuthal_angle` `[nMon]` for every monitor element | forced | Reduced files record no monitor, so `mode` and the geometry come from the instrument profile (POLARIS: `timer`, 14 elements) and an unprofiled instrument writes no group; the arrays describe all monitors because none in particular was used. |
| `monitor` group (monopd) | required at entry level: `mode`, `preset`, `integral` | optional entry-level group with `mode` and `preset` for in-house acquisitions; `integral` in the acquisition subentry, only when its EDF header carries a positive beam-monitor counter | forced | In-house acquisitions count to a time preset shared by all scans; synchrotron sources record no monitor. |
| `monitor/preset` (monopd) | required | optional | forced | Absent when the acquisitions do not share one exposure (each is in `environment/exposure_time`). |
| `monitor/integral` (monopd) | required, entry level | optional, per acquisition subentry | forced | Absent on instruments without a beam monitor (the Pilatus image totals in the header are not monitor counts). |
| `crystal` (monopd) | required NXcrystal with `wavelength [i]` | recommended, scalar `wavelength` | forced/choice | Absent for neutron sources; a monochromatic beam has one wavelength. |
| `source/probe` (monopd) | enumeration incl. electron | narrowed to `x-ray` / `neutron` | choice | Only these probes are in scope. |
| `source` (tofnpd) | absent | required NXsource with `name`, `type`, `probe` | choice (addition) | Records the facility and probe in the same place as monopd. |
| `user` (tofnpd) | required | optional | forced | Harvest-dependent; not present in all reduced files. |
| Detector geometry (tofnpd) | required `[nDet]` `distance`, `polar_angle`, `azimuthal_angle`, `detector_number` | optional per-bank arrays `[nBank]` | forced | Not in reduced files; supplied from the instrument profile, per bank rather than per detector element. `nBank`, not `nDet`. POLARIS writes `detector_number`, `distance` and `polar_angle` for its five banks and omits `azimuthal_angle`: the banks are rings around the beam, so no single azimuth is meaningful. |
| Monitor spectrum and preset (tofnpd) | `preset`, `data [nTimeChan]`, `time_of_flight` | absent (not declared) | forced | Reduced files carry no monitor spectrum (the POLARIS reduction normalises to the integrated proton charge, never to a monitor) and the preset is not recorded; the realised duration of each acquisition is `environment/exposure_time`. |
| 2-D `[nDet, nTimeChan]` data (tofnpd) | required | dropped | forced | Focussed 1-D pattern per bank. |
| `pre_sample_flightpath` (tofnpd) | required | optional | forced | Written whenever the instrument profile supplies it (POLARIS does); optional for unprofiled instruments (see above). |

## Member order

Generated files switch on HDF5 link creation-order tracking for the root and
every group, and the writer emits members in the order of the ratified parent
definition, with each OperaXN addition directly after the item it extends:

- entry: `title`, `start_time`, `end_time`, `definition`,
  `experiment_identifier`, `process`,
  `pre_sample_flightpath` (tofnpd), `user`, `instrument`, `sample`,
  `cycling_protocol`, `monitor`, `operando_electrochemistry`,
  `standard_electrochemistry`, then the `scan_NNNNNN` subentries in numeric
  order
- instrument: `name`, `source` (`type`, `name`, `probe`), `crystal`,
  `detector` (tofnpd: `description`, `detector_number`, `distance`,
  `polar_angle`, `azimuthal_angle`), then the harvest collections;
  `crystal` and `detector` exist only when they have content
- monitor: tofnpd `mode`, `detector_number`, `distance`, `polar_angle`,
  `azimuthal_angle`; monopd `mode`, `preset`
- subentry: `instrument` (soft link to `/entry/instrument`), `environment`
  (`start_time`, `end_time`, `voltage`, `current`, `voltage_min`,
  `voltage_max`, `current_min`, `current_max`, `scan_timestamp`,
  `midpoint_adjusted_timestamp`, `voltage_timestamp`, `exposure_time`,
  `logbook_entry`), `monitor` (`integral`) for acquisitions with a
  beam-monitor reading, then `data`
  (`polar_angle`, `data`, `data_errors`), `image_source`, `image_data` for
  monopd, or `bank_N` / `bank_N_d` (`data`, `data_errors`, then the axis) for
  tofnpd

NeXus attaches no meaning to member order and validators match by name and
class, so this only affects people and viewers reading the file. h5py and
viewers built on it follow the recorded order; `h5dump` needs
`--sort_by=creation_order`, and HDFView lists by name unless its indexing
option is set to creation order. A `standard_electrochemistry` group added to
an existing file from the GUI is deleted and recreated, and therefore lists
after the subentries.

## Instrument profiles

Reduced data files carry no instrument geometry, so the NXtofnpd fields
that describe it (`pre_sample_flightpath`, the per-bank `detector_number` /
`distance` / `polar_angle` / `azimuthal_angle` arrays) and the beam monitor
record (`monitor/mode` and the per-element monitor geometry arrays) come from a
per-instrument profile (`INSTRUMENT_PROFILES` in `bin/core/config.py`), selected by the instrument
name harvested from the Mantid `.dat` headers. Harvested values always win
over profile values; a profile only fills what the source files cannot.
Each profile entry records in a comment where its numbers come from.

POLARIS (ISIS) ships with `pre_sample_flightpath` = 14.0 m, a detector
`description` (ZnS scintillator elements in five banks) and the nominal
sample-to-bank distance and scattering angle of its five focussed banks,
taken from the Mantid instrument definition `Polaris_upgrade` (valid from
2012-05-01) embedded in a Mantid-processed POLARIS file. `azimuthal_angle`
is omitted: the banks are rings around the beam. Its `monitor/mode` is
`timer`: in an operando POLARIS dataset every run lasted 126 or 127 s while
the delivered proton charge varied, and runs that received no beam still
ended on the clock, so acquisitions are time-terminated. Its `monitor`
arrays list all 14 beam-monitor elements of the instrument definition (six
upstream positions and one downstream transmission monitor at 3.270 m, two
elements each); the reduction used none of them in particular (it
normalises to proton charge), so none is singled out.

WISH (ISIS) ships its source and instrument name only (`ISIS`, `WISH`): the
flight path, the five paired banks (1/10 to 5/6; the reduced data imply
scattering angles of 27.1, 58.3, 90.0, 121.7 and 153.2 degrees at a total
flight path of 42.2 m) and the monitors are not written until a
Mantid-processed WISH file supplies them.

To add an instrument, derive its block from any Mantid `SaveNexusProcessed`
file of that instrument (the focussed banks plus the embedded IDF) and
paste it into `INSTRUMENT_PROFILES`:

```bash
operaxn --profile POLARIS164798-164802.nxs
```

The printed block carries the written keys, including the `monitors` table
read from the file's `physical_monitors` group, plus, as comments, the
per-bank ranges and element counts and the IDF identity. `monitor_mode`
cannot be read from the file and prints as `None`; set it from the run
records (`timer` when runs end on the clock, `monitor` when they end on
monitor counts). A value the file cannot supply is `None` and is skipped by
the writer, so nothing is fabricated.

## Validating generated files

Files can be validated with FAIRmat's
[pynxtools](https://github.com/FAIRmat-NFDI/pynxtools). pynxtools resolves
definition names against its own bundled schema library, so the two NXDL
files here must be copied into it once (and again after every pynxtools
upgrade, which replaces the bundle):

```bash
pip install 'pynxtools>=0.15,<0.16'

# 1. locate pynxtools' contributed-definitions folder
python -c "import pynxtools, os; print(os.path.join(os.path.dirname(pynxtools.__file__), 'definitions', 'contributed_definitions'))"

# 2. copy BOTH definitions there
#    Windows:
copy NXoperando_monopd.nxdl.xml <path printed above>
copy NXoperando_tofnpd.nxdl.xml <path printed above>
#    macOS/Linux:
cp NXoperando_*.nxdl.xml <path printed above>

# 3. validate
pynx validate experiment.nxs
```

Success looks like:

```
The entry `entry` in file `experiment.nxs` is valid according to the
`NXoperando_monopd` application definition.
```

(`NXoperando_tofnpd` for neutron files. Older pynxtools versions use
`validate_nexus experiment.nxs` instead of `pynx validate`.)

After editing either NXDL, the copies inside pynxtools' contributed
definitions folder must be refreshed (repeat step 2) or validation silently
runs against the old version. The test suite does this automatically: the
repo `definitions/*.nxdl.xml` are authoritative and are copied over the
installed pynxtools copies whenever the bytes differ.

## Validating the definitions themselves

The NXDL meta-schema (`nxdl.xsd`, `nxdlTypes.xsd`) is vendored in
[../schema/](../schema) at release tag **v2026.01** (the pinned version these
were authored against), and the test suite validates both definitions against
it automatically (`pytest tests/test_nxdl.py`). To check manually with any
XSD validator:

```python
from lxml import etree
xsd = etree.XMLSchema(etree.parse("schema/nxdl.xsd"))
xsd.assertValid(etree.parse("definitions/NXoperando_monopd.nxdl.xml"))
```
