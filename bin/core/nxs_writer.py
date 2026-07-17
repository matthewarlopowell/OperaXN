"""
NeXus/HDF5 file writer for the core pipeline — schema 4.0.

Layout conforms to the custom application definitions shipped in
definitions/ (NXoperando_monopd / NXoperando_tofnpd; validated against
the NXDL schema by the local tests/run_nxdl_checks.py):

/entry (NXentry)                      one entry for the whole experiment
    definition = NXoperando_monopd | NXoperando_tofnpd
    title, start_time, end_time, experiment_identifier
    program_name (@version)           generator provenance
    process (NXprocess)               correlation provenance
    instrument (NXinstrument)         harvested + instrument profile; static
        name, source (NXsource), crystal (NXcrystal), detector (NXdetector)
        edf_metadata / synchrotron_metadata (full raw harvests, NXcollection)
    sample (NXsample)                 name + optional cell description
    user (NXuser)                     when harvestable (logbook / beamline file)
    operando_electrochemistry (NXdata)      time (s, @start ISO)/voltage/current
    standard_electrochemistry (NXenvironment)  file_NNN NXdata children
    scan_000001..scan_N (NXsubentry)  per acquisition
        definition = NXmonopd | NXtofnpd (semantic pointer)
        title, start_time, end_time (ISO 8601)
        environment (NXenvironment)   voltage/current (+units), window
                                      min/max, echem index pointers,
                                      voltage_log/current_log (NXlog)
        monitor (NXmonitor)           mode/preset/integral + raw counters
        instrument                    soft link to /entry/instrument
        data (NXdata)                 XRD: polar_angle/data/errors
        image_source (NXnote)         2D image reference (+image_data NXdata
                                      when embedding was chosen)
        bank_N / bank_N_d (NXdata)    neutron: one group per bank and axis
                                      family (TOF / d-spacing)

Known, deliberate deviations from NXmonopd/NXtofnpd (documented in the
definitions): processed intensities are NX_NUMBER, not NX_INT raw counts;
neutron banks keep their own native axes — nothing is discarded.

All timestamps are timezone-aware ISO 8601 (local wall time + machine
offset), written via _iso().
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from .config import (DEFINITION_URL_BASE, DEFINITION_VERSION,
                     ECHEM_TIME_TOLERANCE, GENERATOR_NAME, GENERATOR_VERSION)
from .classify import extract_edf_scan_fields, parse_mantid_header
from .correlate import EchemParser
from .model import DataSourceType, Scan
from .profiles import get_profile
from .readers import DataReaderFactory, HDFReader

try:
    import fabio

    FABIO_AVAILABLE = True
except ImportError:
    fabio = None
    FABIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Header fields excluded from the *global* EDF metadata dump. Per-scan fields
# (monitor counters, exposure) are harvested separately by
# classify.extract_edf_scan_fields — excluded here only because they vary
# per scan and would be misleading as experiment-wide values.
EDF_EXCLUDE_FIELDS = {
    'Date', 'ExposureTime', 'Image', 'Monitor', 'Intensity1', 'title',
    'SumForIntensity1', 'TransmittedFlux', 'Saturation',
    'pilai0', 'pilai1', 'pilct0', 'pilct1', 'pilroi0', 'pilroi1', 'Pil_Roi0'
}
NXS_EXCLUDE_FIELDS = {'start_time', 'end_time', 'count_time', 'scan_identifier'}

# Targeted pulls from a source synchrotron .nxs for the NXinstrument slots
SYNCHROTRON_FIELD_PATHS = {
    'instrument_name': ['/entry1/instrument/name', '/entry/instrument/name'],
    'source_name': ['/entry1/instrument/source/name', '/entry/instrument/source/name'],
    'source_type': ['/entry1/instrument/source/type', '/entry/instrument/source/type'],
    'probe': ['/entry1/instrument/source/probe', '/entry/instrument/source/probe'],
    'user': ['/entry1/user01/username', '/entry/user01/username'],
    'experiment_identifier': ['/entry1/experiment_identifier', '/entry/experiment_identifier'],
    'title': ['/entry1/title', '/entry/title'],
    'detector_distance': ['/entry1/sample/detector_distance', '/entry/sample/detector_distance'],
}


# ============================================================================
# Global metadata extraction (raw harvests, kept in full)
# ============================================================================

def extract_edf_global_metadata(edf_path: str) -> Dict[str, Any]:
    """Extract non-excluded header fields from the first EDF file."""
    if not FABIO_AVAILABLE:
        return {}
    try:
        image = fabio.open(edf_path)
        metadata = {}
        for key, value in image.header.items():
            if key in EDF_EXCLUDE_FIELDS:
                continue
            if value is None or str(value).strip() == '':
                continue
            metadata[key] = value
        logger.info(f"Extracted {len(metadata)} EDF metadata fields")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting EDF metadata: {e}")
        return {}


def extract_nxs_global_metadata(nxs_path: str) -> Dict[str, Any]:
    """Flatten all small datasets and attributes from a synchrotron .nxs file."""
    try:
        metadata: Dict[str, Any] = {}

        with h5py.File(nxs_path, 'r') as f:
            def extract_item(name: str, obj: Any) -> None:
                leaf = name.split('/')[-1]
                if leaf in NXS_EXCLUDE_FIELDS:
                    return
                if isinstance(obj, h5py.Dataset):
                    if obj.size > 1000:
                        return
                    if 'data' in name.lower():
                        return

                flat_name = name.replace('/', '_')

                for attr_key, attr_value in obj.attrs.items():
                    meta_key = f"{flat_name}_attr_{attr_key}"
                    decoded_value = _decode_h5_value(attr_value)
                    if decoded_value is not None:
                        metadata[meta_key] = decoded_value

                if isinstance(obj, h5py.Dataset):
                    try:
                        decoded_value = _decode_h5_value(obj[()])
                        if decoded_value is not None:
                            metadata[flat_name] = decoded_value
                    except Exception:
                        pass

            for root_attr_name, root_attr_val in f.attrs.items():
                decoded_val = _decode_h5_value(root_attr_val)
                if decoded_val is not None:
                    metadata[f"root_attr_{root_attr_name}"] = decoded_val

            f.visititems(extract_item)

        logger.info(f"Extracted {len(metadata)} NeXus metadata fields")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting NeXus metadata: {e}")
        return {}


def extract_synchrotron_fields(nxs_path: str) -> Dict[str, Any]:
    """Targeted pull of NXinstrument/NXsample/NXuser slot values from a source
    synchrotron .nxs (e.g. Diamond i11-1)."""
    fields: Dict[str, Any] = {}
    try:
        with h5py.File(nxs_path, 'r') as f:
            for name, paths in SYNCHROTRON_FIELD_PATHS.items():
                for path in paths:
                    if path in f:
                        val = _decode_h5_value(f[path][()])
                        if isinstance(val, list) and len(val) == 1:
                            val = val[0]
                        if val is not None and str(val).strip() not in ('', 'undefined'):
                            fields[name] = val
                        break
    except Exception as e:
        logger.debug(f"Could not extract synchrotron fields from {nxs_path}: {e}")
    return fields


def _decode_h5_value(value: Any) -> Any:
    """Decode HDF5 bytes/numpy scalars to native Python types; None if too large."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            item = value.item()
            if isinstance(item, bytes):
                return item.decode('utf-8', errors='ignore')
            return item
        elif value.size <= 10:
            decoded = []
            for x in value.flat:
                if isinstance(x, bytes):
                    decoded.append(x.decode('utf-8', errors='ignore'))
                else:
                    decoded.append(x)
            return decoded
        else:
            return None
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


# ============================================================================
# Shared writing helpers (also used by the GUI's add-echem-after-load path)
# ============================================================================

# Mantid axis labels -> NeXus units; the raw label is kept as @x_label
MANTID_UNIT_MAP = {
    'Time-of-flight': 'microsecond',
    'd-Spacing': 'angstrom',
}

# NXsource/type is an open enumeration; harvested values matching a canonical
# item case-insensitively adopt its casing (e.g. Diamond files say
# "Synchrotron X-Ray Source"). The verbatim harvest keeps the original.
_SOURCE_TYPES = (
    'Spallation Neutron Source', 'Pulsed Reactor Neutron Source',
    'Reactor Neutron Source', 'Synchrotron X-ray Source',
    'Pulsed Muon Source', 'Rotating Anode X-ray', 'Fixed Tube X-ray',
    'UV Laser', 'Free-Electron Laser', 'Optical Laser', 'Ion Source',
    'UV Plasma Source', 'Metal Jet X-ray',
)
_SOURCE_TYPE_CANONICAL = {t.lower(): t for t in _SOURCE_TYPES}


def _iso(value: Any) -> Optional[str]:
    """Timezone-aware ISO 8601 for a timestamp-like value; None if unparseable.
    Naive input is local wall time and gets the machine's DST-aware offset."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime().astimezone().isoformat()


def write_echem_series(group: h5py.Group, echem_df: pd.DataFrame) -> None:
    """Write an EchemParser DataFrame into an NXdata group: float seconds +
    time/@start origin (NXlog convention); an all-NaN current is omitted."""
    timestamps = pd.to_datetime(echem_df['timestamp'])
    start = timestamps.min()
    time_ds = group.create_dataset(
        'time', data=(timestamps - start).dt.total_seconds().to_numpy(dtype=float))
    time_ds.attrs['units'] = 's'
    time_ds.attrs['start'] = _iso(start)

    voltage_ds = group.create_dataset(
        'voltage', data=echem_df['echem_data'].to_numpy(dtype=float))
    voltage_ds.attrs['units'] = 'V'

    group.attrs['signal'] = 'voltage'
    group.attrs['axes'] = 'time'
    group.attrs['time_indices'] = 0

    if 'current' in echem_df.columns:
        current = pd.to_numeric(echem_df['current'], errors='coerce').to_numpy(dtype=float)
        if not np.all(np.isnan(current)):
            current_ds = group.create_dataset('current', data=current)
            current_ds.attrs['units'] = 'mA'
            group.attrs['auxiliary_signals'] = ['current']


def write_standard_echem_group(parent: h5py.Group,
                               datasets: List[Dict[str, Any]]) -> h5py.Group:
    """(Re)write entry/standard_electrochemistry from parsed echem datasets
    ('source_file' + 'data' items). Single source of truth for the layout:
    used by NXSWriter and the GUI's add-echem-after-load path."""
    if 'standard_electrochemistry' in parent:
        del parent['standard_electrochemistry']
    container = parent.create_group('standard_electrochemistry')
    container.attrs['NX_class'] = 'NXenvironment'
    container.attrs['num_files'] = len(datasets)
    container.create_dataset(
        'description', data='Additional (non-correlated) electrochemistry files')

    for index, item in enumerate(datasets, start=1):
        group = container.create_group(f'file_{index:03d}')
        group.attrs['NX_class'] = 'NXdata'
        group.attrs['source_file'] = str(item.get('source_file') or 'echem')
        write_echem_series(group, item['data'])
    return container


# ============================================================================
# Writer
# ============================================================================

class NXSWriter:
    """Writes the canonical schema-4.0 NeXus/HDF5 file."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE,
                 include_2d_images: bool = False,
                 max_display_size: int = 0,
                 correlation_method: str = "absolute",
                 title: Optional[str] = None,
                 sample_name: Optional[str] = None,
                 sample_description: Optional[str] = None):
        self.data_source = data_source
        self.include_2d_images = include_2d_images
        self.max_display_size = max_display_size  # 0 = full resolution
        self.correlation_method = correlation_method
        self.title = title
        self.sample_name = sample_name
        self.sample_description = sample_description
        self.echem_parser = EchemParser()

    # --- top level ---

    def write(self, output_path: str, scans: List[Scan], echem_df: pd.DataFrame,
              standard_echem_files: Optional[List[str]] = None) -> None:
        """Write the full canonical file: metadata, scans, and echem layers."""
        reader_factory = DataReaderFactory()
        harvest = self._harvest_experiment_metadata(scans)

        with h5py.File(output_path, 'w') as f:
            f.attrs['NX_class'] = 'NXroot'
            f.attrs['file_name'] = os.path.basename(output_path)
            f.attrs['file_time'] = _iso(pd.Timestamp.now())
            f.attrs['default'] = 'entry'

            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'

            definition = ('NXoperando_tofnpd'
                          if self.data_source == DataSourceType.NEUTRON
                          else 'NXoperando_monopd')
            def_ds = entry.create_dataset('definition', data=definition)
            def_ds.attrs['version'] = DEFINITION_VERSION
            def_ds.attrs['URL'] = f'{DEFINITION_URL_BASE}{definition}.nxdl.xml'

            prog_ds = entry.create_dataset('program_name', data=GENERATOR_NAME)
            prog_ds.attrs['version'] = GENERATOR_VERSION
            prog_ds.attrs['configuration'] = json.dumps({
                'data_source': self.data_source.value,
                'correlation_method': self.correlation_method,
                'include_2d_images': self.include_2d_images,
                'max_display_size': self.max_display_size,
            })

            self._write_process(entry, scans)
            self._write_entry_fields(entry, scans, harvest)
            self._write_instrument(entry, scans, harvest)
            self._write_sample(entry, harvest)
            self._write_user(entry, harvest)

            for scan in scans:
                self._write_scan(entry, scan, reader_factory, harvest)

            self._write_operando_echem(entry, echem_df)
            self._write_standard_echem(entry, standard_echem_files)

        logger.info(f"NeXus file written to {output_path}")

    def _harvest_experiment_metadata(self, scans: List[Scan]) -> Dict[str, Any]:
        """Collect experiment-level metadata from the first scan's source files
        and merge with the instrument profile (harvested values win)."""
        harvested: Dict[str, Any] = {}

        first = scans[0] if scans else None
        if first is not None:
            # In-house: EDF header of the first 2D file
            twod = first.twod
            if twod and str(twod).lower().endswith('.edf'):
                edf = extract_edf_scan_fields(str(twod))
                harvested.update(edf.get('instrument', {}))
                harvested['_edf_global'] = extract_edf_global_metadata(str(twod))

            # Synchrotron: targeted slots + full dump from the source .nxs
            if self.data_source == DataSourceType.SYNCHROTRON:
                src = first.source_nxs
                if src and os.path.isfile(src):
                    harvested.update(extract_synchrotron_fields(src))
                    harvested['_nxs_global'] = extract_nxs_global_metadata(src)

            # Neutron: Mantid header of the first data file + logbook extras
            if self.data_source == DataSourceType.NEUTRON:
                if first.neutron_files:
                    for meas in first.neutron_files.values():
                        path = meas.get('tof') or meas.get('d')
                        if path:
                            mantid = parse_mantid_header(path)
                            if mantid.get('instrument'):
                                harvested['instrument_name'] = mantid['instrument']
                            break
                if first.logbook:
                    for key in ('run_title', 'users', 'proposal'):
                        if first.logbook.get(key):
                            harvested[key] = first.logbook[key]

        profile = get_profile(self.data_source, harvested.get('instrument_name'))
        # harvested values win over profile defaults
        merged = {**profile, **{k: v for k, v in harvested.items() if v is not None}}
        return merged

    def _write_process(self, entry: h5py.Group, scans: List[Scan]) -> None:
        """NXprocess with correlation provenance (entry attrs in schema v3)."""
        process = entry.create_group('process')
        process.attrs['NX_class'] = 'NXprocess'
        process.create_dataset('program', data=GENERATOR_NAME)
        process.create_dataset('version', data=GENERATOR_VERSION)
        process.create_dataset('date', data=_iso(pd.Timestamp.now()))
        process.create_dataset('data_source', data=self.data_source.value)
        process.create_dataset('correlation_method', data=self.correlation_method)
        tolerance = process.create_dataset('echem_time_tolerance',
                                           data=float(ECHEM_TIME_TOLERANCE))
        tolerance.attrs['units'] = 's'
        process.create_dataset('total_scans', data=len(scans))
        # 2D options only apply to the XRD definitions; NXoperando_tofnpd
        # does not declare them
        if self.data_source != DataSourceType.NEUTRON:
            process.create_dataset('twod_included', data=self.include_2d_images)
            if self.include_2d_images and self.max_display_size > 0:
                process.create_dataset('twod_max_display_size',
                                       data=self.max_display_size)

    def _write_entry_fields(self, entry: h5py.Group, scans: List[Scan],
                            harvest: Dict[str, Any]) -> None:
        """Entry title, time window, and identifier (user values win)."""
        title = (self.title or harvest.get('run_title') or harvest.get('title')
                 or 'operando diffraction experiment')
        entry.create_dataset('title', data=str(title))

        timestamps = [s.original_timestamp or s.timestamp for s in scans
                      if (s.original_timestamp or s.timestamp)]
        if timestamps:
            start = min(timestamps)
            entry.create_dataset('start_time', data=_iso(start) or str(start))
            ends = [s.neutron_end for s in scans if s.neutron_end]
            end = max(ends) if ends else max(timestamps)
            entry.create_dataset('end_time', data=_iso(end) or str(end))

        identifier = harvest.get('proposal') or harvest.get('experiment_identifier')
        if identifier:
            entry.create_dataset('experiment_identifier', data=str(identifier))

    def _write_instrument(self, entry: h5py.Group, scans: List[Scan],
                          harvest: Dict[str, Any]) -> None:
        """NXinstrument: source, crystal, detector, and raw-harvest collections."""
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'
        # NeXus linking convention: the linked-to object names its own path
        # so consumers can tell the original from the per-scan soft links
        inst.attrs['target'] = '/entry/instrument'
        inst.create_dataset('name', data=str(harvest.get('instrument_name', 'unknown')))

        source = inst.create_group('source')
        source.attrs['NX_class'] = 'NXsource'
        source.create_dataset('name', data=str(harvest.get('source_name', 'unknown')))
        source_type = str(harvest.get('source_type', 'unknown'))
        source.create_dataset(
            'type', data=_SOURCE_TYPE_CANONICAL.get(source_type.lower(), source_type))
        source.create_dataset('probe', data=str(harvest.get('probe', 'x-ray')))

        # NXcrystal wavelength (monochromatic sources); EDF stores metres
        wavelength = None
        if 'wavelength_m' in harvest:
            try:
                wavelength = float(harvest['wavelength_m']) * 1e10  # m -> Angstrom
            except (ValueError, TypeError):
                pass
        elif 'wavelength' in harvest:
            wavelength = harvest['wavelength']
        if wavelength and self.data_source != DataSourceType.NEUTRON:
            crystal = inst.create_group('crystal')
            crystal.attrs['NX_class'] = 'NXcrystal'
            ds = crystal.create_dataset('wavelength', data=float(wavelength))
            ds.attrs['units'] = 'angstrom'

        detector = inst.create_group('detector')
        detector.attrs['NX_class'] = 'NXdetector'
        detector_fields = {
            'description': harvest.get('detector_model'),
            # EDF SampleDistance only: the synchrotron detector_distance
            # carries no unit at source, so the typed NX_LENGTH slot stays
            # empty (the verbatim value remains in synchrotron_metadata)
            'distance': harvest.get('distance_m'),
            'x_pixel_size': harvest.get('x_pixel_size_m'),
            'y_pixel_size': harvest.get('y_pixel_size_m'),
            'beam_center_x': harvest.get('beam_center_x'),
            'beam_center_y': harvest.get('beam_center_y'),
        }
        detector_units = {
            'distance': 'm',
            'x_pixel_size': 'm',
            'y_pixel_size': 'm',
            'beam_center_x': 'pixel',
            'beam_center_y': 'pixel',
        }
        for name, value in detector_fields.items():
            if value is not None:
                try:
                    ds = detector.create_dataset(name, data=float(value))
                    if detector_units.get(name):
                        ds.attrs['units'] = detector_units[name]
                except (ValueError, TypeError):
                    detector.create_dataset(name, data=str(value))

        # Full raw harvests preserved as collections
        for key, group_name in (('_edf_global', 'edf_metadata'),
                                ('_nxs_global', 'synchrotron_metadata')):
            dump = harvest.get(key)
            if dump:
                grp = inst.create_group(group_name)
                grp.attrs['NX_class'] = 'NXcollection'
                for k, v in dump.items():
                    if v is None:
                        continue
                    try:
                        grp.create_dataset(k, data=v)
                    except Exception as e:
                        logger.debug(f"Could not write harvested field '{k}': {e}")

    def _write_sample(self, entry: h5py.Group, harvest: Dict[str, Any]) -> None:
        """NXsample; the dialog's sample field overrides harvested names."""
        sample = entry.create_group('sample')
        sample.attrs['NX_class'] = 'NXsample'
        # Harvested values (e.g. the EDF Comment header) are stored verbatim,
        # even when the instrument leaves a trailing template separator
        name = (self.sample_name or harvest.get('comment')
                or harvest.get('run_title') or 'unknown')
        sample.create_dataset('name', data=str(name))
        if self.sample_description:
            sample.create_dataset('description', data=str(self.sample_description))

    @staticmethod
    def _write_user(entry: h5py.Group, harvest: Dict[str, Any]) -> None:
        """NXuser, only when a name was harvestable."""
        user_name = harvest.get('users') or harvest.get('user')
        if user_name:
            user = entry.create_group('user')
            user.attrs['NX_class'] = 'NXuser'
            user.create_dataset('name', data=str(user_name))

    # --- per-scan subentries ---

    def _write_scan(self, entry: h5py.Group, scan: Scan,
                    reader_factory: DataReaderFactory,
                    harvest: Dict[str, Any]) -> None:
        """One scan_N NXsubentry with environment, monitor, and data groups."""
        sub = entry.create_group(f'scan_{scan.scan_num:06d}')
        sub.attrs['NX_class'] = 'NXsubentry'
        sub.attrs['scan_number'] = scan.scan_num

        if scan.neutron_files:
            definition = 'NXtofnpd'
        else:
            definition = 'NXmonopd'
        sub.create_dataset('definition', data=definition)

        title = None
        if scan.logbook:
            title = scan.logbook.get('run_title')
        sub.create_dataset('title', data=str(title or f'scan {scan.scan_num}'))

        # start_time is the acquisition start; for neutron scans the display
        # timestamp is the logbook midpoint and lives in environment instead
        start = scan.neutron_start or scan.original_timestamp or scan.timestamp
        if start:
            sub.create_dataset('start_time', data=_iso(start) or str(start))
        end = scan.neutron_end
        if end is None and start and scan.exposure_time:
            try:
                end = pd.to_datetime(start) + pd.Timedelta(seconds=scan.exposure_time)
            except (ValueError, TypeError):
                end = None
        if end is not None:
            sub.create_dataset('end_time', data=_iso(end) or str(end))

        # Instrument: linked, not copied
        sub['instrument'] = h5py.SoftLink('/entry/instrument')

        self._write_environment(sub, scan)
        self._write_monitor(sub, scan)

        if scan.oned or scan.twod:
            self._write_xrd_data(sub, scan, reader_factory)

        if scan.neutron_files:
            self._write_neutron_banks(sub, scan, reader_factory)

    @classmethod
    def _write_environment(cls, sub: h5py.Group, scan: Scan) -> None:
        """Electrochemical state at acquisition time (the operando extension)."""
        env = sub.create_group('environment')
        env.attrs['NX_class'] = 'NXenvironment'

        def scalar(name: str, value: Optional[float], units: str) -> None:
            if value is None:
                return
            ds = env.create_dataset(name, data=float(value))
            ds.attrs['units'] = units

        def date(name: str, value: Any) -> None:
            iso = _iso(value)
            if iso:
                env.create_dataset(name, data=iso)

        date('scan_timestamp', scan.original_timestamp or scan.timestamp)
        scalar('voltage', scan.echem, 'V')
        scalar('current', scan.current, 'mA')
        scalar('voltage_min', scan.voltage_min, 'V')
        scalar('voltage_max', scan.voltage_max, 'V')
        scalar('current_min', scan.current_min, 'mA')
        scalar('current_max', scan.current_max, 'mA')
        date('voltage_timestamp', scan.echem_timestamp)
        date('midpoint_adjusted_timestamp', scan.timestamp_for_correlation)
        scalar('exposure_time', scan.exposure_time, 's')
        # File names use first/last: "_end" is a reserved NeXus suffix
        if scan.echem_index_start is not None:
            env.create_dataset('echem_index_first', data=int(scan.echem_index_start))
        if scan.echem_index_end is not None:
            env.create_dataset('echem_index_last', data=int(scan.echem_index_end))
        if scan.logbook and scan.logbook.get('full_line'):
            env.create_dataset('logbook_entry', data=str(scan.logbook['full_line']))

        segment = scan.echem_segment
        if segment:
            cls._write_nxlog(env, 'voltage_log', segment, segment['voltage'], 'V')
            if segment.get('current') is not None:
                cls._write_nxlog(env, 'current_log', segment,
                                 segment['current'], 'mA')

    @staticmethod
    def _write_nxlog(env: h5py.Group, name: str, segment: Dict[str, Any],
                     values: Any, value_units: str) -> None:
        """NXlog with float-second time (+@start ISO origin) and value arrays."""
        log = env.create_group(name)
        log.attrs['NX_class'] = 'NXlog'
        time_ds = log.create_dataset(
            'time', data=np.asarray(segment['time_s'], dtype=float))
        time_ds.attrs['units'] = 's'
        time_ds.attrs['start'] = _iso(segment['start'])
        value_ds = log.create_dataset('value', data=np.asarray(values, dtype=float))
        value_ds.attrs['units'] = value_units

    def _write_monitor(self, sub: h5py.Group, scan: Scan) -> None:
        """Beam monitor record: EDF counters when available, timer fallback."""
        monitor = sub.create_group('monitor')
        monitor.attrs['NX_class'] = 'NXmonitor'

        counters: Dict[str, float] = {}
        if scan.twod and str(scan.twod).lower().endswith('.edf'):
            counters = extract_edf_scan_fields(str(scan.twod)).get('monitor', {})

        monitor.create_dataset('mode', data='timer')
        if scan.exposure_time is not None:
            preset = monitor.create_dataset('preset', data=float(scan.exposure_time))
            preset.attrs['units'] = 's'

        # Best available integral: first positive counter in preference order
        integral = None
        for key in ('Monitor', 'pilct1', 'pilai1', 'Intensity1'):
            val = counters.get(key)
            if val:
                integral = float(val)
                break
        if integral is not None:
            integral_ds = monitor.create_dataset('integral', data=integral)
            integral_ds.attrs['units'] = 'counts'

        # Keep every raw counter — nothing harvested is discarded
        for key, val in counters.items():
            try:
                monitor.create_dataset(key, data=float(val))
            except Exception:
                pass

    def _write_xrd_data(self, sub: h5py.Group, scan: Scan,
                        reader_factory: DataReaderFactory) -> None:
        """NXdata with the 1D pattern (plus errors); 2D image at subentry level."""
        if scan.oned:
            data_group = sub.create_group('data')
            data_group.attrs['NX_class'] = 'NXdata'
            try:
                arr = reader_factory.read_file(scan.oned)
                angle_ds = data_group.create_dataset('polar_angle', data=arr[:, 0])
                angle_ds.attrs['units'] = 'degrees'
                data_group.create_dataset('data', data=arr[:, 1])
                if arr.shape[1] >= 3:
                    data_group.create_dataset('errors', data=arr[:, 2])
                data_group.attrs['signal'] = 'data'
                data_group.attrs['axes'] = 'polar_angle'
                data_group.attrs['polar_angle_indices'] = 0
            except Exception as e:
                logger.error(f"Error reading 1D data for scan {scan.scan_num}: {e}")
            data_group.attrs['oned_source_file'] = os.path.basename(scan.oned)

        if scan.twod:
            self._write_2d_data(sub, scan)

    def _write_2d_data(self, sub: h5py.Group, scan: Scan) -> None:
        """Reference the source 2D detector image (image_source NXnote) and
        embed a copy in image_data when embedding was chosen."""
        twod_path = str(scan.twod)
        basename = os.path.basename(twod_path)
        ext = os.path.splitext(basename)[1].lower()
        mime = {'.hdf': 'application/x-hdf5',
                '.edf': 'application/x-esrf-edf'}.get(ext)

        note = sub.create_group('image_source')
        note.attrs['NX_class'] = 'NXnote'
        note.create_dataset('file_name', data=twod_path)
        if mime:
            note.create_dataset('type', data=mime)
        note.create_dataset('description',
                            data=f'2D detector image {basename}')
        note.create_dataset('max_display_size', data=int(self.max_display_size))

        if not self.include_2d_images:
            note.create_dataset('embedded', data=False)
            return

        try:
            if ext == '.hdf':
                hdf_reader = HDFReader(max_display_size=self.max_display_size)
                data_2d = hdf_reader.read(twod_path)
                original_shape = hdf_reader.original_shape
            else:
                data_2d = DataReaderFactory.read_file(twod_path)
                original_shape = data_2d.shape

            image = sub.create_group('image_data')
            image.attrs['NX_class'] = 'NXdata'
            image.attrs['signal'] = 'data'
            image_ds = image.create_dataset('data', data=data_2d)
            image_ds.attrs['interpretation'] = 'image'
            note.create_dataset('original_shape',
                                data=np.asarray(original_shape, dtype=np.int64))
            note.create_dataset('embedded', data=True)

        except Exception as e:
            logger.error(f"Error embedding 2D data for scan {scan.scan_num}: {e}")
            note.create_dataset('embedded', data=False)

    def _write_neutron_banks(self, sub: h5py.Group, scan: Scan,
                             reader_factory: DataReaderFactory) -> None:
        """One NXdata per bank and axis family: bank_N (native TOF) plus
        bank_N_d (d-spacing) — keep-all-data, one axis family per group."""
        for meas_num, meas_files in scan.neutron_files.items():
            for key, (suffix, x_name) in {
                'tof': ('', 'time_of_flight'),
                'd': ('_d', 'd_spacing'),
            }.items():
                if key not in meas_files:
                    continue
                path = meas_files[key]
                bank = sub.create_group(f'bank_{meas_num}{suffix}')
                bank.attrs['NX_class'] = 'NXdata'
                # grouper keys are strings; the schema wants NX_INT
                bank.attrs['measurement_number'] = (
                    int(meas_num) if str(meas_num).isdigit() else meas_num)
                bank.attrs['source_file'] = os.path.basename(path)
                try:
                    arr = reader_factory.read_file(path, is_neutron=True)
                    x_ds = bank.create_dataset(x_name, data=arr[:, 0])
                    bank.create_dataset('data', data=arr[:, 1])
                    if arr.shape[1] >= 3:
                        bank.create_dataset('errors', data=arr[:, 2])

                    header = parse_mantid_header(path)
                    label = header.get('x_unit')
                    if label:
                        # Mantid exports axis labels, not units
                        x_ds.attrs['units'] = MANTID_UNIT_MAP.get(label, label)
                        bank.attrs['x_label'] = label
                    if header.get('spectrum') is not None:
                        bank.attrs['spectrum'] = header['spectrum']
                    if header.get('y_unit'):
                        bank.attrs['y_unit'] = header['y_unit']
                    bank.attrs['signal'] = 'data'
                    bank.attrs['axes'] = x_name
                except Exception as e:
                    logger.error(f"Error reading {key} data: {e}")

    # --- electrochemistry layers ---

    @staticmethod
    def _write_operando_echem(entry: h5py.Group, echem_df: pd.DataFrame) -> None:
        """Full cycling protocol: time (s, @start ISO) / voltage / current."""
        if echem_df is None or echem_df.empty:
            return

        echem_group = entry.create_group('operando_electrochemistry')
        echem_group.attrs['NX_class'] = 'NXdata'
        write_echem_series(echem_group, echem_df)
        entry.attrs['default'] = 'operando_electrochemistry'

    def _write_standard_echem(self, entry: h5py.Group,
                              standard_echem_files: Optional[List[str]]) -> None:
        """Parse and store each additional echem file as a file_NNN NXdata."""
        if not standard_echem_files:
            return

        datasets = []
        for std_echem_path in standard_echem_files:
            if not os.path.isfile(std_echem_path):
                continue
            standard_echem_df = self.echem_parser.parse(std_echem_path)
            if standard_echem_df is None or standard_echem_df.empty:
                continue
            datasets.append({'source_file': os.path.basename(std_echem_path),
                             'data': standard_echem_df})
            logger.info(f"Added standard electrochemistry file"
                        f" {os.path.basename(std_echem_path)}"
                        f" ({len(standard_echem_df)} points)")

        if datasets:
            write_standard_echem_group(entry, datasets)
