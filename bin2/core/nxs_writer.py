"""
NeXus/HDF5 file writer for the core pipeline — schema 3.0.

Layout follows the extended NXmonopd/NXtofnpd application definitions from the
OperaXN perspective paper:

/entry (NXentry)                      one entry for the whole experiment
    title, start_time, end_time, experiment_identifier
    instrument (NXinstrument)         harvested + instrument profile; static
        name, source (NXsource), crystal (NXcrystal), detector (NXdetector)
        edf_metadata / synchrotron_metadata (full raw harvests, NXcollection)
    sample (NXsample)
    user (NXuser)                     when harvestable (logbook / beamline file)
    operando_electrochemistry (NXdata)      cycling protocol layer
    standard_electrochemistry (NXcollection)
    scan_000001..scan_N (NXsubentry)  per acquisition
        definition = NXmonopd | NXtofnpd
        title, start_time, end_time
        environment (NXcollection)    correlated voltage/current/timestamps
        monitor (NXmonitor)           mode/preset/integral + raw counters
        instrument                    soft link to /entry/instrument
        data (NXdata)                 XRD: polar_angle/data/errors (+2D refs)
        bank_N (NXdata)               neutron: time_of_flight/data/errors
                                      + d_spacing/d_data/d_errors (kept, ext.)

Known, deliberate deviations (documented for the paper):
- processed intensities are floats (definitions say NX_INT raw counts)
- neutron banks keep their own TOF axes (no common-axis rebin) and d-spacing
  patterns are stored alongside TOF — nothing is discarded
- 2D images are stored by reference (path attributes) unless embedding was
  chosen at generation
"""

import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from .config import GENERATOR_NAME, GENERATOR_VERSION
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
# Writer
# ============================================================================

class NXSWriter:
    """Writes the canonical schema-3.0 NeXus/HDF5 file."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE,
                 include_2d_images: bool = False,
                 max_display_size: int = 0,
                 correlation_method: str = "absolute",
                 title: Optional[str] = None,
                 sample_name: Optional[str] = None):
        self.data_source = data_source
        self.include_2d_images = include_2d_images
        self.max_display_size = max_display_size  # 0 = full resolution
        self.correlation_method = correlation_method
        self.title = title
        self.sample_name = sample_name
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
            f.attrs['file_time'] = pd.Timestamp.now().isoformat()

            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'
            entry.attrs['data_source'] = self.data_source.value
            entry.attrs['correlation_method'] = self.correlation_method
            entry.attrs['generator'] = GENERATOR_NAME
            entry.attrs['generator_version'] = GENERATOR_VERSION
            entry.attrs['total_scans'] = len(scans)
            entry.attrs['twod_included'] = self.include_2d_images
            if self.include_2d_images and self.max_display_size > 0:
                entry.attrs['twod_max_display_size'] = self.max_display_size

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

    def _write_entry_fields(self, entry: h5py.Group, scans: List[Scan],
                            harvest: Dict[str, Any]) -> None:
        """Entry title, time window, and identifier (user values win)."""
        title = (self.title or harvest.get('run_title') or harvest.get('title')
                 or 'operando diffraction experiment')
        entry.create_dataset('title', data=str(title))

        timestamps = [s.timestamp for s in scans if s.timestamp]
        if timestamps:
            entry.create_dataset('start_time', data=str(min(timestamps)))
            ends = [s.neutron_end for s in scans if s.neutron_end]
            entry.create_dataset('end_time', data=str(max(ends) if ends else max(timestamps)))

        identifier = harvest.get('proposal') or harvest.get('experiment_identifier')
        if identifier:
            entry.create_dataset('experiment_identifier', data=str(identifier))

    def _write_instrument(self, entry: h5py.Group, scans: List[Scan],
                          harvest: Dict[str, Any]) -> None:
        """NXinstrument: source, crystal, detector, and raw-harvest collections."""
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'
        inst.create_dataset('name', data=str(harvest.get('instrument_name', 'unknown')))

        source = inst.create_group('source')
        source.attrs['NX_class'] = 'NXsource'
        source.create_dataset('name', data=str(harvest.get('source_name', 'unknown')))
        source.create_dataset('type', data=str(harvest.get('source_type', 'unknown')))
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
            'distance': harvest.get('distance_m') or harvest.get('detector_distance'),
            'x_pixel_size': harvest.get('x_pixel_size_m'),
            'y_pixel_size': harvest.get('y_pixel_size_m'),
            'beam_center_x': harvest.get('beam_center_x'),
            'beam_center_y': harvest.get('beam_center_y'),
        }
        for name, value in detector_fields.items():
            if value is not None:
                try:
                    detector.create_dataset(name, data=float(value))
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
        start = scan.neutron_start or scan.timestamp
        if start:
            sub.create_dataset('start_time', data=str(start))
        if scan.neutron_end:
            sub.create_dataset('end_time', data=str(scan.neutron_end))

        # Instrument: linked, not copied
        sub['instrument'] = h5py.SoftLink('/entry/instrument')

        self._write_environment(sub, scan)
        self._write_monitor(sub, scan)

        if scan.oned or scan.twod:
            self._write_xrd_data(sub, scan, reader_factory)

        if scan.neutron_files:
            self._write_neutron_banks(sub, scan, reader_factory)

    @staticmethod
    def _write_environment(sub: h5py.Group, scan: Scan) -> None:
        """Electrochemical state at acquisition time (the operando extension)."""
        env = sub.create_group('environment')
        env.attrs['NX_class'] = 'NXcollection'

        if scan.timestamp:
            env.create_dataset('scan_timestamp', data=str(scan.timestamp))
        if scan.echem is not None:
            env.create_dataset('voltage (V)', data=scan.echem)
        if scan.current is not None:
            env.create_dataset('current (mA)', data=scan.current)
        if scan.echem_timestamp:
            env.create_dataset('voltage_timestamp', data=str(scan.echem_timestamp))
        if scan.timestamp_for_correlation is not None:
            env.create_dataset('midpoint_adjusted_timestamp',
                               data=str(scan.timestamp_for_correlation))
        if scan.exposure_time is not None:
            env.create_dataset('exposure_time', data=scan.exposure_time)
        if scan.logbook and scan.logbook.get('full_line'):
            env.create_dataset('logbook_entry', data=str(scan.logbook['full_line']))

    def _write_monitor(self, sub: h5py.Group, scan: Scan) -> None:
        """Beam monitor record: EDF counters when available, timer fallback."""
        monitor = sub.create_group('monitor')
        monitor.attrs['NX_class'] = 'NXmonitor'

        counters: Dict[str, float] = {}
        if scan.twod and str(scan.twod).lower().endswith('.edf'):
            counters = extract_edf_scan_fields(str(scan.twod)).get('monitor', {})

        monitor.create_dataset('mode', data='timer')
        if scan.exposure_time is not None:
            monitor.create_dataset('preset', data=float(scan.exposure_time))

        # Best available integral: first positive counter in preference order
        integral = None
        for key in ('Monitor', 'pilct1', 'pilai1', 'Intensity1'):
            val = counters.get(key)
            if val:
                integral = float(val)
                break
        if integral is not None:
            monitor.create_dataset('integral', data=integral)

        # Keep every raw counter — nothing harvested is discarded
        for key, val in counters.items():
            try:
                monitor.create_dataset(key, data=float(val))
            except Exception:
                pass

    def _write_xrd_data(self, sub: h5py.Group, scan: Scan,
                        reader_factory: DataReaderFactory) -> None:
        """NXdata with the 1D pattern (plus errors) and the 2D image layer."""
        data_group = sub.create_group('data')
        data_group.attrs['NX_class'] = 'NXdata'

        if scan.oned:
            try:
                arr = reader_factory.read_file(scan.oned)
                data_group.create_dataset('polar_angle', data=arr[:, 0])
                data_group.create_dataset('data', data=arr[:, 1])
                if arr.shape[1] >= 3:
                    data_group.create_dataset('errors', data=arr[:, 2])
                data_group.attrs['signal'] = 'data'
                data_group.attrs['axes'] = 'polar_angle'
                data_group.attrs['oned_source_file'] = os.path.basename(scan.oned)
            except Exception as e:
                logger.error(f"Error reading 1D data for scan {scan.scan_num}: {e}")
                data_group.attrs['oned_source_file'] = os.path.basename(scan.oned)

        if scan.twod:
            self._write_2d_data(data_group, scan)

    def _write_2d_data(self, data_group: h5py.Group, scan: Scan) -> None:
        """Embed or reference the 2D detector image depending on configuration."""
        twod_path = str(scan.twod)
        basename = os.path.basename(twod_path)
        ext = os.path.splitext(basename)[1].lower()

        is_hdf = (ext == ".hdf")
        is_edf = (ext == ".edf")

        data_group.attrs["twod_source"] = basename
        data_group.attrs["twod_source_path"] = twod_path
        data_group.attrs["twod_is_hdf"] = is_hdf
        data_group.attrs["twod_is_edf"] = is_edf

        if self.max_display_size > 0:
            data_group.attrs["twod_max_display_size"] = self.max_display_size
        else:
            data_group.attrs["twod_max_display_size"] = "full_resolution"

        if not self.include_2d_images:
            data_group.attrs["twod_embedded"] = False
            return

        try:
            if is_hdf:
                hdf_reader = HDFReader(max_display_size=self.max_display_size)
                data_2d = hdf_reader.read(twod_path)
                data_group.attrs["twod_original_shape"] = hdf_reader.original_shape
            else:
                data_2d = DataReaderFactory.read_file(twod_path)
                data_group.attrs["twod_original_shape"] = data_2d.shape

            data_group.create_dataset("twod_image", data=data_2d)
            data_group.attrs["twod_embedded"] = True

        except Exception as e:
            logger.error(f"Error embedding 2D data for scan {scan.scan_num}: {e}")
            data_group.attrs["twod_embedded"] = False

    def _write_neutron_banks(self, sub: h5py.Group, scan: Scan,
                             reader_factory: DataReaderFactory) -> None:
        """Per-bank NXdata groups. Banks keep their own TOF axes and d-spacing
        patterns are stored alongside TOF (keep-all-data extension)."""
        for meas_num, meas_files in scan.neutron_files.items():
            bank = sub.create_group(f'bank_{meas_num}')
            bank.attrs['NX_class'] = 'NXdata'
            bank.attrs['measurement_number'] = meas_num

            for key, (x_name, y_name, e_name) in {
                'tof': ('time_of_flight', 'data', 'errors'),
                'd': ('d_spacing', 'd_data', 'd_errors'),
            }.items():
                if key not in meas_files:
                    continue
                path = meas_files[key]
                try:
                    arr = reader_factory.read_file(path, is_neutron=True)
                    x_ds = bank.create_dataset(x_name, data=arr[:, 0])
                    bank.create_dataset(y_name, data=arr[:, 1])
                    if arr.shape[1] >= 3:
                        bank.create_dataset(e_name, data=arr[:, 2])

                    header = parse_mantid_header(path)
                    if header.get('x_unit'):
                        x_ds.attrs['units'] = header['x_unit']
                    if header.get('spectrum') is not None:
                        bank.attrs[f'{key}_spectrum'] = header['spectrum']
                    if header.get('y_unit'):
                        bank.attrs[f'{key}_y_unit'] = header['y_unit']
                except Exception as e:
                    logger.error(f"Error reading {key} data: {e}")
                bank.attrs[f'{key}_source_file'] = os.path.basename(path)

            if 'time_of_flight' in bank:
                bank.attrs['signal'] = 'data'
                bank.attrs['axes'] = 'time_of_flight'

    # --- electrochemistry layers ---

    @staticmethod
    def _write_operando_echem(entry: h5py.Group, echem_df: pd.DataFrame) -> None:
        """Full cycling protocol as parallel timestamp/voltage/current arrays."""
        if echem_df is None or echem_df.empty:
            return

        echem_group = entry.create_group('operando_electrochemistry')
        echem_group.attrs['NX_class'] = 'NXdata'

        # Use .to_numpy(dtype=object) to avoid PyArrow-backed arrays in pandas >= 3.0
        timestamps = echem_df['timestamp'].astype(str).to_numpy(dtype=object)
        echem_group.create_dataset('timestamps', data=timestamps)
        echem_group.create_dataset('voltage (V)',
                                   data=echem_df['echem_data'].to_numpy(dtype=float))

        if 'current' in echem_df.columns:
            current_values = echem_df['current'].to_numpy(dtype=float)
            echem_group.create_dataset('current (mA)', data=current_values)

    def _write_standard_echem(self, entry: h5py.Group,
                              standard_echem_files: Optional[List[str]]) -> None:
        """Parse and store each additional echem file as a file_N dataset."""
        if not standard_echem_files:
            return

        std_echem_container = entry.create_group('standard_electrochemistry')
        std_echem_container.attrs['NX_class'] = 'NXcollection'
        std_echem_container.attrs['num_files'] = len(standard_echem_files)

        for idx, std_echem_path in enumerate(standard_echem_files, start=1):
            if not os.path.isfile(std_echem_path):
                continue

            standard_echem_df = self.echem_parser.parse(std_echem_path)
            if standard_echem_df is None or standard_echem_df.empty:
                continue

            group_name = f'file_{idx:03d}'
            std_group = std_echem_container.create_group(group_name)
            std_group.attrs['NX_class'] = 'NXdata'
            std_group.attrs['source_file'] = os.path.basename(std_echem_path)

            std_timestamps = standard_echem_df['timestamp'].astype(str).to_numpy(dtype=object)
            std_group.create_dataset('timestamps', data=std_timestamps)
            std_group.create_dataset('voltage (V)',
                                     data=standard_echem_df['echem_data'].to_numpy(dtype=float))

            if 'current' in standard_echem_df.columns:
                current_vals = standard_echem_df['current'].to_numpy(dtype=float)
                if not np.all(np.isnan(current_vals)):
                    std_group.create_dataset('current (mA)', data=current_vals)

            logger.info(f"Added standard electrochemistry file {idx}:"
                        f" {os.path.basename(std_echem_path)} ({len(standard_echem_df)} points)")
