"""
Instrument profiles: per-instrument constants not recorded in raw data files.

Profiles supply NXmonopd/NXtofnpd required fields. Harvested file metadata
always wins over profile values; profiles only fill the gaps. Keys follow the
NeXus field names they populate.

Neutron geometry (NXoperando_tofnpd): `pre_sample_flightpath_m` and the
per-bank `detector_banks` mapping ({bank: {distance_m, polar_angle_deg,
azimuthal_angle_deg}}) feed entry/pre_sample_flightpath and the
instrument/detector arrays. All-or-skip rule: a geometry array (distance,
polar_angle, azimuthal_angle) is written only when EVERY bank supplies that
key; detector_number is written iff at least one array is. None placeholders
are skipped, so nothing is fabricated until real numbers are supplied.
"""

from typing import Any, Dict, Optional

from .model import DataSourceType

# Per-instrument profiles, keyed by harvested instrument name (lowercase)
INSTRUMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "polaris": {
        "source_name": "ISIS",
        "source_type": "Spallation Neutron Source",
        "probe": "neutron",
        "instrument_name": "POLARIS",
        # TODO(POLARIS geometry): fill in the moderator-to-sample flight
        # path (m) and per-bank detector distance (m) / polar and azimuthal
        # angles (degrees). None placeholders are skipped by the writer.
        "pre_sample_flightpath_m": None,
        "detector_banks": {
            bank: {"distance_m": None, "polar_angle_deg": None,
                   "azimuthal_angle_deg": None}
            for bank in (1, 2, 3, 4, 5)
        },
    },
    "i11-1": {
        "source_name": "Diamond Light Source",
        "source_type": "Synchrotron X-ray Source",
        "probe": "x-ray",
        "instrument_name": "i11-1",
        # Wavelength comes from the .poni calibration at the beamline and is
        # not stored in the scan .nxs; supply per-experiment when known.
    },
}

# Fallbacks by data source when no instrument name could be harvested
DEFAULT_PROFILES: Dict[DataSourceType, Dict[str, Any]] = {
    DataSourceType.INHOUSE: {
        "source_name": "laboratory X-ray source",
        "source_type": "Fixed Tube X-ray",
        "probe": "x-ray",
        "instrument_name": "in-house diffractometer",
    },
    DataSourceType.SYNCHROTRON: {
        "source_name": "synchrotron",
        "source_type": "Synchrotron X-ray Source",
        "probe": "x-ray",
        "instrument_name": "synchrotron beamline",
    },
    DataSourceType.NEUTRON: {
        "source_name": "neutron source",
        "source_type": "Spallation Neutron Source",
        "probe": "neutron",
        "instrument_name": "neutron diffractometer",
    },
}


def get_profile(data_source: DataSourceType,
                instrument_name: Optional[str] = None) -> Dict[str, Any]:
    """Profile for a harvested instrument name, falling back to the
    data-source default; a copy safe to update with harvested values."""
    profile = dict(DEFAULT_PROFILES.get(data_source, {}))
    if instrument_name:
        named = INSTRUMENT_PROFILES.get(str(instrument_name).strip().lower())
        if named:
            profile.update(named)
        else:
            profile["instrument_name"] = str(instrument_name)
    return profile
