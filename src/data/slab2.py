from __future__ import annotations
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path
import utm
import warnings
from typing import Union, Literal
from functools import cached_property
from pyproj import Transformer

base_dir = Path(__file__).parents[2]
DATA_DIR = base_dir / "Datasets" / "Slab2" / "Slab2_TXT"

ALL_SLABS = {
    "cal": "Calabria",
    "cam": "Central_America",
    "cot": "Cotabato",
    "hin": "Hindu_Kush",
    "man": "Manila",
    "sco": "Scotia",
    "sul": "Sulawesi",
    "sam": "South_America",
    "cas": "Cascadia",
    "him": "Himalaya",
    "puy": "Puysegur",
    "mak": "Makran",
    "hal": "Halmahera",
    "kur": "Kuril",
    "mue": "Muertos",
    "alu": "Aleutian",
    "ryu": "Ryukyu",
    "phi": "Philippines",
    "ker": "Kermadec",
    "van": "Vanuatu",
    "png": "New_Guinea",
    "car": "Caribbean",
    "hel": "Hellenic",
    "pam": "Pamir",
    "sol": "Solomon",
    "sum": "Sumatra",
    "izu": "Izu_Bonin",
}

SLAB_PROPERTIES = {
    "dep": "depth",
    "dip": "dip",
    "str": "strike",
    "thk": "thickness",
    "unc": "uncertainty",
}

SLAB_MODEL_DATE = [
    "02.23.18",
    "02.24.18",
    "02.26.18",
]


class Slab:
    def __init__(
        self,
        name: str,
        path: Path = DATA_DIR,
        date: Union[list, str] = SLAB_MODEL_DATE,
    ):
        assert name in ALL_SLABS.keys(), f"Slab name {name} not in {ALL_SLABS}"

        self.name = name
        self.path = path
        self.date = date

        self.raw_xyz = self._get_property_xyz("dep")

        self.longitude = np.where(
            self.raw_xyz[:, 0] > 180, self.raw_xyz[:, 0] - 360, self.raw_xyz[:, 0]
        )
        self.latitude = self.raw_xyz[:, 1]
        self.depth = -self.raw_xyz[:, 2] * 1000

        self.easting, self.northing, self.utm_zone, self.utm_letter = self.force_ll2utm(
            self.latitude, self.longitude
        )

    def _get_property_xyz(self, property: str):
        if type(self.date) == list:
            for idate in self.date:
                file = self.path / f"{self.name}_slab2_{property}_{idate}.xyz"
                if file.exists():
                    break
        xyz = np.genfromtxt(
            file,
            delimiter=",",
            dtype=float,
            missing_values="NaN",
            filling_values=np.nan,
        )

        return xyz

    @cached_property
    def strike(self):
        return self._get_property_xyz("str")[:, 2]

    @cached_property
    def dip(self):
        return self._get_property_xyz("dip")[:, 2]

    @cached_property
    def thickness(self):
        return self._get_property_xyz("thk")[:, 2]

    @cached_property
    def uncertainty(self):
        return self._get_property_xyz("unc")[:, 2]

    def densify(self, step_meter: float = 1000):
        """Densify the slab by linear interpolation between coordinates of the geometry."""

        raise NotImplementedError("densify() not implemented yet")

    def distance(
        self,
        xyz,
        from_latlon: bool = True,  # else from ECEF
        depth_unit: Literal["km", "m"] = "m",
        distance_unit: Literal["km", "m"] = "m",
    ):
        """Calculates the distance between each point in xyz and the nearest point in the slab."""

        xyz = np.atleast_2d(xyz)
        assert xyz.shape[1] == 3, "xyz must have 3 columns"

        if np.any(xyz[:, 2] < 0):
            warnings.warn("xyz contains negative depths")
        if depth_unit == "km":
            xyz[:, 2] = xyz[:, 2] * 1000

        if from_latlon:
            ECEF_xyz = np.atleast_2d(
                np.column_stack(
                    self.gps_to_ecef_pyproj(xyz[:, 0], xyz[:, 1], -xyz[:, 2])
                )
            )

        slab_ECEF_xyz = np.array(
            self.gps_to_ecef_pyproj(self.latitude, self.longitude, -self.depth)
        ).T
        slab_ECEF_xyz = slab_ECEF_xyz[~np.isnan(slab_ECEF_xyz[:, 2]), :]
        tree = BallTree(slab_ECEF_xyz)

        query = tree.query(xyz, return_distance=True)[
            0
        ].squeeze()  # [0] is the distance [1] is index

        if distance_unit == "km":
            query /= 1000.0

        return query

    def interpolate(
        self, property: str, lat: np.ndarray, lon: np.ndarray
    ):  # TODO: use ECEF instead of lat/lon
        """Interpolates the querried property at the given lat, lon using a nearest neighbor search.
        Assumes that the queried location is within the slab geometry.

        Args:
            property: property specified in SLAB_PROPERTIES
            lat: latitudes
            lon: longitudes

        Returns:
            interpolated property
        """
        mask = ~np.isnan(getattr(self, property))
        slab_xyz = np.array(
            [v[mask] for v in [self.latitude, self.longitude, getattr(self, property)]]
        ).T
        tree = BallTree(np.deg2rad(slab_xyz[:, :2]), metric="haversine")

        query = tree.query(
            np.deg2rad(np.column_stack([lat, lon])),
            return_distance=True,
        )[
            1
        ].squeeze()  # [0] is the distance [1] is index

        property = slab_xyz[query, 2]

        return property

    @staticmethod
    def force_ll2utm(lat, lon, force_zone_number=False, force_zone_letter=False):
        # Hack to force utm to use the same zone for all points
        # (note I posted a stackoverflow question about this)
        if not force_zone_letter or not force_zone_letter:
            _, _, zone, letter = utm.from_latlon(np.mean(lat), np.mean(lon))
        else:
            zone, letter = [force_zone_number, force_zone_letter]
        I_positives = lat >= 0

        if np.sum(~I_positives) > 0 and np.sum(I_positives) > 0:
            east_pos, north_pos, _, _ = utm.from_latlon(
                lat[I_positives],
                lon[I_positives],
                force_zone_number=zone,
                force_zone_letter=letter,
            )
            east_neg, north_neg, _, _ = utm.from_latlon(
                lat[~I_positives],
                lon[~I_positives],
                force_zone_number=zone,
                force_zone_letter=letter,
            )

            east = np.concatenate((east_pos, east_neg))
            north = np.concatenate((north_pos, north_neg - 10000000))
        else:
            east, north, _, _ = utm.from_latlon(
                lat, lon, force_zone_number=zone, force_zone_letter=letter
            )
        return east, north, zone, letter

    @staticmethod
    def gps_to_ecef_pyproj(lat, lon, alt):
        transformer = Transformer.from_crs(
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        )
        return transformer.transform(lon, lat, alt, radians=False)


class AllSlabs(Slab):
    def __init__(self):
        slab_array = []
        for name in ALL_SLABS.keys():
            slab_array.append(Slab(name))

        self._slab_array = slab_array

        self.latitude = np.concatenate([slab.latitude for slab in self._slab_array])
        self.longitude = np.concatenate([slab.longitude for slab in self._slab_array])

    def collect_property(self, property: str):
        return np.concatenate([getattr(slab, property) for slab in self._slab_array])

    @cached_property
    def depth(self):
        return self.collect_property("depth")

    @cached_property
    def dip(self):
        return self.collect_property("dip")

    @cached_property
    def strike(self):
        return self.collect_property("strike")

    @cached_property
    def thickness(self):
        return self.collect_property("thikness")

    @cached_property
    def uncertainty(self):
        return self.collect_property("uncertainty")


if __name__ == "__main__":
    all_slabs = AllSlabs()
