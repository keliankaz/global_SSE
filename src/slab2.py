# %%
from __future__ import annotations
import pandas as pd
import numpy as np
import copy
from sklearn.neighbors import BallTree
from pathlib import Path
import utm
import warnings
from typing import Union, Literal

base_dir = Path(__file__).parents[1]
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

SLAB_PROPERTIES = [
    "dep",  # depth
    "dip",  # dip
    "str",  # strike
    "thk",  # slab thickness
    "unc",  # uncertainty
]

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
        property: str = "dep",
        date: Union[list, str] = SLAB_MODEL_DATE,
    ):
        assert name in ALL_SLABS.keys(), f"Slab name {name} not in {ALL_SLABS}"
        assert (
            property in SLAB_PROPERTIES
        ), f"Slab property {property} not in {SLAB_PROPERTIES}"

        self.name = name
        self.path = path

        # The slab xyz files have the following format:
        # {name}_slab2_{property}_{date}.xyz
        # The date is one of SLAB_MODEL_DATE.
        # Check which date in SLAB_MODEL_DATE is available:
        if type(date) == list:
            for idate in date:
                self.file = path / f"{name}_slab2_{property}_{idate}.xyz"
                if self.file.exists():
                    break

        assert self.file.exists(), f"Slab file {self.file} does not exist"

        self.raw_xyz = pd.read_csv(
            self.file, header=None, names=["longitude", "latitude", "z"]
        )

        # xyz files have longitude values from 0 to 360 degrees. Convert to -180 to 180.
        self._xyz = copy.deepcopy(self.raw_xyz)
        self._xyz["longitude"] = self.raw_xyz["longitude"].apply(
            lambda x: x - 360 if x > 180 else x
        )

        # drop NaNs rows
        self._xyz = self._xyz.dropna()

        east, north, zone, letter = self.force_ll2utm(
            self._xyz["latitude"].values, self._xyz["longitude"].values
        )

        # create new dataframe with utm coordinates, positive depth in meters
        self.utm_geometry = pd.DataFrame(
            {"easting": east, "northing": north, "depth": -self._xyz["z"].values * 1000}
        )
        self.utm_zone = zone
        self.utm_letter = letter

    def densify(self, step_meter: float = 1000):
        """Densify the slab by linear interpolation between coordinates of the geometry."""

        raise NotImplementedError("densify() not implemented yet")

    def distance(
        self,
        xyz,
        from_latlon: bool = False,
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
            east, north, _, _ = self.force_ll2utm(
                xyz[:, 0], xyz[:, 1], self.utm_zone, self.utm_letter
            )
            xyz = np.atleast_2d(np.column_stack([east, north, xyz[:, 2]]))

        tree = BallTree(
            self.utm_geometry[["easting", "northing", "depth"]].values,
        )

        query = tree.query(xyz, return_distance=True)[
            0
        ].squeeze()  # [0] is the distance [1] is index

        if distance_unit == "km":
            query /= 1000.0

        return query

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


# %%
