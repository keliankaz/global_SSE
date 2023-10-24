import pandas as pd
import numpy as np
from pathlib import Path

_base_dir = Path(__file__).parents[2]


class Arcs:
    """
    A class to load and store data about subduction arcs from the analysis by
    Syracuse et al., 2010.
    """

    def __init__(self):
        self.file = "Syracuse2010.csv"
        self.dirname = _base_dir / "Datasets" / "Arcs"
        self.path = self.dirname / self.file
        self._column_names = [
            "Name",
            "Lon",
            "Lat",
            "H",
            "Arc_trench_distance",
            "Slab_dip",
            "Vc",
            "Age",
            "Descent_rate",
            "Thermal_parameter",
            "Sediment_thickness",
            "Subducted_sediment_thickness",
            "Upper_plate_type",
            "Upper_plate_thickness",
            "Upper_plate_age",
        ]

        self._column_units = [
            "",
            "°",
            "°",
            "km",
            "km",
            "°",
            "km/Ma",
            "Ma",
            "km/Ma",
            "km",
            "km",
            "km",
            "",
            "km",
            "Ma",
        ]

        self.units = dict(zip(self._column_names, self._column_units))

        self._rawdata = pd.read_csv(
            self.path,
            skiprows=1,
            names=self._column_names,
            sep="\t",
        )

        for col in self._column_names:
            setattr(self, col, self._rawdata[col].values)

        self.Thermal_parameter = self.Thermal_parameter * 100

    # static methods
    @staticmethod
    def get_slab_surface_temprature(depth, thermal_parameter, age):
        """
        Calculate the slab surface temprature from the slab depth and thermal parameter.

        This follows Molnar and England (1995) equatiuon 15 and the assumptions therein.
        """

        b = 0.9
        km = 3.3
        ks = 3.3
        kappa = 1e-6
        Sp = 1 + b * km / ks * np.sqrt(thermal_parameter * depth / age / kappa)

        T0 = 1350

        return (
            (Sp - 1) / (b * Sp) * T0 / np.sqrt(np.pi * (1 + thermal_parameter / depth))
        )


# %% Test
if __name__ == "__main__":
    arcs = Arcs()
    print(arcs._rawdata.head())
    print(arcs.units)

    [print(name) for name in arcs.Name]

    assert len(arcs._column_names) == len(arcs._column_units)

# %%
