from __future__ import annotations
import pandas as pd
import os
from pathlib import Path
from src.data import EarthquakeCatalog

base_dir = Path(__file__).parents[2]


class ESTEarthquakeCatalog(EarthquakeCatalog):
    def __init__(self) -> EarthquakeCatalog:
        self.dir_name = os.path.join(
            os.path.join(base_dir, "Datasets/Seismicity_datasets/Japan")
        )
        self.file_name = "est.csv"
        _catalog = self.read_catalog(self.dir_name, self.file_name)

        super().__init__(_catalog)

    @staticmethod
    def read_catalog(dir_name, file_name):
        """
        Reads in a EST catalog and returns a pandas dataframe. Read each file in
         directory and concatenate them into one dataframe.
        """
        full_file_name = os.path.join(dir_name, file_name)

        column_names = [
            "datetime",
            "lat",
            "lon",
            "depth",
            "mag",
            "hypoflag_est",
            "dotime_est",
            "dolat_est",
            "dolon_est",
            "dodep_est",
            "std_ditp_est",
            "std_dits_est",
            "ppick_est",
            "19_est",
            "20_est",
            "bothps_est",
            "22_est",
            "23_est",
            "24_est",
            "pspicknear_est",
            "26_est",
            "27_est",
            "fname_est",
            "event_idx_est",
            "event_idx_man",
            "datetime_man",
            "lat_man",
            "lon_man",
            "dep_man",
            "mag_man",
            "dt",
            "dlat",
            "dlon",
            "dho",
            "ddep",
            "dmag",
        ]

        df = pd.read_csv(
            full_file_name,
            header=0,
            sep=",",
            skiprows=1,
            names=column_names,
            engine="python",
        )

        df["time"] = pd.to_datetime(df.datetime)

        return df
