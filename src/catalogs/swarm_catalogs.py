from __future__ import annotations
import pandas as pd
import os
from pathlib import Path
from src.data import SwarmCatalog

base_dir = Path(__file__).parents[2]


class NishikawaSwarmCatalog(SwarmCatalog):
    def __init__(self):
        self.dir_name = os.path.join(base_dir, "Datasets/Swarm_datasets/Global")
        self.file_name = "nishikawa2017S3.txt"

        _catalog = self.read_catalog(os.path.join(self.dir_name, self.file_name))
        self._time_columns = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
        _catalog = self._add_time_column(_catalog, "time")
        _catalog[
            "mag"
        ] = 4.5  # Use dummy magnitude (the catalog completeness used for the analysis in Nishikawa and Ide 2017)
        super().__init__(catalog=_catalog)

    @staticmethod
    def read_catalog(filename):
        """
        Reads in a global swarm catalog and returns a pandas dataframe, with the following columns:

        Region_name Time_period Cluseter_id Lon(deg) Lat(deg) Depth(km) Magnitude Year Month Day Hour Minute Second
        """

        df = pd.read_csv(
            filename,
            skiprows=1,
            sep=" ",
            index_col=False,
            names=[
                "Region_name",
                "Time_period",
                "Cluseter_id",
                "lon",
                "lat",
                "depth",
                "Magnitude",
                "Year",
                "Month",
                "Day",
                "Hour",
                "Minute",
                "Second",
            ],
        )

        # remove 360 from longitudes that are greater than 180 degrees
        df.loc[df["lon"] > 180, "lon"] = df.loc[df["lon"] > 180, "lon"] - 360

        return df
