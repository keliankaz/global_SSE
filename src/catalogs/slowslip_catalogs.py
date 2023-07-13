# %%
from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path
from src.data import SlowSlipCatalog, Slab
from src.data.utils import DAY_PER_YEAR, SEC_PER_DAY


base_dir = Path(__file__).parents[2]


class JapanSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self, files=None):
        self.name = "Japan"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Japan/")
        _catalog = self.read_catalog(self.dir_name, files)
        self.files = files
        self.catalog = self._add_time_column(_catalog, "time")

        super().__init__(self.catalog)

    @staticmethod
    def _add_time_column(df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(df[["year", "month", "day"]])
        return df

    @staticmethod
    def read_catalog(dir_name, files=None):
        """
        Reads in a catalog of slow slip events in Japan and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Japan/ directory and concatenate them into one dataframe.
        """
        if files is None:
            directory_content = os.listdir(dir_name)
            files = [
                f
                for f in directory_content
                if os.path.isfile(os.path.join(dir_name, f))
            ]

        if type(files) is str:
            files = [files]
        elif type(files) is not list:
            raise TypeError("files must be a string or a list of strings.")

        for i, file in enumerate(files):
            df = pd.read_csv(
                os.path.join(dir_name, file),
                comment='"',
                na_values="\s",
                skipinitialspace=True,
            )
            if i == 0:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])

        # For some reason some of the days are set to 0. I am assuming that they can be set to 1.
        df_all.loc[df_all.day == 0, "day"] = 1

        # At the moment one of the catalogs, (Yano 2022) has no magnitudes. This column gets parsed as
        # a string. I am setting all fields that were read as a string of spaces to NaN.
        df_all = df_all.replace(r"^\s*$", np.nan, regex=True)

        # There are some duplicates in the catalog that we remove
        df_all = df_all.drop_duplicates(["year", "month", "day", "lat", "lon"])

        # Reset indices to avoid issues with duplicate indices
        df_all = df_all.reset_index()

        df_all["depth"] = df_all.dep

        return df_all


class RoussetSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Mexico (Rousset et al. 2017)"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Mexico/")
        self.file_name = "Rousset2017.txt"
        _catalog = self.read_catalog(self.dir_name, self.file_name)
        self.catalog = self._add_time_column(_catalog, "time")

        super().__init__(self.catalog)

    @staticmethod
    def _deciyear_to_datetime(deciyear):
        """
        Converts decimal year to datetime.
        """
        year = int(deciyear)
        rem = deciyear - year
        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
        )
        return result

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with date converted from deciyear.
        """

        for i, row in df.iterrows():
            df.loc[i, column] = self._deciyear_to_datetime(row["date"])

        return df

    @staticmethod
    def read_catalog(dir_name, file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """
        full_dir_name = os.path.join(os.path.dirname(__file__), dir_name)
        df = pd.read_csv(
            os.path.join(full_dir_name, file_name),
            header=0,
            sep="\s",
            skiprows=1,
            names=["lon", "lat", "date", "duration", "mag"],
            engine="python",
        )
        df.duration = df.duration * 24 * 3600  # convert from days to seconds

        return df


class XieSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Costa Rica (Xie et al. 2020)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Costa_Rica/"),
        )
        self.file_name = "Xie2020.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(df["Average date of GPS maximum speed"])

        return df

    @staticmethod
    def read_catalog(file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """

        columns = {
            "Named_SSEs": None,
            "Start": None,
            "End": None,
            "Average date of GPS maximum speed": "mm/day",
            "Date of maximum moment rate": None,
            "Maximum moment rate": "10^17 N m day^-1",
            "Total moment released": "10^19 N m",
            "mag": "Mw",
            "lon": "deg",
            "lat": "deg",
            "depth": "km",
        }

        df = pd.read_csv(
            file_name,
            skiprows=1,
            index_col=False,
            delimiter=",",
            names=columns.keys(),
            na_values="-",
        )

        # remove rows that have an nan value
        df = df.dropna()
        df = df.reset_index(drop=True)

        return df


class WilliamsSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "New Zealand (Williams et al., in prep.)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/New_Zealand/"),
        )
        self.file_name = "v005_sf_nm_test018_m1_catalog.txt"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(df[["year", "month", "day"]])

        return df

    @staticmethod
    def read_catalog(file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """

        df = pd.read_csv(
            file_name,
            index_col=False,
            delimiter="\t",
        )

        df["lat"] = df["Lat_geom"]
        df["lon"] = df["Lon_geom"]
        df["depth"] = -df["Z_geom"]  # Note minus sign
        df["mag"] = df["Mw"]
        df["year"] = df["Start_year"]
        df["month"] = df["Start_month"]
        df["day"] = df["Start_day"]

        # TODO: add duration to this catalog

        return df


class IkariSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "New Zealand (Ikari et al. 2020)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/New_Zealand"),
        )
        self.file_name = "ikari2020.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str).fillna("1")
        )

        return df

    @staticmethod
    def read_catalog(file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """

        df = pd.read_csv(
            file_name,
            skiprows=1,
            index_col=False,
            names=[
                "Event",
                "Month",
                "Year",
                "duration",
                "Depth_of_Maximum_Slip_km",
                "Maximum_Slip_mm",
                "Stress_Drop_kPa",
                "Peak_Slip_Velocity_cm_per_yr",
                "V_Vo",
                "Downdip_Width_km",
                "D_over_r",
                "mag",
                "Reference",
            ],
        )

        return df


class MichelSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Cascadia (Michel et al. 2018)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Cascadia"),
        )
        self.file_name = "Michel2018.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with date converted from deciyear.
        """

        for i, row in df.iterrows():
            df.loc[i, column] = self._deciyear_to_datetime(row["year_time"])

        return df

    @staticmethod
    def _deciyear_to_datetime(deciyear):
        """
        Converts decimal year to datetime.
        """
        year = int(deciyear)
        rem = deciyear - year
        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
        )
        return result

    @staticmethod
    def read_catalog(file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """

        df = pd.read_csv(
            file_name,
            skiprows=1,
            sep="\t",
            index_col=False,
            names=[
                "ID",
                "year_time",
                "mag",
                "lat",
                "lon",
                "SSE_number_michel2018",
                "min_start",
                "max_start",
                "min_end",
                "max_end",
                "start",
                "end",
            ],
        )

        # for the purpose of having one representative number for the duration
        # take the average of the maximum and the minumum duration
        df["duration"] = (
            ((df["max_end"] - df["min_start"]) + (df["min_end"] - df["max_start"]))
            / 2
            * DAY_PER_YEAR
            * SEC_PER_DAY
        )
        df["lon"] = -df["lon"]

        slab = Slab("cas")
        # used the geometry of slab2.0 to derive a depth for each event
        depth = slab.interpolate(
            "depth", lat=df["lat"].to_numpy(), lon=df["lon"].to_numpy()
        )
        df["depth"] = depth / 1000  # convert to km

        return df


class ChenSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Taiwan ( Chen et al. 2018)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Taiwan"),
        )
        self.file_name = "Chen2018.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with date converted from deciyear.
        """

        for i, row in df.iterrows():
            df.loc[i, column] = self._deciyear_to_datetime(row["mid_time_deciyear"])

        return df

    @staticmethod
    def _deciyear_to_datetime(deciyear):
        """
        Converts decimal year to datetime.
        """
        year = int(deciyear)
        rem = deciyear - year
        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
        )
        return result

    @staticmethod
    def read_catalog(file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """

        df = pd.read_csv(
            file_name,
            skiprows=1,
            sep=",",
            index_col=False,
            names=["start", "end", "lat", "lon", "mag"],
        )

        # for the purpose of having one representative number for the duration
        # take the average of the maximum and the minumum duration
        df["duration"] = (df["end"] - df["start"]) * DAY_PER_YEAR * SEC_PER_DAY
        df["mid_time_deciyear"] = (df["start"] + df["end"]) / 2

        slab = Slab("ryu")
        # used the geometry of slab2.0 to derive a depth for each event
        depth = slab.interpolate(
            "depth", lat=df["lat"].to_numpy(), lon=df["lon"].to_numpy()
        )
        df["depth"] = depth / 1000  # convert to km

        return df
