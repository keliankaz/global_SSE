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


class JapanSlowslipDatabase(SlowSlipCatalog):
    def __init__(self, files=None):
        self.name = "Japan"
        self.dir_name = os.path.join(
            base_dir, "Datasets/Slow_slip_datasets/Japan/SlowEqDatabase"
        )
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

        # For catalogs that do not have a specified magnitude column,
        # (currently just Takagi, 2016) but do have the slip, and dimensions
        # we can estimate the magntidue:
        I = (
            df_all.mag.isna()
            & ~df_all.slip.isna()
            & ~df_all.width.isna()
            & ~df_all.length.isna()
        )
        df_all.loc[I, "mag"] = (
            2
            / 3
            * np.log10(
                3e10
                * df_all.slip[I]  # rigidity [Pa]
                * df_all.width[I]  # slip [m]
                * 1e3
                * df_all.length[I]  # width [m]
                * 1e3  # length [m]
            )
            - 6.0
        )

        return df_all


class RoussetSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Mexico (Rousset et al. 2017)"
        self.ref = "Rousset et al. 2017"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Mexico/")
        self.file_name = "Rousset2017.txt"
        _catalog = self.read_catalog(self.dir_name, self.file_name)
        self.catalog = self._add_time_column(_catalog, "time")
        self.catalog["ref"] = self.ref

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

        slab = Slab("cam")
        # used the geometry of slab2.0 to derive a depth for each event
        depth = slab.interpolate(
            "depth", lat=df["lat"].to_numpy(), lon=df["lon"].to_numpy()
        )
        df["depth"] = depth / 1000  # convert to km

        return df


class LouSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Japan trench (Lou et al. in prep.)"
        self.ref = "Lou et al. in prep."
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Japan/")
        self.file_name = "Lou2023.csv"
        _catalog = self.read_catalog(self.dir_name, self.file_name)
        self.catalog = self._add_time_column(_catalog, "time")
        self.catalog["ref"] = self.ref

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
            sep=",",
            skiprows=1,
            names=[
                "ID",
                "date",
                "lon",
                "lat",
                "duration",
                "mag",
            ],
            engine="python",
        )
        df.duration = df.duration * 24 * 3600  # convert from days to seconds

        slab = Slab("izu")
        # used the geometry of slab2.0 to derive a depth for each event
        depth = slab.interpolate(
            "depth", lat=df["lat"].to_numpy(), lon=df["lon"].to_numpy()
        )
        df["depth"] = depth / 1000  # convert to km

        return df


class JapanSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Japan"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Japan/")

        # Digitized with QGIS
        self.japan_trench_bounding_polygon = [
            (149.08366037274828386, 46.70988278437626207),
            (150.61391815425150753, 44.3838909564913493),
            (149.98141160456350462, 44.03703252601728479),
            (148.38994351180014064, 43.09847442002863716),
            (147.00250978990388262, 42.56798505577418013),
            (145.34983138588040674, 41.99668881734631043),
            (143.90118735272400841, 40.38481728749624011),
            (143.90118735272400841, 39.22182137355378018),
            (143.41150486264297115, 38.48729763843223139),
            (143.22787392886257862, 36.87542610858216108),
            (140.26937555128967006, 32.97836962502059777),
            (139.55525525325484182, 33.77410367140227976),
            (139.14718651152062989, 34.67185490321750763),
            (138.5758902730927673, 35.24315114164538443),
            (138.51467996183262699, 36.50816424102138313),
            (140.02453430624916564, 39.26262824772719995),
            (141.08551303475806549, 41.97628538025959699),
            (143.22787392886257862, 44.01662908893057136),
            (144.61530765075883664, 45.54688687043380213),
            (149.08366037274828386, 46.7098827843762620),
        ]
        self.nankai_trough_bounding_polygon = [
            (138.59248657995570397, 35.27462259320842719),
            (139.1070635303310894, 34.73860493656739834),
            (139.53587765564390111, 33.75233244834791435),
            (139.66452189323774746, 32.5516528974720174),
            (136.08392394687570004, 32.5516528974720174),
            (132.58908882557619791, 31.32953264033047702),
            (131.52777386542697968, 30.80423533682227344),
            (130.84167126492647526, 30.99720169321304297),
            (130.52006067094185937, 31.70474499997919438),
            (130.92743408998904897, 32.9268652571207312),
            (132.57836847244340106, 34.06322268919970497),
            (135.15125322432032817, 34.94229164609099314),
            (137.61693444486903104, 35.52119071526330174),
            (138.06718927644749328, 35.69271636538842785),
            (138.59248657995570397, 35.2746225932084271),
        ]
        self.ryukyu_trench_bounding_polygon = [
            (130.41094287826504683, 31.40796420301290937),
            (131.82559511444495115, 30.31088695862848681),
            (131.1904451308539592, 28.17447337745882407),
            (129.80466334847363896, 25.98031888868997896),
            (124.781204387344971, 23.06440305493138965),
            (122.61592035237572418, 22.89118033213384962),
            (121.69206583078884876, 23.38197804672687852),
            (120.62385904020401028, 24.59453710630965872),
            (121.3456203851937687, 26.18241206528711018),
            (123.91509077335727795, 26.73095068747932146),
            (125.87828163172939355, 27.04852567927481033),
            (128.21678838949620172, 29.70460742883709671),
            (129.89127470987241963, 31.63892783340962822),
            (130.41094287826504683, 31.40796420301290937),
        ]

        _combined = (
            JapanSlowslipDatabase() + LouSlowSlipCatalog() + ChenSlowSlipCatalog()
        )
        self.catalog = _combined.catalog

        super().__init__(self.catalog)

    def get_japan_trench(self) -> SlowSlipCatalog:
        """
        Returns a catalog of slow slip events in the Japan trench.
        """

        new = self.get_polygon_slice(self.japan_trench_bounding_polygon)
        new.name = "Japan Trench"

        return new

    def get_nankai_trough(self) -> SlowSlipCatalog:
        """
        Returns a catalog of slow slip events in the Nankai trough.
        """
        new = self.get_polygon_slice(self.nankai_trough_bounding_polygon)
        new.name = "Nankai Trough"

        return new

    def get_ryukyu_trench(self) -> SlowSlipCatalog:
        """
        Returns a catalog of slow slip events in the Ryukyu trench.
        """

        new = self.get_polygon_slice(self.ryukyu_trench_bounding_polygon)
        new.name = "Ryukyu Trench"

        return new


class XieSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Costa Rica (Xie et al. 2020)"
        self.ref = "Xie et al. 2020"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Costa_Rica/"),
        )
        self.file_name = "Xie2020.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )
        self.catalog["ref"] = self.ref

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
        self.ref = "Williams et al., in prep."
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/New_Zealand/"),
        )
        self.file_name = "v005_sf_nm_test018_m1_catalog.txt"

        raw_catalog = self.read_catalog(os.path.join(self.dir_name, self.file_name))
        super().__init__(raw_catalog)

        self.catalog["ref"] = self.ref

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

        # We consider the event centroid to be the mid-point between the start and end times
        df["Start_time"] = pd.to_datetime(
            df[["Start_year", "Start_month", "Start_day"]].apply(
                lambda x: "-".join(x.astype(str)), axis=1
            )
        )

        df["End_time"] = pd.to_datetime(
            df[["End_year", "End_month", "End_day"]].apply(
                lambda x: "-".join(x.astype(str)), axis=1
            )
        )

        df["time"] = df["Start_time"] + (df["End_time"] - df["Start_time"]) / 2

        df["duration"] = (df["End_time"] - df["Start_time"]).dt.seconds

        return df


class IkariSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "New Zealand (Ikari et al. 2020)"
        self.ref = "Ikari et al. 2020"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/New_Zealand"),
        )
        self.file_name = "ikari2020.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )
        self.catalog["ref"] = self.ref

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
        self.ref = "Michel et al. 2018"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Cascadia"),
        )
        self.file_name = "Michel2018.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )
        self.catalog["ref"] = self.ref

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
        self.name = "Taiwan (Chen et al. 2018)"
        self.ref = "Chen et al. 2018"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Taiwan"),
        )
        self.file_name = "Chen2018.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )
        self.catalog["ref"] = self.ref

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
