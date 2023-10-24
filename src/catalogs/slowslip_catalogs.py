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
        self.region = "Japan"
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
        self.region = "Mexico"
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


class ElYousfiSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "El Yousfi  et al., 2022"
        self.region = "Mexico"
        self.ref = "El Yousfi  et al., 2022"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Mexico/")
        self.file_name = "ElYousfi2022.txt"
        _catalog = self.read_catalog(self.dir_name, self.file_name)

        self.catalog = _catalog

        super().__init__(self.catalog)

    @staticmethod
    def read_catalog(dir_name, file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """
        full_dir_name = os.path.join(os.path.dirname(__file__), dir_name)
        df = pd.read_csv(
            os.path.join(full_dir_name, file_name),
            sep=",",
            names=[
                "type",
                "year",
                "month",
                "day",
                "lat",
                "lon",
                "depth",
                "mag",
                "duration",
                "ref",
            ],
        )

        df.duration = df.duration * 24 * 3600  # convert from days to seconds

        # add date based on year, month, day
        df["time"] = pd.to_datetime(
            df[["year", "month", "day"]].apply(
                lambda x: "-".join(x.astype(str)), axis=1
            )
        )

        return df


class MexicoSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Mexico"
        self.region = "Mexico"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Mexico/")

        _combined = RoussetSlowSlipCatalog() + ElYousfiSlowSlipCatalog()
        self.catalog = _combined.catalog

        super().__init__(self.catalog)


class LouSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Japan trench (Lou et al. in prep.)"
        self.name = "Japan trench"
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

        slab = Slab("kur")
        # used the geometry of slab2.0 to derive a depth for each event
        depth = slab.interpolate(
            "depth", lat=df["lat"].to_numpy(), lon=df["lon"].to_numpy()
        )
        df["depth"] = depth / 1000  # convert to km

        return df


class JapanSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Japan"
        self.region = "Japan"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Japan/")

        # Digitized with QGIS
        self.japan_trench_bounding_polygon = [
            (141.550130021335, 34.5913079754225),
            (140.910125678055, 34.6419527168039),
            (139.925503611469, 34.8745200714673),
            (139.39626925068, 35.0459961217758),
            (139.236268164859, 35.628324070063),
            (139.211652613195, 35.9976109583285),
            (139.150113734033, 36.1766421822648),
            (139.150113734033, 36.3056887246082),
            (139.199344837363, 36.4741201642055),
            (139.457808129841, 36.6224328475793),
            (139.531654784835, 36.7606010703024),
            (139.691655870655, 37.2128311820651),
            (139.691655870655, 37.3401475213849),
            (139.777810301481, 37.8180933989398),
            (139.851656956475, 38.0607596268774),
            (139.950119163134, 38.3412474975774),
            (140.147043576451, 38.774344604867),
            (140.257813558942, 38.9659936975189),
            (140.40550686893, 39.3762873371753),
            (140.442430196427, 39.4998579800184),
            (140.528584627253, 39.7558037697164),
            (140.639354609744, 40.0579197672273),
            (140.676277937241, 40.283631758802),
            (140.713201264738, 40.5834124917836),
            (140.627046833911, 40.798053825331),
            (140.528584627253, 41.0120033732259),
            (140.294736886439, 41.5668711362225),
            (140.147043576451, 41.979921993966),
            (140.048581369792, 42.2993435875736),
            (139.974734714799, 42.6624226814425),
            (140.023965818128, 42.8251206626614),
            (140.220890231445, 43.0323910131286),
            (141.449496099166, 44.1178935067269),
            (143.836814909796, 44.8215580019539),
            (144.355797259933, 44.9686167943544),
            (145.757049605303, 45.0053227312198),
            (147.573487830782, 44.5263122385939),
            (148.48964360506, 42.884005135429),
            (146.894447594349, 41.8553071798156),
            (145.93462626587, 41.46140746568),
            (145.43443768624, 41.2381454055641),
            (145.218139922076, 41.0752923041463),
            (145.001842157912, 40.8200247699799),
            (144.89369327583, 40.6766441109159),
            (144.569246629583, 40.1000336013456),
            (144.37998608594, 39.7994937636949),
            (144.150169711515, 38.6264267061171),
            (144.150169711515, 38.4466619897739),
            (143.86627889605, 37.5839827845772),
            (143.839241675529, 37.4338503255281),
            (143.420164757461, 36.7003513868742),
            (143.352571706159, 36.646138071497),
            (143.122755331735, 36.4071461132897),
            (142.690159803406, 35.8502863888669),
            (142.514417870023, 35.6747725301775),
            (142.460343428982, 35.2342972642307),
            (141.550130021335, 34.5913079754225),
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
        new.region = "Japan Trench"

        return new

    def get_nankai_trough(self, ref=None) -> SlowSlipCatalog:
        """
        Returns a catalog of slow slip events in the Nankai trough.
        """
        new = self.get_polygon_slice(self.nankai_trough_bounding_polygon)
        if ref is not None:
            new.catalog = new.catalog.loc[new.catalog["ref"] == ref]
        new.name = "Nankai Trough, " + ref if ref else "Nankai Trough"
        new.region = "Nankai Trough"

        return new

    def get_ryukyu_trench(self) -> SlowSlipCatalog:
        """
        Returns a catalog of slow slip events in the Ryukyu trench.
        """

        new = self.get_polygon_slice(self.ryukyu_trench_bounding_polygon)
        new.name = "Ryukyu Trench"
        new.region = "Ryukyu Trench"

        return new


class XieSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Costa Rica (Xie et al. 2020)"
        self.region = "Costa Rica"
        self.ref = "Xie et al. (2020)"
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
        self.region = "New Zealand"
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

        df["duration"] = (df["End_time"] - df["Start_time"]) / np.timedelta64(1, "s")

        return df


class IkariSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "New Zealand"
        self.region = "New Zealand"
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
        self.name = "Cascadia"
        self.region = "Cascadia"
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
        self.name = "Taiwan"
        self.ref = "Chen et al. (2018)"
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


class OkadaAlaskaSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Alaska (Okada et al., 2023)"
        self.region = "Alaska"
        self.ref = "Okada et al. (2023)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Alaska/"),
        )
        self.file_name = "okada2023.txt"

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
            delim_whitespace=True,
        )

        df["depth"] = df["dep"]
        df["mag"] = df["Mw"]

        # We consider the event centroid to be the mid-point between the start and end times
        df["time"] = pd.to_datetime(
            df[["year", "month", "day"]].apply(
                lambda x: "-".join(x.astype(str)), axis=1
            )
        )

        df["duration"] = df["dur"] * 24 * 60 * 60

        return df
