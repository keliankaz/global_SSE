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
            (142.225499326711, 35.3328270936419),
            (142.138291100781, 34.9837696861799),
            (141.674058385192, 35.878750607878),
            (140.955495021331, 36.6668884777978),
            (140.681833088636, 36.9169072802688),
            (141.643896466215, 42.5179069793497),
            (143.120272958242, 42.9614880737278),
            (145.625639126532, 43.6532837147515),
            (146.855952869888, 42.1541255535136),
            (145.78222451205, 41.5877823510286),
            (144.753234835788, 40.9995468674037),
            (144.339402031205, 40.448584938636),
            (144.227555327263, 39.4971907575807),
            (144.026231260169, 38.1026055845413),
            (143.053164935878, 36.3663017477545),
            (142.35971537144, 35.6424648084837),
            (142.225499326711, 35.3328270936419),
        ]
        self.nankai_trough_bounding_polygon = [
            (139.417748318578, 35.2846435617882),
            (139.5647910681, 34.8333672470613),
            (138.498359788482, 34.1307186002105),
            (137.415413686099, 33.2888492231574),
            (137.105697637772, 33.0700281020563),
            (136.610825256204, 32.8209086718795),
            (135.661478238504, 32.6492182537847),
            (135.205320362977, 32.6256529022815),
            (135.104325999392, 32.4943602296208),
            (134.659950799618, 32.1812777025068),
            (133.942890818163, 31.8378968663172),
            (133.463874551678, 31.6121744637044),
            (133.355743253067, 31.5089582241204),
            (133.01168912112, 31.0666029116174),
            (131.999186961391, 30.5062861824469),
            (131.360229287775, 30.6930584255037),
            (130.146209707906, 31.1255836199511),
            (130.072483822489, 31.6318346998157),
            (130.426368072491, 32.2904526095424),
            (130.755402612863, 33.1716311574826),
            (133.022343484937, 34.1854054501365),
            (134.985360236421, 34.4810028824915),
            (135.160220689363, 34.7724369707288),
            (135.955419415839, 35.0638710589661),
            (137.293533216223, 35.0568598974459),
            (137.632560065477, 35.1681030823572),
            (138.127857103058, 35.6686974144583),
            (139.438937496656, 35.8885351370212),
            (139.39920778776, 35.6263190583016),
            (139.417748318578, 35.2846435617882),
        ]
        self.ryukyu_trench_bounding_polygon = [
            (130.146525809775, 31.1305417139102),
            (131.991418162898, 30.5002575131796),
            (132.401181215602, 30.310530915731),
            (131.904926638543, 29.7456883076969),
            (131.489363862632, 29.1082230786299),
            (131.263426819419, 28.6724873524322),
            (130.864002403738, 28.0874717941112),
            (130.488785528401, 27.4742141053885),
            (129.460863174568, 26.140588652173),
            (129.267423925241, 25.9886006705591),
            (128.9219966943, 25.5961953362106),
            (128.325098439235, 25.2341875981848),
            (126.794482345324, 24.0056840819772),
            (126.181855061297, 23.6728988906543),
            (125.039796791076, 23.2796073009091),
            (124.933910593837, 23.2493541016979),
            (124.661631800936, 23.2039743028812),
            (124.306156710205, 23.1661578038672),
            (124.026314617501, 23.1812844034728),
            (123.34561763525, 23.17372110367),
            (122.823749948857, 23.1964110030784),
            (122.233812564239, 23.3401136993314),
            (121.848084274297, 23.4611264961761),
            (121.727071477452, 23.0375817072197),
            (122.184215772543, 22.3686496557512),
            (121.689522779944, 22.1403298130135),
            (120.738190101871, 22.1403298130135),
            (120.585976873379, 22.9394492625952),
            (120.547923566256, 23.8146753264228),
            (120.662083487625, 24.5376881617586),
            (120.966509944609, 24.9943278472338),
            (121.917842622682, 25.1084877686026),
            (122.831121993633, 25.1084877686026),
            (123.363868293354, 25.2226476899714),
            (124.467414199919, 25.489020839832),
            (125.418746877992, 25.7173406825696),
            (126.408132863189, 26.1739803680449),
            (126.978932470033, 26.5925667463972),
            (128.767437904811, 28.3430188740523),
            (130.146525809775, 31.1305417139102),
        ]
        self.boso_bounding_polygon = [
            (140.680918420881, 36.9116282904888),
            (140.974673210963, 36.6574808878336),
            (141.681004953407, 35.8719343705356),
            (142.136489908815, 34.977467537814),
            (141.97145912787, 34.4922770418358),
            (141.85263696559, 34.2414302547995),
            (141.819630809401, 34.1721173268026),
            (141.591919298494, 34.238377346744),
            (141.532596767491, 34.3471353202481),
            (141.483161324989, 34.39657076275),
            (141.404064616986, 34.4114013955006),
            (141.374403351485, 34.421288484001),
            (141.211266391229, 34.4558932937523),
            (141.147000315977, 34.4657803822526),
            (140.870161837966, 34.5152158247545),
            (140.563662094455, 34.5201593690047),
            (140.089030221633, 34.4118028744922),
            (139.831351831521, 34.6718084384554),
            (139.57442491085, 34.8318243276454),
            (139.414087033404, 35.2887299153903),
            (139.555879414161, 35.439424467008),
            (139.869978203325, 35.6780478786145),
            (140.016258422594, 35.9655641716608),
            (140.03139085907, 36.2883894831515),
            (140.06683190218, 36.4903009758718),
            (140.090535883583, 36.5956520043293),
            (140.382884987552, 36.8142553883786),
            (140.680918420881, 36.9116282904888),
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
    
    def get_boso_peninsula(self) -> SlowSlipCatalog:
        """
        Returns a catalog of slow slip events in the Japan trench.
        """

        new = self.get_polygon_slice(self.boso_bounding_polygon)
        new.name = "Boso Peninsula"
        new.region = "Boso Peninsula"

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


class PerrySlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Costa Rica (Perry et al. 2023)"
        self.region = "Costa Rica"
        self.ref = "Perry et al. (2023)"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Costa_Rica/"),
        )
        self.file_name = "Perry2023.csv"

        super().__init__(
            filename=os.path.join(self.dir_name, self.file_name),
        )
        
        self.catalog["ref"] = self.ref
    
    def _add_time_column(self, df, column):
        return df
    
    @staticmethod
    def read_catalog(file_name):
        """
        Reads in a catalog of slow slip events in Mexico and returns a pandas dataframe. Read each file in
        Datasets/Slow_slip_datasets/Mexico/ directory and concatenate them into one dataframe.
        """

        columns = {
            "start_yr":"year",
            "start_doy":"day",
            "start_dec_year":"decimal_year",
            "end_yr":"year",
            "end_doy":"day",
            "end_dec_year":"decimal_year",
            "Mw_Est":"Mw_equivalent",
            "cent_long":"degrees",
            "cent_lat":"degrees"
        }

        df = pd.read_csv(
            file_name,
            skiprows=1,
            index_col=False,
            delimiter=",",
            names=columns.keys(),
        )

        df = df.reset_index(drop=True)
        
        # standard column notation for SSEs is time, lat, lon, mag
        df['lat'] = df['cent_lat']
        df['lon'] = df['cent_long']
        df['mag'] = df['Mw_Est']
        
        for i, row in df.iterrows():
            df.loc[i,"time"] = (
                _deciyear_to_datetime((row.start_dec_year+row.end_dec_year)/2) 
            )
            
            df.loc[i,"duration"] =  (row.end_dec_year - row.start_dec_year)*365*24*60*30 # in seconds

        # use the geometry of slab2.0 to derive a depth for each event
        slab = Slab("cam")
        depth = slab.interpolate(
            "depth", lat=df["lat"].to_numpy(), lon=df["lon"].to_numpy()
        )
        df["depth"] = depth / 1000  # convert to km
        
        return df

class CostaRicaSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.name = "Costa Rica"
        self.region = "Costa Rica"
        self.dir_name = os.path.join(base_dir, "Datasets/Slow_slip_datasets/Costa_Rica/")

        _combined = XieSlowSlipCatalog() + PerrySlowSlipCatalog()
        self.catalog = _combined.catalog

        super().__init__(self.catalog)

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
        self.ref = "Michel et al. 2019"
        self.dir_name = os.path.join(
            os.path.dirname(__file__),
            os.path.join(base_dir, "Datasets/Slow_slip_datasets/Cascadia"),
        )
        self.file_name = "Michel2019.csv"

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
                "SSE_number_michel2019",
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

# %%

