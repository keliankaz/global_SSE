#%%

from __future__ import annotations
import pandas as pd
from sklearn.neighbors import BallTree
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy
from cartopy.geodesic import Geodesic
import shapely
from obspy.clients.fdsn import Client
from typing import Optional, Literal, Tuple
import copy
import warnings
from pathlib import Path

base_dir = Path(__file__).parents[1]

EARTH_RADIUS_KM = 6371
DAY_PER_YEAR = 365
SEC_PER_DAY = 86400
    

def get_xyz_from_lonlat(
    lon: np.ndarray, lat: np.ndarray, depth_km: np.ndarray = None
) -> np.ndarray:
    """Converts longitude, latitude, and depth to x, y, and z Cartesian
    coordinates.

    Args:
        lon: The longitude, in degrees.
        lat: The latitude, in degrees.
        depth_km: The depth, in kilometers.

    Returns:
        The Cartesian coordinates (x, y, z), in kilometers.
    """
    # Check the shapes of the input arrays
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape")

    assert -180 <= lon.all() <= 180, "Longitude must be between -180 and 180"
    assert -90 <= lat.all() <= 90, "Latitude must be between -90 and 90"
    assert depth_km is None or depth_km.all() >= 0, "Depth must be positive"

    # Assign zero depth if not provided:
    if depth_km is None:
        depth_km = np.zeros_like(lat)

    # Convert to radians
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180

    # Calculate the distance from the center of the earth using the depth
    # and the radius of the earth (6371 km)
    r = EARTH_RADIUS_KM + depth_km

    # Calculate the x, y, z coordinates
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z]).T


class Catalog:
    def __init__(
        self,
        catalog: pd.DataFrame,
        mag_completeness: Optional[float] = None,
        units: Optional[dict] = None,
    ):

        self.raw_catalog = (
            catalog.copy()
        )  # Save a copy of the raw catalog in case of regret

        _catalog = catalog.copy()
        self.catalog = _catalog
        self.__mag_completeness = mag_completeness
        self.__mag_completeness_method = None

        self.units = {k: None for k in self.catalog.keys()}
        if units is not None:
            assert set(units.keys()).issubset(
                set(self.catalog.keys())
            ), "Invalid keys in units"
            self.units.update(units)

        self._stress_drop = 3e9  # Pa

        self.__update__()

    def __update__(self):
        self.catalog = self.catalog.sort_values(by="time")

        # Save catalog attributes to self
        self.start_time = self.catalog["time"].min()
        self.end_time = self.catalog["time"].max()
        self.duration = self.end_time - self.start_time

        if "lat" in self.catalog.keys() and "lon" in self.catalog.keys():
            self.latitude_range = (self.catalog["lat"].min(), self.catalog["lat"].max())
            self.longitude_range = (
                self.catalog["lon"].min(),
                self.catalog["lon"].max(),
            )

        assert "time" in self.catalog.keys() is not None, "No time column"
        assert "mag" in self.catalog.keys() is not None, "No magnitude column"

    @property
    def mag_completeness(
        self,
        magnitude_key: str = "mag",
        method: Literal["minimum", "maximum curvature"] = "minimum",
        filter_catalog: bool = True,
    ):
        if (
            self.__mag_completeness is None
            or self.__mag_completeness_method is not method
        ) and magnitude_key in self.catalog.keys():
            f = {
                "minimum": lambda M: min(M),
                "maximum curvature": lambda M: np.histogram(M)[1][
                    np.argmax(np.histogram(M)[0])
                ]
                + 0.2,
            }
            self.__mag_completeness = f[method](self.catalog[magnitude_key])
            self.__mag_completeness_method = method

            if filter_catalog:
                self.catalog = self.catalog[self.catalog.mag >= self.__mag_completeness]

        return self.__mag_completeness

    @mag_completeness.setter
    def mag_completeness(self, value):
        self.catalog = self.catalog[self.catalog.mag >= value]
        self.__mag_completeness = value

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, index: int) -> pd.DataFrame:
        return self.catalog[index]

    def __getslice__(
        self,
        start: int,
        stop: int,
        step: int = None,
    ) -> Catalog:

        new = copy.deepcopy(self)
        new.catalog = self.catalog[start:stop:step]
        new.__update__()

        return new

    def __iter__(self):
        return self.catalog.iterrows()

    def __add__(self, other) -> Catalog:
        combined_catalog = pd.concat(
            [self.catalog, other.catalog], ignore_index=True, sort=False
        )
        new = copy.deepcopy(self)
        new.catalog = combined_catalog
        new.__update__()

        return new
    
    def __radd__(self, other):
        return self.__add__(other)

    def slice_by(
        self,
        col_name: str,
        start=None,
        stop=None,
    ) -> Catalog:
        if start is None:
            start = self.catalog[col_name].min()
        if stop is None:
            stop = self.catalog[col_name].max()

        assert start <= stop
        in_range = (self.catalog[col_name] >= start) & (self.catalog[col_name] <= stop)

        new = copy.deepcopy(self)
        new.catalog = self.catalog.loc[in_range]
        new.__update__()

        return new

    def get_time_slice(self, start_time, end_time):
        return self.slice_by("time", start_time, end_time)

    def get_space_slice(self, latitude_range, longitude_range):
        return self.slice_by("lat", *latitude_range).slice_by("lon", *longitude_range)

    def intersection(self, other: Catalog, buffer_radius_km: float = 50.0) -> Catalog:
        """returns a new catalog with the events within `buffer_radius_km` of the events in `other`"""
        tree = BallTree(
            np.deg2rad([self.catalog.lat.values, self.catalog.lon.values]).T,
            metric="haversine",
        )

        indices = tree.query_radius(
            np.deg2rad([other.catalog.lat.values, other.catalog.lon.values]).T,
            r=buffer_radius_km / EARTH_RADIUS_KM,
            return_distance=False,
        )

        indices = np.unique(np.concatenate(indices))

        new = copy.deepcopy(self)
        new.catalog = self.catalog.iloc[indices]
        new.__update__()

        return new

    def get_neighboring_indices(
        self, other: Catalog, buffer_radius_km: float = 50.0
    ) -> np.ndarray:
        """gets the indices of events in `other` that are within `buffer_radius_km` from self.

        The ouput therefore has dimensions [len(other),k] where k is the number of neibors for each event.

        For instance:

        ```
        [other[indices] for indices in self.get_neighboring_indices(other)]
        ```

        Returns a list of catalogs for each neighborhood of events in self."""

        tree = BallTree(
            np.deg2rad(other.catalog[["lat", "lon"]]).values.T,
            metric="haversine",
        )

        return tree.query_radius(
            np.deg2rad(self.catalog[["lat", "lon"]]).values.T,
            r=buffer_radius_km / EARTH_RADIUS_KM,
            return_distance=False,
        )

    def plot_time_series(self, column: str = "mag", ax=None) -> plt.axes.Axes:
        """
        Plots a time series of a given column in a dataframe.
        """
        if ax is None:
            fig, ax = plt.subplots()
            
        if column == "mag" and self.mag_completeness is not None:
            bottom = self.mag_completeness-0.05
        else:
            bottom = 0

        markers, stems, _ = ax.stem(
            self.catalog["time"],
            self.catalog[column],
            markerfmt=".",
            bottom = bottom,
        )
        plt.setp(stems, linewidth=0.5, alpha=0.5)
        plt.setp(markers, markersize=0.5, alpha=0.5)

        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        axb = ax.twinx()
        sns.ecdfplot(self.catalog["time"], c="C1", stat="count", ax=axb)

        return ax

    def plot_space_time_series(
        self,
        p1: list[float, float] = None,
        p2: list[float, float] = None,
        column: str = "mag",
        kwargs: dict = None,
        ax: Optional[plt.axes.Axes] = None,
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if p1 is None and p2 is None:
            p1 = np.array([self.longitude_range[0], self.latitude_range[0]])
            p2 = np.array([self.longitude_range[1], self.latitude_range[1]])

        default_kwargs = {
            "alpha": 0.5,
            "color": "C0",
        }

        if kwargs is None:
            kwargs = {}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        p1, p2, x = [
            get_xyz_from_lonlat(np.atleast_2d(ll)[:, 0], np.atleast_2d(ll)[:, 1])
            for ll in [p1, p2, self.catalog[["lon", "lat"]].values]
        ]

        distance_along_section = np.matmul((x - p1), (p2 - p1).T) / np.linalg.norm(
            p2 - p1
        )

        marker_size = getattr(self.catalog,column) if isinstance(column,str) else 1

        ax.scatter(
            self.catalog.time,
            distance_along_section,
            **kwargs,
            label="Unknown duration and size",
            s=marker_size,
        )

        # horizonta histogram of distance along section on the right side of the plot pointing left
        axb = ax.twiny()
        axb.hist(
            distance_along_section,
            orientation="horizontal",
            density=True,
            alpha=0.3,
        )

        axb.set(
            xlim=np.array(axb.get_xlim()[::-1]) * 10,
            xticks=[],
        )

        return ax

    def plot_base_map(
        self,
        extent: Optional[Tuple[float, float,float, float]] = None, 
        ax=None,
    )-> plt.axes.Axes: 
        
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": cartopy.crs.PlateCarree()})

        usemap_proj = cartopy.crs.PlateCarree()
        # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
        if extent is None:
            buffer = 1
            if self.longitude_range is None or self.latitude_range is None:
                extent = (
                    np.array(
                        [
                            self.catalog["lon"].min(),
                            self.catalog["lon"].max(),
                            self.catalog["lat"].min(),
                            self.catalog["lat"].max(),
                        ]
                    )
                    + np.array([-1, 1, -1, 1]) * buffer
                )

            else:
                extent = (
                    np.array(self.longitude_range + self.latitude_range)
                    + np.array([-1, 1, -1, 1]) * buffer
                )

        if extent[0] < -180:
            extent[0] = -179.99
        if extent[1] > 180:
            extent[1] = 179.99

        if extent[2] < -90:
            extent[2] = 90
        if extent[3] > 90:
            extent[3] = 90

        ax.set_extent(
            extent,
            crs=cartopy.crs.PlateCarree(),
        )

        ax.add_feature(cartopy.feature.LAND, color="lightgray")
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")

        # plot grid lines
        ax.gridlines(draw_labels=True, crs=usemap_proj, color="gray", linewidth=0.3)
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        return ax
    
    def plot_map(
        self, 
        columm: str = "mag", 
        scatter_kwarg: dict = None, 
        extent: Optional[Tuple[float, float,float, float]] = None, 
        ax=None, 
    ) -> plt.axes.Axes:
        
        ax = self.plot_base_map(extent=extent, ax=ax)

        if scatter_kwarg is None:
            scatter_kwarg = {}
        default_scatter_kawrg = {
            "color": "lightgray",
            "marker": "o",
            "edgecolors": "brown",
            "transform": cartopy.crs.PlateCarree(),
        }
        default_scatter_kawrg.update(scatter_kwarg)
        
        ax.scatter(
            self.catalog["lon"],
            self.catalog["lat"],
            s=self.catalog[columm],
            **default_scatter_kawrg,
        )

        return ax

    def plot_hist(
        self, columm: str = "mag", log_scale: bool = True, ax=None
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.catalog[columm], log=log_scale)
        ax.set_xlabel(columm)

        return ax

    def plot_scaling(
        self, column: str = "duration", log_scale: bool = True, ax=None
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        assert self.catalog[column] is not None, "No duration column"
        ax.scatter(self.catalog["mag"], self.catalog[column])
        ax.set(
            xlabel="Magnitude",
            ylabel=column,
        )
        if log_scale:
            ax.set_yscale("log")

        return ax

    def plot_summary(
        self, kwarg={"time series": None, "map": None, "hist": None}, ax=None
    ) -> list[plt.axes.Axes, plt.axes.Axes, plt.axes.Axes]:
        if ax is None:
            fig = plt.figure(figsize=(6.5, 7))
            gs = fig.add_gridspec(4, 3)
            ax1 = fig.add_subplot(gs[0:2, 0:2], projection=cartopy.crs.PlateCarree())
            ax2 = fig.add_subplot(gs[0:2, 2])
            ax3 = fig.add_subplot(gs[2, :])
            ax4 = fig.add_subplot(gs[3, :])
        else:
            print("Bold decision")
            ax1, ax2, ax3 = ax

        self.plot_map(ax=ax1)
        self.plot_hist(ax=ax2)
        self.plot_time_series(ax=ax3)
        self.plot_space_time_series(ax=ax4)

        plt.tight_layout()

        return (ax1, ax2, ax3)

    @staticmethod
    def read_catalog(self, filename):
        """
        Reads a catalog from a file and returns a dataframe.
        """
        raise NotImplementedError


class EarthquakeCatalog(Catalog):
    def __init__(
        self,
        catalog: Optional[pd.DataFrame] = None,
        filename: str = None,
        use_other_catalog: bool = False,
        kwargs: dict = None,
        other_catalog: Catalog = None,
        other_catalog_buffer: float = 0.0,
    ) -> Catalog:

        if catalog is None:
            if kwargs is None:
                kwargs = {}

            if use_other_catalog and other_catalog is not None:
                metadata = {
                    "starttime": other_catalog.start_time,
                    "endtime": other_catalog.end_time,
                    "latitude_range": other_catalog.latitude_range
                    + np.array([-1, 1]) * other_catalog_buffer,
                    "longitude_range": other_catalog.longitude_range
                    + np.array([-1, 1]) * other_catalog_buffer,
                }
                metadata.update(kwargs)
            elif not use_other_catalog:
                metadata = kwargs
            else:
                raise ValueError("No other catalog provided")

            _catalog = self.get_and_save_catalog(filename, **metadata)
            self.catalog = self._add_time_column(_catalog, "time")
        else:
            self.catalog = catalog

        super().__init__(self.catalog)

    @staticmethod
    def _add_time_column(df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(pd.to_datetime(df["time"], unit="d"))
        return df

    @staticmethod
    def get_and_save_catalog(
        filename: str = "_temp_local_catalog.csv",
        starttime: str = "2019-01-01",
        endtime: str = "2020-01-01",
        latitude_range: list[float, float] = [-90, 90],
        longitude_range: list[float, float] = [-180, 180],
        minimum_magnitude: float = 4.5,
        use_local_client: bool = False,
        default_client_name: str = "IRIS",
    ) -> pd.DataFrame:
        """
        Gets earthquake catalog for the specified region and minimum event
        magnitude and writes the catalog to a file.

        By default, events are retrieved from the NEIC PDE catalog for recent
        events and then the ISC catalog when it becomes available. These default
        results include only that catalog's "primary origin" and
        "primary magnitude" for each event.
        """
        
        if longitude_range[1] > 180:
            longitude_range[1] = 180
            warnings.warn("Longitude range exceeds 180 degrees. Setting to 180.")
            
        if longitude_range[0] < -180:
            longitude_range[0] = -180
            warnings.warn("Longitude range exceeds -180 degrees. Setting to -180.")
        
        def is_within(lat_range_querry, lon_range_querry, lat_range, lon_range):
            """
            Checks if a point is within a latitude and longitude range.
            """
            return (lat_range[0] <= lat_range_querry[0] <= lat_range[1]) and (
                lon_range[0] <= lon_range_querry[0] <= lon_range[1]) and (
                lat_range[0] <= lat_range_querry[1] <= lat_range[1]) and (
                lon_range[0] <= lon_range_querry[1] <= lon_range[1]
            )
        
        local_client_coverage = {
            "GEONET": [[-49.18, -32.28], [163.52, 179.99]],
        }

        # Note that using local client supersedes the any specified default_client_name 
        if use_local_client:
            ## use local clients if lat and long are withing the coverage of the local catalogs
            index = []
            for i,key in enumerate(local_client_coverage.keys()):
                if is_within(latitude_range, longitude_range, *local_client_coverage[key]):
                    index.append(i)
                else:
                    index.append(i)

            if len(index) > 1:
                raise ValueError("Multiple local clients found")
            elif len(index) == 1:
                if default_client_name is not None:
                    warnings.warn("Using local client instead of default client")
                client_name=list(local_client_coverage.keys())[index[0]]
            else:
                client_name = default_client_name
        else:
            client_name = default_client_name
        
        
        # Use obspy api to ge  events from the IRIS earthquake client    
        client = Client(client_name)
                
        cat = client.get_events(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minimum_magnitude,
            minlatitude=latitude_range[0],
            maxlatitude=latitude_range[1],
            minlongitude=longitude_range[0],
            maxlongitude=longitude_range[1],
        )

        # Write the earthquakes to a file
        f = open(filename, "w")
        f.write("time,lat,lon,dep,mag\n")
        for event in cat:
            loc = event.preferred_origin()
            lat = loc.latitude
            lon = loc.longitude
            dep = loc.depth
            time = loc.time.matplotlib_date
            mag = event.preferred_magnitude().mag
            f.write("{}, {}, {}, {}, {}\n".format(time, lat, lon, dep, mag))
        f.close()
        df = pd.read_csv(filename)

        return df


class SlowSlipCatalog(Catalog):
    def __init__(
        self,
        catalog: pd.DataFrame = None,
        filename: str = None,
        time_columns: list[str] = ["year", "month", "day"],
        time_alignment: Literal[
            "centroid", "start"
        ] = "centroid",  # assumes SSEs times are centroid-time
    ):

        if catalog is not None:
            self.catalog = catalog
        else:
            assert filename is not None and time_columns is not None
            _catalog = self.read_catalog(filename)
            self._time_columns = time_columns

            self.catalog = self._add_time_column(_catalog, "time")

        if "duration" not in self.catalog.keys():
            self.catalog["duration"] = np.nan * np.ones(len(self.catalog))

        super().__init__(self.catalog)

        self.time_alignment = time_alignment
        self._stress_drop = 1e4  # Pa

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(df[self._time_columns])
        return df

    def _add_duration_column(self, df, column):
        """
        Adds a duration colum with the standardized key name: duration.
        """
        raise NotImplementedError

    def plot_slowslip_timeseries(self, column: str = "mag", ax=None) -> plt.axes.Axes:
        """
        Plots a time series of a given column in a dataframe.
        """
        if ax is None:
            fig, ax = plt.subplots()

        for t, d, c in zip(
            self.catalog["time"],
            self.catalog["duration"],
            self.catalog[column],
        ):
            if np.isnan(d) and not np.isnan(c):
                ax.axvline(t, alpha=c / np.nanmax(self.catalog[column]))
            elif not np.isnan(d):
                ax.axvspan(
                    t - d * pd.Timedelta(1, "s") / 2,
                    t + d * pd.Timedelta(1, "s") / 2,
                    alpha=0.2,
                )

        ax.set_xlabel("Time")
        axb = ax.twinx()
        sns.ecdfplot(self.catalog["time"], c="C1", stat="count", ax=axb)

        return ax

    def plot_space_time_series(
        self,
        p1: list[float, float] = None,
        p2: list[float, float] = None,
        kwargs: dict = None,
        ax: Optional[plt.axes.Axes] = None,
    ) -> plt.axes.Axes:

        if ax is None:
            fig, ax = plt.subplots()

        if p1 is None and p2 is None:
            p1 = np.array([self.longitude_range[0], self.latitude_range[0]])
            p2 = np.array([self.longitude_range[1], self.latitude_range[1]])

        default_kwargs = {
            "alpha": 0.5,
            "color": "C0",
        }

        if kwargs is None:
            kwargs = {}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        p1, p2, x = [
            get_xyz_from_lonlat(np.atleast_2d(ll)[:, 0], np.atleast_2d(ll)[:, 1])
            for ll in [p1, p2, self.catalog[["lon", "lat"]].values]
        ]
        distance_along_section = np.matmul((x - p1), (p2 - p1).T) / np.linalg.norm(
            p2 - p1
        )

        for x, y, d, L in zip(
            self.catalog.time,
            distance_along_section,
            (self.catalog.duration * pd.Timedelta(1, "s")).values,
            Scaling.magnitude_to_size(self.catalog.mag, self._stress_drop, "km"),
        ):
            if (np.isnan(d) or d == 0) and np.isnan(L):
                ax.scatter(
                    x,
                    y,
                    marker="x",
                    **kwargs,
                    label="Unknown duration and size",
                )
            elif not np.isnan(L) and np.isnan(d):
                ax.plot(
                    [x, x],
                    [y - L / 2, y + L / 2],
                    **kwargs,
                    label="Unknown duration",
                )
            else:
                start = mdates.date2num(x - d / 2)
                end = mdates.date2num(x + d / 2)
                width = end - start
                rh = plt.Rectangle(
                    xy=[start, y - L / 2],
                    width=width,
                    height=L,
                    **kwargs,
                    label="Known duration and size",
                )
                ax.add_patch(rh)

        ax.scatter(self.catalog.time, distance_along_section, s=0)

        ax.set(
            xlabel="Time",
            yticks=[],
            ylabel="Distance along cross-section",
        )

        axb = ax.twiny()
        axb.hist(
            distance_along_section,
            orientation="horizontal",
            density=True,
            **kwargs,
        )

        axb.set(
            xlim=np.array(axb.get_xlim()[::-1]) * 10,
            xticks=[],
        )

        return ax

    def plot_map(
        self, 
        columm: str = "mag", 
        scatter_kwarg: dict = None, 
        extent: Optional[Tuple[float, float,float, float]] = None, 
        ax=None, 
    ) -> plt.axes.Axes:
    
        ax = self.plot_base_map(extent=extent, ax=ax)

        if scatter_kwarg is None:
            scatter_kwarg = {}
        default_scatter_kawrg = {
            "color": "indianred",
            "edgecolors": None,
            "crs": cartopy.crs.PlateCarree(),
            "alpha": 0.2,
        }
        default_scatter_kawrg.update(scatter_kwarg)
        geoms = []
        gd = Geodesic()
        for lon, lat, R in zip(
            self.catalog["lon"],
            self.catalog["lat"],
            Scaling.magnitude_to_size(self.catalog[columm], self._stress_drop, "m")/2, # divide by 2 to get radius
        ):
            cp = gd.circle(lon=lon, lat=lat, radius=R)
            geoms.append(shapely.geometry.Polygon(cp))
        
        ax.add_geometries(geoms, **default_scatter_kawrg)

        return ax

    @staticmethod
    def read_catalog(filename):
        """
        Reads in a catalog of galaxies and returns a pandas dataframe.
        """
        df = pd.read_csv(filename)
        return df


class Scaling:
    """A collection of scaling relationships for earthquakes"""

    @staticmethod
    def magnitude_to_size(
        MW: np.ndarray, stress_drop_Pa=3e6, out_unit: Literal["km", "m"] = "km"
    ) -> np.ndarray:

        # M0 = mu * A * D ~ \delta \sigma * a^3                   # MISSING CONSTANTS HERE!
        # Mw = (2/3) * (log10(M0) - 9.1)
        # a ~ [(1/(\delta \sigma)) 10^((3/2 * Mw) + 9.1)]^(1/3)   # CHECK THIS! e.g. dyne cm vs Pa
        # for SSEs \delta \sigma ~ 10 kPa
        # for Earthquake \delta \sigma ~ 3 MPa (default)
        # returns the dimensions of the earthquake in km

        unit_conversion_factor = {"km": 1 / 1000, "m": 1}

        return (10 ** ((3 / 2) * MW + 9.1) / stress_drop_Pa) ** (
            1 / 3
        ) * unit_conversion_factor[out_unit]


class JapanSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self, files=None):
        self.dir_name = os.path.join(base_dir,"Datasets/Slow_slip_datasets/Japan/")
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

        return df_all


class RoussetSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.dir_name = os.path.join(base_dir,"Datasets/Slow_slip_datasets/Mexico/")
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
        self.dir_name = os.path.join(
            os.path.dirname(__file__), os.path.join(base_dir,"Datasets/Slow_slip_datasets/Costa_Rica/"),
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
        self.dir_name = os.path.join(
            os.path.dirname(__file__), os.path.join(base_dir,"Datasets/Slow_slip_datasets/New_Zealand/"),
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
        df["depth"] = df["Z_geom"]
        df["mag"] = df["Mw"]
        df["year"] = df["Start_year"]
        df["month"] = df["Start_month"]
        df["day"] = df["Start_day"]

        # TODO: add duration to this catalog

        return df


class IkariSlowSlipCatalog(SlowSlipCatalog):
    def __init__(self):
        self.dir_name = os.path.join(
            os.path.dirname(__file__), os.path.join(base_dir,"Datasets/Slow_slip_datasets/New_Zealand"),
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
        self.dir_name = os.path.join(
            os.path.dirname(__file__), os.path.join(base_dir,"Datasets/Slow_slip_datasets/Cascadia")
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

        return df

class SwarmCatalog(Catalog):
    def __init__(
        self,
        catalog: pd.DataFrame = None,
        filename: str = None,
        time_columns: list[str] = ["Year", "Month", "Day", "Hour", "Minute", "Second"],
    ):
        """
        Swarm catalog.
        """
        
        if catalog is not None:
            _catalog = catalog
        else:
            assert filename is not None and time_columns is not None
            _catalog = self.read_catalog(filename)
            self._time_columns = time_columns

            _catalog =  self._add_time_column(_catalog, "time")
        
        
        super().__init__(_catalog)
        
        return self

    def _add_time_column(self, df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(df[self._time_columns])
        return df

class NishikawaSwarmCatalog(SwarmCatalog):

    def __init__(self):
        self.dir_name = os.path.join(base_dir,"Datasets/Swarm_datasets/Global")
        self.file_name = "nishikawa2017S3.txt"
        
        _catalog = self.read_catalog(os.path.join(self.dir_name, self.file_name))
        self._time_columns = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
        _catalog =  self._add_time_column(_catalog, "time")
        _catalog["mag"] = 4.5 # Use dummy magnitude (the catalog completeness used for the analysis in Nishikawa and Ide 2017)
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
            sep=' ',
            index_col=False,
            names= [ 
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


if __name__ == "__main__":

    # A few tests to ensure that everything still runs smoothly:
    william_catalog = WilliamsSlowSlipCatalog()
    japan_catalog = JapanSlowSlipCatalog()
    rousset_catalog = RoussetSlowSlipCatalog()
    michel_catalog = MichelSlowSlipCatalog()

    print("Japan Slow Slip Catalog")

    # test all visualizations:
    plotting_methods = [k for k in dir(japan_catalog) if "plot" in k]
    for catalog in [william_catalog, japan_catalog, rousset_catalog, michel_catalog]:
        print(catalog)
        [getattr(catalog, k)() for k in plotting_methods]

    combined_catalog = japan_catalog + rousset_catalog
    