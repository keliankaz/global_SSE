"""Slow slip events"""
from __future__ import annotations
from typing import Optional, Literal, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy
from cartopy.geodesic import Geodesic
import shapely

from .catalog import Catalog
from .utils import Scaling, get_xyz_from_lonlat

base_dir = Path(__file__).parents[2]


class SlowSlipCatalog(Catalog):
    """Slow slip catalog
    
    Args:
        catalog: Dataframe with source charactersistic of slow slip events
        filenemae: location where the data is stored
        time_columns: format for time data
        time_alignments: determines whether time are reported with respect to start time
            or centroid time
    """
    def __init__(
        self,
        catalog: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        time_columns: list[str] = ("year", "month", "day"),
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
        self._short_term_event_cutoff = 90  # days

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

    def get_duration_from_scaling(
        self,
        mag_querry: np.ndarray, 
        mag: Optional[np.ndarray]=None,
        duration:  Optional[np.ndarray]=None,
        filter_na: bool = True, 
        filter_null: bool = True, 
    ):
        """Get the duration of slow slip events using the 
        magntitude log-duration scaling.
        
        If mag and duration are not specified, uses self.catalog instead.
        """
        if mag is not None and duration is not None:
            assert len(mag)==len(duration)
        elif mag is not None or duration is not None:
            raise ValueError('Must specify both mag and duration')
        else:
            mag = self.catalog.mag
            duration = self.catalog.duration
        
        if filter_na is True:
            is_nan = np.isnan(mag) | np.isnan(duration)
            mag, duration = [v[~is_nan] for v in [mag, duration]]
            
        if filter_null is True:
            is_null = (mag==0) | (duration==0)
            mag, duration = [v[~is_null] for v in [mag, duration]]
            
        assert not np.any(duration<=0)
        
        coefficients = np.polyfit(mag, np.log(duration),1)
        poly = np.poly1d(coefficients)
        
        return np.exp(poly(mag_querry))
    
    def impute_duration(
        self,
        mag: Optional[np.ndarray]=None,
        duration:  Optional[np.ndarray]=None
    ):
        missing_duration = self.catalog.duration.isna() | (self.catalog.duration==0) 
        self.catalog.loc[missing_duration,"duration"] = self.get_duration_from_scaling(
            self.catalog.mag.loc[missing_duration].values,
            mag,
            duration,
        )
        
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
        p1: list[float] = None,
        p2: list[float] = None,
        kwargs: dict = None,
        plot_histogram: bool = False,
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
            ylabel="Distance along cross-section",
        )
        if plot_histogram:
            axb = ax.twiny()
            axb.hist(
                distance_along_section,
                orientation="horizontal",
                density=True,
                bins=20,
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
        scatter_kwarg: Optional[dict] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
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
            Scaling.magnitude_to_size(self.catalog[columm], self._stress_drop, "m")
            / 2,  # divide by 2 to get radius
        ):
            if np.isnan(R):
                continue
            cp = gd.circle(lon=lon, lat=lat, radius=R)
            geoms.append(shapely.geometry.Polygon(cp))

        ax.add_geometries(geoms, **default_scatter_kawrg)

        return ax

    def get_deep_cluster(self):
        """
        returns a new slow slip catalog with the events that fall in the deeper of two clusters using kmeans

        note that this is not a good method for all cases
        """
        return self.get_clusters("depth", 2)[2]

    def get_shallow_cluster(self):
        """
        returns a new slow slip catalog with the events that fall in the shallower of two clusters using kmeans

        note that this is not a good method for all cases
        """
        return self.get_clusters("depth", 2)[2]

    def get_short_term_events(self):
        return self.slice_by(
            "duration",
            1,
            self._short_term_event_cutoff * 60 * 60 * 24,  # note duration is in seconds
        )  # exclude events without a specified duration

    def get_long_term_events(self):
        return self.slice_by("duration", self._short_term_event_cutoff, np.inf)

    @staticmethod
    def read_catalog(filename):
        """
        Reads in a catalog of galaxies and returns a pandas dataframe.
        """
        df = pd.read_csv(filename)
        return df
