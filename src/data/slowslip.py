from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy
from cartopy.geodesic import Geodesic
import shapely
from typing import Optional, Literal, Tuple
from pathlib import Path
from .catalog import Catalog
from .utils import Scaling, get_xyz_from_lonlat

base_dir = Path(__file__).parents[2]


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

    @staticmethod
    def read_catalog(filename):
        """
        Reads in a catalog of galaxies and returns a pandas dataframe.
        """
        df = pd.read_csv(filename)
        return df
