import numpy as np
from src.data import Catalog, AllSlabs, Scaling
from typing import Union


def center_sequences(
    slowslipevents: Catalog,
    earthquakes: Catalog,
    time_window: Union[float, np.ndarray] = 100,
    space_window: float = 100,
    lag=0,
    slab_model=AllSlabs(),
    concatenate_output: bool = True,
    return_indices: bool = False,
):
    """
    Center events within T seconds and R km of each other
    """

    # time window
    delta_t = (
        earthquakes.catalog.time.values.reshape(1, -1)
        - slowslipevents.catalog.time.values.reshape(-1, 1)
    ) / np.timedelta64(1, "D")

    time_window, lag = [np.atleast_2d(i) for i in [time_window, lag]]
    is_in_time_window = (
        np.abs(delta_t - lag.T) < time_window.T / 2
    )  # dimensions: (len(slowslipevents), len(earthquakes))

    # before iterating over each event some operations can be done once
    slowslipevents.catalog["strike"] = slab_model.interpolate(
        "strike",
        lon=slowslipevents.catalog["lon"].to_numpy(),
        lat=slowslipevents.catalog["lat"].to_numpy(),
    )
    slowslipevents.catalog["_stress_drop"] = slowslipevents._stress_drop

    # space window
    is_in_space_window = np.zeros_like(is_in_time_window)
    neighboring_indices = slowslipevents.get_neighboring_indices(
        earthquakes, space_window
    )
    for i in range(len(slowslipevents)):
        if len(neighboring_indices[i]) > 0:
            is_in_space_window[i, neighboring_indices[i]] = True

    # combine time and space windows
    is_in_window = is_in_time_window & is_in_space_window
    indices = [np.where(b)[0] for b in is_in_window]

    # get all the xy differences
    xy_delta_array, time_delta_array = [], []
    for i, ((_, SSE), i_window) in enumerate(zip(slowslipevents, is_in_window)):
        if np.any(i_window):
            centroid = SSE[["lat", "lon"]].values
            strike = SSE.strike
            sse_east, sse_north, _, _ = slab_model.force_ll2utm(
                centroid[0], centroid[1]
            )

            # TODO: use existing dimensions if available?
            sse_dimension = Scaling.magnitude_to_size(SSE.mag, SSE._stress_drop, "m")

            east, north, _, _ = slab_model.force_ll2utm(
                earthquakes.catalog["lat"].values[i_window],
                earthquakes.catalog["lon"].values[i_window],
            )

            xy = np.column_stack((east, north))
            xy = xy - np.array([sse_east, sse_north])
            xy = xy / sse_dimension

            R = np.array(
                [
                    [np.cos(strike * np.pi / 180), -np.sin(strike * np.pi / 180)],
                    [np.sin(strike * np.pi / 180), np.cos(strike * np.pi / 180)],
                ]
            )
            xy_rotated = (R @ xy.T).T

            time_delta_array.append(delta_t[i, i_window])
            xy_delta_array.append(xy_rotated)

            if np.isnan(xy_rotated).sum():
                print("Found NaN", i, SSE.ref)

        # Somewhat counter intuitive decision here. It is useful to know what
        # events have no events associated with them.
        elif concatenate_output is False:
            time_delta_array.append(np.array([]))
            xy_delta_array.append(np.array([]))

        weights = [np.ones_like(t) / len(t) for t in time_delta_array]

    if concatenate_output is True:
        time_delta_array, xy_delta_array, weights = (
            np.concatenate(time_delta_array),
            np.concatenate(xy_delta_array),
            np.concatenate(weights),
        )

    out = [time_delta_array, xy_delta_array, weights]
    if return_indices is True:
        out.append(indices)

    return out
