import numpy as np


EARTH_RADIUS_KM = 6371.0


def latlon_to_xy(latitudes, longitudes):
    """
    Convert latitude/longitude into local Cartesian
    coordinates measured approximately in kilometres.

    Uses an equirectangular projection suitable for
    city-scale distances.
    """

    latitudes = np.asarray(
        latitudes,
        dtype=float
    )

    longitudes = np.asarray(
        longitudes,
        dtype=float
    )

    # Reference point
    lat0 = np.mean(latitudes)
    lon0 = np.mean(longitudes)

    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    x = (
        EARTH_RADIUS_KM
        * (lon_rad - lon0_rad)
        * np.cos(lat0_rad)
    )

    y = (
        EARTH_RADIUS_KM
        * (lat_rad - lat0_rad)
    )

    return (
        np.column_stack((x, y)),
        lat0,
        lon0
    )


def xy_to_latlon(
    x,
    y,
    lat0,
    lon0
):
    """
    Convert local Cartesian kilometres back into
    latitude/longitude.
    """

    lat0_rad = np.radians(lat0)

    latitude = (
        lat0
        + np.degrees(
            y / EARTH_RADIUS_KM
        )
    )

    longitude = (
        lon0
        + np.degrees(
            x
            / (
                EARTH_RADIUS_KM
                * np.cos(lat0_rad)
            )
        )
    )

    return latitude, longitude