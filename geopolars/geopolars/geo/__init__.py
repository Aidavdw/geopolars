"""Provides the functional API for Expressions over GeoArrow geometry columns.
Everything in the 'geo' namespace forwards to here.
"""

from geopolars.geo.affine import translate
from geopolars.geo.area import area, area_rsgeo
from geopolars.geo.construct import (
    linestring,
    linestring_from_columns,
    linestring_from_vertices,
    multilinestring,
    multilinestring_from_columns,
    multilinestring_from_linestrings,
    multipoint,
    multipoint_from_columns,
    multipoint_from_points,
    point,
    polygon,
    polygon_from_columns,
    polygon_from_rings,
    validate,
)
from geopolars.geo.crs import to_crs
from geopolars.geo.distance import distance, distance_squared
from geopolars.geo.mean_coordinate import mean_coordinate

__all__ = [
    "area",
    "area_rsgeo",
    "distance",
    "distance_squared",
    "linestring",
    "linestring_from_columns",
    "linestring_from_vertices",
    "mean_coordinate",
    "multilinestring",
    "multilinestring_from_columns",
    "multilinestring_from_linestrings",
    "multipoint",
    "multipoint_from_columns",
    "multipoint_from_points",
    "point",
    "polygon",
    "polygon_from_columns",
    "polygon_from_rings",
    "to_crs",
    "translate",
    "validate",
]
