"""Provides the functional API for Expressions over GeoArrow geometry columns.
Everything in the 'geo' namespace forwards to here.
"""

from geopolars.geo.affine import translate
from geopolars.geo.area import area
from geopolars.geo.construct import (
    box,
    linestring,
    linestring_from_columns,
    linestring_from_vertices,
    multilinestring,
    multilinestring_from_columns,
    multilinestring_from_linestrings,
    multipoint,
    multipoint_from_columns,
    multipoint_from_points,
    multipolygon,
    multipolygon_from_columns,
    multipolygon_from_polygons,
    point,
    polygon,
    polygon_from_columns,
    polygon_from_rings,
    validate,
)
from geopolars.geo.coordinates import m, x, y, z
from geopolars.geo.crs import to_crs
from geopolars.geo.distance import distance, distance_squared
from geopolars.geo.is_empty import is_empty
from geopolars.geo.length import length
from geopolars.geo.mean_coordinate import mean_coordinate
from geopolars.geo.wkb import from_wkb, to_wkb
from geopolars.geo.wkt import from_wkt, to_wkt

__all__ = [
    "area",
    "box",
    "distance",
    "distance_squared",
    "from_wkb",
    "from_wkt",
    "is_empty",
    "length",
    "linestring",
    "linestring_from_columns",
    "linestring_from_vertices",
    "m",
    "mean_coordinate",
    "multilinestring",
    "multilinestring_from_columns",
    "multilinestring_from_linestrings",
    "multipoint",
    "multipoint_from_columns",
    "multipoint_from_points",
    "multipolygon",
    "multipolygon_from_columns",
    "multipolygon_from_polygons",
    "point",
    "polygon",
    "polygon_from_columns",
    "polygon_from_rings",
    "to_crs",
    "to_wkb",
    "to_wkt",
    "translate",
    "validate",
    "x",
    "y",
    "z",
]
