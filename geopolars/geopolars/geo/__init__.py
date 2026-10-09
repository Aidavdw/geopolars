"""Provides the functional API for Expressions over GeoArrow geometry columns.
Everything in the 'geo' namespace forwards to here.
"""

from geopolars.geo.affine import (
    affine_transform,
    rotate,
    rotate_about_centroid,
    scale,
    scale_about_centroid,
    skew,
    skew_about_centroid,
    translate,
)
from geopolars.geo.area import area, area_geodesic, area_planar
from geopolars.geo.boundary import boundary
from geopolars.geo.bounds import bounds
from geopolars.geo.centroid import centroid
from geopolars.geo.construct import (
    box,
    line_string,
    line_string_from_columns,
    line_string_from_vertices,
    multi_line_string,
    multi_line_string_from_columns,
    multi_line_string_from_line_strings,
    multi_point,
    multi_point_from_columns,
    multi_point_from_points,
    multi_polygon,
    multi_polygon_from_columns,
    multi_polygon_from_polygons,
    point,
    polygon,
    polygon_from_columns,
    polygon_from_rings,
    validate,
)
from geopolars.geo.coordinates import m, x, y, z
from geopolars.geo.count_coordinates import count_coordinates
from geopolars.geo.crs import is_geographic, set_crs, to_crs
from geopolars.geo.distance import distance, distance_squared
from geopolars.geo.exterior import exterior
from geopolars.geo.interior import interior
from geopolars.geo.is_empty import is_empty
from geopolars.geo.length import length, length_geodesic, length_planar
from geopolars.geo.mean_coordinate import mean_coordinate
from geopolars.geo.to_polygon import to_polygon
from geopolars.geo.wkb import from_wkb, to_wkb
from geopolars.geo.wkt import from_wkt, to_wkt
from geopolars.geo.wrap_longitude import wrap_longitude

__all__ = [
    "affine_transform",
    "area",
    "area_geodesic",
    "area_planar",
    "boundary",
    "bounds",
    "box",
    "centroid",
    "count_coordinates",
    "distance",
    "distance_squared",
    "exterior",
    "from_wkb",
    "from_wkt",
    "interior",
    "is_empty",
    "is_geographic",
    "length",
    "length_geodesic",
    "length_planar",
    "line_string",
    "line_string_from_columns",
    "line_string_from_vertices",
    "m",
    "mean_coordinate",
    "multi_line_string",
    "multi_line_string_from_columns",
    "multi_line_string_from_line_strings",
    "multi_point",
    "multi_point_from_columns",
    "multi_point_from_points",
    "multi_polygon",
    "multi_polygon_from_columns",
    "multi_polygon_from_polygons",
    "point",
    "polygon",
    "polygon_from_columns",
    "polygon_from_rings",
    "rotate",
    "rotate_about_centroid",
    "scale",
    "scale_about_centroid",
    "set_crs",
    "skew",
    "skew_about_centroid",
    "to_crs",
    "to_polygon",
    "to_wkb",
    "to_wkt",
    "translate",
    "validate",
    "wrap_longitude",
    "x",
    "y",
    "z",
]
