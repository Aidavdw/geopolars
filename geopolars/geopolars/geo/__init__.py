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
)
from geopolars.geo.convex_hull import convex_hull
from geopolars.geo.coordinates import m, x, y, z
from geopolars.geo.count_coordinates import count_coordinates
from geopolars.geo.crs import is_geographic, set_crs, to_crs
from geopolars.geo.distance import distance
from geopolars.geo.exterior import exterior
from geopolars.geo.interior import interior
from geopolars.geo.is_ccw import is_ccw
from geopolars.geo.is_close import is_close
from geopolars.geo.is_degenerate import is_degenerate, null_out_degenerate
from geopolars.geo.is_empty import is_empty
from geopolars.geo.is_ring import is_ring
from geopolars.geo.is_valid import is_valid, null_out_invalid
from geopolars.geo.length import length
from geopolars.geo.mean_coordinate import mean_coordinate
from geopolars.geo.rings_overlap import rings_overlap
from geopolars.geo.self_intersects import self_intersects
from geopolars.geo.simplify import simplify
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
    "convex_hull",
    "count_coordinates",
    "distance",
    "exterior",
    "from_wkb",
    "from_wkt",
    "interior",
    "is_ccw",
    "is_close",
    "is_degenerate",
    "is_empty",
    "is_geographic",
    "is_ring",
    "is_valid",
    "length",
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
    "null_out_degenerate",
    "null_out_invalid",
    "point",
    "polygon",
    "polygon_from_columns",
    "polygon_from_rings",
    "rings_overlap",
    "rotate",
    "rotate_about_centroid",
    "scale",
    "scale_about_centroid",
    "self_intersects",
    "set_crs",
    "simplify",
    "skew",
    "skew_about_centroid",
    "to_crs",
    "to_polygon",
    "to_wkb",
    "to_wkt",
    "translate",
    "wrap_longitude",
    "x",
    "y",
    "z",
]
