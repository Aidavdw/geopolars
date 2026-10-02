"""Polars plugin expressions, with a statically-typed namespace accessor.

In code organisation, the layering is one-directional:
`expr` (namespaces) -> the functional modules -> `datatypes`.
There should be no links the other direction.
"""

from geopolars import datatypes, geo
from geopolars.datatypes import (
    GeoLineString,
    GeoMultiLineString,
    GeoMultiPoint,
    GeoPoint,
    GeoPolygon,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    LineStringXYZM,
    MultiLineStringXY,
    MultiLineStringXYM,
    MultiLineStringXYZ,
    MultiLineStringXYZM,
    MultiPointXY,
    MultiPointXYM,
    MultiPointXYZ,
    MultiPointXYZM,
    PointXY,
    PointXYM,
    PointXYZ,
    PointXYZM,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)

# Importing this registers the namespaces on pl.Expr as a side effect.
from geopolars.expr import Geometry, PluginExpr, as_plugin, col

__all__ = [
    "GeoLineString",
    "GeoMultiLineString",
    "GeoMultiPoint",
    "GeoPoint",
    "GeoPolygon",
    "Geometry",
    "LineStringXY",
    "LineStringXYM",
    "LineStringXYZ",
    "LineStringXYZM",
    "MultiLineStringXY",
    "MultiLineStringXYM",
    "MultiLineStringXYZ",
    "MultiLineStringXYZM",
    "MultiPointXY",
    "MultiPointXYM",
    "MultiPointXYZ",
    "MultiPointXYZM",
    "PluginExpr",
    "PointXY",
    "PointXYM",
    "PointXYZ",
    "PointXYZM",
    "PolygonXY",
    "PolygonXYM",
    "PolygonXYZ",
    "PolygonXYZM",
    "as_plugin",
    "col",
    "datatypes",
    "geo",
]
