"""Polars plugin expressions, with a statically-typed namespace accessor.

# TODO: move this to a readme?

We present two equivalent APIs:
```python

    # functional
    from geopolars import geo
    geo.translate("route", dx=1.0, dy=2.0)

    # expressions using a namespace
    import geopolars as gp
    gpl.col("route").geo.translate(1.0, 2.0)
```

Both are fully type-checked, but with an asterisk.
Plain `pl.col("route").geo...` works at runtime,
but a checker cannot see namespaces that `register_expr_namespace`
patches onto `pl.Expr`.
To avoid that, prefer `gpl.col` to keep static typing.

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
