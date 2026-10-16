"""The convex hull of a linestring or polygon, or their multi-variants.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_ACCEPTED = (LineStringType, MultiLineStringType, PolygonType, MultiPolygonType)


def _convex_hull(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, _ACCEPTED):
        msg = (
            "convex_hull expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="convex_hull",
        is_elementwise=True,
    )


def convex_hull(geometry: IntoExprColumn) -> pl.Expr:
    """The smallest convex polygon that contains every vertex of a linestring or polygon,
    with its CRS.
    A multilinestring or multipolygon gives a single polygon:
    the hull of the geometry as a whole, not one per part.
    An XY or XYM geometry gives a `PolygonXY`, an XYZ or XYZM one a `PolygonXYZ`:

    | in                     | out             |
    |------------------------|-----------------|
    | `LineStringXY`         | `PolygonXY`     |
    | `MultiLineStringXYM`   | `PolygonXY`     |
    | `PolygonXYZ`           | `PolygonXYZ`    |
    | `MultiPolygonXYZM`     | `PolygonXYZ`    |

    Computed by GEOS.
    This operation keeps z, but drops m.
    An empty geometry gives an empty polygon, and a missing one a missing polygon.
    One that encloses nothing (all its vertices on one point or one line,
    such as a straight linestring) has a point or a line for a hull,
    so it gives a missing polygon.

    ```python
    df.select(geo.convex_hull("parcel"))
    ```
    """
    return on_geometry(geometry, _convex_hull)
