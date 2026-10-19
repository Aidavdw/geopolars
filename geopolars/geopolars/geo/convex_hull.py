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
    MultiPointType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_ACCEPTED = (
    LineStringType,
    MultiPointType,
    MultiLineStringType,
    PolygonType,
    MultiPolygonType,
)


def convex_hull(
    geometry: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    ccw_outer: bool = False,
) -> pl.Expr:
    """The smallest convex polygon that contains every vertex of a linestring or polygon,
    with its CRS.
    A multipoint, multilinestring or multipolygon gives a single polygon:
    the hull of the geometry as a whole, not one per part.
    Only the exterior of a polygon counts, so its holes are ignored.
    An XY or XYM geometry gives a `PolygonXY`, an XYZ or XYZM one a `PolygonXYZ`:

    | in                     | out             |
    |------------------------|-----------------|
    | `LineStringXY`         | `PolygonXY`     |
    | `MultiPointXYM`        | `PolygonXY`     |
    | `MultiLineStringXYM`   | `PolygonXY`     |
    | `PolygonXYZ`           | `PolygonXYZ`    |
    | `MultiPolygonXYZM`     | `PolygonXYZ`    |

    Computed by GEOS.
    This operation keeps z, but drops m.
    An empty geometry gives an empty polygon, and a missing one a missing polygon.
    One that encloses nothing (all its vertices on one point or one line,
    such as a straight linestring, or a multipoint of one or two points)
    has a point or a line for a hull,
    so it gives a missing polygon.

    CRS is ignored:
    for longitude/latitude the hull's edges are straight in degrees, not along the globe.
    A geometry crossing the antimeridian is hulled the long way round.

    A geometry GEOS fails on raises GEOS's error, failing the query.
    With `ignore_per_row_errors=True`, it gives a missing polygon instead.

    The exterior runs as GEOS gives it, clockwise.
    With `ccw_outer=True`, it runs counter-clockwise instead at a minimal performance cost.

    ```python
    df.select(geo.convex_hull("parcel"))
    ```
    """

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if not isinstance(dtype, _ACCEPTED):
            msg = (
                "convex_hull expects a linestring, polygon, multipoint, multilinestring"
                f" or multipolygon column, got: {dtype!r}"
            )
            raise TypeError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="convex_hull",
            kwargs={
                "ignore_per_row_errors": ignore_per_row_errors,
                "ccw_outer": ccw_outer,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
