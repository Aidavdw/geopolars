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


def convex_hull(
    geometry: IntoExprColumn, *, stop_on_error: bool = False, ccw_outer: bool = False
) -> pl.Expr:
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

    CRS is ignored:
    for longitude/latitude the hull's edges are straight in degrees, not along the globe.
    A geometry crossing the antimeridian is hulled the long way round.

    A geometry GEOS fails on gives a missing polygon too.
    With `stop_on_error`, GEOS's error is raised instead, failing the query.

    The exterior runs as GEOS gives it, clockwise.
    With `ccw_outer`, it runs counter-clockwise instead at a minimal performance cost.

    ```python
    df.select(geo.convex_hull("parcel"))
    ```
    """

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if not isinstance(dtype, _ACCEPTED):
            msg = (
                "convex_hull expects a linestring, polygon, multilinestring"
                f" or multipolygon column, got: {dtype!r}"
            )
            raise TypeError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="convex_hull",
            kwargs={"stop_on_error": stop_on_error, "ccw_outer": ccw_outer},
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
