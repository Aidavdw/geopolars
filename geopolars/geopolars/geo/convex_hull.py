"""The convex hull of a linestring or polygon, or their multi-variants.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
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


@doc(output="polygon")
def convex_hull(
    geometry: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    ccw_outer: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """The smallest convex polygon that contains every vertex of a linestring or polygon,
    with its CRS.
    A multipoint, multilinestring or multipolygon gives a single polygon:
    the hull of the geometry as a whole, not one per part.
    Only the exterior of a polygon counts, so its holes are ignored.
    An XY or XYM geometry gives a `PolygonXY`, an XYZ or XYZM one a `PolygonXYZ`:

    | in | out |
    | - | - |
    | `PointXY[ZM]` | refused |
    | `LineStringXY[M]` | `PolygonXY` |
    | `LineStringXYZ[M]` | `PolygonXYZ` |
    | `PolygonXY[M]` | `PolygonXY` |
    | `PolygonXYZ[M]` | `PolygonXYZ` |
    | `MultiPointXY[M]` | `PolygonXY` |
    | `MultiPointXYZ[M]` | `PolygonXYZ` |
    | `MultiLineStringXY[M]` | `PolygonXY` |
    | `MultiLineStringXYZ[M]` | `PolygonXYZ` |
    | `MultiPolygonXY[M]` | `PolygonXY` |
    | `MultiPolygonXYZ[M]` | `PolygonXYZ` |

    {computed_by_geos_keeping_z}
    {empty_and_missing}
    One that encloses nothing (all its vertices on one point or one line,
    such as a straight linestring, or a multipoint of one or two points)
    has a point or a line for a hull,
    so it gives a missing polygon.

    CRS is ignored:
    for longitude/latitude the hull's edges are straight in degrees, not along the globe.
    A geometry crossing the antimeridian is hulled the long way round.

    {geos_errors}

    {geos_non_finite}
    Depending on where such a vertex sits, it is left out of the hull,
    the polygon is missing, or GEOS raises.
    {early_finite_check}

    {ccw_outer}

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
                "early_finite_check": early_finite_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
