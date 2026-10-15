"""The convex hull of a polygon.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _convex_hull(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, PolygonType):
        msg = f"convex_hull expects a polygon column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="convex_hull",
        is_elementwise=True,
    )


def convex_hull(geometry: IntoExprColumn) -> pl.Expr:
    """The smallest convex polygon that contains every vertex of a polygon,
    with the polygon's dimension and CRS.

    | in              | out             |
    |-----------------|-----------------|
    | `PolygonXY`     | `PolygonXY`     |
    | `PolygonXYZM`   | `PolygonXYZM`   |

    Computed by GEOS, on x and y. GEOS keeps z, but drops m, so the hull's m is NaN.
    An empty polygon gives an empty polygon, and a missing polygon a missing one.
    A polygon that encloses nothing (all its vertices on one point or one line)
    has a point or a line for a hull, so it gives a missing polygon.

    ```python
    df.select(geo.convex_hull("parcel"))
    ```
    """
    return on_geometry(geometry, _convex_hull)
