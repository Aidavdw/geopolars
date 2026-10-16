"""The convex hull of a polygon or multipolygon.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import MultiPolygonType, PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _convex_hull(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (PolygonType, MultiPolygonType)):
        msg = f"convex_hull expects a polygon or multipolygon column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="convex_hull",
        is_elementwise=True,
    )


def convex_hull(geometry: IntoExprColumn) -> pl.Expr:
    """The smallest convex polygon that contains every vertex of a polygon,
    with the polygon's CRS.
    A multipolygon gives a single polygon:
    the hull of the multipolygon as a whole, not one per part.

    | in                                    | out             |
    |---------------------------------------|-----------------|
    | `PolygonXY`, `PolygonXYM`             | `PolygonXY`     |
    | `PolygonXYZ`, `PolygonXYZM`           | `PolygonXYZ`    |
    | `MultiPolygonXY`, `MultiPolygonXYM`   | `PolygonXY`     |
    | `MultiPolygonXYZ`, `MultiPolygonXYZM` | `PolygonXYZ`    |

    Computed by GEOS.
    This operation keeps z, but drops m.
    An empty (multi)polygon gives an empty polygon, and a missing one a missing polygon.
    One that encloses nothing (all its vertices on one point or one line)
    has a point or a line for a hull, so it gives a missing polygon.

    ```python
    df.select(geo.convex_hull("parcel"))
    ```
    """
    return on_geometry(geometry, _convex_hull)
