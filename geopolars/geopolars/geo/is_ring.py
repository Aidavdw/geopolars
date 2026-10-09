"""Whether a linestring is a ring: closed, and long enough to enclose something.

A native kernel (A-tier) compares the endpoints in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, MultiLineStringType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _is_ring(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (LineStringType, MultiLineStringType)):
        msg = (
            f"is_ring expects a linestring or multilinestring column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="is_ring",
        is_elementwise=True,
    )


def is_ring(geometry: IntoExprColumn) -> pl.Expr:
    """Whether a linestring is a ring: it has at least 4 vertices,
    and its last vertex lies coincides with its first (within margin default margin).
    A multilinestring gives a list instead, one per part.

    | in                      | out          |
    |-------------------------|--------------|
    | `LineStringXY`          | `bool`       |
    | `MultiLineStringXYZM`   | `list[bool]` |

    m is a measure along the line rather than a position, so it is not compared.
    4 vertices is a triangle plus the one closing it, so a closed line with fewer is no ring.

    An empty linestring is no ring. A missing linestring gives a missing result.

    This is what the polygon constructors check every ring against.

    ```python
    df.select(geo.is_ring("route"))
    ```
    """
    return on_geometry(geometry, _is_ring)
