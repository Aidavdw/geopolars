"""Whether a linestring or polygon crosses or touches itself."""

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


def _self_intersects(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, _ACCEPTED):
        msg = (
            "self_intersects expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="self_intersects",
        is_elementwise=True,
    )


def self_intersects(geometry: IntoExprColumn) -> pl.Expr:
    """Whether a linestring or polygon crosses or touches itself.
    A multi-geometry gives a list instead, one per part.
    Parts crossing each other do not count.

    | in                      | out          |
    |-------------------------|--------------|
    | `LineStringXY`          | `bool`       |
    | `PolygonXY`             | `bool`       |
    | `MultiLineStringXYZM`   | `list[bool]` |
    | `MultiPolygonXYZM`      | `list[bool]` |

    A polygon intersects itself when one of its rings does,
    or when two of its holes overlap (see `rings_overlap`; holes that only touch do not).
    How the holes sit against the exterior is not looked at; see `is_valid` for that.
    Every ring has to be a ring, as for `rings_overlap`, or this raises.
    A polygon without rings does not intersect itself.

    This looks at a projected plane.
    Only x and y are looked at. z and m are ignored.
    Going back and forth between two points is considered crossing.
    Any other two segments may not touch at all,
    so a line whose end lies on itself intersects itself.
    A closed line (its last vertex at its first) may meet itself there, so a simple ring does not.
    A segment whose ends lie at the same place is skipped, so repeated vertices do not count.
    "At the same place" is within the default margin of `pl.Expr.is_close`.

    An empty linestring, a single vertex or a single segment does not intersect itself.
    A missing linestring gives a missing result,
    and so does one with a missing vertex, or one that is not finite in x or y.
    The same goes for a polygon and its rings.

    ```python
    df.filter(~geo.self_intersects("route"))
    ```
    """
    return on_geometry(geometry, _self_intersects)
