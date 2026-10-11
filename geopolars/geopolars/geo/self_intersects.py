"""Whether a linestring crosses or touches itself."""

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


def _self_intersects(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (LineStringType, MultiLineStringType)):
        msg = (
            "self_intersects expects a linestring or multilinestring column,"
            f" got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="self_intersects",
        is_elementwise=True,
    )


def self_intersects(geometry: IntoExprColumn) -> pl.Expr:
    """Whether a linestring crosses or touches itself.
    A multilinestring gives a list instead.
    For a Multilinestring, parts crossing each other do not count.

    | in                      | out          |
    |-------------------------|--------------|
    | `LineStringXY`          | `bool`       |
    | `MultiLineStringXYZM`   | `list[bool]` |

    This looks at a projected plane.
    Only x and y are looked at. z and m are ignored.
    If two segments in a row share a vertex, the are not considerend intersecting.
    Going back and forth between two points is considered crossing.
    Any other two segments may not touch at all,
    so a line whose end lies on itself intersects itself.
    A closed line (its last vertex at its first) may meet itself there, so a simple ring does not.
    A segment whose ends lie at the same place is skipped, so repeated vertices do not count.
    "At the same place" is within the default margin of `pl.Expr.is_close`.

    An empty linestring, a single vertex or a single segment does not intersect itself.
    A missing linestring gives a missing result,
    and so does one with a missing vertex, or one that is not finite in x or y.

    ```python
    df.filter(~geo.self_intersects("route"))
    ```
    """
    return on_geometry(geometry, _self_intersects)
