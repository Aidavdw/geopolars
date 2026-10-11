"""Whether a geometry runs in the standard orientation:
counter-clockwise, with a polygon's holes the other way round.

A native kernel (A-tier) takes the sign of each ring's shoelace sum in place.
"""

from __future__ import annotations

from functools import partial
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


def _is_standard_orientation(
    column: pl.Expr, geometry: GeoArrowType, *, holes_clockwise: bool
) -> pl.Expr:
    if not isinstance(
        geometry,
        (LineStringType, PolygonType, MultiLineStringType, MultiPolygonType),
    ):
        msg = (
            "is_standard_orientation expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="is_standard_orientation",
        kwargs={"holes_clockwise": holes_clockwise},
        is_elementwise=True,
    )


def is_standard_orientation(
    geometry: IntoExprColumn, *, holes_clockwise: bool = True
) -> pl.Expr:
    """Whether a geometry runs in the standard orientation, as GeoParquet and OGC wind it.

    To be in the standard orientation,
    a linestring has to run counter-clockwise.
    A polygon has to have its exterior counter-clockwise and its holes clockwise,
    (standard for GeoParquet) or counter-clockwise too with `holes_clockwise=False`.
    A multilinestring or multipolygon gives a list instead, one per part.

    | in                      | out          |
    |-------------------------|--------------|
    | `LineStringXY`          | `bool`       |
    | `PolygonXY`             | `bool`       |
    | `MultiLineStringXYZM`   | `list[bool]` |
    | `MultiPolygonXYZM`      | `list[bool]` |

    The orientation is the sign of the area the ring encloses in x and y; z and m are ignored.
    An empty linestring, ring or polygon is always in the standard orientation.
    Any other ring that encloses nothing (too few vertices, all on a line,
    or with a missing vertex) runs neither way, so it is never in the standard orientation.

    Whether a line is closed is not checked: an open line is taken as closed back to its start.
    Combine with `is_ring` or `is_valid` to check that too.

    A missing geometry gives a missing result, and so does a polygon with a missing ring.

    ```python
    df.select(geo.is_standard_orientation("parcel"))
    ```
    """
    return on_geometry(
        geometry,
        partial(_is_standard_orientation, holes_clockwise=holes_clockwise),
    )
