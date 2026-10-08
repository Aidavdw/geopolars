"""Whether a geometry holds anything at all."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _empty(parts: pl.Expr, layers: int) -> pl.Expr:
    """Whether every innermost list of coordinates is empty."""
    if layers == 1:
        return parts.list.len() == 0
    # `all` of no parts is true: a polygon without rings is empty.
    return parts.list.eval(_empty(pl.element(), layers - 1)).list.all()


def _is_empty(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    storage = column.ext.storage()
    # A point always stores one coordinate,
    # so GeoArrow spells the empty point as a NaN `x` and `y`.
    if geometry._nesting == 0:
        return storage.struct.field("x").is_nan() & storage.struct.field("y").is_nan()
    return _empty(storage, geometry._nesting)


def is_empty(geometry: IntoExprColumn) -> pl.Expr:
    """Whether a geometry is empty: it has no coordinates at all.

    | in                 | empty when                             |
    |--------------------|----------------------------------------|
    | `Point…`           | `x` and `y` are both NaN               |
    | `LineString…`      | it has no vertices                     |
    | `Polygon…`         | it has no rings, or only empty ones    |
    | `MultiPoint…`      | it has no points                       |
    | `MultiLineString…` | it has no parts, or only empty ones    |
    | `MultiPolygon…`    | it has no polygons, or only empty ones |

    A NaN point inside a multipoint is a point like any other,
    not an empty one.
    A missing geometry is neither empty nor not: it gives null.

    ```python
    df.filter(~geo.is_empty("parcel"))
    ```
    """
    return on_geometry(geometry, _is_empty)
