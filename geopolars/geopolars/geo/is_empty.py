"""Whether a geometry holds anything at all."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GeoLineString, GeoMultiPoint
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


def _empty_point(point: pl.Expr) -> pl.Expr:
    return point.struct.field("x").is_nan() & point.struct.field("y").is_nan()


def _is_empty(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    storage = column.ext.storage()
    if geometry._nesting == 0:
        return _empty_point(storage)
    # Their vertices or parts are points, which can be empty.
    if isinstance(geometry, (GeoLineString, GeoMultiPoint)):
        return storage.list.eval(_empty_point(pl.element())).list.all()
    return _empty(storage, geometry._nesting)


def is_empty(geometry: IntoExprColumn) -> pl.Expr:
    """Whether a geometry is empty: it has no coordinates, or only empty points.

    | in                 | empty when                             |
    |--------------------|----------------------------------------|
    | `Point…`           | `x` and `y` are both NaN               |
    | `LineString…`      | it has no vertices, or only empty ones |
    | `Polygon…`         | it has no rings, or only empty ones    |
    | `MultiPoint…`      | it has no points, or only empty ones   |
    | `MultiLineString…` | it has no parts, or only empty ones    |
    | `MultiPolygon…`    | it has no polygons, or only empty ones |

    A linestring or multipoint whose points are all empty is empty as well,
    but a single non-empty point makes it non-empty.
    A missing geometry is neither empty nor not: it gives null.

    ```python
    df.filter(~geo.is_empty("parcel"))
    ```
    """
    return on_geometry(geometry, _is_empty)
