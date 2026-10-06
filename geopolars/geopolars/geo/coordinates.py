"""Getters for the coordinates a geometry is stored as."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GeoPoint
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _axis(axis: str) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    """Reads one axis off a point's storage."""

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        if not isinstance(geometry, GeoPoint):
            msg = f"`{axis}` is read off a `geoarrow.point` column, got: {geometry!r}"
            raise TypeError(msg)
        return column.ext.storage().struct.field(axis)

    return build


def x(geometry: IntoExprColumn) -> pl.Expr:
    """The `x` coordinate of a point, as an `f64`.

    The value is returned as it is stored, in the unit of the CRS.
    The CRS itself is not carried over: the result is a plain float column.
    Other geometries are refused.
    A missing point has no `x`.

    ```python
    df.select(geo.x("location"))
    ```
    """
    return on_geometry(geometry, _axis("x"))


def y(geometry: IntoExprColumn) -> pl.Expr:
    """The `y` coordinate of a point, as an `f64`.

    The value is returned as it is stored, in the unit of the CRS.
    The CRS itself is not carried over: the result is a plain float column.
    Other geometries are refused.
    A missing point has no `y`.

    ```python
    df.select(geo.y("location"))
    ```
    """
    return on_geometry(geometry, _axis("y"))
