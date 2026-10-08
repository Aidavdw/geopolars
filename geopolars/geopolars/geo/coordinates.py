"""Getters for the coordinates a geometry is stored as."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _axis(values: pl.Expr, layers: int, axis: str) -> pl.Expr:
    """One axis of the coordinates."""
    # should keep this element-wise,
    # so not all geometries are walked at the same time.
    if layers == 0:
        return values.struct.field(axis)
    return values.list.eval(_axis(pl.element(), layers - 1, axis))


def _getter(axis: str) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    """Reads one axis off a geometry's storage, keeping its nesting."""

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        if axis not in geometry._dimension:
            msg = (
                f"{type(geometry).__name__} has no `{axis}` coordinate; "
                f"`{axis}` needs a geometry whose dimension includes it"
            )
            raise TypeError(msg)
        return _axis(column.ext.storage(), geometry._nesting, axis)

    return build


def x(geometry: IntoExprColumn) -> pl.Expr:
    """The `x` coordinates of a geometry.

    The nesting of the output of this function depends on how nested the geometry is.
    A point will return a single value, while a linestring will return a list of values.

    | in                  | out                     |
    |---------------------|-------------------------|
    | `Point`             | `f64`                   |
    | `LineString`        | `list[f64]`             |
    | `MultiPoint`        | `list[f64]`             |
    | `Polygon`           | `list[list[f64]]`       |
    | `MultiLineString`   | `list[list[f64]]`       |
    | `MultiPolygon`      | `list[list[list[f64]]]` |

    The values are returned as they are stored, in the unit of the CRS.
    The CRS itself is not carried over: the result is a plain float column.
    Every stored coordinate is included, so a polygon's rings keep their closing coordinate.
    A missing geometry has no `x`; an empty one gives an empty list.

    ```python
    df.select(geo.x("location"))
    ```
    """
    return on_geometry(geometry, _getter("x"))


def y(geometry: IntoExprColumn) -> pl.Expr:
    """The `y` coordinates of a geometry.

    Works exactly like `x`, for the `y` axis.

    ```python
    df.select(geo.y("location"))
    ```
    """
    return on_geometry(geometry, _getter("y"))


def z(geometry: IntoExprColumn) -> pl.Expr:
    """The `z` coordinates of a geometry.

    Works exactly like `x`, for the `z` axis.
    Only a geometry that has a `z` (`…XYZ` or `…XYZM`) has one to read:
    any other is refused.

    ```python
    df.select(geo.z("location"))
    ```
    """
    return on_geometry(geometry, _getter("z"))


def m(geometry: IntoExprColumn) -> pl.Expr:
    """The `m` (measure) values of a geometry.

    Works exactly like `x`, for the `m` axis.
    Only a geometry that has an `m` (`…XYM` or `…XYZM`) has one to read:
    any other is refused while the plan is built.

    ```python
    df.select(geo.m("location"))
    ```
    """
    return on_geometry(geometry, _getter("m"))
