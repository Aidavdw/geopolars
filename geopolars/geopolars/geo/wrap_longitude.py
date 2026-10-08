"""Bringing longitudes that went past the antimeridian back into range."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _turns(x: pl.Expr, turn: float) -> pl.Expr:
    """How many whole turns `x` lies outside `[-turn/2, turn/2]`; 0 inside it (or NaN)."""
    half = turn / 2
    return (
        pl.when(x > half)
        .then(((x - half) / turn).ceil())
        .when(x < -half)
        .then(((x + half) / turn).floor())
        .otherwise(0.0)
    )


def _shifts(values: pl.Expr, layers: int, turn: float) -> pl.Expr:
    """`_turns` for every coordinate, keeping the geometry's nesting."""
    if layers == 0:
        return _turns(values.struct.field("x"), turn)
    return values.list.eval(_shifts(pl.element(), layers - 1, turn))


def _wrapped(values: pl.Expr, layers: int, turn: float) -> pl.Expr:
    """Every coordinate with its `x` moved back by its own whole turns.
    The other coordinates and the list offsets are left as they are."""
    if layers == 0:
        x = pl.field("x")
        return values.struct.with_fields(x=x - _turns(x, turn) * turn)
    return values.list.eval(_wrapped(pl.element(), layers - 1, turn))


def _crossing(storage: pl.Expr, layers: int, turn: float) -> pl.Expr:
    """Whether a geometry's coordinates would not all move by the same turns."""
    shifts = _shifts(storage, layers, turn)
    # One flat list of shifts per geometry, over every part and ring.
    for _ in range(layers - 1):
        shifts = shifts.list.eval(pl.element().explode())
    return shifts.list.min() != shifts.list.max()


def _wrap(skip_crossing: bool) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        turn = geometry._longitude_turn()
        if turn is None:
            msg = (
                f"{geometry!r} has no longitude to wrap: "
                "`wrap_longitude` needs a geographic CRS, such as EPSG:4326"
            )
            raise TypeError(msg)

        storage = column.ext.storage()
        layers = geometry._nesting
        wrapped = _wrapped(storage, layers, turn)
        if skip_crossing and layers > 0:
            crossing = _crossing(storage, layers, turn)
            wrapped = pl.when(crossing).then(storage).otherwise(wrapped)
        # The input's own dtype: the coordinates moved, the CRS did not.
        return wrapped.ext.to(geometry)

    return build


def wrap_longitude(geometry: IntoExprColumn, *, skip_crossing: bool = False) -> pl.Expr:
    """Move every `x` that went past the antimeridian back into range by whole turns.

    The range is half a turn either way, in the CRS's unit:
    `[-180, 180]` for degrees, `[-200, 200]` for grads.
    Coordinates already in range are left as they are, including exactly 180.
    So 190 becomes -170, -200 becomes 160, and 540 becomes 180.
    Only `x` moves: `y`, `z` and `m` are untouched.

    Each vertex moves on its own, which can tear a geometry apart.
    A line from 170 to 190 becomes one from 170 to -170, which runs the long way round.
    With `skip_crossing`, a geometry whose vertices would not all move by the same number
    of turns is left as it is, still past the antimeridian.
    A geometry that lies entirely past it is still moved, and moves as a whole.

    The geometry needs a geographic CRS: without one there is no longitude to wrap,
    and the expression is refused while the plan is built.

    ```python
    df.select(geo.wrap_longitude("route", skip_crossing=True))
    ```
    """
    return on_geometry(geometry, _wrap(skip_crossing))
