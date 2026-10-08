"""The bounding box of a geometry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import (
    GeoBox,
    GeoMultiLineString,
    GeoMultiPoint,
    GeoMultiPolygon,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

# Reduces a list of values to its smallest or largest one.
Reduce = Callable[[pl.Expr], pl.Expr]


def _extreme(values: pl.Expr, layers: int, axis: str, reduce: Reduce) -> pl.Expr:
    """The smallest or largest value of one axis, `layers` lists deep."""
    if layers == 0:
        return values.struct.field(axis)
    # `min` and `max` skip null (an empty part) and NaN (an empty point),
    # so only what has no coordinates at all comes out null or NaN.
    return reduce(values.list.eval(_extreme(pl.element(), layers - 1, axis, reduce)))


def _around_the_globe(
    low: pl.Expr, high: pl.Expr, turn: float
) -> tuple[pl.Expr, pl.Expr]:
    """A longitude range brought back to start in `[-turn / 2, turn / 2)`.
    A range that went past the antimeridian ends up with `low > high`,
    the way a box crosses it. One spanning a full turn or more covers the globe.
    """
    start = -turn / 2
    full = (high - low) >= turn
    # Both ends move by the turns that bring `low` into range,
    # so a range lying entirely past the antimeridian moves as a whole.
    turns = ((low - start) / turn).floor()
    low, high = low - turns * turn, high - turns * turn
    high = pl.when(high > -start).then(high - turn).otherwise(high)

    return (
        pl.when(full).then(start).otherwise(low),
        pl.when(full).then(-start).otherwise(high),
    )


def _box(
    part: pl.Expr,
    layers: int,
    box: GeoBox,
    turn: float | None,
    margin_x: float,
    margin_y: float,
) -> pl.Expr:
    """The box around one part, `layers` lists deep."""
    margin = {"x": margin_x, "y": margin_y}
    dimension = box._dimension
    low, high = {}, {}
    for axis in dimension:
        low[axis] = _extreme(part, layers, axis, lambda e: e.list.min())
        high[axis] = _extreme(part, layers, axis, lambda e: e.list.max())
        if axis in margin:
            low[axis] = low[axis] - margin[axis]
            high[axis] = high[axis] + margin[axis]

    if turn is not None:
        # After the margin, which can push a box over the antimeridian too.
        low["x"], high["x"] = _around_the_globe(low["x"], high["x"], turn)

    for axis in dimension:
        # Nothing to bound: the spec writes an empty range as `inf` to `-inf`.
        low[axis] = low[axis].fill_nan(None).fill_null(float("inf"))
        high[axis] = high[axis].fill_nan(None).fill_null(float("-inf"))

    bounds = [low[axis].alias(f"{axis}min") for axis in dimension] + [
        high[axis].alias(f"{axis}max") for axis in dimension
    ]
    return pl.when(part.is_not_null()).then(pl.struct(bounds)).ext.to(box)


def _bounds(
    column: pl.Expr, geometry: GeoArrowType, margin_x: float, margin_y: float
) -> pl.Expr:
    storage = column.ext.storage()
    box = GeoBox.of_dimension(geometry._dimension)._with_metadata_of(geometry)
    turn = geometry._longitude_turn()
    if isinstance(geometry, (GeoMultiPoint, GeoMultiLineString, GeoMultiPolygon)):
        # One box per part. A missing multi-geometry stays missing.
        layers = geometry._nesting - 1
        return storage.list.eval(
            _box(pl.element(), layers, box, turn, margin_x, margin_y)
        )
    return _box(storage, geometry._nesting, box, turn, margin_x, margin_y)


def _envelope(geometry: IntoExprColumn) -> pl.Expr:
    """same as `bounds`, except:
    - One box per row around the whole geometry
    - all parts of a multi-geometry together.
    Used for a GeoParquet covering.
    """

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        box = GeoBox.of_dimension(dtype._dimension)._with_metadata_of(dtype)
        storage = column.ext.storage()
        return _box(storage, dtype._nesting, box, dtype._longitude_turn(), 0.0, 0.0)

    return on_geometry(geometry, build)


def bounds(
    geometry: IntoExprColumn, *, margin_x: float = 0.0, margin_y: float = 0.0
) -> pl.Expr:
    """The bounding box of a geometry, as a `geoarrow.box` with its dimension and CRS.
    A multi-geometry gives a list of boxes instead, one per part.

    | in                  | out                   |
    |---------------------|-----------------------|
    | `PointXY`           | `BoxXY`               |
    | `LineStringXYZ`     | `BoxXYZ`              |
    | `PolygonXYM`        | `BoxXYM`              |
    | `MultiPointXY`      | `list[BoxXY]`         |
    | `MultiPolygonXYZM`  | `list[BoxXYZM]`       |

    A point's box repeats its own coordinates: `xmin == xmax`.
    z and m are bounded like x and y.
    `margin_x` and `margin_y` widen the box by that much on both sides
    (a negative margin shrinks it); z and m get no margin.

    The bounds are the minimum and maximum of the stored coordinates.
    With a geographic CRS, `x` is then brought back into `[-180, 180]`
    (or the same in the CRS's unit).
    This considers the margin as well.
    `xmin` lands in `[-180, 180]`, and `xmax` moves by the same turns:
    a geometry running from 170 to 190 gets a box from 170 to -170,
    which is how a box crosses the antimeridian (`xmin > xmax`),
    and one from 540 to 545 gets a box from -180 to -175.
    One spanning a full turn or more gets `-180` to `180`.
    A geometry that crosses the antimeridian without leaving that range,
    such as a line from 170 straight to -170, cannot be told apart
    from one going the long way round, and gets a box from -170 to 170.
    Without a geographic CRS the bounds are left as they are.
    A geometry with no coordinates (or only empty points) gets an empty box,
    `inf` to `-inf` in every range. A missing geometry gives a missing box.

    ```python
    df.select(geo.bounds("parcel", margin_x=10.0, margin_y=10.0))
    ```
    """
    return on_geometry(
        geometry,
        lambda column, dtype: _bounds(column, dtype, margin_x, margin_y),
    )
