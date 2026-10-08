"""Moving a geometry through space."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _affine(column: pl.Expr, matrix: Sequence[float]) -> pl.Expr:
    """Transform every position (`x`, `y`, `z`) of a geometry by an affine matrix,
    given row by row as the twelve numbers of

    ```text
    | x' |   | a  b  c  xoff |   | x |
    | y' | = | d  e  f  yoff | · | y |
    | z' |   | g  h  i  zoff |   | z |
                                 | 1 |
    ```

    A geometry without a `z` reads it as 0 and has no `z'`.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="affine",
        is_elementwise=True,
        kwargs={"coefficients": [float(c) for c in matrix]},
    )


def _translate(
    dx: float, dy: float, dz: float
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # Without this, translating an XY column by dz would quietly do nothing.
        if dz != 0.0 and "z" not in geometry._dimension:
            msg = f"cannot translate by dz: {geometry!r} has no z coordinate"
            raise TypeError(msg)
        # fmt: off
        return _affine(column, (
            1, 0, 0, dx,
            0, 1, 0, dy,
            0, 0, 1, dz,
        ))
        # fmt: on

    return build


def translate(
    geometry: IntoExprColumn, dx: float, dy: float, dz: float = 0.0
) -> pl.Expr:
    """Shift every coordinate of a geometry by a constant offset.

    Works on any geometry, of any dimension. A non-zero `dz` is rejected for a
    geometry with no `z` rather than silently ignored. An `m` value is a measure,
    not a position, so it is always carried through untouched.

    This does not check if the new position might have wrapped around the antimeridian (± 180°).
    If your data might have, it would be best to call `wrap_longitude` on it as well.
    """
    return on_geometry(geometry, _translate(dx, dy, dz))


def _cos_sin(degrees: float) -> tuple[float, float]:
    """The cosine and sine of an angle, exact for whole quarter turns.

    `math.cos(math.pi / 2)` is 6e-17 rather than 0, which would leave a quarter turn
    of `(1, 0)` at `(6e-17, 1)` and make every row of the matrix read every coordinate.
    """
    quarters, rest = divmod(degrees, 90.0)
    if rest == 0.0:
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(quarters) % 4]
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def _rotate(
    degrees: float,
    axis: Literal["x", "y", "z"],
    origin: tuple[float, float, float],
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    cos, sin = _cos_sin(degrees)
    # Counter-clockwise for a positive angle, looking down the axis from its positive end.
    # fmt: off
    rotation = {
        "x": (
            (1,    0,    0),
            (0,    cos, -sin),
            (0,    sin,  cos),
        ),
        "y": (
            (cos,  0,    sin),
            (0,    1,    0),
            (-sin, 0,    cos),
        ),
        "z": (
            (cos, -sin,  0),
            (sin,  cos,  0),
            (0,    0,    1),
        ),
    }[axis]
    # fmt: on
    # Turning about `origin` is moving it to (0, 0, 0), turning, and moving it back:
    # R.(p - origin) + origin, which is R.p + constant offset origin - R.origin.
    matrix: list[float] = []
    for row, at in zip(rotation, origin, strict=True):
        turned = sum(r * o for r, o in zip(row, origin, strict=True))
        matrix += [*row, at - turned]

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # Turning about `x` or `y` moves positions into and out of `z`,
        # which a geometry without one cannot hold.
        if axis != "z" and "z" not in geometry._dimension:
            msg = (
                f"cannot rotate about the {axis} axis: {geometry!r} has no z coordinate"
            )
            raise TypeError(msg)
        return _affine(column, matrix)

    return build


def rotate(
    geometry: IntoExprColumn,
    amount: float,
    unit: Literal["deg", "pi"] = "deg",
    axis: Literal["x", "y", "z"] = "z",
    origin: tuple[float, float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Turn every position of a geometry about an axis through `origin`.

    `amount` is in degrees, or with `unit="pi"` in multiples of π radians:
    `rotate("shape", 90)` and `rotate("shape", 0.5, unit="pi")` are the same quarter turn.
    A positive amount turns counter-clockwise, looking down the axis from its positive end
    towards the origin (the right-hand rule).

    By default this turns about the `z` axis: in the plane, leaving `z` as it is,
    so it works on any geometry.
    Turning about `x` or `y` needs a geometry with a `z`, and is refused for one without.
    `m`, being a measure, is always carried through untouched.

    `origin` is the point the axis runs through, `(x, y)` or `(x, y, z)`,
    by default `(0, 0, 0)`. A `z` left out is 0.
    It is the same point for every geometry in the column, not the middle of each one.
    Coordinates are turned as they are, in the units of their CRS:
    in longitude/latitude, this does not turn anything on the globe,
    and can leave positions outside of the valid range.
    """
    if unit not in ("deg", "pi"):
        msg = f'`unit` has to be "deg" or "pi", got: {unit!r}'
        raise ValueError(msg)
    if axis not in ("x", "y", "z"):
        msg = f'`axis` has to be "x", "y" or "z", got: {axis!r}'
        raise ValueError(msg)
    if not math.isfinite(amount):
        msg = f"`amount` has to be a finite number, got: {amount!r}"
        raise ValueError(msg)
    if len(origin) not in (2, 3) or not all(math.isfinite(at) for at in origin):
        msg = f"`origin` has to be two or three finite numbers, got: {origin!r}"
        raise ValueError(msg)
    degrees = amount * 180.0 if unit == "pi" else amount
    x, y, *z = origin
    return on_geometry(geometry, _rotate(degrees, axis, (x, y, z[0] if z else 0.0)))
