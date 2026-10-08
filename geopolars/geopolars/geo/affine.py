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


Origin = tuple[float, float] | tuple[float, float, float]


def _degrees(amount: float, unit: Literal["deg", "pi"], name: str) -> float:
    """`amount` in degrees, refused while the expression is built if it is not a number."""
    if unit not in ("deg", "pi"):
        msg = f'`unit` has to be "deg" or "pi", got: {unit!r}'
        raise ValueError(msg)
    if not math.isfinite(amount):
        msg = f"`{name}` has to be a finite number, got: {amount!r}"
        raise ValueError(msg)
    return amount * 180.0 if unit == "pi" else amount


def _origin(origin: Origin) -> tuple[float, float, float]:
    """`origin` as `(x, y, z)`, a `z` left out being 0."""
    if len(origin) not in (2, 3) or not all(math.isfinite(at) for at in origin):
        msg = f"`origin` has to be two or three finite numbers, got: {origin!r}"
        raise ValueError(msg)
    x, y, *z = origin
    return x, y, z[0] if z else 0.0


def _about(
    linear: tuple[tuple[float, float, float], ...], origin: tuple[float, float, float]
) -> list[float]:
    """The affine matrix that applies the 3×3 `linear` part about `origin` instead of (0, 0, 0).

    That is moving `origin` to (0, 0, 0), applying it, and moving it back:
    L.(p - origin) + origin, which is L.p + constant offset origin - L.origin.
    """
    matrix: list[float] = []
    for row, at in zip(linear, origin, strict=True):
        moved = sum(r * o for r, o in zip(row, origin, strict=True))
        matrix += [*row, at - moved]
    return matrix


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


def _tan(degrees: float, name: str) -> float:
    """The tangent of an angle, exact for whole eighth turns,
    refused where it does not exist (a quarter turn, either way).

    `math.tan(math.pi / 4)` is 0.9999999999999999 rather than 1.
    """
    eighths, rest = divmod(degrees, 45.0)
    if rest == 0.0:
        tangent = (0.0, 1.0, None, -1.0)[int(eighths) % 4]
        if tangent is None:
            msg = f"`{name}` cannot be a quarter turn (±90°): it would skew without end"
            raise ValueError(msg)
        return tangent
    return math.tan(math.radians(degrees))


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
    matrix = _about(rotation, origin)

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
    origin: Origin = (0.0, 0.0, 0.0),
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
    if axis not in ("x", "y", "z"):
        msg = f'`axis` has to be "x", "y" or "z", got: {axis!r}'
        raise ValueError(msg)
    degrees = _degrees(amount, unit, "amount")
    return on_geometry(geometry, _rotate(degrees, axis, _origin(origin)))


def skew(
    geometry: IntoExprColumn,
    xs: float = 0.0,
    ys: float = 0.0,
    unit: Literal["deg", "pi"] = "deg",
    origin: Origin = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Shear every position of a geometry in the plane, by an angle along each axis.

    `xs` leans the geometry along `x`: every position moves along `x`
    by `tan(xs)` times how far it is from `origin` along `y`,
    so a line that ran parallel to the `y` axis ends up at an angle `xs` from it,
    leaning towards positive `x` for a positive angle.
    `ys` does the same along `y`, by `tan(ys)` times the distance along `x`.
    Both are applied to the positions as they were:

    ```text
    x' = x + tan(xs)·(y - origin_y)
    y' = y + tan(ys)·(x - origin_x)
    ```

    The angles are in degrees, or with `unit="pi"` in multiples of π radians.
    A quarter turn (±90°) has no tangent and is refused.

    This works on any geometry: `z` stays as it is,
    and `m`, being a measure, is always carried through untouched.

    `origin` is the point that stays where it is, `(x, y)` or `(x, y, z)`.
    It is the same point for every geometry in the column, not the middle of each one.
    Coordinates are sheared as they are, in the units of their CRS.
    """
    tan_x = _tan(_degrees(xs, unit, "xs"), "xs")
    tan_y = _tan(_degrees(ys, unit, "ys"), "ys")
    # fmt: off
    shear = (
        (1,     tan_x, 0),
        (tan_y, 1,     0),
        (0,     0,     1),
    )
    # fmt: on
    matrix = _about(shear, _origin(origin))
    return on_geometry(geometry, lambda column, _: _affine(column, matrix))


def _scale(
    xfact: float, yfact: float, zfact: float, origin: tuple[float, float, float]
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    # fmt: off
    stretch = (
        (xfact, 0,     0),
        (0,     yfact, 0),
        (0,     0,     zfact),
    )
    # fmt: on
    matrix = _about(stretch, origin)

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # Without this, scaling an XY column by zfact would quietly do nothing.
        if zfact != 1.0 and "z" not in geometry._dimension:
            msg = f"cannot scale by zfact: {geometry!r} has no z coordinate"
            raise TypeError(msg)
        return _affine(column, matrix)

    return build


def scale(
    geometry: IntoExprColumn,
    xfact: float = 1.0,
    yfact: float = 1.0,
    zfact: float = 1.0,
    origin: Origin = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Stretch every position of a geometry away from (or towards) `origin`,
    by a factor along each axis:

    ```text
    x' = origin_x + xfact·(x - origin_x)
    ```

    and the same for `y` and `z`.
    A factor above 1 stretches, one between 0 and 1 shrinks,
    a negative one mirrors as well, and 0 flattens the geometry onto `origin` along that axis.

    A `zfact` other than 1 is refused for a geometry with no `z`,
    rather than silently ignored.
    `m`, being a measure, is always carried through untouched.

    `origin` is the point that stays where it is, `(x, y)` or `(x, y, z)`,
    by default `(0, 0, 0)`. A `z` left out is 0.
    It is the same point for every geometry in the column, not the middle of each one.
    Coordinates are scaled as they are, in the units of their CRS.
    """
    for name, factor in (("xfact", xfact), ("yfact", yfact), ("zfact", zfact)):
        if not math.isfinite(factor):
            msg = f"`{name}` has to be a finite number, got: {factor!r}"
            raise ValueError(msg)
    return on_geometry(geometry, _scale(xfact, yfact, zfact, _origin(origin)))
