"""Moving a geometry through space."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _translate(
    dx: float, dy: float, dz: float
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # Without this, translating an XY column by dz would quietly do nothing.
        if dz != 0.0 and "z" not in geometry._dimension:
            msg = f"cannot translate by dz: {geometry!r} has no z coordinate"
            raise TypeError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="translate",
            is_elementwise=True,
            kwargs={"dx": dx, "dy": dy, "dz": dz},
        )

    return build


def translate(
    geometry: IntoExprColumn, dx: float, dy: float, dz: float = 0.0
) -> pl.Expr:
    """Shift every coordinate of a geometry by a constant offset.

    Works on any geometry, of any dimension. A non-zero `dz` is rejected for a
    geometry with no `z` rather than silently ignored. An `m` value is a measure,
    not a position, so it is always carried through untouched.
    """
    return on_geometry(geometry, _translate(dx, dy, dz))
