"""Choosing an expression by the dtype of the column it is given.

Which geometry a column holds is in its dtype,
and every operation has to be written differently per geometry.

`Expr.pipe_with_dtype` hands us that dtype while Polars resolves the plan,
before anything is optimised or run.
So `build` is called once, with the one concrete geometry type,
and gives back a plain expression for just that geometry.
It works the same for a column, an expression or a Series.

A dtype that is no geometry of ours raises right there,
so a bad input is a schema error rather than something that waits for the data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GEOMETRIES, GeoArrowType

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn

# analog of `Kind::ALL` x `Dimension::ALL` on the Rust side.
GEOMETRY_TYPES: tuple[type[GeoArrowType], ...] = tuple(
    concrete for geometry in GEOMETRIES for concrete in geometry.dimensions()
)


def _names() -> str:
    """The extension names, for an error message. Mirrors `Kind::names`."""
    names = [f"`{geometry._extension_name}`" for geometry in GEOMETRIES]
    return f"{', '.join(names[:-1])} or {names[-1]}"


UNSUPPORTED = (
    f"expected a {_names()} column with separated x/y[/z][/m] coordinates "
    "and no extension metadata, which this version cannot carry through"
)


def _geometry_of(dtype: pl.DataType) -> type[GeoArrowType]:
    geometry = type(dtype)
    if geometry not in GEOMETRY_TYPES or dtype.ext_metadata() is not None:
        msg = f"{UNSUPPORTED}, got: {dtype!r}"
        raise TypeError(msg)
    return geometry


def on_geometry(
    value: IntoExprColumn,
    build: Callable[[pl.Expr, type[GeoArrowType]], pl.Expr],
) -> pl.Expr:
    """Wrapper for callbacks for pipe_with_dtype."""
    if isinstance(value, str):
        expr = pl.col(value)
    elif isinstance(value, pl.Series):
        expr = pl.lit(value)
    else:
        expr = value

    def resolved(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        # Whatever `build` returns is named after what it was built from last,
        # so put the input's name back on.
        return build(expr, _geometry_of(dtype)).alias(expr.meta.output_name())

    return expr.pipe_with_dtype(resolved)
