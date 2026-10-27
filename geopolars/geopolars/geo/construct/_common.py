"""Helpers shared by the constructors of every geometry."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.geo.close_rings import CloseRings


def _coord(value: IntoExprColumn, name: str) -> pl.Expr:
    """One coordinate column of a geometry's storage struct."""
    return _named(value, name).cast(pl.Float64)


def _named(value: IntoExprColumn, name: str) -> pl.Expr:
    """One argument, under the name of the coordinate it carries.

    The plugin reads the dimension back off these names, so they are the spec's
    (`x`, `y`, `z`, `m`) rather than whatever the column was called.
    """
    if isinstance(value, str):
        expr = pl.col(value)
    elif isinstance(value, pl.Series):
        expr = pl.lit(value)
    else:
        expr = value
    return expr.alias(name)


def _given(
    x: IntoExprColumn,
    y: IntoExprColumn | None,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
) -> dict[str, IntoExprColumn]:
    """The coordinates that were passed, in the field order the spec fixes."""
    given = {"x": x, "y": y, "z": z, "m": m}
    return {name: value for name, value in given.items() if value is not None}


def _metadata(crs: str | None) -> dict[str, str | None]:
    """Mirrors `MetadataKwargs` in `src/geoarrow/crs.rs`."""
    return {"crs": crs}


def _polygon_kwargs(crs: str | None, close: CloseRings | None) -> dict[str, Any]:
    """Mirrors `PolygonKwargs` in `src/expr/construct.rs`."""
    return {**_metadata(crs), "close": None if close is None else asdict(close)}


def _decode_kwargs(dtype: type[GeoArrowType], crs: str | None) -> dict[str, str | None]:
    """Mirrors `DecodeKwargs` in `src/geoarrow/decode.rs`:
    the concrete geometry `from_wkb` / `from_wkt` decode into.
    """
    dimension = getattr(dtype, "_dimension", ())
    if not dimension:
        msg = f"expected a concrete geometry type such as `PolygonXY`, got: {dtype!r}"
        raise TypeError(msg)
    return {
        "kind": dtype._display,
        "dimension": "".join(dimension),
        **_metadata(crs),
    }


def _from_columns(
    function_name: str,
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
    kwargs: dict[str, Any],
) -> pl.Expr:
    """Zip one nested coordinate column per axis into one geometry per row."""
    # A plugin call rather than expressions:
    # Polars can transpose a struct of lists into a list of structs only by exploding and regrouping,
    # where the plugin reuses the offsets the columns already have.
    return register_plugin_function(
        plugin_path=LIB,
        args=[_named(value, name) for name, value in _given(x, y, z, m).items()],
        function_name=function_name,
        is_elementwise=True,
        kwargs=kwargs,
    )


def _gather(
    function_name: str, parts: IntoExprColumn, kwargs: dict[str, Any]
) -> pl.Expr:
    """Gather a list column of a geometry's parts into one geometry per row."""
    # Unlike `point`, this is a plugin call: the output dimension is only
    # knowable from the input's dtype, which `output_type_func` gets to see and
    # Python does not.
    return register_plugin_function(
        plugin_path=LIB,
        args=[parts],
        function_name=function_name,
        is_elementwise=True,
        kwargs=kwargs,
    )
