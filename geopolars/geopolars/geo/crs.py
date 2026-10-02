"""Moving a geometry from one coordinate reference system to another."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn


def to_crs(expr: IntoExprColumn, crs: str) -> pl.Expr:
    """Reproject every coordinate from the column's CRS to `crs`.

    `crs` is anything PROJ accepts:
    e.g. PROJJSON, WKT or an `AUTH:CODE` format such as `"EPSG:4326"`.
    The source CRS is read from the column's metadata.
    Only `x` and `y` are reprojected, so `z` and `m` are carried through untouched.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="to_crs",
        is_elementwise=True,
        kwargs={"to": crs},
    )
