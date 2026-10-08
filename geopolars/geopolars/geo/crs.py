"""Labelling a geometry with a coordinate reference system, or moving it to another."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def set_crs(geometry: IntoExprColumn, crs: str, *, force: bool = False) -> pl.Expr:
    """Declare the CRS the coordinates are in, without moving them.

    `crs` is anything PROJ accepts:
    e.g. PROJJSON, WKT or an `AUTH:CODE` format such as `"EPSG:4326"`.
    It is only a label: nothing is checked or reprojected (that is `to_crs`).
    A geometry that already declares a CRS is refused while the plan is built,
    unless `force` is set, in which case its label is overwritten.
    Any other metadata is kept.

    ```python
    df.select(geo.set_crs("parcel", "EPSG:28992"))
    ```
    """

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        if geometry._declares_crs() and not force:
            msg = (
                f"{geometry!r} already declares a CRS; use `to_crs` to reproject it, "
                "or `force=True` to overwrite the label"
            )
            raise ValueError(msg)
        return column.ext.storage().ext.to(geometry._with_crs(crs))

    return on_geometry(geometry, build)


def is_geographic(geometry: IntoExprColumn, *, ignore_errors: bool = False) -> pl.Expr:
    """Whether the column's CRS is geographic: its `x` and `y` are longitude and latitude.

    False if the column has no CRS, or with a projected one.
    The CRS is the same for every row, so this is decided while the plan is built
    and gives a single bool (broadcast by `with_columns`), not one per row.
    A CRS that cannot be read is refused while the plan is built,
    unless `ignore_errors` is set, in which case it gives False.

    ```python
    df.with_columns(geo.is_geographic("parcel"))
    ```
    """

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        try:
            turn = geometry._longitude_turn()
        except (pl.exceptions.ComputeError, ValueError):
            if not ignore_errors:
                raise
            turn = None
        return pl.lit(turn is not None)

    return on_geometry(geometry, build)


def to_crs(geometry: IntoExprColumn, crs: str) -> pl.Expr:
    """Reproject every coordinate from the column's CRS to `crs`.

    `crs` is anything PROJ accepts:
    e.g. PROJJSON, WKT or an `AUTH:CODE` format such as `"EPSG:4326"`.
    The source CRS is read from the column's metadata.
    Only `x` and `y` are reprojected, so `z` and `m` are carried through untouched.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="to_crs",
        is_elementwise=True,
        kwargs={"to": crs},
    )
