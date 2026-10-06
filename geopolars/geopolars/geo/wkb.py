"""Converting between `geoarrow.wkb` and (binary) native geometries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo.construct import _metadata

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def to_wkb(geometry: IntoExprColumn) -> pl.Expr:
    """Encode every geometry as (little-endian, ISO) WKB.
    The output is in a `Wkb` column.
    The geometry's metadata, such as its CRS, is carried over.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="to_wkb",
        is_elementwise=True,
    )


def from_wkb(
    wkb: IntoExprColumn,
    geometry: type[GeoArrowType],
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Decode a `Wkb` or plain binary column into `geometry`.

    WKB can mix geometries row by row, but a Polars column has one dtype,
    so `geometry` names the concrete type to decode into, e.g. `PolygonXY`.
    A single geometry is promoted into its multi geometry
    (e.g. a polygon into `MultiPolygonXY`).
    Anything else that does not fit, including a different dimension, is an error.

    The metadata of a `Wkb` column is carried over, and `crs` labels the result
    the same way it does for the constructors.

    ```python
    pl.read_parquet("parcels.parquet").select(
        geo.from_wkb("geometry", MultiPolygonXY, crs="EPSG:4326")
    )
    ```
    """
    dimension = getattr(geometry, "_dimension", ())
    if not dimension:
        msg = (
            f"expected a concrete geometry type such as `PolygonXY`, got: {geometry!r}"
        )
        raise TypeError(msg)

    return register_plugin_function(
        plugin_path=LIB,
        args=[wkb],
        function_name="from_wkb",
        is_elementwise=True,
        kwargs={
            "kind": geometry._display,
            "dimension": "".join(dimension),
            **_metadata(crs),
        },
    )
