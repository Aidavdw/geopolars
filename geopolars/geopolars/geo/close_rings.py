"""Closing the open rings (and linestrings) of a geometry."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _close_rings(
    column: pl.Expr, geometry: GeoArrowType, *, snap_z: bool, snap_m: bool
) -> pl.Expr:
    if not isinstance(
        geometry,
        (LineStringType, PolygonType, MultiLineStringType, MultiPolygonType),
    ):
        msg = (
            "close_rings expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="close_rings",
        kwargs={"snap_z": snap_z, "snap_m": snap_m},
        is_elementwise=True,
    )


@doc()
def close_rings(
    geometry: IntoExprColumn, *, snap_z: bool = False, snap_m: bool = False
) -> pl.Expr:
    """Closes any unclosed lines into rings.
    A ring is closed when its last vertex lies at exactly the same place as its first,
    in x, y and z (see `is_valid_geoarrow`).
    For a polygon, this acts on all of the rings.

    As a repairing function, it does not require the linestring or polygon to be valid (yet).

    {keeps_type_dimension_and_crs}
    {ignores_empty_and_invalid}

    For a geometry with the first vertex as F and the last as L:

    | ends | result |
    | - | - |
    | x or y differ | F is appended |
    | x and y the same, z differs | F is appended, or with `snap_z=True`, L gets the z of F |
    | x, y and z the same, m differs | left as it is, or with `snap_m=True`, L gets the m of F |

    If `snap_z=True` and `snap_m=True`, a snapped L gets both the z and the m of F.
    An appended F brings its own z and m, so the ring is closed in every axis.
    A flag for an axis the column does not have does nothing.

    ```python
    pl.scan_parquet("parcels.parquet").select(geo.close_rings("parcel"))
    ```

    This expression leaves rings that are already closed as-is.
    """
    return on_geometry(geometry, partial(_close_rings, snap_z=snap_z, snap_m=snap_m))
