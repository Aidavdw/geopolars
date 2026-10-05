"""How long a linestring is.

Measured the same way as `distance`, segment by segment:
along the ellipsoid in metres for a geometry that declares a CRS,
on the plane in coordinate units for one that doesn't.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import GeoLineString, GeoMultiLineString
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.distance import _squared_norm

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _planar(line: pl.Expr) -> pl.Expr:
    """The planar length of one list of coordinates.

    Each vertex is diffed against the one before it, per axis.
    That is the distance between the two written for a whole line at once,
    and twice as fast as shifting the coordinate structs to pair them up.
    The first vertex has nothing before it, which `sum` skips,
    so an empty line or a single vertex has a length of `0.0`.
    """
    x = pl.element().struct.field("x")
    y = pl.element().struct.field("y")
    return line.list.eval(_squared_norm(x.diff(), y.diff()).sqrt()).list.sum()


def _geodesic(column: pl.Expr) -> pl.Expr:
    """The length along the ellipsoid of the CRS, in metres."""
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="length_geodesic",
        is_elementwise=True,
    )


def _length(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (GeoLineString, GeoMultiLineString)):
        msg = (
            "a length is measured on a `geoarrow.linestring` "
            f"or `geoarrow.multilinestring` column, got: {geometry!r}"
        )
        raise TypeError(msg)

    if geometry._declares_crs():
        return _geodesic(column)

    storage = column.ext.storage()
    if isinstance(geometry, GeoLineString):
        return _planar(storage)
    # As long as its parts together.
    return storage.list.eval(_planar(pl.element())).list.sum()


def length(geometry: IntoExprColumn) -> pl.Expr:
    """How long a linestring is, as an `f64`.

    | in                  | measured                     | in                  |
    |---------------------|------------------------------|---------------------|
    | no CRS              | on the plane, by Pythagoras  | coordinate units    |
    | a CRS               | along the CRS's ellipsoid    | metres              |

    Every segment is measured as `distance` measures two points, and added up.
    A multilinestring is as long as its parts together.
    Other geometries are refused.

    `z` and `m` are ignored.
    An empty linestring, or one of a single vertex, has a length of `0.0`.
    A missing geometry has no length.

    ```python
    df.select(geo.length("route"))
    ```
    """
    return on_geometry(geometry, _length)
