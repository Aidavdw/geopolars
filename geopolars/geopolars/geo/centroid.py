"""The centroid of a polygon: the centre of the area it encloses.
Note that this is not the mean coordinate, which weighs every vertex the same!

A native kernel (A-tier) applies the shoelace formula over the planar area,
in one pass over the storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _centroid(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    """The centroid of a polygon, per row."""
    if not isinstance(geometry, PolygonType):
        msg = f"centroid expects a `{PolygonType._extension_name}` column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="centroid",
        is_elementwise=True,
    )


def centroid(geometry: IntoExprColumn) -> pl.Expr:
    """The centroid (centre of area) of a polygon, as a `PointXY`.

    It is measured on the plane the coordinates lie in, even when they carry a CRS,
    and the CRS is carried over to the result.
    Holes are taken out.
    Only the two-dimensional part is considered: 'z' and 'm' are ignored.
    For a 3D polygon that gives the centroid of its footprint
    (its shadow on the x/y plane), which is not its centroid in 3D,
    and the result has no 'z'.
    A vertical polygon has a footprint that encloses nothing, so it has no centroid.

    A null geometry, an empty polygon, and a polygon that encloses nothing
    (all of its vertices on a line) have no centroid.

    | in                     | out          |
    |------------------------|--------------|
    | `PolygonXY`            | `PointXY`    |
    | `PolygonXYZM`          | `PointXY`    |

    ```python
    df.select(geo.centroid("parcel"))
    ```
    """
    return on_geometry(geometry, _centroid)
