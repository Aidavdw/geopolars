"""The centroid of a linestring or polygon: the centre of its length or area.
Note that this is not the mean coordinate, which weighs every vertex the same!

A native kernel (A-tier) does either in one pass over the storage:
it weighs every segment of a linestring by its length,
and applies the shoelace formula over a polygon's planar area.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _centroid(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    """The centroid of a linestring or polygon, per row."""
    if not isinstance(geometry, (LineStringType, PolygonType)):
        msg = (
            f"centroid expects a `{LineStringType._extension_name}` "
            f"or `{PolygonType._extension_name}` column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="centroid",
        is_elementwise=True,
    )


def centroid(geometry: IntoExprColumn) -> pl.Expr:
    """The centroid of a polygon (centre of area) or linestring (centre of length).


    A polygon's centroid is measured on the plane the coordinates lie in,
    even when they carry a CRS, and the CRS is carried over to the result.
    Holes are taken out.
    Only the two-dimensional part is considered: 'z' and 'm' are ignored.
    For a 3D polygon that gives the centroid of its footprint
    (its shadow on the x/y plane), which is not its centroid in 3D,
    and the result has no 'z'.
    A vertical polygon has a footprint that encloses nothing, so it has no centroid.

    A linestring's centroid is the midpoint of each of its segments,
    weighted by that segment's length.
    Fully three-dimensional if 'z'.
    If the line has an 'm', it is averaged.
    The result keeps the line's dimension.
    A null linestring, an empty one, one with a missing coordinate,
    and one without length (a single vertex, or all on one spot) have no centroid.

    A null geometry, an empty polygon, and a polygon that encloses nothing
    (all of its vertices on a line) have no centroid.

    | in                     | out          |
    |------------------------|--------------|
    | `LineStringXY`         | `PointXY`    |
    | `LineStringXYZ`        | `PointXYZ`   |
    | `LineStringXYM`        | `PointXYM`   |
    | `LineStringXYZM`       | `PointXYZM`  |
    | `PolygonXY`            | `PointXY`    |
    | `PolygonXYZM`          | `PointXY`    |

    ```python
    df.select(geo.centroid("parcel"), geo.centroid("road"))
    ```
    """
    return on_geometry(geometry, _centroid)
