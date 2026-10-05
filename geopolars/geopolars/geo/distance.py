"""How far apart two points are.

Points that declare a CRS are measured along the ellipsoid, in metres.
Points without one are measured on the flat plane their coordinates lie in,
in whatever units those are: for longitude/latitude that is degrees,
which shrink east-west towards the poles. Declare the CRS to get metres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import GeoPoint
from geopolars.geo._dispatch import on_geometry_pair

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _squared_norm(dx: pl.Expr, dy: pl.Expr) -> pl.Expr:
    """The squared length of an offset, by Pythagoras.

    Shared by everything measured on the plane:
    two points subtract their coordinates, a linestring diffs its vertices.
    """
    return dx.pow(2) + dy.pow(2)


def _planar_squared(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """The squared distance between two coordinate structs."""
    dx = b.struct.field("x") - a.struct.field("x")
    dy = b.struct.field("y") - a.struct.field("y")
    return _squared_norm(dx, dy)


def _geodesic_squared(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """The squared geodesic distance between two points in the same CRS, in metres²."""
    return register_plugin_function(
        plugin_path=LIB,
        args=[a, b],
        function_name="distance_squared_geodesic",
        is_elementwise=True,
    )


# TODO: add expr for things with multiple points, get the 'closest distance'.
# with flag to also interpolate between points / line?


def _distance_squared(
    a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
) -> pl.Expr:
    # TODO: for other shapes, calculate centroid, and take distance on that.
    if not (isinstance(a_dtype, GeoPoint) and isinstance(b_dtype, GeoPoint)):
        msg = (
            "a distance is measured between two `geoarrow.point` columns, "
            f"got: {a_dtype!r} and {b_dtype!r}"
        )
        raise TypeError(msg)

    # Whether both declare the same CRS, and whether PROJ can use it,
    # is for the kernel to check while resolving the schema.
    if a_dtype._declares_crs() or b_dtype._declares_crs():
        return _geodesic_squared(a, b)
    return _planar_squared(a.ext.storage(), b.ext.storage())


def distance_squared(a: IntoExprColumn, b: IntoExprColumn) -> pl.Expr:
    """The square of `distance`, as an `f64`.

    Cheaper than `distance` when only comparing distances,
    because ordering by the square orders by the distance as well.
    See `distance` for more information.

    ```python
    df.select(geo.distance_squared("home", "work"))
    ```
    """
    return on_geometry_pair(a, b, _distance_squared)


def distance(a: IntoExprColumn, b: IntoExprColumn) -> pl.Expr:
    """How far apart two points are, row by row, as an `f64`.

    | `a` and `b`         | measured                     | in                  |
    |---------------------|------------------------------|---------------------|
    | no CRS              | on the plane, by Pythagoras  | coordinate units    |
    | the same CRS        | along the CRS's ellipsoid    | metres              |

    Without a CRS, longitude/latitude comes out in degrees,
    which are not the same length east-west as north-south.
    With one, the points are taken to the longitude/latitude the CRS is defined on
    (for a projected CRS, the one it projects from; no datum is shifted)
    and measured along the geodesic between them, on that CRS's own ellipsoid.
    Points in different CRSs, or a CRS on only one of them, are refused:
    reproject one onto the other with `to_crs` first.

    `z` and `m` are ignored. A missing point has no distance to anything.
    Either side may be a single point, to measure every row against it.

    ```python
    df.select(geo.distance("home", "work"))
    ```
    """
    return distance_squared(a, b).sqrt()
