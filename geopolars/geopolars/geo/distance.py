"""How far apart two points are.

Points that declare a CRS are measured along the ellipsoid, in the unit of the CRS
(metres for a CRS in longitude/latitude).
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


def _offset(a: pl.Expr, b: pl.Expr, axis: str) -> pl.Expr:
    return b.struct.field(axis) - a.struct.field(axis)


def _planar_squared(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """The squared distance between two coordinate structs, in the plane."""
    return _squared_norm(_offset(a, b, "x"), _offset(a, b, "y"))


def _has_z(dtype: GeoArrowType) -> bool:
    return "z" in dtype._dimension


def _both_have_z(a_dtype: GeoArrowType, b_dtype: GeoArrowType) -> bool:
    return _has_z(a_dtype) and _has_z(b_dtype)


def _geodesic_squared(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """The squared geodesic distance between two points in the same CRS, in its unit²."""
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
    # The kernel counts the height itself, when both points carry one.
    if a_dtype._declares_crs() or b_dtype._declares_crs():
        return _geodesic_squared(a, b)

    a, b = a.ext.storage(), b.ext.storage()
    squared = _planar_squared(a, b)
    if _both_have_z(a_dtype, b_dtype):
        squared = squared + _offset(a, b, "z").pow(2)
    return squared


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
    | no CRS              | in space, by Pythagoras      | coordinate units    |
    | the same CRS        | along the CRS's ellipsoid    | CRS units           |

    Without a CRS, longitude/latitude comes out in degrees,
    which are not the same length east-west as north-south.
    With one, the points are taken to the longitude/latitude the CRS is defined on
    (for a projected CRS, the one it projects from; no datum is shifted)
    and measured along the geodesic between them, on that CRS's own ellipsoid.
    The result is in the unit of the CRS's `x` and `y` (such as US survey feet for EPSG:2263),
    or in metres when those are degrees.
    Points in different CRSs, or a CRS on only one of them, are refused:
    reproject one onto the other with `to_crs` first.

    When both points have a `z`, the difference in height counts too:
    the distance is `sqrt(d² + Δz²)`, with `d` the distance above.
    With a CRS, `z` is taken to be in the same unit as the result,
    and the geodesic is measured at the surface of the ellipsoid
    (a few parts per million short at aircraft heights).
    If only one of the points has a `z`, there is no height difference to count,
    and they are measured as if neither had one.
    `m` is always ignored.
    Taking the distance from a two-dimensional point to a three-dimensional point (or vice-versa)
    ignores any height difference.
    Either side may be a single point, in which case its distance to all the other points is measured.

    ```python
    df.select(geo.distance("home", "work"))
    ```
    """
    return distance_squared(a, b).sqrt()
