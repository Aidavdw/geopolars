"""How long a linestring is.

`length_planar` measures in the space the coordinates lie in, in their unit.
`length_geodesic` measures along the ellipsoid of the CRS, in the unit of the CRS
(metres for a CRS in longitude/latitude).
`length` picks between them by the CRS.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, MultiLineStringType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _has_no_length(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_lines: bool
) -> pl.Expr | None:
    """Only linestrings and multilinestrings have a length: `None` for those.
    Other geometries are refused, or given `0.0` with `allow_non_lines`."""
    if isinstance(geometry, (LineStringType, MultiLineStringType)):
        return None
    if not allow_non_lines:
        msg = (
            "a length is measured on a `geoarrow.linestring` "
            f"or `geoarrow.multilinestring` column, got: {geometry!r};"
            " pass `allow_non_lines=True` to give it a length of 0.0"
        )
        raise TypeError(msg)
    return pl.when(column.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))


def _planar(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_lines: bool
) -> pl.Expr:
    zero = _has_no_length(column, geometry, allow_non_lines=allow_non_lines)
    if zero is not None:
        return zero
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="length_planar",
        is_elementwise=True,
    )


def _geodesic(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_lines: bool
) -> pl.Expr:
    # Whether PROJ can use the CRS is for the kernel to check.
    if not geometry._declares_crs():
        msg = (
            "length_geodesic needs a CRS to measure on, but the geometry declares none"
        )
        raise TypeError(msg)
    zero = _has_no_length(column, geometry, allow_non_lines=allow_non_lines)
    if zero is not None:
        return zero
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="length_geodesic",
        is_elementwise=True,
    )


def length_planar(
    geometry: IntoExprColumn, *, allow_non_lines: bool = False
) -> pl.Expr:
    """How long a linestring is in the space its coordinates lie in, as an `f64`,
    or a list of them for a multilinestring.

    A CRS is ignored: for longitude/latitude that gives degrees,
    which is rarely what you want (see `length_geodesic`).

    Every segment is measured by Pythagoras, as `distance` measures two points
    without a CRS, and added up, in the unit of the coordinates.
    A multilinestring gets the length of each of its parts, in order, as a list;
    sum it with `.list.sum()` for the length of the whole.
    For a line with a `z`, every segment counts its difference in height too.
    `m` is ignored.

    An empty linestring, or one of a single vertex, has a length of `0.0`.
    A missing geometry, or one with a missing coordinate, has no length.
    Every other geometry (points, polygons and their multi forms)
    is refused while the plan is built,
    unless `allow_non_lines` is set: then each of them gets `0.0`
    (and a null stays null).

    | in                     | out          |
    |------------------------|--------------|
    | `LineStringXY`         | `0.0` and up |
    | `LineStringXYZ`        | `0.0` and up |
    | `MultiLineStringXY`    | `list[f64]`  |
    | `PointXY`              | refused      |
    | `PolygonXY`            | refused      |
    | `MultiPointXYZM`       | refused      |

    ```python
    df.select(geo.length_planar("route"))
    ```
    """
    return on_geometry(geometry, partial(_planar, allow_non_lines=allow_non_lines))


def length_geodesic(
    geometry: IntoExprColumn, *, allow_non_lines: bool = False
) -> pl.Expr:
    """How long a linestring is along the ellipsoid of its CRS, as an `f64`,
    or a list of them for a multilinestring.

    Every segment is measured as `distance` measures two points with a CRS:
    along the ellipsoid, in the unit of the CRS's `x` and `y`
    (such as US survey feet for EPSG:2263), or in metres when those are degrees.
    With a `z`, every segment counts its difference in height too,
    taken to be in that same unit.

    A geometry without a CRS is refused while the plan is built,
    as is a CRS that is not defined on longitude/latitude.

    Multilinestrings, `m`, and other geometries
    (including `allow_non_lines`) are treated as in `length_planar`.

    ```python
    df.select(geo.length_geodesic("route"))
    ```
    """
    return on_geometry(geometry, partial(_geodesic, allow_non_lines=allow_non_lines))


def _length(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_lines: bool
) -> pl.Expr:
    """`length_planar` without a CRS, `length_geodesic` with one.

    No projection keeps lengths everywhere, so every CRS is measured on its ellipsoid.
    """
    measure = _geodesic if geometry._declares_crs() else _planar
    return measure(column, geometry, allow_non_lines=allow_non_lines)


def length(geometry: IntoExprColumn, *, allow_non_lines: bool = False) -> pl.Expr:
    """How long a linestring is.

    Forwards to `length_planar` for a geometry without a CRS,
    and to `length_geodesic` for one with a CRS.

    ```python
    df.select(geo.length("route"))
    ```
    """
    return on_geometry(geometry, partial(_length, allow_non_lines=allow_non_lines))
