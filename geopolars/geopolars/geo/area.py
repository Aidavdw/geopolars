"""The area a geometry encloses.

`area_planar` measures on the flat plane the coordinates lie in,
in whatever units those are squared.
`area_geodesic` measures along the ellipsoid of the CRS, in the unit of the CRS squared
(square metres for a CRS in longitude/latitude).
`area` picks between them by the CRS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import MultiPointType, PointType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _refuse_points(geometry: GeoArrowType) -> None:
    """Points are refused rather than given an area of `0.0`."""
    if isinstance(geometry, (PointType, MultiPointType)):
        msg = f"area does not accept a `{geometry._extension_name}` column, got: {geometry!r}"
        raise TypeError(msg)


def _planar(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    _refuse_points(geometry)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_planar",
        is_elementwise=True,
    )


def _geodesic(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    _refuse_points(geometry)
    # Whether PROJ can use the CRS is for the kernel to check.
    if not geometry._declares_crs():
        msg = "area_geodesic needs a CRS to measure on, but the geometry declares none"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_geodesic",
        is_elementwise=True,
    )


def area_planar(geometry: IntoExprColumn) -> pl.Expr:
    """The area a geometry encloses on the plane its coordinates lie in, as an `f64`,
    or a list of them for a multipolygon.

    A CRS is ignored: for longitude/latitude that gives square degrees,
    which is rarely what you want (see `area_geodesic`).

    The area is computed with the shoelace formula, in the unit of the coordinates squared.
    Only a polygon actually encloses something.
    This also considers its holes.
    A multipolygon gets the area of each of its polygons, in order, as a list;
    sum it with `.list.sum()` for the area of the whole
    (which assumes its polygons do not overlap, as the spec requires).
    'z' and 'm' are ignored: this is the area of the footprint.

    An empty polygon, as well as linestrings and multilinestrings
    (that do not have an 'area') return '0.0'.
    Points and multipoints are refused while the plan is built.

    | in                     | out          |
    |------------------------|--------------|
    | `PolygonXY`            | `0.0` and up |
    | `PolygonXYM`           | `0.0` and up |
    | `MultiPolygonXY`       | `list[f64]`  |
    | `LineStringXY`         | `0.0`        |
    | `PointXY`              | refused      |
    | `MultiPointXYZM`       | refused      |

    A self-intersecting ring is not a valid polygon, and the parts of it that
    wind the other way will cancel rather than add.

    ```python
    df.select(geo.area_planar("parcel"))
    ```
    """
    return on_geometry(geometry, _planar)


def area_geodesic(geometry: IntoExprColumn) -> pl.Expr:
    """The area a geometry encloses along the ellipsoid of its CRS, as an `f64`,
    or a list of them for a multipolygon.

    The coordinates are taken to the longitude/latitude
    the CRS is defined on (for a projected CRS, the one it projects from;
    no datum is shifted),
    and every ring is measured as the region its geodesic edges enclose
    on that CRS's own ellipsoid, as `distance` measures two points.
    The result is in the unit of the CRS's `x` and `y` squared
    (such as square US survey feet for EPSG:2263),
    or in square metres when those are degrees.
    That is the smaller side of the ring:
    a ring around more than half the ellipsoid is taken to enclose the rest of it.

    A geometry without a CRS is refused while the plan is built,
    as is a CRS that is not defined on longitude/latitude.

    Polygons, multipolygons, holes, 'z' and 'm', points and other types
    are treated as in `area_planar`.

    ```python
    df.select(geo.area_geodesic("parcel"))
    ```
    """
    return on_geometry(geometry, _geodesic)


def _area(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    """`area_planar` where the plane keeps areas, `area_geodesic` everywhere else."""
    if not geometry._declares_crs() or geometry._is_equal_area():
        return _planar(column, geometry)
    return _geodesic(column, geometry)


def area(geometry: IntoExprColumn) -> pl.Expr:
    """The area a geometry encloses.

    Forwards to `area_planar` for a geometry without a CRS or with an equal-area CRS,
    and to `area_geodesic` for every other CRS.

    ```python
    df.select(geo.area("parcel"))
    ```
    """
    return on_geometry(geometry, _area)
