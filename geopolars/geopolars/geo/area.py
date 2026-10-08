"""The area a geometry encloses.

Measured the same way as `distance`:
geometries that declare a CRS are measured along the ellipsoid, in the unit of the CRS squared
(square metres for a CRS in longitude/latitude).
Geometries without one are measured on the flat plane their coordinates lie in,
in whatever units those are squared:
for longitude/latitude that is square degrees, which might be undesireable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import MultiPolygonType, PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _twice_signed(ring: pl.Expr) -> pl.Expr:
    """Twice the signed area of one closed ring, positive when it runs CCW."""
    # Use shift(-1) so we don't count the repeated last coordinate
    x = pl.element().struct.field("x")
    y = pl.element().struct.field("y")
    dx, dy = x - x.first(), y - y.first()
    return ring.list.eval(dx * dy.shift(-1) - dx.shift(-1) * dy).list.sum()


def _polygon(rings: pl.Expr) -> pl.Expr:
    """A polygon's area, from its list of rings: its exterior ring, less the holes inside it."""
    # Per ring, unsigned
    # First ring is the 'positive' area,
    # everything after that is a hole.
    areas = rings.list.eval(_twice_signed(pl.element()).abs())

    # exterior - holes == first - (sum - first).
    return (2 * areas.list.first().fill_null(0.0) - areas.list.sum()) / 2


def _geodesic(column: pl.Expr) -> pl.Expr:
    """The area along the ellipsoid of the CRS, in its unit²."""
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_geodesic",
        is_elementwise=True,
    )


def _area(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    """The area of one geometry type, per row."""
    # Whether PROJ can use the CRS is for the kernel to check.
    if geometry._declares_crs():
        return _geodesic(column)
    if isinstance(geometry, PolygonType):
        return _polygon(column.ext.storage())
    if isinstance(geometry, MultiPolygonType):
        return column.ext.storage().list.eval(_polygon(pl.element()))
    # Everything else has no defined area.
    return pl.when(column.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))


def area(geometry: IntoExprColumn) -> pl.Expr:
    """The area a geometry encloses, as an `f64`,
    or a list of them for a multipolygon.

    | in                  | measured                     | in                  |
    |---------------------|------------------------------|---------------------|
    | no CRS              | on the plane, by shoelace    | coordinate units²   |
    | a CRS               | along the CRS's ellipsoid    | CRS units²          |

    With a CRS, the coordinates are taken to the longitude/latitude
    the CRS is defined on (for a projected CRS, the one it projects from;
    no datum is shifted),
    and every ring is measured as the region its geodesic edges enclose
    on that CRS's own ellipsoid, as `distance` measures two points.
    The result is in the unit of the CRS's `x` and `y` squared
    (such as square US survey feet for EPSG:2263),
    or in square metres when those are degrees.
    That is the smaller side of the ring:
    a ring around more than half the ellipsoid is taken to enclose the rest of it.

    Only a polygon actually encloses something.
    This also considers its holes.
    A multipolygon gets the area of each of its polygons, in order, as a list;
    sum it with `.list.sum()` for the area of the whole
    (which assumes its polygons do not overlap, as the spec requires).
    'z' and 'm' are ignored: this is the area of the footprint.

    This calculation assumes that the (closed) polygon is not self-intersecting.

    An empty polygon, as well as other types
    (that do not have an 'area') return '0.0'.

    | in                     | out          |
    |------------------------|--------------|
    | `PolygonXY`            | `0.0` and up |
    | `PolygonXYZ`           | `0.0` and up |
    | `MultiPolygonXY`       | `list[f64]`  |
    | `LineStringXY`         | `0.0`        |
    | `PointXYZM`            | `0.0`        |

    A self-intersecting ring is not a valid polygon, and the parts of it that
    wind the other way will cancel rather than add.

    ```python
    df.select(geo.area("parcel"))
    ```
    """
    return on_geometry(geometry, _area)
