"""The area a geometry encloses.

Measured the same way as `distance`:
geometries that declare a CRS are measured along the ellipsoid, in square metres.
Geometries without one are measured on the flat plane their coordinates lie in,
in whatever units those are squared:
for longitude/latitude that is square degrees, which might be undesireable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import GeoPolygon
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


def _polygon(column: pl.Expr) -> pl.Expr:
    """A polygon's area: its exterior ring, less the holes inside it."""
    # Per ring, unsigned
    # First ring is the 'positive' area,
    # everything after that is a hole.
    rings = column.ext.storage().list.eval(_twice_signed(pl.element()).abs())

    # exterior - holes == first - (sum - first).
    return (2 * rings.list.first().fill_null(0.0) - rings.list.sum()) / 2


def _geodesic(column: pl.Expr) -> pl.Expr:
    """The area along the ellipsoid of the CRS, in metres²."""
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
    if isinstance(geometry, GeoPolygon):
        return _polygon(column)
    # Everything else has no defined area.
    return pl.when(column.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))


def area(geometry: IntoExprColumn) -> pl.Expr:
    """The area a geometry encloses, as an `f64`.

    | in                  | measured                     | in                  |
    |---------------------|------------------------------|---------------------|
    | no CRS              | on the plane, by shoelace    | coordinate units²   |
    | a CRS               | along the CRS's ellipsoid    | metres²             |

    With a CRS, the coordinates are taken to the longitude/latitude
    the CRS is defined on (for a projected CRS, the one it projects from;
    no datum is shifted),
    and every ring is measured as the region its geodesic edges enclose
    on that CRS's own ellipsoid, as `distance` measures two points.
    That is the smaller side of the ring:
    a ring around more than half the ellipsoid is taken to enclose the rest of it.

    Only a polygon actually encloses something.
    This also considers its holes.
    'z' and 'm' are ignored: this is the area of the footprint.

    This calculation assumes that the (closed) polygon is not self-intersecting.

    An empty polygon, as well as other types
    (that do not have an 'area') return '0.0'.

    | in                     | out          |
    |------------------------|--------------|
    | `PolygonXY`            | `0.0` and up |
    | `PolygonXYZ`           | `0.0` and up |
    | `LineStringXY`         | `0.0`        |
    | `PointXYZM`            | `0.0`        |

    A self-intersecting ring is not a valid polygon, and the parts of it that
    wind the other way will cancel rather than add.

    ```python
    df.select(geo.area("parcel"))
    ```
    """
    return on_geometry(geometry, _area)


def _area_rsgeo(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    """The area of the geometry"""
    # rsgeo only measures on the plane, or on WGS 84 alone.
    if geometry._declares_crs():
        return _geodesic(column)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_rsgeo",
        is_elementwise=True,
    )


def area_rsgeo(geometry: IntoExprColumn) -> pl.Expr:
    """The area a geometry encloses, as an `f64`, computed by the `geo` crate.

    Gives the same answers as `area`, which is written in plain Polars expressions;
    see there for what counts as an area. The two sit side by side so they can be
    compared, for correctness and for speed.
    With a CRS, both measure along the ellipsoid in the same kernel.

    ```python
    df.select(geo.area_rsgeo("parcel"))
    ```
    """
    return on_geometry(geometry, _area_rsgeo)
