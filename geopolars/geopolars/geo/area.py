"""The area a geometry encloses.

Planar area, in whatever units the coordinates are in. Nothing here knows about
a CRS, so this is the area on the flat plane the coordinates lie in, not the
area on the ellipsoid: for degrees (EPSG:4326) the number comes out in square
degrees, which is not a unit anyone wants. Project first.
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


def _area(column: pl.Expr, geometry: type[GeoArrowType]) -> pl.Expr:
    """The area of one geometry type, per row."""
    if issubclass(geometry, GeoPolygon):
        return _polygon(column)
    # Everything else has no defined area.
    return pl.when(column.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))


def area(geometry: IntoExprColumn) -> pl.Expr:
    """The planar area a geometry encloses, as an `f64`.


    Only a polygon actually encloses something.
    This also considers its holes.
    'z' and 'm' are ignored.

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
    # CHECK: CCW/CW orientation of polygon should not matter, area should be positive only.
    return on_geometry(geometry, _area)


def _area_rsgeo(column: pl.Expr, geometry: type[GeoArrowType]) -> pl.Expr:
    """The area of the geometry"""
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_rsgeo",
        is_elementwise=True,
    )


def area_rsgeo(geometry: IntoExprColumn) -> pl.Expr:
    """The planar area a geometry encloses, as an `f64`, computed by the `geo` crate.

    Gives the same answers as `area`, which is written in plain Polars expressions;
    see there for what counts as an area. The two sit side by side so they can be
    compared, for correctness and for speed.

    ```python
    df.select(geo.area_rsgeo("parcel"))
    ```
    """
    return on_geometry(geometry, _area_rsgeo)
