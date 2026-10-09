"""Whether a geometry is valid, and nulling out the ones that are not.

A native kernel (A-tier) checks every coordinate and part in place.
The CRS bounds are looked up once, while the plan is built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _kernel(
    function_name: str, allow_wrapped_longitude: bool, within_area_of_use: bool
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        bounds = geometry._crs_bounds(
            allow_wrapped_longitude=allow_wrapped_longitude,
            within_area_of_use=within_area_of_use,
        )
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name=function_name,
            is_elementwise=True,
            kwargs={"bounds": bounds},
        )

    return build


def is_valid(
    geometry: IntoExprColumn,
    *,
    allow_wrapped_longitude: bool = True,
    within_area_of_use: bool = False,
) -> pl.Expr:
    """Whether each geometry is valid: one `bool` per row, a multi-geometry included.
    A missing geometry gives a missing result.

    This checks structure and coordinates, not topology:
    a self-intersecting polygon can still be valid here.

    Every coordinate has to be:

    - there, with every axis: GeoArrow allows nulls only at the outermost level
      (<https://geoarrow.org/format.html#missing-values-null>);
    - finite on every axis (x, y, z and m): no NaN, no infinity;
    - inside the CRS's bounds, if it has any (see below).

    On top of that, per geometry:

    | geometry          | valid when                                                       |
    |-------------------|------------------------------------------------------------------|
    | point             | a valid coordinate, or NaN on every axis (the empty point)       |
    | linestring        | no vertices, or at least 2; no two in a row at the same place    |
    | polygon           | no missing rings; all rings empty, or every ring a valid ring    |
    | multi-geometries  | no missing parts, and every part valid as the geometry above     |

    A ring is a valid linestring that is also a ring (see `is_ring`):
    at least 4 vertices, the last at the same place as the first.
    "At the same place" is within the default margin of `pl.Expr.is_close`, in x, y and z.

    CRS bounds only apply to a column that declares a CRS PROJ can read.
    An opaque SRID is skipped, unless `within_area_of_use` is asked for, which then raises.

    - In a geographic CRS, latitude has to lie within a quarter turn of the equator
      (±90 for degrees), and longitude within half a turn of the prime meridian (±180).
      `allow_wrapped_longitude` lets longitude run up to a full turn east (360),
      so that data in the 0 to 360 convention is valid as well.
    - `within_area_of_use` also requires `x` and `y` to lie inside the area
      PROJ says the CRS is meant for (UTM zone 31N covers 0 to 6 degrees east, for one).
      Data just past it is common and often fine, so this is off by default.

    ```python
    df.filter(geo.is_valid("parcel"))
    ```
    """
    return on_geometry(
        geometry, _kernel("is_valid", allow_wrapped_longitude, within_area_of_use)
    )


def null_out_invalid(
    geometry: IntoExprColumn,
    *,
    allow_wrapped_longitude: bool = True,
    within_area_of_use: bool = False,
) -> pl.Expr:
    """Null out every geometry that `is_valid` rejects, keeping the rest as they are.
    The arguments are those of `is_valid`.

    Every constructor in `geo` already holds to GeoArrow's rules on missing values,
    but nothing checks a column that arrives another way,
    such as from a file or relabelled with `ext.to`:

    ```python
    pl.scan_parquet("routes.parquet").select(geo.null_out_invalid("route"))
    ```

    On a column that is entirely valid, it copies nothing.
    """
    return on_geometry(
        geometry,
        _kernel("null_out_invalid", allow_wrapped_longitude, within_area_of_use),
    )
