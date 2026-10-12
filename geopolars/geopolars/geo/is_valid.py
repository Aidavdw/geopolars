"""Whether a geometry is valid, and nulling out the ones that are not.

A native kernel (A-tier) checks every coordinate and part in place.
A polygon's rings are related to each other through rsgeo's `Relate` (B-tier).
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
    function_name: str,
    *,
    allow_wrapped_longitude: bool,
    within_area_of_use: bool,
    check_missing_coordinates: bool,
    check_finite_coordinates: bool,
    check_crs_bounds: bool,
    check_line_string_zero_length_sections: bool,
    check_ring_self_intersect: bool,
    check_polygon_self_overlap: bool,
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # The bounds are only for the bounds check, so PROJ is not asked without it.
        bounds = (
            geometry._crs_bounds(
                allow_wrapped_longitude=allow_wrapped_longitude,
                within_area_of_use=within_area_of_use,
            )
            if check_crs_bounds
            else None
        )
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name=function_name,
            is_elementwise=True,
            kwargs={
                "bounds": bounds,
                "check_missing_coordinates": check_missing_coordinates,
                "check_finite_coordinates": check_finite_coordinates,
                "check_crs_bounds": check_crs_bounds,
                "check_line_string_zero_length_sections": (
                    check_line_string_zero_length_sections
                ),
                "check_ring_self_intersect": check_ring_self_intersect,
                "check_polygon_self_overlap": check_polygon_self_overlap,
            },
        )

    return build


def is_valid(
    geometry: IntoExprColumn,
    *,
    allow_wrapped_longitude: bool = True,
    within_area_of_use: bool = False,
    check_missing_coordinates: bool = True,
    check_finite_coordinates: bool = True,
    check_crs_bounds: bool = True,
    check_line_string_zero_length_sections: bool = True,
    check_ring_self_intersect: bool = True,
    check_polygon_self_overlap: bool = True,
) -> pl.Expr:
    """Whether each geometry is valid: one `bool` per row, a multi-geometry included.
    A missing geometry gives a missing result.

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
    | polygon           | no missing rings; all empty, or valid rings that fit (see below) |
    | multi-geometries  | no missing parts, and every part valid as the geometry above     |

    A ring is a valid linestring that is also a ring (see `is_ring`):
    at least 4 vertices, the last at the same place as the first.
    "At the same place" is within the default margin of `pl.Expr.is_close`, in x, y and z.

    A polygon's rings also have to fit together, in x and y:

    - no ring crosses or touches itself (see `self_intersects`);
    - every hole lies inside the exterior;
    - no two holes share any area, nor does one lie inside another;
    - rings may touch each other only at points, never along a line.
      Whether the interior stays in one piece is not checked.

    A linestring may still cross itself.

    CRS bounds only apply to a column that declares a CRS PROJ can read.
    An opaque SRID is skipped, unless `within_area_of_use` is asked for, which then raises.

    - In a geographic CRS, latitude has to lie within a quarter turn of the equator
      (±90 for degrees), and longitude within half a turn of the prime meridian (±180).
      `allow_wrapped_longitude` lets longitude run up to a full turn east (360),
      so that data in the 0 to 360 convention is valid as well.
    - `within_area_of_use` also requires `x` and `y` to lie inside the area
      PROJ says the CRS is meant for (UTM zone 31N covers 0 to 6 degrees east, for one).
      Data just past it is common and often fine, so this is off by default.

    Parts of the check can be individually turned off.
    This can be used to reduce computation time when you know for sure that
    your data is valid in some way.
    The other shape rules in the table (vertex counts, closed rings) always apply.

    - `check_missing_coordinates=False` lets a coordinate, or one of its axes, be missing.
    - `check_finite_coordinates=False` lets coordinates use NaN (partial missing) or infinite.
    - `check_crs_bounds=False` means it is not checked if the coordinates are
      within the range of validity of the CRS. If this flag is set,
      `allow_wrapped_longitude` and `within_area_of_use` do nothing.
    - `check_line_string_zero_length_sections=False` lets two vertices in a row
      lie at the same place, in a linestring as well as in a ring.
    - `check_ring_self_intersect=False` skips checking for a polygon's ring crossing itself.
    - `check_polygon_self_overlap=False` drops the rules on how the rings sit together
      (inside the exterior, not overlapping, touching only at points).
      This has the biggest performance impact out of all of the checks for polygons.

    How a polygon's rings cross or fit is only known for valid points,
    so a ring without them stays invalid while either of the last two checks is on.

    ```python
    df.filter(geo.is_valid("parcel"))
    ```
    """
    return on_geometry(
        geometry,
        _kernel(
            "is_valid",
            allow_wrapped_longitude=allow_wrapped_longitude,
            within_area_of_use=within_area_of_use,
            check_missing_coordinates=check_missing_coordinates,
            check_finite_coordinates=check_finite_coordinates,
            check_crs_bounds=check_crs_bounds,
            check_line_string_zero_length_sections=check_line_string_zero_length_sections,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
        ),
    )


def null_out_invalid(
    geometry: IntoExprColumn,
    *,
    allow_wrapped_longitude: bool = True,
    within_area_of_use: bool = False,
    check_missing_coordinates: bool = True,
    check_finite_coordinates: bool = True,
    check_crs_bounds: bool = True,
    check_line_string_zero_length_sections: bool = True,
    check_ring_self_intersect: bool = True,
    check_polygon_self_overlap: bool = True,
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
        _kernel(
            "null_out_invalid",
            allow_wrapped_longitude=allow_wrapped_longitude,
            within_area_of_use=within_area_of_use,
            check_missing_coordinates=check_missing_coordinates,
            check_finite_coordinates=check_finite_coordinates,
            check_crs_bounds=check_crs_bounds,
            check_line_string_zero_length_sections=check_line_string_zero_length_sections,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
        ),
    )
