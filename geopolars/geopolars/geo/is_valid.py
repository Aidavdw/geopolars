"""Whether a geometry is valid, and nulling out the ones that are not.

A native kernel (A-tier) checks every coordinate first, then every row's shape.
The CRS bounds are looked up once, while the plan is built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

CrsBounds = Literal[
    "off",
    "crs",
    "crs_wrapped_longitude",
    "area_of_use",
    "area_of_use_wrapped_longitude",
]

# What each `crs_bounds` asks of `_crs_bounds`, as
# `(allow_wrapped_longitude, within_area_of_use)`. `None` asks nothing.
_CRS_BOUNDS: dict[str, tuple[bool, bool] | None] = {
    "off": None,
    "crs": (False, False),
    "crs_wrapped_longitude": (True, False),
    "area_of_use": (False, True),
    "area_of_use_wrapped_longitude": (True, True),
}


def _kernel(
    function_name: str,
    *,
    crs_bounds: CrsBounds,
    check_missing_coordinates: bool,
    check_finite_coordinates: bool,
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    if crs_bounds not in _CRS_BOUNDS:
        msg = f"`crs_bounds` must be one of {list(_CRS_BOUNDS)}, got {crs_bounds!r}"
        raise ValueError(msg)
    asked = _CRS_BOUNDS[crs_bounds]

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # PROJ is only asked when there are bounds to check.
        bounds = None
        if asked is not None:
            allow_wrapped_longitude, within_area_of_use = asked
            bounds = geometry._crs_bounds(
                allow_wrapped_longitude=allow_wrapped_longitude,
                within_area_of_use=within_area_of_use,
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
            },
        )

    return build


def is_valid(
    geometry: IntoExprColumn,
    *,
    crs_bounds: CrsBounds = "crs_wrapped_longitude",
    check_missing_coordinates: bool = True,
    check_finite_coordinates: bool = True,
) -> pl.Expr:
    """Whether each geometry is valid: one `bool` per row, a multi-geometry included.
    A missing geometry gives a missing result.

    Valid means it follows the GeoArrow spec, plus a few rules of ours on top of it.
    A valid geometry is safe to hand to every other operation.
    For geometries that are valid, but not useful (repeated points, empty,
    or rings that cross), see `is_degenerate`.

    Every coordinate has to be:

    - there, with every axis: GeoArrow allows nulls only at the outermost level
      (<https://geoarrow.org/format.html#missing-values-null>);
    - finite on every axis (x, y, z and m): no NaN, no infinity.
      The exception is the empty point, which is NaN in x and y. It may stand anywhere, also as a vertex;
    - inside the CRS's bounds, if it has any (see `crs_bounds` below).
      The empty point is exempt.

    On top of that, per geometry:

    | geometry          | valid when                                                     |
    |-------------------|----------------------------------------------------------------|
    | point             | a valid coordinate                                             |
    | linestring        | no vertices, or at least 2                                     |
    | polygon           | all rings empty, or all of them rings                          |
    | multi-geometries  | every part valid as the geometry above                         |

    The GeoArrow spec only requires a ring to be closed.
    On top of that, a ring needs at least 4 vertices
    (see `is_ring`), and the last has to lie at exactly the same place as the first,
    in x, y and z.
    A linestring of 1 vertex, and a polygon with an empty exterior around a hole, are invalid too.

    `crs_bounds` says which bounds the coordinates have to lie in.
    They only apply to a column that declares a CRS PROJ can read.
    An opaque SRID is skipped, unless an area of use is asked for, which then raises.

    - `"off"` does not check bounds, and does not ask PROJ for any.
    - `"crs"`: in a geographic CRS, latitude has to lie within a quarter turn of the equator
      (±90 for degrees), and longitude within half a turn of the prime meridian (±180).
      A projected CRS has no bounds of its own.
    - `"crs_wrapped_longitude"`, the default: as `"crs"`, but longitude may run up to a full turn east (360),
      so that data in the 0 to 360 convention is valid as well.
    - `"area_of_use"`: as `"crs"`, and `x` and `y` also have to lie inside the area
      PROJ says the CRS is meant for (UTM zone 31N covers 0 to 6 degrees east, for one).
      Data just past it is common and often fine.
    - `"area_of_use_wrapped_longitude"`: as `"area_of_use"`, with longitude as in `"crs_wrapped_longitude"`.

    The other checks can be individually turned off.
    This can be used to reduce computation time when you know for sure that
    your data is valid in some way. The shape rules in the table always apply.

    - `check_missing_coordinates=False` skips looking for missing values below the row,
      and trusts the column to hold to GeoArrow's rule on them.
      A missing coordinate is then read as whatever values lie under it.
      Do not use this unless you know for sure the inner data is valid!
    - `check_finite_coordinates=False` lets coordinates use NaN (partial missing) or infinite.

    ```python
    df.filter(geo.is_valid("parcel"))
    ```
    """
    return on_geometry(
        geometry,
        _kernel(
            "is_valid",
            crs_bounds=crs_bounds,
            check_missing_coordinates=check_missing_coordinates,
            check_finite_coordinates=check_finite_coordinates,
        ),
    )


def null_out_invalid(
    geometry: IntoExprColumn,
    *,
    crs_bounds: CrsBounds = "crs_wrapped_longitude",
    check_missing_coordinates: bool = True,
    check_finite_coordinates: bool = True,
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
            crs_bounds=crs_bounds,
            check_missing_coordinates=check_missing_coordinates,
            check_finite_coordinates=check_finite_coordinates,
        ),
    )
