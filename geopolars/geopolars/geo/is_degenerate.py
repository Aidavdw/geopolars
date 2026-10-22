"""Whether a geometry is degenerate, and nulling out the ones that are.

A native kernel (A-tier) checks the points of every row.
A polygon's rings, and a multipolygon's parts,
are related to each other through rsgeo's `Relate` (B-tier).
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
    function_name: str, **checks: bool
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, _geometry: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name=function_name,
            is_elementwise=True,
            kwargs=checks,
        )

    return build


def is_degenerate(
    geometry: IntoExprColumn,
    *,
    check_repeated_points: bool = True,
    check_empty: bool = True,
    check_ring_self_intersect: bool = True,
    check_polygon_self_overlap: bool = True,
    check_multi_polygon_self_overlap: bool = True,
) -> pl.Expr:
    """Whether each geometry is degenerate: technically valid, but practically not of much use.
    One `bool` per row, `True` where a check fails. A missing geometry gives a missing result.

    The geometries have to be valid already (see `is_valid`): this trusts its input.
    On a geometry that is not, the result means nothing.

    | check                              | geometry                       | degenerate when                                      |
    |------------------------------------|--------------------------------|------------------------------------------------------|
    | `check_repeated_points`            | linestring, every ring         | two vertices in a row lie at the same place          |
    | `check_repeated_points`            | multipoint                     | any two points lie at the same place                 |
    | `check_empty`                      | point                          | it is the empty point: NaN in x and y                |
    | `check_empty`                      | any other                      | it has no points, or only empty ones (as `is_empty`) |
    | `check_ring_self_intersect`        | polygon, multipolygon          | a ring crosses or touches itself                     |
    | `check_polygon_self_overlap`       | polygon, multipolygon          | its rings do not fit together (see below)            |
    | `check_multi_polygon_self_overlap` | multipolygon                   | its parts do not fit together (see below)            |

    "At the same place" is within the default margin of `pl.Expr.is_close`, in x, y and z.
    A linestring may still cross itself.

    A polygon's rings fit together, in x and y, when:

    - every hole lies inside the exterior;
    - no two holes share any area, nor does one lie inside another;
    - rings touch each other only at points, never along a line.
      Whether the interior stays in one piece is not checked.

    A multipolygon's parts fit together, in x and y, when
    no two parts share any area (one may lie inside a hole of another),
    and they touch each other only at points, never along a line.

    How rings cross or fit is only known for finite coordinates.
    So with any of the last three checks on, a polygon with a non-finite x or y is degenerate,
    the empty point included.

    Every check can be turned off on its own, which can save computation time.
    `check_polygon_self_overlap` has the biggest impact on polygons.
    `check_multi_polygon_self_overlap` compares every pair of parts,
    so its cost grows quickly with their number.

    ```python
    df.filter(~geo.is_degenerate("parcel"))
    ```
    """
    return on_geometry(
        geometry,
        _kernel(
            "is_degenerate",
            check_repeated_points=check_repeated_points,
            check_empty=check_empty,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        ),
    )


def null_out_degenerate(
    geometry: IntoExprColumn,
    *,
    check_repeated_points: bool = True,
    check_empty: bool = True,
    check_ring_self_intersect: bool = True,
    check_polygon_self_overlap: bool = True,
    check_multi_polygon_self_overlap: bool = True,
) -> pl.Expr:
    """Null out every geometry that `is_degenerate` finds, keeping the rest as they are.
    The arguments are those of `is_degenerate`, and so is the need for valid input.

    ```python
    df.select(geo.null_out_degenerate(geo.null_out_invalid("parcel")))
    ```

    On a column without degenerate geometries, it copies nothing.
    """
    return on_geometry(
        geometry,
        _kernel(
            "null_out_degenerate",
            check_repeated_points=check_repeated_points,
            check_empty=check_empty,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        ),
    )
