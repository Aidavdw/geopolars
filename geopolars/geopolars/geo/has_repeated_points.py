"""Whether a geometry has two points that coincide (A-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def has_repeated_points(
    geometry: IntoExprColumn,
    *,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
) -> pl.Expr:
    """Whether each geometry has two points that lie at the same place.
    One `bool` per row, a multi-geometry included. A missing geometry gives a missing result.

    The geometries have to be valid by `is_valid_geoarrow` already. This trusts its input.

    | geometry | has repeated points when |
    | - | - |
    | `PointXY[ZM]` | never |
    | `MultiPointXY[ZM]` | any two points lie at the same place |
    | `LineStringXY[ZM]` | two vertices in a row lie at the same place |
    | `PolygonXY[ZM]` | two vertices in a row of one ring lie at the same place |
    | `MultiLineStringXY[ZM]`, `MultiPolygonXY[ZM]` | one of its parts has repeated points |

    The ends of a ring lie at the same place, but they are not in a row, so they do not count.

    "At the same place" is compared in x, y and z.
    With `abs_tol=None` and `rel_tol=None`, the default, the coordinates have to be exactly equal.
    Otherwise they only have to be close on every axis, as in `pl.Expr.is_close` (see `geo.is_close`).
    A tolerance that is not given then counts as `0.0`.

    Repeated points are valid, also by OGC (see `is_valid_ogc`), but some tools stumble over them.

    ```python
    df.filter(~geo.has_repeated_points("route"))
    ```
    """

    def build(column: pl.Expr, _geometry: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="has_repeated_points",
            is_elementwise=True,
            kwargs={"abs_tol": abs_tol, "rel_tol": rel_tol},
        )

    return on_geometry(geometry, build)
