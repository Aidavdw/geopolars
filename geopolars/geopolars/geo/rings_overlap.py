"""Check two rings for overlapping
This is a hacky implementation that wraps two linestrings into a Polygon,
and calls rsgeo's Relate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType
from geopolars.geo._dispatch import on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def rings_overlap(
    a: IntoExprColumn, b: IntoExprColumn, *, include_touching: bool = False
) -> pl.Expr:
    """Whether two rings overlap, row by row, as a `bool`:
    whether the areas they enclose share area.
    One ring inside the other, or two equal rings, overlap too.

    | `a` and `b`        | out    |
    |--------------------|--------|
    | `LineStringXY`     | `bool` |
    | `LineStringXYZM`   | `bool` |

    Rings that only touch, at a vertex or along an edge, do not overlap,
    unless `include_touching` is set.

    This looks at a projected plane.
    Only x and y are looked at. z and m are ignored.
    Both have to declare the same CRS, or neither one.
    A side of a single ring is compared against every ring of the other.

    Every linestring has to be a ring:
    at least four vertices, the last at the first
    ("at" is within the default margin of `pl.Expr.is_close`).
    Anything else, including an empty linestring, raises.
    A ring that crosses itself (see `self_intersects`) gives no meaningful answer.
    A missing ring gives a missing result,
    and so does one with a missing vertex, or one that is not finite in x or y.

    ```python
    df.filter(~geo.rings_overlap("hole", "other_hole", include_touching=True))
    ```
    """

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        if not (
            isinstance(a_dtype, LineStringType) and isinstance(b_dtype, LineStringType)
        ):
            msg = (
                "rings_overlap expects two linestring columns,"
                f" got: {a_dtype!r} and {b_dtype!r}"
            )
            raise TypeError(msg)
        a_crs, b_crs = a_dtype._crs(), b_dtype._crs()
        if (a_crs is None) != (b_crs is None):
            msg = "only one of the rings declares a CRS; declare the same one on both"
            raise ValueError(msg)
        if a_crs != b_crs:
            msg = (
                f"the rings declare different CRSs, {a_crs} and {b_crs}; "
                "use `to_crs` to bring one onto the other first"
            )
            raise ValueError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="rings_overlap",
            kwargs={"include_touching": include_touching},
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
