"""The bounding box of a geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _bounds(
    column: pl.Expr,
    geometry: GeoArrowType,
    margin_x: float,
    margin_y: float,
    *,
    per_part: bool,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="bounds",
        kwargs={
            "margin_x": margin_x,
            "margin_y": margin_y,
            "turn": geometry._longitude_turn(),
            "per_part": per_part,
        },
        is_elementwise=True,
    )


def _envelope(geometry: IntoExprColumn) -> pl.Expr:
    """same as `bounds`, except:
    - One box per row around the whole geometry
    - all parts of a multi-geometry together.
    Used for a GeoParquet covering.
    """
    return on_geometry(
        geometry,
        lambda column, dtype: _bounds(column, dtype, 0.0, 0.0, per_part=False),
    )


def bounds(
    geometry: IntoExprColumn, *, margin_x: float = 0.0, margin_y: float = 0.0
) -> pl.Expr:
    """The bounding box of a geometry, as a `geoarrow.box` with its dimension and CRS.
    A multi-geometry gives a list of boxes instead, one per part.

    | in | out |
    | - | - |
    | `PointXY[ZM]` | `BoxXY[ZM]` |
    | `LineStringXY[ZM]` | `BoxXY[ZM]` |
    | `PolygonXY[ZM]` | `BoxXY[ZM]` |
    | `MultiPointXY[ZM]` | `list[BoxXY[ZM]]` |
    | `MultiLineStringXY[ZM]` | `list[BoxXY[ZM]]` |
    | `MultiPolygonXY[ZM]` | `list[BoxXY[ZM]]` |

    A point's box repeats its own coordinates: `xmin == xmax`.
    z and m are bounded like x and y.
    `margin_x` and `margin_y` widen the box by that much on both sides
    (a negative margin shrinks it); z and m get no margin.

    The bounds are the minimum and maximum of the stored coordinates.
    With a geographic CRS, `x` is then brought back into `[-180, 180]`
    (or the same in the CRS's unit).
    This considers the margin as well.
    `xmin` lands in `[-180, 180]`, and `xmax` moves by the same turns:
    a geometry running from 170 to 190 gets a box from 170 to -170,
    which is how a box crosses the antimeridian (`xmin > xmax`),
    and one from 540 to 545 gets a box from -180 to -175.
    One spanning a full turn or more gets `-180` to `180`.
    A geometry that crosses the antimeridian without leaving that range,
    such as a line from 170 straight to -170, cannot be told apart
    from one going the long way round, and gets a box from -170 to 170.
    Without a geographic CRS the bounds are left as they are.
    A geometry with no coordinates (or only empty points) gets an empty box,
    `inf` to `-inf` in every range. A missing geometry gives a missing box.

    ```python
    df.select(geo.bounds("parcel", margin_x=10.0, margin_y=10.0))
    ```
    """
    return on_geometry(
        geometry,
        lambda column, dtype: _bounds(column, dtype, margin_x, margin_y, per_part=True),
    )
