"""The buffer of a geometry: every point within a distance of it.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, MultiLineStringType, PointType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_CAP_STYLES = ("round", "flat", "square")
_JOIN_STYLES = ("round", "mitre", "bevel")
_SIDES = ("both", "left", "right")
# GEOS's own default.
_MITRE_LIMIT = 5.0
_SINGLE_SIDED = (PointType, LineStringType, MultiLineStringType)


def _check_one_of(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        msg = f"buffer expects `{name}` to be one of {allowed}, got: {value!r}"
        raise ValueError(msg)


def buffer(
    geometry: IntoExprColumn,
    distance: float,
    *,
    quadrant_segments: int = 8,
    cap_style: Literal["round", "flat", "square"] | None = None,
    join_style: Literal["round", "mitre", "bevel"] = "round",
    mitre_limit: float | None = None,
    side: Literal["both", "left", "right"] = "both",
    allow_splitting: bool = False,
    ccw_outer: bool = False,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """Draws a 'buffer' around the geometry.
    Think of it like an outset/inset of the original shape.
    Every point of this buffer is a polygon with points is within `distance` of the original.
    `distance` is in the units of the coordinates
    (degrees for longitude/latitude).

    A point, linestring or polygon gives a polygon.
    Their multi-variants give a multipolygon,
    in which parts whose buffers overlap are merged into one.
    The output is always XY:

    | in | out | with `allow_splitting` and `distance < 0` |
    | - | - | - |
    | `PointXY[Z][M]` | `PolygonXY` | `MultiPolygonXY` |
    | `LineStringXY[Z][M]` | `PolygonXY` | `MultiPolygonXY` |
    | `PolygonXY[Z][M]` | `PolygonXY` | `MultiPolygonXY` |
    | `MultiPointXY[Z][M]` | `MultiPolygonXY` | `MultiPolygonXY` |
    | `MultiLineStringXY[Z][M]` | `MultiPolygonXY` | `MultiPolygonXY` |
    | `MultiPolygonXY[Z][M]` | `MultiPolygonXY` | `MultiPolygonXY` |

    Computed by GEOS.
    This operation drops z and m.
    An empty geometry gives an empty polygon, and a missing one a missing polygon.
    A `distance` of at most 0 gives an empty polygon for a point or linestring.

    With `allow_splitting=True` and a negative `distance`, every kind gives a multipolygon.
    A negative `distance` shrinks a polygon, which can split it into several polygons.
    Splitting can only happen when `distance` is negative:
    a positive buffer of a geometry in one piece stays in one piece.
    Take for example a dumbbell-like shape: two 4 by 4 squares, joined by a neck 0.4 wide.
    Shrunk by 0.5, the section in the neck vanishes, and two 3 by 3 squares are left.
    That shape can only be represented as a multipolygon.
    With `allow_splitting=True` and a negative `distance`, every row is a multipolygon,
    which holds both squares.

    With `distance>0`, splitting cannot occur, because the outset will always expand outwards.
    This means you can never end up with 'islands', so it is not necessary to output multipolygons.
    `allow_splitting=true` does not have to change the output type in this case.

    For geometries with a CRS, longitude/latitude the distance is in degrees, not along the globe.

    You can also customise the shape of the buffer!

    - `quadrant_segments` is the number of segments that approximate a quarter circle.
      It must be at least 1.
    - `cap_style` is how a line ends: `"round"` (the default), `"flat"` or `"square"`.
    - `join_style` is how two segments meet at a corner: `"round"`, `"mitre"` or `"bevel"`.
    - `mitre_limit` only goes with `join_style="mitre"`.
      A corner that sticks out more than `mitre_limit` times `distance` is bevelled.
      It defaults to 5.
    - `side="left"` or `side="right"` buffers only that side of a line,
      relative to its direction.
      It goes with a `distance` of at least 0, and has flat ends, so it takes no `cap_style`.
      A point has no sides, so it gets its regular buffer.
      A polygon or multipoint is refused.

    A geometry GEOS fails on (or a polygon it splits, without `allow_splitting`) raises GEOS's error,
    failing the query.
    With `ignore_per_row_errors=True`, it gives a missing polygon instead.

    GEOS supports neither empty points inside a geometry (a vertex with NaN x and y)
    nor any other NaN or infinite x or y.
    By default they go to GEOS as they are.
    With `early_finite_check=True`, a geometry with such a vertex is caught before it reaches GEOS.
    It then raises, or gives a missing polygon with `ignore_per_row_errors=True`.
    This only adds a short path.
    It pays off when you expect many non-finite coordinates in your data.

    The exterior runs as GEOS gives it, clockwise.
    With `ccw_outer=True`, it runs counter-clockwise instead at a minimal performance cost.

    ```python
    df.select(geo.buffer("parcel", 2.0))
    df.select(geo.buffer("road", 5.0, cap_style="flat"))
    ```
    """

    # TODO: we could add some image examples here?
    # - a normal outset
    # - a normal inset
    # - a dumbbell shape inset that causes splitting

    if not math.isfinite(distance):
        msg = f"buffer expects a finite distance, got: {distance!r}"
        raise ValueError(msg)
    if quadrant_segments < 1:
        msg = f"buffer expects `quadrant_segments` of at least 1, got: {quadrant_segments!r}"
        raise ValueError(msg)
    _check_one_of("join_style", join_style, _JOIN_STYLES)
    _check_one_of("side", side, _SIDES)
    if cap_style is not None:
        _check_one_of("cap_style", cap_style, _CAP_STYLES)
    if mitre_limit is not None and join_style != "mitre":
        msg = f"buffer only takes `mitre_limit` with join_style='mitre', got: {join_style!r}"
        raise ValueError(msg)
    single_sided = side != "both"
    if single_sided and cap_style is not None:
        msg = f"buffer has flat ends with side={side!r}, so it takes no `cap_style`"
        raise ValueError(msg)
    if single_sided and distance < 0:
        msg = f"buffer expects a distance of at least 0 with side={side!r}, got: {distance!r}"
        raise ValueError(msg)

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if single_sided and not isinstance(dtype, _SINGLE_SIDED):
            msg = (
                "a buffer of one side expects a point, linestring or multilinestring column,"
                f" got: {dtype!r}"
            )
            raise TypeError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="buffer",
            kwargs={
                # GEOS buffers the right side for a negative distance.
                "distance": -float(distance) if side == "right" else float(distance),
                "quadrant_segments": int(quadrant_segments),
                "cap_style": cap_style or "round",
                "join_style": join_style,
                "mitre_limit": _MITRE_LIMIT
                if mitre_limit is None
                else float(mitre_limit),
                "single_sided": single_sided,
                "allow_splitting": allow_splitting,
                "ccw_outer": ccw_outer,
                "ignore_per_row_errors": ignore_per_row_errors,
                "early_finite_check": early_finite_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
