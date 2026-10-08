"""The rectangle a box describes, as a polygon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import BoxXY, PolygonXY
from geopolars.geo._dispatch import on_box

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoBox


def _box_to_polygon(column: pl.Expr, box: GeoBox) -> pl.Expr:
    # FIXME: allow creation of PolygonXYZM or PolygonXYZ or PolygonXYM
    if not isinstance(box, BoxXY):
        # A polygon is flat: it has nowhere to keep a range of z or m.
        msg = (
            f"expected a `BoxXY`, as a polygon cannot hold z or m bounds, got: {box!r}"
        )
        raise TypeError(msg)

    bounds = column.ext.storage()
    xmin, ymin, xmax, ymax = (
        bounds.struct.field(name) for name in ("xmin", "ymin", "xmax", "ymax")
    )
    # The spec writes an empty range as `inf` to `-inf`.
    # `xmin > xmax` is otherwise a box crossing the antimeridian;
    # the exception does not apply to y.
    empty_x = (xmin > xmax) & ((xmin == float("inf")) | (xmax == float("-inf")))
    empty = empty_x | (ymin > ymax)
    crosses_antimeridian = xmin > xmax

    turn = box._longitude_turn()
    if turn is not None:
        # Continue east past the antimeridian (to 190 rather than -170),
        # so the polygon covers the side the box means.
        xmax = pl.when(crosses_antimeridian).then(xmax + turn).otherwise(xmax)

    # The exterior ring, counter-clockwise and closed.
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    ring = pl.concat_arr([pl.struct(x=x, y=y) for x, y in corners])
    # Polars has no elementwise way to wrap a list in another list,
    # so a single ring is reshaped into a list of one ring.
    storage = PolygonXY._geo_storage
    rings = ring.reshape((-1, 1, len(corners))).cast(storage)

    polygon = (
        pl.when(bounds.is_null()).then(None).when(empty).then(pl.lit([], dtype=storage))
    )
    if turn is None:
        # Without a longitude there is no way around: no polygon describes it.
        polygon = polygon.when(crosses_antimeridian).then(None)
    polygon = polygon.otherwise(rings)
    return polygon.ext.to(PolygonXY._with_metadata_of(box))


def box_to_polygon(box: IntoExprColumn) -> pl.Expr:
    """The rectangle a `BoxXY` describes, as a `PolygonXY` with the box's CRS.

    The polygon has a single, counter-clockwise exterior ring of five vertices,
    starting and ending at `(xmin, ymin)`.

    | box                                     | polygon                         |
    |-----------------------------------------|---------------------------------|
    | missing                                 | missing                         |
    | an empty x or y range (`inf` to `-inf`) | empty (no rings)                |
    | `xmin > xmax`, geographic CRS           | continues past the antimeridian |
    | `xmin > xmax`, any other CRS, or none   | missing                         |

    A box from 170 to -170 crosses the antimeridian.
    With a geographic CRS, its polygon runs from 170 to 190 instead:
    one full turn (360 degrees, or 400 grads) is added to `xmax`,
    so the polygon covers the same side of the globe as the box.
    Subtract the turn again to bring a vertex back into range.
    Planar operations on such a polygon only see it as running to 190,
    so a point at -175 is not inside it to them.
    Without a geographic CRS there is no way around, and no polygon describes the box.

    A box with z or m bounds is refused, as a polygon has nowhere to keep them.

    ```python
    df.select(geo.box_to_polygon("extent"))
    ```
    """
    return on_box(box, _box_to_polygon)
