"""Removing coinciding points in succession (A-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

CollapsedLines = Literal["keep_ends", "single_vertex"]


@doc()
def remove_repeated_points(
    geometry: IntoExprColumn,
    *,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    unique_z: bool = True,
    unique_m: bool = False,
    collapsed_lines: CollapsedLines = "keep_ends",
    drop_short_lines: bool = False,
) -> pl.Expr:
    """Removes coincident points in direct succession, leaving only one.
    If `n` repeating points are all at the same place, all but the first are removed,
    except at the end of a linestring or ring.
    Does not remove duplicates that are not directly after each other (see XXXXX)

    The last vertex of a linestring or ring is always kept, so a ring stays closed.
    In such a case any repeating point before that is removed rather than the very last point
    to ensure exact coincidence (a prerequisite for GeoArrow validity),
    unless that one is the first vertex (see `collapsed_lines` below).

    | geometry | removed |
    | - | - |
    | `PointXY[ZM]` | nothing, returns the input |
    | `LineStringXY[ZM]`, `MultiPointXY[ZM]` | coincidence in direct succeccesion |
    | `PolygonXY[ZM]` | the same, per ring |
    | `MultiLineStringXY[ZM]`, `MultiPolygonXY[ZM]` | the same, per part |

    Each vertex is compared with the first vertex in a coinciding set, not the one directly in front of it.
    This means that a group of vertices that slowly drifts will at some point be cut off.

    {coincidence_axes}
    {coincidence_tolerance}

    This operation removes coordinates, which means that linestrings can be left
    with an insufficient amount of vertices.
    What is done with these is determined by the argument `collapsed_lines`.
    `collapsed_lines` says what a linestring keeps when all its vertices lie at one place.
    With `"keep_ends"`, it keeps its first and last vertex, as GEOS does.
    With `"single_vertex"`, it keeps only its first.
    This argument is not considered for polygon rings.
    A ring always keeps both ends in order to stay closed (and therefore valid GeoArrow).

    `collapsed_lines` works together with `drop_short_lines`.
    `drop_short_lines` sets any resulting linestring that is left with only a single vertex,
    and a polygon ring with fewer than 3 (all at one place, closed), to empty.

    For a linestring:

    | `collapsed_lines` | `drop_short_lines` | `A A A` gives | 1-vertex input gives |
    | - | - | - | - |
    | `"keep_ends"` | `False` | `[A, A]` (as GEOS does) | unchanged |
    | `"keep_ends"` | `True` | `[A, A]` | empty |
    | `"single_vertex"` | `False` | `[A]` | unchanged |
    | `"single_vertex"` | `True` | empty | empty |

    Only the last combination never leaves a linestring that `is_valid_ogc` rejects.

    A (multi)polygon with a ring that is left with fewer than 4 vertices becomes missing,
    unless `drop_short_lines` empties that ring.
    For a ring, which `collapsed_lines` does not affect:

    | `drop_short_lines` | `A A A A` gives | `A B B A` gives |
    | - | - | - |
    | `False` | missing row | missing row |
    | `True` | empty ring | missing row |

    Note that leaving empty (outer) rings may make a polygon invalid by OGC,
    unless every hole of it is empty too (the empty polygon).

    {ignores_empty_and_invalid}

    ```python
    df.select(geo.remove_repeated_points("route", abs_tol=0.01))
    ```
    """

    if collapsed_lines not in get_args(CollapsedLines):
        msg = (
            f"`collapsed_lines` must be one of {list(get_args(CollapsedLines))}, "
            f"got {collapsed_lines!r}"
        )
        raise ValueError(msg)

    def build(column: pl.Expr, _geometry: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="remove_repeated_points",
            is_elementwise=True,
            kwargs={
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "unique_z": unique_z,
                "unique_m": unique_m,
                "collapsed_lines": collapsed_lines,
                "drop_short_lines": drop_short_lines,
            },
        )

    return on_geometry(geometry, build)
