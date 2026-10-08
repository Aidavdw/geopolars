"""How many coordinates a geometry is made of."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo._dispatch import on_geometry
from geopolars.geo.is_empty import _empty_point
from geopolars.geo.mean_coordinate import _coordinates

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _count_coordinates(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if geometry._nesting == 0:
        # A null point stays null: `is_nan` of null is null.
        count = ~_empty_point(column.ext.storage())
    else:
        count = _coordinates(geometry, column)
    # The same dtype whatever the nesting, the one Polars counts lengths in.
    return count.cast(pl.UInt32)


def count_coordinates(geometry: IntoExprColumn) -> pl.Expr:
    """How many coordinates a geometry is made of, over all of its parts.

    This gives one count per geometry, however deeply it is nested.

    | in                 | counts                                       |
    |--------------------|----------------------------------------------|
    | `Point…`           | `1`, or `0` when it is empty                 |
    | `LineString…`      | its vertices                                 |
    | `Polygon…`         | the vertices of every ring                   |
    | `MultiPoint…`      | its points                                   |
    | `MultiLineString…` | the vertices of every linestring             |
    | `MultiPolygon…`    | the vertices of every ring of every polygon  |

    This skips a ring's last coordinate, as this repeats its first.
    A geometry without coordinates counts `0`; a missing geometry gives null.

    ```python
    df.select(geo.count_coordinates("parcel"))
    ```
    """
    return on_geometry(geometry, _count_coordinates)
