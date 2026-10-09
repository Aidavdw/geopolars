"""The exterior (outer) ring of a polygon, as a linestring.

A polygon's is plain Polars (C-tier).
A multipolygon's runs in a native kernel (A-tier), which gathers the rings in bulk:
in Polars it would take a list operation inside `list.eval`, which runs per row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, MultiPolygonType, PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _ring(polygon: pl.Expr, line_storage: PolarsDataType) -> pl.Expr:
    return (
        pl.when(polygon.list.len() == 0)
        .then(pl.lit([], dtype=line_storage))
        .otherwise(polygon.list.first())
    )


def _exterior(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (PolygonType, MultiPolygonType)):
        msg = f"exterior expects a polygon or multipolygon column, got: {geometry!r}"
        raise TypeError(msg)
    if isinstance(geometry, MultiPolygonType):
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="exterior",
            is_elementwise=True,
        )
    line_type = LineStringType.of_dimension(geometry._dimension)
    line = line_type._with_metadata_of(geometry)
    storage = column.ext.storage()
    return _ring(storage, line_type._geo_storage).ext.to(line)


def exterior(geometry: IntoExprColumn) -> pl.Expr:
    """The exterior (outer) ring of a polygon (holes left out),
    as a closed linestring with the polygon's dimension and CRS.
    A multipolygon gives a list of exteriors instead, one per polygon.

    | in                  | out                     |
    |---------------------|-------------------------|
    | `PolygonXY`         | `LineStringXY`          |
    | `PolygonXYZM`       | `LineStringXYZM`        |
    | `MultiPolygonXYZ`   | `list[LineStringXYZ]`   |

    A polygon without rings gives an empty linestring.
    A missing polygon gives a missing linestring.

    ```python
    df.select(geo.exterior("parcel"))
    ```
    """
    return on_geometry(geometry, _exterior)
