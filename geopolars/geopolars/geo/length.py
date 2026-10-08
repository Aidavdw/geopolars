"""How long a linestring is.

Measured the same way as `distance`, segment by segment:
along the ellipsoid in the unit of the CRS for a geometry that declares one
(metres for a CRS in longitude/latitude),
in coordinate units for one that doesn't, with the height counted when there is a `z`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, MultiLineStringType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.distance import _has_z, _squared_norm

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _planar(line: pl.Expr, *, has_z: bool) -> pl.Expr:
    """The length in space of one list of coordinates.

    Each vertex is diffed against the one before it, per axis,
    `z` included only when the line has one.
    That is the distance between the two written for a whole line at once,
    and twice as fast as shifting the coordinate structs to pair them up.
    The first vertex has nothing before it, which `sum` skips,
    so an empty line or a single vertex has a length of `0.0`.
    """
    coord = pl.element().struct
    squared = _squared_norm(coord.field("x").diff(), coord.field("y").diff())
    if has_z:
        squared = squared + coord.field("z").diff().pow(2)
    return line.list.eval(squared.sqrt()).list.sum()


def _geodesic(column: pl.Expr) -> pl.Expr:
    """The length along the ellipsoid of the CRS, in its unit."""
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="length_geodesic",
        is_elementwise=True,
    )


def _length(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (LineStringType, MultiLineStringType)):
        msg = (
            "a length is measured on a `geoarrow.linestring` "
            f"or `geoarrow.multilinestring` column, got: {geometry!r}"
        )
        raise TypeError(msg)

    # The kernel counts the height itself, when the line has one.
    if geometry._declares_crs():
        return _geodesic(column)

    storage = column.ext.storage()
    has_z = _has_z(geometry)
    if isinstance(geometry, LineStringType):
        return _planar(storage, has_z=has_z)
    return storage.list.eval(_planar(pl.element(), has_z=has_z))


def length(geometry: IntoExprColumn) -> pl.Expr:
    """How long a linestring is, as an `f64`.

    | in                  | measured                     | in                  |
    |---------------------|------------------------------|---------------------|
    | no CRS              | in space, by Pythagoras      | coordinate units    |
    | a CRS               | along the CRS's ellipsoid    | CRS units           |

    Every segment is measured as `distance` measures two points, and added up.
    A multilinestring gets the length of each of its parts, in order, as a list;
    sum it with `.list.sum()` for the length of the whole.
    Other geometries are refused.

    For a line with a `z`, every segment counts its difference in height too,
    as `distance` does between two points with a `z`
    (with a CRS, `z` is taken to be in the same unit as the length:
    that of `x` and `y`, or metres when those are degrees).
    `m` is ignored.
    An empty linestring, or one of a single vertex, has a length of `0.0`.
    A missing geometry has no length.

    ```python
    df.select(geo.length("route"))
    ```
    """
    return on_geometry(geometry, _length)
