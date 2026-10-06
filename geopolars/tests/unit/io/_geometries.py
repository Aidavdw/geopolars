"""Geometries of every kind, as native columns and as Shapely sees them.
Shared by the WKB and WKT tests.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import shapely

from geopolars.datatypes import GeoArrowType
from tests.unit.conftest import Dimension

Coordinate = tuple[float, float]

#: One geometry per kind, as nested `(x, y)` coordinates.
#: The polygon has a hole, so interior rings are covered.
SQUARE = [[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 0.0)]]
POLYGON = [*SQUARE, [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 1.0)]]
GEOMETRIES: dict[str, Any] = {
    "point": (1.0, 2.0),
    "linestring": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
    "polygon": POLYGON,
    "multipoint": [(1.0, 2.0), (3.0, 4.0)],
    "multilinestring": [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0), (7.0, 8.0)]],
    "multipolygon": [
        POLYGON,
        [[(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 10.0)]],
    ],
}


def kind_dtype(kind: str, dimension: Dimension) -> type[GeoArrowType]:
    return getattr(dimension, f"{kind}_dtype")  # type: ignore[no-any-return]


def extend(xy: Coordinate, dimension: Dimension) -> tuple[float, ...]:
    """The coordinate with every axis `dimension` has, `z`/`m` derived from x/y."""
    extra = {"z": xy[0] * 10, "m": xy[1] * 100}
    return xy + tuple(extra[name] for name in dimension.coords[2:])


def storage(nested: Any, dimension: Dimension) -> Any:
    """The geometry as Python values for its storage dtype."""
    if isinstance(nested, tuple):
        return dict(zip(dimension.coords, extend(nested, dimension)))
    return [storage(part, dimension) for part in nested]


def reference_wkt(kind: str, dimension: Dimension) -> str:
    def coord(xy: Coordinate) -> str:
        return " ".join(str(v) for v in extend(xy, dimension))

    def seq(texts: Any) -> str:
        return "(" + ", ".join(texts) + ")"

    def polygon(rings: Any) -> str:
        return seq(seq(map(coord, ring)) for ring in rings)

    geometry = GEOMETRIES[kind]
    text = {
        "point": lambda: seq([coord(geometry)]),
        "linestring": lambda: seq(map(coord, geometry)),
        "polygon": lambda: polygon(geometry),
        "multipoint": lambda: seq(seq([coord(p)]) for p in geometry),
        "multilinestring": lambda: polygon(geometry),
        "multipolygon": lambda: seq(map(polygon, geometry)),
    }[kind]()
    tag = "".join(dimension.coords[2:]).upper()
    return " ".join(part for part in (kind.upper(), tag, text) if part)


def column(kind: str, dimension: Dimension, crs: str | None = None) -> pl.Series:
    """One geometry of `kind`, then a null."""
    dtype = kind_dtype(kind, dimension)(crs=crs)
    values = [storage(GEOMETRIES[kind], dimension), None]
    return pl.Series("g", values, dtype=dtype.ext_storage()).ext.to(dtype)


def shapely_wkb(kind: str, dimension: Dimension, *, byte_order: int = 1) -> bytes:
    geometry = shapely.from_wkt(reference_wkt(kind, dimension))
    return shapely.to_wkb(  # type: ignore[no-any-return]
        geometry, flavor="iso", output_dimension=4, byte_order=byte_order
    )
