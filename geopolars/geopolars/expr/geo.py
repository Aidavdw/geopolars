"""Provides `geo` expression namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo import affine, construct, coordinates, wkb, wkt

# bound directly: `geopolars.geo.area` is the function, not the module.
from geopolars.geo.area import area as _area

# bound directly: the method's `crs` argument would shadow the module.
from geopolars.geo.crs import to_crs as _to_crs

# bound directly: `geopolars.geo.distance` is the function, not the module.
from geopolars.geo.distance import distance as _distance
from geopolars.geo.distance import distance_squared as _distance_squared

# bound directly: `geopolars.geo.length` is the function, not the module.
from geopolars.geo.length import length as _length

# bound directly: `geopolars.geo.mean_coordinate` is the function, not the module.
from geopolars.geo.mean_coordinate import mean_coordinate as _mean_coordinate

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@pl.api.register_expr_namespace("geo")
class Geometry:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def linestring(self, *, crs: str | None = None) -> pl.Expr:
        return construct.linestring_from_vertices(self._expr, crs=crs)

    def multipoint(self, *, crs: str | None = None) -> pl.Expr:
        return construct.multipoint_from_points(self._expr, crs=crs)

    def multilinestring(self, *, crs: str | None = None) -> pl.Expr:
        return construct.multilinestring_from_linestrings(self._expr, crs=crs)

    def polygon(self, *, crs: str | None = None) -> pl.Expr:
        return construct.polygon_from_rings(self._expr, crs=crs)

    def multipolygon(self, *, crs: str | None = None) -> pl.Expr:
        return construct.multipolygon_from_polygons(self._expr, crs=crs)

    def validate(self) -> pl.Expr:
        return construct.validate(self._expr)

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> pl.Expr:
        return affine.translate(self._expr, dx=dx, dy=dy, dz=dz)

    def to_crs(self, crs: str) -> pl.Expr:
        return _to_crs(self._expr, crs)

    def mean_coordinate(self) -> pl.Expr:
        return _mean_coordinate(self._expr)

    def distance(self, other: IntoExprColumn) -> pl.Expr:
        return _distance(self._expr, other)

    def distance_squared(self, other: IntoExprColumn) -> pl.Expr:
        return _distance_squared(self._expr, other)

    def length(self) -> pl.Expr:
        return _length(self._expr)

    def area(self) -> pl.Expr:
        return _area(self._expr)

    def x(self) -> pl.Expr:
        return coordinates.x(self._expr)

    def y(self) -> pl.Expr:
        return coordinates.y(self._expr)

    def z(self) -> pl.Expr:
        return coordinates.z(self._expr)

    def m(self) -> pl.Expr:
        return coordinates.m(self._expr)

    def to_wkb(self) -> pl.Expr:
        return wkb.to_wkb(self._expr)

    def from_wkb(
        self, geometry: type[GeoArrowType], *, crs: str | None = None
    ) -> pl.Expr:
        return wkb.from_wkb(self._expr, geometry, crs=crs)

    def to_wkt(self) -> pl.Expr:
        return wkt.to_wkt(self._expr)

    def from_wkt(
        self, geometry: type[GeoArrowType], *, crs: str | None = None
    ) -> pl.Expr:
        return wkt.from_wkt(self._expr, geometry, crs=crs)
