"""Provides `geo` expression namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo import affine, construct, coordinates, wkb, wkt

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@pl.api.register_expr_namespace("geo")
class ExprGeoNameSpace:
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

    def set_crs(self, crs: str, *, force: bool = False) -> pl.Expr:
        from geopolars.geo.crs import set_crs

        return set_crs(self._expr, crs, force=force)

    def to_crs(self, crs: str) -> pl.Expr:
        from geopolars.geo.crs import to_crs

        return to_crs(self._expr, crs)

    def mean_coordinate(self) -> pl.Expr:
        from geopolars.geo.mean_coordinate import mean_coordinate

        return mean_coordinate(self._expr)

    def distance(self, other: IntoExprColumn) -> pl.Expr:
        from geopolars.geo.distance import distance

        return distance(self._expr, other)

    def distance_squared(self, other: IntoExprColumn) -> pl.Expr:
        from geopolars.geo.distance import distance_squared

        return distance_squared(self._expr, other)

    def length(self) -> pl.Expr:
        from geopolars.geo.length import length

        return length(self._expr)

    def area(self) -> pl.Expr:
        from geopolars.geo.area import area

        return area(self._expr)

    def is_empty(self) -> pl.Expr:
        from geopolars.geo.is_empty import is_empty

        return is_empty(self._expr)

    def count_coordinates(self) -> pl.Expr:
        from geopolars.geo.count_coordinates import count_coordinates

        return count_coordinates(self._expr)

    def bounds(self, *, margin_x: float = 0.0, margin_y: float = 0.0) -> pl.Expr:
        from geopolars.geo.bounds import bounds

        return bounds(self._expr, margin_x=margin_x, margin_y=margin_y)

    def to_polygon(self) -> pl.Expr:
        from geopolars.geo.to_polygon import to_polygon

        return to_polygon(self._expr)

    def wrap_longitude(
        self, *, start: float | None = None, skip_crossing: bool = False
    ) -> pl.Expr:
        from geopolars.geo.wrap_longitude import wrap_longitude

        return wrap_longitude(self._expr, start=start, skip_crossing=skip_crossing)

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

    def from_wkb(self, dtype: type[GeoArrowType], *, crs: str | None = None) -> pl.Expr:
        return wkb.from_wkb(self._expr, dtype, crs=crs)

    def to_wkt(self) -> pl.Expr:
        return wkt.to_wkt(self._expr)

    def from_wkt(self, dtype: type[GeoArrowType], *, crs: str | None = None) -> pl.Expr:
        return wkt.from_wkt(self._expr, dtype, crs=crs)
