"""Provides `geo` expression namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from geopolars.geo import affine, construct, coordinates, wkb, wkt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@pl.api.register_expr_namespace("geo")
class ExprGeoNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def line_string(self, *, crs: str | None = None) -> pl.Expr:
        return construct.line_string_from_vertices(self._expr, crs=crs)

    def multi_point(self, *, crs: str | None = None) -> pl.Expr:
        return construct.multi_point_from_points(self._expr, crs=crs)

    def multi_line_string(self, *, crs: str | None = None) -> pl.Expr:
        return construct.multi_line_string_from_line_strings(self._expr, crs=crs)

    def polygon(self, *, crs: str | None = None) -> pl.Expr:
        return construct.polygon_from_rings(self._expr, crs=crs)

    def multi_polygon(self, *, crs: str | None = None) -> pl.Expr:
        return construct.multi_polygon_from_polygons(self._expr, crs=crs)

    def validate(self) -> pl.Expr:
        return construct.validate(self._expr)

    def translate(self, offset: affine.Offset | IntoExprColumn) -> pl.Expr:
        return affine.translate(self._expr, offset)

    def affine_transform(
        self,
        matrix: Sequence[float],
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.affine_transform(self._expr, matrix, origin=origin)

    def rotate(
        self,
        amount: float,
        unit: Literal["deg", "pi"] = "deg",
        axis: Literal["x", "y", "z"] = "z",
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.rotate(self._expr, amount, unit=unit, axis=axis, origin=origin)

    def rotate_about_centroid(
        self,
        amount: float,
        unit: Literal["deg", "pi"] = "deg",
        axis: Literal["x", "y", "z"] = "z",
    ) -> pl.Expr:
        return affine.rotate_about_centroid(self._expr, amount, unit=unit, axis=axis)

    def skew(
        self,
        xs: float = 0.0,
        ys: float = 0.0,
        unit: Literal["deg", "pi"] = "deg",
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.skew(self._expr, xs=xs, ys=ys, unit=unit, origin=origin)

    def skew_about_centroid(
        self, xs: float = 0.0, ys: float = 0.0, unit: Literal["deg", "pi"] = "deg"
    ) -> pl.Expr:
        return affine.skew_about_centroid(self._expr, xs=xs, ys=ys, unit=unit)

    def scale(
        self,
        xfact: float = 1.0,
        yfact: float = 1.0,
        zfact: float = 1.0,
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.scale(
            self._expr, xfact=xfact, yfact=yfact, zfact=zfact, origin=origin
        )

    def scale_about_centroid(
        self, xfact: float = 1.0, yfact: float = 1.0, zfact: float = 1.0
    ) -> pl.Expr:
        return affine.scale_about_centroid(
            self._expr, xfact=xfact, yfact=yfact, zfact=zfact
        )

    def set_crs(self, crs: str, *, force: bool = False) -> pl.Expr:
        from geopolars.geo.crs import set_crs

        return set_crs(self._expr, crs, force=force)

    def to_crs(self, crs: str) -> pl.Expr:
        from geopolars.geo.crs import to_crs

        return to_crs(self._expr, crs)

    def is_geographic(self, *, ignore_errors: bool = False) -> pl.Expr:
        from geopolars.geo.crs import is_geographic

        return is_geographic(self._expr, ignore_errors=ignore_errors)

    def mean_coordinate(self) -> pl.Expr:
        from geopolars.geo.mean_coordinate import mean_coordinate

        return mean_coordinate(self._expr)

    def centroid(self) -> pl.Expr:
        from geopolars.geo.centroid import centroid

        return centroid(self._expr)

    def exterior(self) -> pl.Expr:
        from geopolars.geo.exterior import exterior

        return exterior(self._expr)

    def interior(self) -> pl.Expr:
        from geopolars.geo.interior import interior

        return interior(self._expr)

    def boundary(self) -> pl.Expr:
        from geopolars.geo.boundary import boundary

        return boundary(self._expr)

    def is_ring(self) -> pl.Expr:
        from geopolars.geo.is_ring import is_ring

        return is_ring(self._expr)

    def distance(self, other: IntoExprColumn) -> pl.Expr:
        from geopolars.geo.distance import distance

        return distance(self._expr, other)

    def distance_squared(self, other: IntoExprColumn) -> pl.Expr:
        from geopolars.geo.distance import distance_squared

        return distance_squared(self._expr, other)

    def length(self, *, allow_non_lines: bool = False) -> pl.Expr:
        from geopolars.geo.length import length

        return length(self._expr, allow_non_lines=allow_non_lines)

    def length_planar(self, *, allow_non_lines: bool = False) -> pl.Expr:
        from geopolars.geo.length import length_planar

        return length_planar(self._expr, allow_non_lines=allow_non_lines)

    def length_geodesic(self, *, allow_non_lines: bool = False) -> pl.Expr:
        from geopolars.geo.length import length_geodesic

        return length_geodesic(self._expr, allow_non_lines=allow_non_lines)

    def area(self, *, allow_non_polygons: bool = False) -> pl.Expr:
        from geopolars.geo.area import area

        return area(self._expr, allow_non_polygons=allow_non_polygons)

    def area_planar(self, *, allow_non_polygons: bool = False) -> pl.Expr:
        from geopolars.geo.area import area_planar

        return area_planar(self._expr, allow_non_polygons=allow_non_polygons)

    def area_geodesic(self, *, allow_non_polygons: bool = False) -> pl.Expr:
        from geopolars.geo.area import area_geodesic

        return area_geodesic(self._expr, allow_non_polygons=allow_non_polygons)

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
