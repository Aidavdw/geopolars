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

    def convex_hull(
        self, *, stop_on_error: bool = False, ccw_outer: bool = False
    ) -> pl.Expr:
        from geopolars.geo.convex_hull import convex_hull

        return convex_hull(self._expr, stop_on_error=stop_on_error, ccw_outer=ccw_outer)

    def simplify(
        self,
        tolerance: float,
        *,
        preserve_topology: bool = False,
        ignore_per_row_errors: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.simplify import simplify

        return simplify(
            self._expr,
            tolerance,
            preserve_topology=preserve_topology,
            ignore_per_row_errors=ignore_per_row_errors,
        )

    def boundary(self) -> pl.Expr:
        from geopolars.geo.boundary import boundary

        return boundary(self._expr)

    def is_ring(self) -> pl.Expr:
        from geopolars.geo.is_ring import is_ring

        return is_ring(self._expr)

    def self_intersects(self) -> pl.Expr:
        from geopolars.geo.self_intersects import self_intersects

        return self_intersects(self._expr)

    def rings_overlap(
        self, other: IntoExprColumn, *, include_touching: bool = False
    ) -> pl.Expr:
        from geopolars.geo.rings_overlap import rings_overlap

        return rings_overlap(self._expr, other, include_touching=include_touching)

    def is_standard_orientation(self, *, holes_clockwise: bool = True) -> pl.Expr:
        from geopolars.geo.is_standard_orientation import is_standard_orientation

        return is_standard_orientation(self._expr, holes_clockwise=holes_clockwise)

    def is_valid(
        self,
        *,
        allow_wrapped_longitude: bool = True,
        within_area_of_use: bool = False,
        check_missing_coordinates: bool = True,
        check_finite_coordinates: bool = True,
        check_crs_bounds: bool = True,
        check_line_string_zero_length_sections: bool = True,
        check_ring_self_intersect: bool = True,
        check_polygon_self_overlap: bool = True,
        check_multi_polygon_self_overlap: bool = True,
    ) -> pl.Expr:
        from geopolars.geo.is_valid import is_valid

        return is_valid(
            self._expr,
            allow_wrapped_longitude=allow_wrapped_longitude,
            within_area_of_use=within_area_of_use,
            check_missing_coordinates=check_missing_coordinates,
            check_finite_coordinates=check_finite_coordinates,
            check_crs_bounds=check_crs_bounds,
            check_line_string_zero_length_sections=check_line_string_zero_length_sections,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        )

    def null_out_invalid(
        self,
        *,
        allow_wrapped_longitude: bool = True,
        within_area_of_use: bool = False,
        check_missing_coordinates: bool = True,
        check_finite_coordinates: bool = True,
        check_crs_bounds: bool = True,
        check_line_string_zero_length_sections: bool = True,
        check_ring_self_intersect: bool = True,
        check_polygon_self_overlap: bool = True,
        check_multi_polygon_self_overlap: bool = True,
    ) -> pl.Expr:
        from geopolars.geo.is_valid import null_out_invalid

        return null_out_invalid(
            self._expr,
            allow_wrapped_longitude=allow_wrapped_longitude,
            within_area_of_use=within_area_of_use,
            check_missing_coordinates=check_missing_coordinates,
            check_finite_coordinates=check_finite_coordinates,
            check_crs_bounds=check_crs_bounds,
            check_line_string_zero_length_sections=check_line_string_zero_length_sections,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        )

    def distance(
        self,
        other: IntoExprColumn,
        *,
        method: Literal["planar", "geodesic"] | None = None,
    ) -> pl.Expr:
        from geopolars.geo.distance import distance

        return distance(self._expr, other, method=method)

    def length(
        self,
        *,
        method: Literal["planar", "geodesic"] | None = None,
        allow_non_lines: bool = False,
        ignore_nan: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.length import length

        return length(
            self._expr,
            method=method,
            allow_non_lines=allow_non_lines,
            ignore_nan=ignore_nan,
        )

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
