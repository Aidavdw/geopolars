"""Whether a geometry is valid: whole, finite, inside its CRS, and shaped right."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZM,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PointXYZM,
    PolygonXY,
)

WGS84 = "EPSG:4326"
UTM31N = "EPSG:32631"

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]
Line = list[Coord]

NAN = math.nan
INF = math.inf


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


_SQUARE = _line((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_HOLE = _line((1, 1), (2, 1), (2, 2), (1, 1))


def _storage(dtype: GeoArrowType) -> pl.DataType:
    """The storage of `dtype`, with every coordinate as `_XY` or `_XYZM`."""
    coords = _XYZM if isinstance(dtype, (PointXYZM, LineStringXYZM)) else _XY
    for _ in range(dtype._nesting):
        coords = pl.List(coords)
    return coords


def _is_valid(dtype: GeoArrowType, *rows: object, **kwargs: bool) -> list[bool | None]:
    """`is_valid` over `rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=_storage(dtype))
    df = pl.DataFrame(storage.ext.to(dtype))
    return df.select(geo.is_valid("g", **kwargs))["g"].to_list()


# Coordinates.


@pytest.mark.parametrize("axis", ["x", "y", "z", "m"])
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_axis_is_invalid(axis: str, value: float) -> None:
    point = {"x": 1.0, "y": 2.0, "z": 3.0, "m": 4.0} | {axis: value}

    assert _is_valid(PointXYZM(), point) == [False]


def test_a_point_that_is_nan_on_every_axis_is_the_empty_point() -> None:
    assert _is_valid(PointXYZM(), dict.fromkeys("xyzm", NAN)) == [True]


def test_a_point_that_is_only_partly_nan_is_invalid() -> None:
    assert _is_valid(PointXY(), {"x": NAN, "y": 1.0}) == [False]


def test_a_missing_axis_is_invalid() -> None:
    assert _is_valid(PointXY(), {"x": 1.0, "y": None}) == [False]


def test_a_nan_vertex_is_invalid_even_in_a_line() -> None:
    assert _is_valid(LineStringXY(), _line((0, 0), (NAN, NAN))) == [False]


# Linestrings.


def test_a_line_of_two_vertices_is_valid() -> None:
    assert _is_valid(LineStringXY(), _line((0, 0), (1, 1))) == [True]


def test_an_empty_line_is_valid() -> None:
    assert _is_valid(LineStringXY(), []) == [True]


def test_a_line_of_one_vertex_is_invalid() -> None:
    assert _is_valid(LineStringXY(), _line((0, 0))) == [False]


def test_two_vertices_in_a_row_at_the_same_place_are_invalid() -> None:
    assert _is_valid(LineStringXY(), _line((0, 0), (1, 1), (1, 1), (2, 0))) == [False]


def test_vertices_in_a_row_that_are_almost_the_same_are_invalid() -> None:
    """Same place as `pl.Expr.is_close` with its defaults."""
    line = _line((0, 0), (1, 1), (1 + 1e-12, 1), (2, 0))

    assert _is_valid(LineStringXY(), line) == [False]


def test_vertices_in_a_row_that_differ_only_in_m_are_invalid() -> None:
    """m is a measure along the line, not a place on it."""
    a = {"x": 1.0, "y": 1.0, "z": 0.0, "m": 0.0}
    b = a | {"m": 1.0}

    assert _is_valid(LineStringXYZM(), [a, b]) == [False]


def test_a_vertex_repeated_further_on_is_valid() -> None:
    assert _is_valid(LineStringXY(), _line((0, 0), (1, 1), (0, 0))) == [True]


def test_a_missing_vertex_is_invalid() -> None:
    assert _is_valid(LineStringXY(), [{"x": 0.0, "y": 0.0}, None]) == [False]


# Polygons.


def test_a_polygon_with_a_hole_is_valid() -> None:
    assert _is_valid(PolygonXY(), [_SQUARE, _HOLE]) == [True]


def test_an_open_ring_is_invalid() -> None:
    assert _is_valid(PolygonXY(), [_SQUARE[:-1]]) == [False]


def test_a_closed_ring_of_three_vertices_is_invalid() -> None:
    assert _is_valid(PolygonXY(), [_line((0, 0), (1, 1), (0, 0))]) == [False]


def test_a_ring_with_vertices_in_a_row_at_the_same_place_is_invalid() -> None:
    ring = _line((0, 0), (4, 0), (4, 0), (4, 4), (0, 0))

    assert _is_valid(PolygonXY(), [ring]) == [False]


def test_an_invalid_hole_makes_the_polygon_invalid() -> None:
    assert _is_valid(PolygonXY(), [_SQUARE, _HOLE[:-1]]) == [False]


@pytest.mark.parametrize("rings", [[], [[]], [[], []]])
def test_an_empty_polygon_is_valid(rings: list[Line]) -> None:
    assert _is_valid(PolygonXY(), rings) == [True]


def test_an_empty_exterior_around_a_hole_is_invalid() -> None:
    assert _is_valid(PolygonXY(), [[], _HOLE]) == [False]


def test_a_missing_ring_is_invalid() -> None:
    assert _is_valid(PolygonXY(), [_SQUARE, None]) == [False]


# Multi-geometries: valid when every part is.


def test_a_multipoint_with_one_invalid_point_is_invalid() -> None:
    good, bad = {"x": 0.0, "y": 0.0}, {"x": INF, "y": 0.0}

    assert _is_valid(MultiPointXY(), [good, good], [good, bad]) == [True, False]


def test_a_multipoint_may_repeat_a_point() -> None:
    """Its points are not a line, so they have no order to be in a row in."""
    point = {"x": 0.0, "y": 0.0}

    assert _is_valid(MultiPointXY(), [point, point]) == [True]


def test_a_multilinestring_with_one_invalid_line_is_invalid() -> None:
    good, bad = _line((0, 0), (1, 1)), _line((0, 0))

    assert _is_valid(MultiLineStringXY(), [good, good], [good, bad]) == [True, False]


def test_a_multipolygon_with_one_invalid_polygon_is_invalid() -> None:
    good, bad = [_SQUARE, _HOLE], [_SQUARE[:-1]]

    assert _is_valid(MultiPolygonXY(), [good, good], [good, bad]) == [True, False]


@pytest.mark.parametrize(
    "dtype", [MultiPointXY(), MultiLineStringXY(), MultiPolygonXY()]
)
def test_an_empty_multi_geometry_is_valid(dtype: GeoArrowType) -> None:
    assert _is_valid(dtype, []) == [True]


@pytest.mark.parametrize(
    "dtype", [MultiPointXY(), MultiLineStringXY(), MultiPolygonXY()]
)
def test_a_missing_part_is_invalid(dtype: GeoArrowType) -> None:
    assert _is_valid(dtype, [None]) == [False]


@pytest.mark.parametrize(
    "dtype", [PointXY(), LineStringXY(), PolygonXY(), MultiPolygonXY()]
)
def test_a_missing_geometry_gives_a_missing_result(dtype: GeoArrowType) -> None:
    assert _is_valid(dtype, None) == [None]


# CRS bounds.


def _point(x: float, y: float) -> Coord:
    return {"x": x, "y": y}


def test_a_latitude_past_the_pole_is_invalid() -> None:
    points = [_point(0, 90), _point(0, 91), _point(0, -91)]

    assert _is_valid(PointXY(crs=WGS84), *points) == [True, False, False]


def test_a_longitude_up_to_a_full_turn_east_is_valid_by_default() -> None:
    points = [_point(-180, 0), _point(270, 0), _point(360, 0)]

    assert _is_valid(PointXY(crs=WGS84), *points) == [True, True, True]


def test_a_longitude_past_a_full_turn_or_half_a_turn_west_is_invalid() -> None:
    points = [_point(361, 0), _point(-181, 0)]

    assert _is_valid(PointXY(crs=WGS84), *points) == [False, False]


def test_a_longitude_past_the_antimeridian_can_be_refused() -> None:
    points = [_point(180, 0), _point(181, 0)]
    out = _is_valid(PointXY(crs=WGS84), *points, allow_wrapped_longitude=False)

    assert out == [True, False]


def test_every_vertex_is_held_to_the_bounds() -> None:
    line = _line((0, 0), (0, 95))

    assert _is_valid(LineStringXY(crs=WGS84), line) == [False]


def test_without_a_crs_there_are_no_bounds() -> None:
    assert _is_valid(PointXY(), _point(1000, 1000), within_area_of_use=True) == [True]


def test_a_projected_crs_is_not_bounded_by_default() -> None:
    assert _is_valid(PointXY(crs=UTM31N), _point(-1e7, 1e8)) == [True]


def test_a_point_outside_the_area_of_use_can_be_refused() -> None:
    """UTM 31N is meant for 0 to 6 degrees east, which lies around easting 500 km."""
    points = [_point(500_000, 5_000_000), _point(-5_000_000, 5_000_000)]
    out = _is_valid(PointXY(crs=UTM31N), *points, within_area_of_use=True)

    assert out == [True, False]


def test_the_area_of_use_narrows_a_geographic_crs() -> None:
    """ETRS89 is meant for Europe only."""
    points = [_point(5, 52), _point(-100, 40)]
    out = _is_valid(PointXY(crs="EPSG:4258"), *points, within_area_of_use=True)

    assert out == [True, False]


def _srid() -> PointXY:
    metadata = '{"crs":"1234","crs_type":"srid"}'
    return PointXY.ext_from_params(PointXY._extension_name, _XY, metadata)


def test_an_opaque_srid_is_not_bounded() -> None:
    assert _is_valid(_srid(), _point(1000, 1000)) == [True]


def test_an_opaque_srid_has_no_area_of_use() -> None:
    with pytest.raises(ComputeError, match="opaque SRID"):
        _is_valid(_srid(), _point(0, 0), within_area_of_use=True)


# The expression itself.


def test_it_runs_on_the_streaming_engine() -> None:
    lines = [_SQUARE, _line((0, 0)), None, []] * 100
    lf = (
        pl.LazyFrame({"g": lines}, schema={"g": pl.List(_XY)})
        .select(pl.col("g").ext.to(LineStringXY()))
        .select(geo.is_valid("g"))
    )

    assert_frame_equal(lf.collect(engine="streaming"), lf.collect())


def test_a_slice_is_checked_on_its_own_rows() -> None:
    rings = [[_SQUARE[:-1]], [_SQUARE], [_SQUARE, None], [_SQUARE, _HOLE]]
    storage = pl.Series("g", rings, dtype=pl.List(pl.List(_XY)))
    df = pl.DataFrame(storage.ext.to(PolygonXY())).slice(1, 2)

    assert df.select(geo.is_valid("g"))["g"].to_list() == [True, False]


def test_namespace_matches_the_functional_api() -> None:
    storage = pl.Series("g", [_point(0, 0), _point(0, 95)], dtype=_XY)
    df = pl.DataFrame(storage.ext.to(PointXY(crs=WGS84)))

    assert_frame_equal(
        df.select(gpl.col("g").geo.is_valid(allow_wrapped_longitude=False)),
        df.select(geo.is_valid("g", allow_wrapped_longitude=False)),
    )


def test_rejects_a_non_geometry_while_building_the_plan() -> None:
    lf = pl.LazyFrame({"g": [1.0]}).select(geo.is_valid("g"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
