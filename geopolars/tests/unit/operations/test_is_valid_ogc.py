"""Whether a geometry is valid by OGC Simple Features: GeoArrow-valid, finite, shaped right, and fitting together."""

from __future__ import annotations

import math

import polars as pl
import pytest
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

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]
Line = list[Coord]

NAN = math.nan
INF = math.inf


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _square(x0: float, y0: float, x1: float, y1: float) -> Line:
    return _line((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))


_SQUARE = _square(0, 0, 4, 4)
_HOLE = _line((1, 1), (2, 1), (2, 2), (1, 1))
_BOWTIE = _line((0, 0), (4, 4), (4, 0), (0, 4), (0, 0))

_NO_TOPOLOGY_CHECKS = {
    "check_ring_self_intersect": False,
    "check_polygon_self_overlap": False,
    "check_multi_polygon_self_overlap": False,
}


def _storage(dtype: GeoArrowType) -> pl.DataType:
    """The storage of `dtype`, with every coordinate as `_XY` or `_XYZM`."""
    coords = _XYZM if isinstance(dtype, (PointXYZM, LineStringXYZM)) else _XY
    for _ in range(dtype._nesting):
        coords = pl.List(coords)
    return coords


def _frame(dtype: GeoArrowType, *rows: object) -> pl.DataFrame:
    """`rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=_storage(dtype))
    return pl.DataFrame(storage.ext.to(dtype))


def _is_valid_ogc(
    dtype: GeoArrowType, *rows: object, **kwargs: object
) -> list[bool | None]:
    return _frame(dtype, *rows).select(geo.is_valid_ogc("g", **kwargs))["g"].to_list()


# GeoArrow's rules still hold.


def test_a_null_vertex_is_invalid() -> None:
    assert _is_valid_ogc(LineStringXY(), [{"x": 0.0, "y": 0.0}, None]) == [False]


def test_an_open_ring_is_invalid() -> None:
    assert _is_valid_ogc(PolygonXY(), [_SQUARE[:-1]], **_NO_TOPOLOGY_CHECKS) == [False]


def test_the_crs_bounds_can_be_checked() -> None:
    point = {"x": 0.0, "y": 91.0}

    assert _is_valid_ogc(PointXY(crs=WGS84), point) == [True]
    out_of_bounds = geo.OutOfBounds()
    assert _is_valid_ogc(PointXY(crs=WGS84), point, out_of_bounds=out_of_bounds) == [
        False
    ]


# Coordinates have to be finite.


@pytest.mark.parametrize("axis", ["x", "y", "z", "m"])
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_axis_is_invalid(axis: str, value: float) -> None:
    point = {"x": 1.0, "y": 2.0, "z": 3.0, "m": 4.0} | {axis: value}

    assert _is_valid_ogc(PointXYZM(), point) == [False]


def test_the_empty_point_is_valid() -> None:
    assert _is_valid_ogc(PointXY(), {"x": NAN, "y": NAN}) == [True]


def test_a_multipoint_may_hold_the_empty_point() -> None:
    points = [{"x": 0.0, "y": 0.0}, {"x": NAN, "y": NAN}]

    assert _is_valid_ogc(MultiPointXY(), points) == [True]


@pytest.mark.parametrize(
    ("dtype", "row"),
    [
        (LineStringXY(), _line((0, 0), (1, 1), (NAN, NAN))),
        (PolygonXY(), [[*_SQUARE[:2], {"x": NAN, "y": NAN}, *_SQUARE[3:]]]),
    ],
    ids=["line", "ring"],
)
def test_the_empty_point_may_not_be_a_vertex(dtype: GeoArrowType, row: object) -> None:
    assert _is_valid_ogc(dtype, row, **_NO_TOPOLOGY_CHECKS) == [False]


# Linestrings: none, or at least 2 distinct vertices.


@pytest.mark.parametrize(
    "line",
    [[], _line((0, 0), (1, 1)), _line((0, 0), (0, 0), (1, 1))],
    ids=["empty", "two", "two-distinct-of-three"],
)
def test_a_line_of_none_or_two_distinct_vertices_is_valid(line: Line) -> None:
    assert _is_valid_ogc(LineStringXY(), line) == [True]


@pytest.mark.parametrize(
    "line", [_line((0, 0)), _line((1, 1), (1, 1))], ids=["one", "the-same-twice"]
)
def test_a_line_of_one_distinct_vertex_is_invalid(line: Line) -> None:
    assert _is_valid_ogc(LineStringXY(), line) == [False]


def test_a_line_may_cross_itself() -> None:
    assert _is_valid_ogc(LineStringXY(), _BOWTIE[:-1]) == [True]


def test_a_multilinestring_with_one_invalid_line_is_invalid() -> None:
    good, bad = _line((0, 0), (1, 1)), _line((0, 0))

    assert _is_valid_ogc(MultiLineStringXY(), [good, good], [good, bad]) == [
        True,
        False,
    ]


# Polygons: all rings empty, or each with at least 4 distinct vertices.


def test_a_polygon_with_a_hole_is_valid() -> None:
    assert _is_valid_ogc(PolygonXY(), [_SQUARE, _HOLE]) == [True]


@pytest.mark.parametrize(
    "ring",
    [_line((0, 0), (1, 1), (0, 0)), _line((0, 0), (1, 0), (1, 0), (0, 0))],
    ids=["three", "three-distinct-of-four"],
)
def test_a_ring_of_three_distinct_vertices_is_invalid(ring: Line) -> None:
    assert _is_valid_ogc(PolygonXY(), [ring], **_NO_TOPOLOGY_CHECKS) == [False]


def test_a_ring_may_repeat_a_vertex_in_a_row() -> None:
    ring = _line((0, 0), (4, 0), (4, 0), (4, 4), (0, 0))

    assert _is_valid_ogc(PolygonXY(), [ring]) == [True]


@pytest.mark.parametrize("rings", [[], [[]], [[], []]])
def test_an_empty_polygon_is_valid(rings: list[Line]) -> None:
    assert _is_valid_ogc(PolygonXY(), rings) == [True]


def test_an_empty_exterior_around_a_hole_is_invalid() -> None:
    assert _is_valid_ogc(PolygonXY(), [[], _HOLE]) == [False]


def test_an_empty_hole_is_invalid() -> None:
    assert _is_valid_ogc(PolygonXY(), [_SQUARE, []]) == [False]


# Polygons: how the rings fit together.


def test_a_self_crossing_exterior_is_invalid() -> None:
    assert _is_valid_ogc(PolygonXY(), [_BOWTIE]) == [False]


def test_a_self_crossing_hole_is_invalid() -> None:
    bowtie = _line((1, 1), (3, 3), (3, 1), (1, 3), (1, 1))

    assert _is_valid_ogc(PolygonXY(), [_SQUARE, bowtie]) == [False]


@pytest.mark.parametrize(
    "hole",
    [_line((0, 0), (2, 1), (1, 2), (0, 0)), _line((2, 0), (3, 1), (1, 1), (2, 0))],
    ids=["at-a-corner", "on-an-edge"],
)
def test_a_hole_may_touch_the_exterior_at_a_point(hole: Line) -> None:
    assert _is_valid_ogc(PolygonXY(), [_SQUARE, hole]) == [True]


def test_a_hole_touching_the_exterior_along_a_line_is_invalid() -> None:
    hole = _line((0, 0), (2, 0), (1, 1), (0, 0))

    assert _is_valid_ogc(PolygonXY(), [_SQUARE, hole]) == [False]


@pytest.mark.parametrize(
    "hole",
    [_square(3, 3, 5, 5), _square(5, 5, 6, 6), _SQUARE],
    ids=["partly-outside", "outside", "the-exterior"],
)
def test_a_hole_not_inside_the_exterior_is_invalid(hole: Line) -> None:
    assert _is_valid_ogc(PolygonXY(), [_SQUARE, hole]) == [False]


def test_holes_may_touch_each_other_at_a_point() -> None:
    holes = [_square(1, 1, 2, 2), _square(2, 2, 3, 3)]

    assert _is_valid_ogc(PolygonXY(), [_SQUARE, *holes]) == [True]


def test_holes_touching_each_other_along_a_line_are_invalid() -> None:
    holes = [_square(1, 1, 2, 2), _square(2, 1, 3, 2)]

    assert _is_valid_ogc(PolygonXY(), [_SQUARE, *holes]) == [False]


@pytest.mark.parametrize(
    "holes",
    [
        [_square(1, 1, 2, 2), _square(1.5, 1.5, 3, 3)],
        [_square(1, 1, 3, 3), _square(1.5, 1.5, 2, 2)],
        [_square(1, 1, 2, 2), _square(1, 1, 2, 2)],
    ],
    ids=["overlapping", "nested", "equal"],
)
def test_holes_sharing_area_are_invalid(holes: list[Line]) -> None:
    assert _is_valid_ogc(PolygonXY(), [_SQUARE, *holes]) == [False]


def test_without_the_ring_check_a_ring_may_cross_itself() -> None:
    out = _is_valid_ogc(
        PolygonXY(),
        [_BOWTIE],
        [_SQUARE, _square(5, 5, 6, 6)],
        check_ring_self_intersect=False,
    )

    assert out == [True, False]


def test_without_the_overlap_check_the_holes_may_lie_anywhere() -> None:
    out = _is_valid_ogc(
        PolygonXY(),
        [_SQUARE, _square(1, 1, 2, 2), _square(1.5, 1.5, 3, 3)],
        [_SQUARE, _square(5, 5, 6, 6)],
        [_BOWTIE],
        check_polygon_self_overlap=False,
    )

    assert out == [True, True, False]


# Multipolygons: every part valid, and the parts fit together.


def test_a_multipolygon_with_one_invalid_polygon_is_invalid() -> None:
    good, other = [_SQUARE, _HOLE], [_square(5, 5, 9, 9)]
    bad = [_square(5, 5, 9, 9)[:-1]]

    assert _is_valid_ogc(MultiPolygonXY(), [good, other], [good, bad]) == [True, False]


def test_a_multipolygon_checks_how_the_rings_of_each_part_fit() -> None:
    good, other = [_SQUARE, _HOLE], [_square(5, 5, 9, 9)]
    bad = [_square(5, 5, 9, 9), _square(6, 6, 8, 8), _square(6.5, 6.5, 7, 7)]

    assert _is_valid_ogc(MultiPolygonXY(), [good, other], [good, bad]) == [True, False]


def test_multipolygon_parts_apart_are_valid() -> None:
    parts = [[_square(0, 0, 1, 1)], [_square(2, 2, 3, 3)]]

    assert _is_valid_ogc(MultiPolygonXY(), parts) == [True]


def test_multipolygon_parts_may_touch_at_a_point() -> None:
    parts = [[_square(0, 0, 1, 1)], [_square(1, 1, 2, 2)]]

    assert _is_valid_ogc(MultiPolygonXY(), parts) == [True]


def test_multipolygon_parts_touching_along_a_line_are_invalid() -> None:
    parts = [[_square(0, 0, 1, 1)], [_square(1, 0, 2, 1)]]

    assert _is_valid_ogc(MultiPolygonXY(), parts) == [False]


@pytest.mark.parametrize(
    "other",
    [_square(2, 2, 6, 6), _square(1, 1, 2, 2), _SQUARE],
    ids=["overlapping", "nested", "equal"],
)
def test_multipolygon_parts_sharing_area_are_invalid(other: Line) -> None:
    assert _is_valid_ogc(MultiPolygonXY(), [[_SQUARE], [other]]) == [False]


def test_a_multipolygon_part_may_lie_in_a_hole_of_another() -> None:
    parts = [[_SQUARE, _square(1, 1, 3, 3)], [_square(1.5, 1.5, 2.5, 2.5)]]

    assert _is_valid_ogc(MultiPolygonXY(), parts) == [True]


def test_a_multipolygon_part_filling_a_hole_of_another_is_invalid() -> None:
    """It meets the hole all along its edge."""
    parts = [[_SQUARE, _square(1, 1, 3, 3)], [_square(1, 1, 3, 3)]]

    assert _is_valid_ogc(MultiPolygonXY(), parts) == [False]


def test_an_empty_multipolygon_part_fits_with_any_other() -> None:
    assert _is_valid_ogc(MultiPolygonXY(), [[], [_SQUARE]]) == [True]


def test_without_the_multipolygon_overlap_check_parts_may_overlap() -> None:
    out = _is_valid_ogc(
        MultiPolygonXY(),
        [[_SQUARE], [_square(2, 2, 6, 6)]],
        [[_BOWTIE]],
        check_multi_polygon_self_overlap=False,
    )

    assert out == [True, False]


@pytest.mark.parametrize(
    "dtype", [MultiPointXY(), MultiLineStringXY(), MultiPolygonXY()]
)
def test_an_empty_multi_geometry_is_valid(dtype: GeoArrowType) -> None:
    assert _is_valid_ogc(dtype, []) == [True]


# The expressions themselves.


@pytest.mark.parametrize(
    "dtype", [PointXY(), LineStringXY(), PolygonXY(), MultiPolygonXY()]
)
def test_a_null_geometry_gives_a_null_result(dtype: GeoArrowType) -> None:
    assert _is_valid_ogc(dtype, None) == [None]


def test_null_out_invalid_keeps_the_rest() -> None:
    df = _frame(PolygonXY(), [_SQUARE], [_BOWTIE], None, [])

    out = df.select(geo.null_out_invalid_ogc("g"))["g"]

    assert out.ext.storage().to_list() == [[_SQUARE], None, None, []]


def test_null_out_invalid_copies_nothing_without_invalid_rows() -> None:
    df = _frame(PolygonXY(), [_SQUARE], None)

    assert_frame_equal(df.select(geo.null_out_invalid_ogc("g")), df)


def test_a_slice_is_checked_on_its_own_rows() -> None:
    df = _frame(PolygonXY(), [_BOWTIE], [_SQUARE], [_SQUARE, None], [_SQUARE]).slice(
        1, 2
    )

    assert df.select(geo.is_valid_ogc("g"))["g"].to_list() == [True, False]


def test_namespace_matches_the_functional_api() -> None:
    df = _frame(PolygonXY(), [_SQUARE], [_BOWTIE])

    assert_frame_equal(
        df.select(gpl.col("g").geo.is_valid_ogc(**_NO_TOPOLOGY_CHECKS)),
        df.select(geo.is_valid_ogc("g", **_NO_TOPOLOGY_CHECKS)),
    )
    assert_frame_equal(
        df.select(gpl.col("g").geo.null_out_invalid_ogc()),
        df.select(geo.null_out_invalid_ogc("g")),
    )


def test_rejects_a_non_geometry_while_building_the_plan() -> None:
    lf = pl.LazyFrame({"g": [1.0]}).select(geo.is_valid_ogc("g"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
