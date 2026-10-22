"""Whether a valid geometry is degenerate: repeated points, empty, or rings that do not fit."""

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
    PolygonXY,
)

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]
Line = list[Coord]

NAN = math.nan


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
    coords = _XYZM if isinstance(dtype, LineStringXYZM) else _XY
    for _ in range(dtype._nesting):
        coords = pl.List(coords)
    return coords


def _frame(dtype: GeoArrowType, *rows: object) -> pl.DataFrame:
    """`rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=_storage(dtype))
    return pl.DataFrame(storage.ext.to(dtype))


def _is_degenerate(
    dtype: GeoArrowType, *rows: object, **kwargs: bool
) -> list[bool | None]:
    return _frame(dtype, *rows).select(geo.is_degenerate("g", **kwargs))["g"].to_list()


# Repeated points.


def test_two_vertices_in_a_row_at_the_same_place_are_degenerate() -> None:
    line = _line((0, 0), (1, 1), (1, 1), (2, 0))

    assert _is_degenerate(LineStringXY(), line) == [True]


def test_vertices_in_a_row_that_are_almost_the_same_are_degenerate() -> None:
    """Same place as `pl.Expr.is_close` with its defaults."""
    line = _line((0, 0), (1, 1), (1 + 1e-12, 1), (2, 0))

    assert _is_degenerate(LineStringXY(), line) == [True]


def test_vertices_in_a_row_that_differ_only_in_m_are_degenerate() -> None:
    """m is a measure along the line, not a place on it."""
    a = {"x": 1.0, "y": 1.0, "z": 0.0, "m": 0.0}
    b = a | {"m": 1.0}

    assert _is_degenerate(LineStringXYZM(), [a, b]) == [True]


def test_a_vertex_repeated_further_on_is_not_degenerate() -> None:
    line = _line((0, 0), (1, 1), (0, 0))

    assert _is_degenerate(LineStringXY(), line) == [False]


def test_a_ring_with_vertices_in_a_row_at_the_same_place_is_degenerate() -> None:
    ring = _line((0, 0), (4, 0), (4, 0), (4, 4), (0, 0))

    assert _is_degenerate(PolygonXY(), [ring]) == [True]


def test_a_multipoint_with_a_repeated_point_is_degenerate() -> None:
    """Its points have no order, so any two count, not only two in a row."""
    a, b = {"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}

    assert _is_degenerate(MultiPointXY(), [a, b, a], [a, b]) == [True, False]


@pytest.mark.parametrize(
    ("dtype", "row"),
    [
        (LineStringXY(), _line((0, 0), (1, 1), (1, 1), (2, 0))),
        (PolygonXY(), [_line((0, 0), (4, 0), (4, 0), (4, 4), (0, 4), (0, 0))]),
        (MultiPointXY(), _line((0, 0), (0, 0))),
    ],
    ids=["line", "ring", "multipoint"],
)
def test_without_the_repeated_points_check_a_point_may_repeat(
    dtype: GeoArrowType, row: object
) -> None:
    assert _is_degenerate(dtype, row, check_repeated_points=False) == [False]


# Empty geometries.


@pytest.mark.parametrize(
    ("dtype", "row"),
    [
        (PointXY(), {"x": NAN, "y": NAN}),
        (LineStringXY(), []),
        (LineStringXY(), _line((NAN, NAN), (NAN, NAN))),
        (PolygonXY(), []),
        (PolygonXY(), [[]]),
        (MultiPointXY(), []),
        (MultiLineStringXY(), [[], []]),
        (MultiPolygonXY(), [[[]]]),
    ],
    ids=[
        "point",
        "line",
        "line-of-empty-points",
        "polygon",
        "polygon-of-an-empty-ring",
        "multipoint",
        "multilinestring",
        "multipolygon",
    ],
)
def test_an_empty_geometry_is_degenerate(dtype: GeoArrowType, row: object) -> None:
    assert _is_degenerate(dtype, row) == [True]
    assert _is_degenerate(dtype, row, check_empty=False) == [False]


def test_one_point_that_is_not_empty_makes_it_not_empty() -> None:
    line = _line((NAN, NAN), (0, 0))

    assert _is_degenerate(LineStringXY(), line) == [False]


# Polygons: how the rings fit together.


def test_a_self_crossing_exterior_is_degenerate() -> None:
    assert _is_degenerate(PolygonXY(), [_BOWTIE]) == [True]


def test_a_self_crossing_hole_is_degenerate() -> None:
    bowtie = _line((1, 1), (3, 3), (3, 1), (1, 3), (1, 1))

    assert _is_degenerate(PolygonXY(), [_SQUARE, bowtie]) == [True]


@pytest.mark.parametrize(
    "hole",
    [_line((0, 0), (2, 1), (1, 2), (0, 0)), _line((2, 0), (3, 1), (1, 1), (2, 0))],
    ids=["at-a-corner", "on-an-edge"],
)
def test_a_hole_may_touch_the_exterior_at_a_point(hole: Line) -> None:
    assert _is_degenerate(PolygonXY(), [_SQUARE, hole]) == [False]


def test_a_hole_touching_the_exterior_along_a_line_is_degenerate() -> None:
    hole = _line((0, 0), (2, 0), (1, 1), (0, 0))

    assert _is_degenerate(PolygonXY(), [_SQUARE, hole]) == [True]


@pytest.mark.parametrize(
    "hole",
    [_square(3, 3, 5, 5), _square(5, 5, 6, 6), _SQUARE],
    ids=["partly-outside", "outside", "the-exterior"],
)
def test_a_hole_not_inside_the_exterior_is_degenerate(hole: Line) -> None:
    assert _is_degenerate(PolygonXY(), [_SQUARE, hole]) == [True]


def test_holes_may_touch_each_other_at_a_point() -> None:
    holes = [_square(1, 1, 2, 2), _square(2, 2, 3, 3)]

    assert _is_degenerate(PolygonXY(), [_SQUARE, *holes]) == [False]


def test_holes_touching_each_other_along_a_line_are_degenerate() -> None:
    holes = [_square(1, 1, 2, 2), _square(2, 1, 3, 2)]

    assert _is_degenerate(PolygonXY(), [_SQUARE, *holes]) == [True]


@pytest.mark.parametrize(
    "holes",
    [
        [_square(1, 1, 2, 2), _square(1.5, 1.5, 3, 3)],
        [_square(1, 1, 3, 3), _square(1.5, 1.5, 2, 2)],
        [_square(1, 1, 2, 2), _square(1, 1, 2, 2)],
    ],
    ids=["overlapping", "nested", "equal"],
)
def test_holes_sharing_area_are_degenerate(holes: list[Line]) -> None:
    assert _is_degenerate(PolygonXY(), [_SQUARE, *holes]) == [True]


def test_a_ring_without_finite_xy_cannot_be_checked_for_how_it_fits() -> None:
    """The empty point counts as not finite here."""
    ring = [*_SQUARE[:2], {"x": NAN, "y": NAN}, *_SQUARE[3:]]

    assert _is_degenerate(PolygonXY(), [ring]) == [True]
    assert _is_degenerate(PolygonXY(), [ring], **_NO_TOPOLOGY_CHECKS) == [False]


def test_without_the_ring_check_a_ring_may_cross_itself() -> None:
    out = _is_degenerate(
        PolygonXY(),
        [_BOWTIE],
        [_SQUARE, _square(5, 5, 6, 6)],
        check_ring_self_intersect=False,
    )

    assert out == [False, True]


def test_without_the_overlap_check_the_holes_may_lie_anywhere() -> None:
    out = _is_degenerate(
        PolygonXY(),
        [_SQUARE, _square(1, 1, 2, 2), _square(1.5, 1.5, 3, 3)],
        [_SQUARE, _square(5, 5, 6, 6)],
        [_BOWTIE],
        check_polygon_self_overlap=False,
    )

    assert out == [False, False, True]


# Multipolygons: how the parts fit together.


def test_a_multipolygon_checks_how_the_rings_of_each_part_fit() -> None:
    good, other = [_SQUARE, _HOLE], [_square(5, 5, 9, 9)]
    bad = [_square(5, 5, 9, 9), _square(6, 6, 8, 8), _square(6.5, 6.5, 7, 7)]

    assert _is_degenerate(MultiPolygonXY(), [good, other], [good, bad]) == [False, True]


def test_multipolygon_parts_apart_are_not_degenerate() -> None:
    parts = [[_square(0, 0, 1, 1)], [_square(2, 2, 3, 3)]]

    assert _is_degenerate(MultiPolygonXY(), parts) == [False]


def test_multipolygon_parts_may_touch_at_a_point() -> None:
    parts = [[_square(0, 0, 1, 1)], [_square(1, 1, 2, 2)]]

    assert _is_degenerate(MultiPolygonXY(), parts) == [False]


def test_multipolygon_parts_touching_along_a_line_are_degenerate() -> None:
    parts = [[_square(0, 0, 1, 1)], [_square(1, 0, 2, 1)]]

    assert _is_degenerate(MultiPolygonXY(), parts) == [True]


@pytest.mark.parametrize(
    "other",
    [_square(2, 2, 6, 6), _square(1, 1, 2, 2), _SQUARE],
    ids=["overlapping", "nested", "equal"],
)
def test_multipolygon_parts_sharing_area_are_degenerate(other: Line) -> None:
    assert _is_degenerate(MultiPolygonXY(), [[_SQUARE], [other]]) == [True]


def test_a_multipolygon_part_may_lie_in_a_hole_of_another() -> None:
    parts = [[_SQUARE, _square(1, 1, 3, 3)], [_square(1.5, 1.5, 2.5, 2.5)]]

    assert _is_degenerate(MultiPolygonXY(), parts) == [False]


def test_a_multipolygon_part_filling_a_hole_of_another_is_degenerate() -> None:
    """It meets the hole all along its edge."""
    parts = [[_SQUARE, _square(1, 1, 3, 3)], [_square(1, 1, 3, 3)]]

    assert _is_degenerate(MultiPolygonXY(), parts) == [True]


def test_an_empty_multipolygon_part_fits_with_any_other() -> None:
    assert _is_degenerate(MultiPolygonXY(), [[], [_SQUARE]]) == [False]


def test_without_the_multipolygon_overlap_check_parts_may_overlap() -> None:
    out = _is_degenerate(
        MultiPolygonXY(),
        [[_SQUARE], [_square(2, 2, 6, 6)]],
        [[_BOWTIE]],
        check_multi_polygon_self_overlap=False,
    )

    assert out == [False, True]


# The expressions themselves.


@pytest.mark.parametrize(
    "dtype", [PointXY(), LineStringXY(), PolygonXY(), MultiPolygonXY()]
)
def test_a_missing_geometry_gives_a_missing_result(dtype: GeoArrowType) -> None:
    assert _is_degenerate(dtype, None) == [None]


def test_null_out_degenerate_keeps_the_rest() -> None:
    df = _frame(PolygonXY(), [_SQUARE], [_BOWTIE], None, [])

    out = df.select(geo.null_out_degenerate("g"))["g"]

    assert out.ext.storage().to_list() == [[_SQUARE], None, None, None]


def test_null_out_degenerate_copies_nothing_without_degenerate_rows() -> None:
    df = _frame(PolygonXY(), [_SQUARE], None)

    assert_frame_equal(df.select(geo.null_out_degenerate("g")), df)


def test_namespace_matches_the_functional_api() -> None:
    df = _frame(PolygonXY(), [_SQUARE], [_BOWTIE])

    assert_frame_equal(
        df.select(gpl.col("g").geo.is_degenerate(**_NO_TOPOLOGY_CHECKS)),
        df.select(geo.is_degenerate("g", **_NO_TOPOLOGY_CHECKS)),
    )
    assert_frame_equal(
        df.select(gpl.col("g").geo.null_out_degenerate()),
        df.select(geo.null_out_degenerate("g")),
    )
