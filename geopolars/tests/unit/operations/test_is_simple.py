"""Whether a geometry is simple: it does not cross or touch itself, in x and y."""

from __future__ import annotations

import random

import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZ,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)
from tests.unit.conftest import Dimension

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})

Line = list[dict[str, float]]

NAN = float("nan")
INF = float("inf")


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _square(x0: float, y0: float, x1: float, y1: float) -> Line:
    return _line((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))


_SQUARE = _square(0, 0, 1, 1)
_BOWTIE = _line((0, 0), (1, 1), (1, 0), (0, 1))
_CLOSED_BOWTIE = [*_BOWTIE, _BOWTIE[0]]
_OUTER = _square(0, 0, 10, 10)


def _storage(dtype: GeoArrowType) -> pl.DataType:
    coords: pl.DataType = _XY
    for _ in range(dtype._nesting):
        coords = pl.List(coords)
    return coords


def _frame(dtype: GeoArrowType, *rows: object) -> pl.DataFrame:
    """`rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=_storage(dtype))
    return pl.DataFrame(storage.ext.to(dtype))


def _is_simple(dtype: GeoArrowType, *rows: object) -> list[bool | None]:
    out = _frame(dtype, *rows).select(geo.is_simple("g"))
    assert out.schema["g"] == pl.Boolean
    return out["g"].to_list()


def _lines_are_simple(*lines: Line | None) -> list[bool | None]:
    return _is_simple(LineStringXY(), *lines)


# Points.


def test_a_point_is_simple() -> None:
    points = [{"x": 0.0, "y": 0.0}, {"x": NAN, "y": NAN}, None]

    assert _is_simple(PointXY(), *points) == [True, True, None]


def test_a_multipoint_is_simple_without_two_equal_points() -> None:
    a, b = {"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}

    assert _is_simple(MultiPointXY(), [a, b], [a, b, a], []) == [True, False, True]


def test_empty_points_of_a_multipoint_are_not_equal() -> None:
    empty = {"x": NAN, "y": NAN}

    assert _is_simple(MultiPointXY(), [empty, empty]) == [True]


# Linestrings.


def test_a_simple_open_line_is_simple() -> None:
    assert _lines_are_simple(_line((0, 0), (1, 0), (1, 1), (2, 1))) == [True]


def test_a_simple_ring_is_simple() -> None:
    triangle = _line((0, 0), (1, 0), (0, 1), (0, 0))
    assert _lines_are_simple(triangle, _SQUARE) == [True, True]


def test_a_crossing_is_not_simple() -> None:
    assert _lines_are_simple(_BOWTIE, _CLOSED_BOWTIE) == [False, False]


def test_an_end_on_its_own_interior_is_not_simple() -> None:
    """A "6": the last vertex lies on the first segment."""
    six = _line((0, 0), (2, 0), (2, 2), (0, 2), (1, 0))
    assert _lines_are_simple(six) == [False]


def test_coming_back_to_an_earlier_vertex_is_not_simple() -> None:
    revisit = _line((0, 0), (1, 0), (1, 1), (0, 1), (1, 0))
    assert _lines_are_simple(revisit) == [False]


def test_a_ring_that_passes_its_start_again_is_not_simple() -> None:
    """Closing only lets the first and last segment meet, not any other two."""
    eight = _line((0, 0), (1, 0), (1, 1), (0, 0), (-1, 0), (-1, -1), (0, 0))
    assert _lines_are_simple(eight) == [False]


def test_doubling_back_is_not_simple() -> None:
    """Segments in a row may share their vertex, but not overlap."""
    spike = _line((0, 0), (2, 0), (1, 0))
    out_and_back = _line((0, 0), (1, 0), (0, 0))
    assert _lines_are_simple(spike, out_and_back) == [False, False]


def test_a_collinear_overlap_is_not_simple() -> None:
    overlap = _line((0, 0), (3, 0), (3, 1), (1, 1), (1, 0), (2, 0))
    assert _lines_are_simple(overlap) == [False]


def test_repeated_vertices_do_not_count() -> None:
    open_line = _line((0, 0), (1, 0), (1, 0), (1, 1))
    ring = _line((0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 0))
    assert _lines_are_simple(open_line, ring) == [True, True]


def test_the_ends_have_to_be_exactly_equal() -> None:
    """Here the last vertex lies on the first segment, which only a closed line allows."""
    exact = _line((1e6, 0), (2e6, 0), (2e6, 1), (1e6, 0))
    almost = _line((1e6, 0), (2e6, 0), (2e6, 1), (1e6 + 1e-4, 0))
    assert _lines_are_simple(exact, almost) == [True, False]


def test_an_empty_line_and_a_single_segment_are_simple() -> None:
    segment = _line((0, 0), (1, 1))
    point_twice = _line((1, 1), (1, 1))
    assert _lines_are_simple([], segment, point_twice) == [True, True, True]


def test_a_single_vertex_is_not_simple() -> None:
    assert _lines_are_simple(_line((0, 0))) == [False]


def test_missing_stays_missing() -> None:
    assert _lines_are_simple(None, _SQUARE) == [None, True]


@pytest.mark.parametrize("value", [NAN, INF])
def test_a_not_finite_vertex_is_not_simple(value: float) -> None:
    assert _lines_are_simple(_line((0, 0), (value, 1), (1, 1))) == [False]


def test_z_does_not_count() -> None:
    """Crossing in x and y counts at any height, and so does closing."""
    crossing = [
        {"x": x, "y": y, "z": z}
        for x, y, z in [(0, 0, 0), (1, 1, 0), (1, 0, 5), (0, 1, 5)]
    ]
    ring = [
        {"x": x, "y": y, "z": z}
        for x, y, z in [(0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 1, 3), (0, 0, 4)]
    ]
    storage = pl.Series("line", [crossing, ring]).cast(
        pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}))
    )
    df = pl.DataFrame(storage.ext.to(LineStringXYZ()))

    assert df.select(geo.is_simple("line"))["line"].to_list() == [False, True]


def test_every_dimension(dimension: Dimension) -> None:
    def line(corners: list[tuple[float, float]]) -> Line:
        return [
            {
                axis: {"x": float(x), "y": float(y)}.get(axis, 7.0)
                for axis in dimension.coords
            }
            for x, y in corners
        ]

    square = line([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    bowtie = line([(0, 0), (1, 1), (1, 0), (0, 1)])
    storage = pl.Series("line", [square, bowtie])
    df = pl.DataFrame(storage.ext.to(dimension.linestring_dtype()))

    assert df.select(geo.is_simple("line"))["line"].to_list() == [True, False]


def test_agrees_with_shapely() -> None:
    """Small integer grids give many shared vertices and collinear segments."""
    rng = random.Random(0)
    lines = [
        _line(*((rng.randint(0, 4), rng.randint(0, 4)) for _ in range(n)))
        for n in [rng.randint(2, 9) for _ in range(500)]
    ]
    expected = [
        shapely.LineString([(v["x"], v["y"]) for v in line]).is_simple for line in lines
    ]

    assert _lines_are_simple(*lines) == expected


def test_a_sliced_linestring_column() -> None:
    df = _frame(LineStringXY(), _BOWTIE, _SQUARE).slice(1, 1)

    assert df.select(geo.is_simple("g"))["g"].to_list() == [True]


# Multilinestrings: every part simple, meeting only where both have an end.


def test_a_multilinestring_is_simple_when_every_part_is() -> None:
    rows = [[_SQUARE], [_line((5, 5), (6, 6)), _BOWTIE], [], None]

    assert _is_simple(MultiLineStringXY(), *rows) == [True, False, True, None]


def test_parts_crossing_each_other_are_not_simple() -> None:
    cross = [_line((0, 0), (2, 2)), _line((0, 2), (2, 0))]

    assert _is_simple(MultiLineStringXY(), cross) == [False]


def test_parts_may_meet_at_their_ends() -> None:
    chained = [_line((0, 0), (1, 0)), _line((1, 0), (1, 1))]
    star = [_line((0, 0), (1, 0)), _line((0, 0), (0, 1)), _line((0, 0), (-1, 0))]

    assert _is_simple(MultiLineStringXY(), chained, star) == [True, True]


def test_an_end_on_the_interior_of_another_part_is_not_simple() -> None:
    tee = [_line((0, 0), (2, 0)), _line((1, 0), (1, 1))]

    assert _is_simple(MultiLineStringXY(), tee) == [False]


def test_nothing_may_touch_a_closed_part() -> None:
    """A closed part has no ends, so even its first vertex is inside it."""
    touching_ring = [_SQUARE, _square(1, 1, 2, 2)]
    line_at_its_start = [_SQUARE, _line((0, 0), (-1, -1))]

    assert _is_simple(MultiLineStringXY(), touching_ring, line_at_its_start) == [
        False,
        False,
    ]


def test_equal_parts_are_not_simple() -> None:
    assert _is_simple(MultiLineStringXY(), [_SQUARE, _SQUARE]) == [False]


def test_a_single_vertex_part_is_not_simple() -> None:
    parts = [_line((5, 5), (6, 6)), _line((0, 0))]

    assert _is_simple(MultiLineStringXY(), parts) == [False]


def test_a_sliced_multilinestring_reads_only_its_own_parts() -> None:
    df = _frame(MultiLineStringXY(), [_BOWTIE], [_SQUARE], [_BOWTIE]).slice(1, 1)

    assert df.select(geo.is_simple("g"))["g"].to_list() == [True]


# Polygons: every ring simple.


def test_a_simple_polygon_is_simple() -> None:
    holes = [_OUTER, _square(1, 1, 2, 2), _square(3, 3, 4, 4)]
    assert _is_simple(PolygonXY(), [_SQUARE], holes) == [True, True]


def test_a_polygon_with_a_crossing_ring_is_not_simple() -> None:
    assert _is_simple(PolygonXY(), [_CLOSED_BOWTIE], [_OUTER, _CLOSED_BOWTIE]) == [
        False,
        False,
    ]


def test_how_the_rings_fit_together_does_not_count() -> None:
    """That is up to `is_valid_ogc`."""
    overlapping = [_OUTER, _square(1, 1, 3, 3), _square(2, 2, 4, 4)]
    outside = [_SQUARE, _square(5, 5, 6, 6)]

    assert _is_simple(PolygonXY(), overlapping, outside) == [True, True]


@pytest.mark.parametrize(
    "ring",
    [_line((0, 0)), _line((0, 0), (0, 0)), _line((0, 0), (1, 0), (0, 0))],
    ids=["one", "two", "three"],
)
def test_a_ring_of_fewer_than_four_vertices_is_not_simple(ring: Line) -> None:
    assert _is_simple(PolygonXY(), [ring]) == [False]


def test_an_empty_ring_is_simple() -> None:
    assert _is_simple(PolygonXY(), [], [[]], [[], _SQUARE]) == [True, True, True]


def test_a_polygon_not_finite_in_x_or_y_is_not_simple() -> None:
    ring = _line((0, 0), (1, 0), (NAN, 1), (0, 0))

    assert _is_simple(PolygonXY(), None, [ring]) == [None, False]


def test_a_multipolygon_is_simple_when_every_part_is() -> None:
    """Parts overlapping each other do not count."""
    overlapping = [_OUTER, _square(1, 1, 3, 3), _square(2, 2, 4, 4)]
    rows = [[[_OUTER], overlapping], [], [[_SQUARE], [_CLOSED_BOWTIE]], None]

    assert _is_simple(MultiPolygonXY(), *rows) == [True, True, False, None]


def test_a_sliced_multipolygon_reads_only_its_own_parts() -> None:
    df = _frame(MultiPolygonXY(), [[_CLOSED_BOWTIE]], [[_SQUARE]], [[_CLOSED_BOWTIE]])

    assert df.slice(1, 1).select(geo.is_simple("g"))["g"].to_list() == [True]


# The expression itself.


def test_namespace_matches_the_functional_api() -> None:
    df = _frame(LineStringXY(), _BOWTIE, _SQUARE)

    assert_frame_equal(
        df.select(gpl.col("g").geo.is_simple()), df.select(geo.is_simple("g"))
    )


def test_rejects_a_non_geometry_while_building_the_plan() -> None:
    lf = pl.LazyFrame({"g": [1.0]}).select(geo.is_simple("g"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
