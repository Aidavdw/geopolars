"""Whether a linestring or polygon crosses or touches itself, in x and y."""

from __future__ import annotations

import random

import polars as pl
import pytest
import shapely
from polars.exceptions import ComputeError

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    LineStringXYZ,
    MultiLineStringXY,
    MultiPolygonXY,
    PolygonXY,
)
from tests.unit.conftest import XY, Dimension

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

Line = list[dict[str, float]]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


_SQUARE = _line((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
_BOWTIE = _line((0, 0), (1, 1), (1, 0), (0, 1))


def _self_intersects(*lines: Line | None) -> list[bool | None]:
    storage = pl.Series("line", list(lines), dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))
    return df.select(geo.self_intersects("line"))["line"].to_list()


def test_a_simple_open_line_does_not() -> None:
    assert _self_intersects(_line((0, 0), (1, 0), (1, 1), (2, 1))) == [False]


def test_a_simple_ring_does_not() -> None:
    triangle = _line((0, 0), (1, 0), (0, 1), (0, 0))
    assert _self_intersects(triangle, _SQUARE) == [False, False]


def test_a_crossing_does() -> None:
    closed_bowtie = [*_BOWTIE, _BOWTIE[0]]
    assert _self_intersects(_BOWTIE, closed_bowtie) == [True, True]


def test_an_end_on_its_own_interior_does() -> None:
    """A "6": the last vertex lies on the first segment."""
    six = _line((0, 0), (2, 0), (2, 2), (0, 2), (1, 0))
    assert _self_intersects(six) == [True]


def test_coming_back_to_an_earlier_vertex_does() -> None:
    revisit = _line((0, 0), (1, 0), (1, 1), (0, 1), (1, 0))
    assert _self_intersects(revisit) == [True]


def test_a_ring_that_passes_its_start_again_does() -> None:
    """Closing only lets the first and last segment meet, not any other two."""
    eight = _line((0, 0), (1, 0), (1, 1), (0, 0), (-1, 0), (-1, -1), (0, 0))
    assert _self_intersects(eight) == [True]


def test_doubling_back_does() -> None:
    """Segments in a row may share their vertex, but not overlap."""
    spike = _line((0, 0), (2, 0), (1, 0))
    out_and_back = _line((0, 0), (1, 0), (0, 0))
    assert _self_intersects(spike, out_and_back) == [True, True]


def test_a_collinear_overlap_does() -> None:
    overlap = _line((0, 0), (3, 0), (3, 1), (1, 1), (1, 0), (2, 0))
    assert _self_intersects(overlap) == [True]


def test_repeated_vertices_do_not_count() -> None:
    open_line = _line((0, 0), (1, 0), (1, 0), (1, 1))
    ring = _line((0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 0))
    assert _self_intersects(open_line, ring) == [False, False]


def test_the_ends_have_to_be_close_not_equal() -> None:
    """As `pl.Expr.is_close`: within a relative 1e-9.
    Here the last vertex lies on the first segment, which only a closed line allows.
    """
    almost = _line((1e6, 0), (2e6, 0), (2e6, 1), (1e6 + 1e-4, 0))
    off = _line((1e6, 0), (2e6, 0), (2e6, 1), (1e6 + 1e-2, 0))
    assert _self_intersects(almost, off) == [False, True]


def test_too_short_to_intersect() -> None:
    single = _line((0, 0))
    segment = _line((0, 0), (1, 1))
    point_twice = _line((1, 1), (1, 1))
    assert _self_intersects([], single, segment, point_twice) == [False] * 4


def test_missing_stays_missing() -> None:
    assert _self_intersects(None, _SQUARE) == [None, False]


def test_a_missing_or_not_finite_vertex_gives_missing() -> None:
    nan, inf = float("nan"), float("inf")
    storage = pl.Series(
        "line",
        [
            [{"x": 0.0, "y": 0.0}, None, {"x": 1.0, "y": 1.0}],
            [{"x": 0.0, "y": None}, {"x": 1.0, "y": 1.0}],
            _line((0, 0), (nan, 1), (1, 1)),
            _line((0, 0), (inf, 1), (1, 1)),
        ],
        dtype=_XY_VERTICES,
    )
    df = pl.DataFrame(storage.ext.to(LineStringXY()))

    assert df.select(geo.self_intersects("line"))["line"].to_list() == [None] * 4


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

    assert df.select(geo.self_intersects("line"))["line"].to_list() == [True, False]


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

    assert df.select(geo.self_intersects("line"))["line"].to_list() == [False, True]


def test_agrees_with_shapely() -> None:
    """Small integer grids give many shared vertices and collinear segments."""
    rng = random.Random(0)
    lines = [
        _line(*((rng.randint(0, 4), rng.randint(0, 4)) for _ in range(n)))
        for n in [rng.randint(2, 9) for _ in range(500)]
    ]
    expected = [
        not shapely.LineString([(v["x"], v["y"]) for v in line]).is_simple
        for line in lines
    ]

    assert _self_intersects(*lines) == expected


def test_a_multilinestring_gives_one_per_part() -> None:
    """Parts crossing each other do not count."""
    storage = pl.Series(
        "multi",
        [[_SQUARE, _BOWTIE], [], [None, _SQUARE], None, [_SQUARE, _SQUARE]],
        dtype=pl.List(_XY_VERTICES),
    )
    df = pl.DataFrame(storage.ext.to(MultiLineStringXY()))

    out = df.select(geo.self_intersects("multi"))

    assert out.schema["multi"] == pl.List(pl.Boolean)
    assert out["multi"].to_list() == [
        [False, True],
        [],
        [None, False],
        None,
        [False, False],
    ]


def test_a_sliced_multilinestring_reads_only_its_own_parts() -> None:
    storage = pl.Series(
        "multi",
        [[_BOWTIE], [_SQUARE, _BOWTIE], [_SQUARE]],
        dtype=pl.List(_XY_VERTICES),
    )
    df = pl.DataFrame(storage.ext.to(MultiLineStringXY())).slice(1, 2)

    assert df.select(geo.self_intersects("multi"))["multi"].to_list() == [
        [False, True],
        [False],
    ]


def test_a_sliced_linestring_column() -> None:
    storage = pl.Series("line", [_BOWTIE, _SQUARE], dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY())).slice(1, 1)

    assert df.select(geo.self_intersects("line"))["line"].to_list() == [False]


def test_the_namespace_forwards() -> None:
    storage = pl.Series("line", [_BOWTIE], dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))

    assert df.select(gpl.col("line").geo.self_intersects())["line"].to_list() == [True]


@pytest.mark.parametrize("geometry", ["point", "multipoint"])
def test_refuses_points(
    geometry: str, coords: pl.DataFrame, line_coords: pl.DataFrame
) -> None:
    df = (
        coords.select(XY.point())
        if geometry == "point"
        else XY.multipoints(line_coords)
    )

    with pytest.raises(TypeError, match="self_intersects expects a linestring"):
        df.select(geo.self_intersects(geometry))


# Polygons: a ring crossing itself, or two holes overlapping.

_XY_RINGS = pl.List(_XY_VERTICES)


def _square(x0: float, y0: float, x1: float, y1: float) -> Line:
    return _line((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))


_OUTER = _square(0, 0, 10, 10)
_CLOSED_BOWTIE = [*_BOWTIE, _BOWTIE[0]]


def _polygons_self_intersect(*polygons: list[Line | None] | None) -> list[bool | None]:
    storage = pl.Series("polygon", list(polygons), dtype=_XY_RINGS)
    df = pl.DataFrame(storage.ext.to(PolygonXY()))
    out = df.select(geo.self_intersects("polygon"))
    assert out.schema["polygon"] == pl.Boolean
    return out["polygon"].to_list()


def test_a_simple_polygon_does_not() -> None:
    holes = [_OUTER, _square(1, 1, 2, 2), _square(3, 3, 4, 4)]
    assert _polygons_self_intersect([_SQUARE], holes) == [False, False]


def test_a_polygon_with_a_crossing_ring_does() -> None:
    assert _polygons_self_intersect([_CLOSED_BOWTIE], [_OUTER, _CLOSED_BOWTIE]) == [
        True,
        True,
    ]


def test_overlapping_holes_do() -> None:
    overlapping = [_OUTER, _square(1, 1, 3, 3), _square(2, 2, 4, 4)]
    nested = [_OUTER, _square(1, 1, 4, 4), _square(2, 2, 3, 3)]
    assert _polygons_self_intersect(overlapping, nested) == [True, True]


def test_holes_that_only_touch_do_not() -> None:
    at_a_vertex = [_OUTER, _square(1, 1, 2, 2), _square(2, 2, 3, 3)]
    along_an_edge = [_OUTER, _square(1, 1, 2, 2), _square(2, 1, 3, 2)]
    assert _polygons_self_intersect(at_a_vertex, along_an_edge) == [False, False]


def test_the_exterior_is_not_compared_against_its_holes() -> None:
    """Whether a hole lies inside the exterior is up to `is_valid`."""
    outside = [_SQUARE, _square(5, 5, 6, 6)]
    assert _polygons_self_intersect(outside) == [False]


def test_a_missing_polygon_or_ring_is_missing() -> None:
    nan = _line((0, 0), (1, 0), (float("nan"), 1), (0, 0))
    assert _polygons_self_intersect(None, [_OUTER, None], [nan], [], [_SQUARE]) == [
        None,
        None,
        None,
        False,
        False,
    ]


def test_a_ring_that_is_not_one_raises() -> None:
    with pytest.raises(ComputeError, match="expects rings"):
        _polygons_self_intersect([_line((0, 0), (1, 0), (1, 1))])


def test_a_multipolygon_gives_one_per_part() -> None:
    """Parts overlapping each other do not count."""
    overlapping = [_OUTER, _square(1, 1, 3, 3), _square(2, 2, 4, 4)]
    storage = pl.Series(
        "multi",
        [[[_OUTER], overlapping], [], [None, [_SQUARE]], None, [[_OUTER], [_OUTER]]],
        dtype=pl.List(_XY_RINGS),
    )
    df = pl.DataFrame(storage.ext.to(MultiPolygonXY()))

    out = df.select(geo.self_intersects("multi"))

    assert out.schema["multi"] == pl.List(pl.Boolean)
    assert out["multi"].to_list() == [
        [False, True],
        [],
        [None, False],
        None,
        [False, False],
    ]


def test_a_sliced_multipolygon_reads_only_its_own_parts() -> None:
    storage = pl.Series(
        "multi",
        [[[_CLOSED_BOWTIE]], [[_SQUARE], [_CLOSED_BOWTIE]], [[_SQUARE]]],
        dtype=pl.List(_XY_RINGS),
    )
    df = pl.DataFrame(storage.ext.to(MultiPolygonXY())).slice(1, 2)

    assert df.select(geo.self_intersects("multi"))["multi"].to_list() == [
        [False, True],
        [False],
    ]
