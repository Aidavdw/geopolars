"""Whether a geometry runs counter-clockwise, with a polygon's holes the other way."""

from __future__ import annotations

import polars as pl
import pytest

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    MultiLineStringXY,
    MultiPolygonXY,
    PolygonXY,
)
from tests.unit.conftest import XY, Dimension

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Line = list[dict[str, float]]
Polygon = list[Line | None]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _reversed(line: Line) -> Line:
    return line[::-1]


_CCW = _line((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_CW = _reversed(_CCW)
_HOLE_CW = _line((1, 1), (1, 2), (2, 2), (2, 1), (1, 1))
_HOLE_CCW = _reversed(_HOLE_CW)


def _lines(*lines: Line | None) -> list[bool | None]:
    storage = pl.Series("line", list(lines), dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))
    return df.select(geo.is_standard_orientation("line"))["line"].to_list()


def _polygons(
    *polygons: Polygon | None, holes_clockwise: bool = True
) -> list[bool | None]:
    storage = pl.Series("polygon", list(polygons), dtype=_XY_RINGS)
    df = pl.DataFrame(storage.ext.to(PolygonXY()))
    out = df.select(
        geo.is_standard_orientation("polygon", holes_clockwise=holes_clockwise)
    )
    return out["polygon"].to_list()


def test_a_counter_clockwise_line_is_standard() -> None:
    assert _lines(_CCW, _CW) == [True, False]


def test_an_open_line_is_taken_as_closed() -> None:
    assert _lines(_CCW[:-1], _CW[:-1]) == [True, False]


def test_a_line_enclosing_nothing_runs_neither_way() -> None:
    straight = _line((0, 0), (1, 1), (2, 2), (0, 0))
    out_and_back = _line((0, 0), (1, 0), (0, 0))
    assert _lines(straight, out_and_back, _line((0, 0))) == [False] * 3


def test_an_empty_line_is_standard() -> None:
    assert _lines([]) == [True]


def test_a_missing_line_stays_missing() -> None:
    assert _lines(None, _CCW) == [None, True]


def test_a_nan_or_missing_vertex_runs_neither_way() -> None:
    nan = float("nan")
    with_nan = _line((0, 0), (4, 0), (nan, 4), (0, 4), (0, 0))
    missing: list[dict[str, float] | None] = [*_CCW]
    missing[2] = None
    storage = pl.Series("line", [with_nan, missing], dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))

    out = df.select(geo.is_standard_orientation("line"))["line"].to_list()

    assert out == [False, False]


def test_a_polygon_without_holes_needs_a_counter_clockwise_exterior() -> None:
    assert _polygons([_CCW], [_CW]) == [True, False]


def test_by_default_holes_run_clockwise() -> None:
    assert _polygons([_CCW, _HOLE_CW], [_CCW, _HOLE_CCW], [_CW, _HOLE_CW]) == [
        True,
        False,
        False,
    ]


def test_holes_can_be_asked_to_run_counter_clockwise() -> None:
    assert _polygons(
        [_CCW, _HOLE_CW],
        [_CCW, _HOLE_CCW],
        [_CW, _HOLE_CCW],
        holes_clockwise=False,
    ) == [False, True, False]


def test_every_hole_has_to_run_the_right_way() -> None:
    assert _polygons([_CCW, _HOLE_CW, _HOLE_CW], [_CCW, _HOLE_CW, _HOLE_CCW]) == [
        True,
        False,
    ]


def test_an_empty_polygon_or_ring_is_standard() -> None:
    assert _polygons([], [_CCW, []], [[]], [_CW, []]) == [True, True, True, False]


def test_a_missing_polygon_or_ring_gives_missing() -> None:
    """Even when another ring already runs the wrong way."""
    assert _polygons(None, [_CCW, None], [_CW, None]) == [None, None, None]


def test_z_and_m_do_not_matter(dimension: Dimension) -> None:
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    ring = [
        {axis: {"x": x, "y": y}.get(axis, float(i)) for axis in dimension.coords}
        for i, (x, y) in enumerate(corners)
    ]
    lines = pl.Series("line", [ring, ring[::-1]]).ext.to(dimension.linestring_dtype())
    polygons = pl.Series("polygon", [[ring], [ring[::-1]]]).ext.to(
        dimension.polygon_dtype()
    )

    out = pl.DataFrame([lines, polygons]).select(
        geo.is_standard_orientation("line"), geo.is_standard_orientation("polygon")
    )

    assert out["line"].to_list() == [True, False]
    assert out["polygon"].to_list() == [True, False]


def test_a_multilinestring_gives_one_per_part() -> None:
    storage = pl.Series(
        "multi",
        [[_CCW, _CW], [], [None, _CCW], None],
        dtype=pl.List(_XY_VERTICES),
    )
    df = pl.DataFrame(storage.ext.to(MultiLineStringXY()))

    out = df.select(geo.is_standard_orientation("multi"))

    assert out.schema["multi"] == pl.List(pl.Boolean)
    assert out["multi"].to_list() == [[True, False], [], [None, True], None]


def test_a_multipolygon_gives_one_per_part() -> None:
    storage = pl.Series(
        "multi",
        [[[_CCW, _HOLE_CW], [_CW]], [], [None, [_CCW]], None],
        dtype=pl.List(_XY_RINGS),
    )
    df = pl.DataFrame(storage.ext.to(MultiPolygonXY()))

    out = df.select(geo.is_standard_orientation("multi"))
    every_ccw = df.select(geo.is_standard_orientation("multi", holes_clockwise=False))

    assert out.schema["multi"] == pl.List(pl.Boolean)
    assert out["multi"].to_list() == [[True, False], [], [None, True], None]
    assert every_ccw["multi"].to_list() == [[False, False], [], [None, True], None]


def test_a_sliced_multipolygon_reads_only_its_own_parts() -> None:
    storage = pl.Series(
        "multi",
        [[[_CW]], [[_CCW], [_CW]], [[_CCW, _HOLE_CW]]],
        dtype=pl.List(_XY_RINGS),
    )
    df = pl.DataFrame(storage.ext.to(MultiPolygonXY())).slice(1, 2)

    out = df.select(geo.is_standard_orientation("multi"))["multi"].to_list()

    assert out == [[True, False], [True]]


def test_a_sliced_multilinestring_reads_only_its_own_parts() -> None:
    storage = pl.Series(
        "multi",
        [[_CW], [_CCW, _CW], [_CCW]],
        dtype=pl.List(_XY_VERTICES),
    )
    df = pl.DataFrame(storage.ext.to(MultiLineStringXY())).slice(1, 2)

    out = df.select(geo.is_standard_orientation("multi"))["multi"].to_list()

    assert out == [[True, False], [True]]


def test_a_sliced_polygon_column() -> None:
    storage = pl.Series("polygon", [[_CW], [_CCW, _HOLE_CW]], dtype=_XY_RINGS)
    df = pl.DataFrame(storage.ext.to(PolygonXY())).slice(1, 1)

    out = df.select(geo.is_standard_orientation("polygon"))["polygon"].to_list()

    assert out == [True]


def test_the_namespace_forwards() -> None:
    storage = pl.Series("polygon", [[_CCW, _HOLE_CCW]], dtype=_XY_RINGS)
    df = pl.DataFrame(storage.ext.to(PolygonXY()))
    namespace = gpl.col("polygon").geo

    out = df.select(
        namespace.is_standard_orientation().alias("default"),
        namespace.is_standard_orientation(holes_clockwise=False).alias("every_ccw"),
    )

    assert out.row(0) == (False, True)


@pytest.mark.parametrize("geometry", ["point", "multipoint"])
def test_refuses_points(
    geometry: str, coords: pl.DataFrame, line_coords: pl.DataFrame
) -> None:
    df = (
        coords.select(XY.point())
        if geometry == "point"
        else XY.multipoints(line_coords)
    )

    with pytest.raises(TypeError, match="is_standard_orientation expects"):
        df.select(geo.is_standard_orientation(geometry))
