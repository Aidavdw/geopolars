"""Making a geometry run counter-clockwise (or clockwise)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)

if TYPE_CHECKING:
    from geopolars.datatypes import GeoArrowType

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})

Vertices = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Vertices:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


_CCW = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_CW = _ring((0, 0), (0, 4), (4, 4), (4, 0))
_HOLE_CCW = _ring((1, 1), (2, 1), (2, 2), (1, 2))
_HOLE_CW = _HOLE_CCW[::-1]
_FLAT = _ring((0, 0), (1, 1), (2, 2))


def _column(dtype: GeoArrowType, nesting: int, *rows: object) -> pl.DataFrame:
    """A `geometry` column of `dtype`, its rows `nesting` lists deep around the vertices."""
    storage = _XY
    for _ in range(nesting):
        storage = pl.List(storage)
    series = pl.Series("geometry", list(rows), dtype=storage)
    return pl.DataFrame(series.ext.to(dtype))


def _storage(df: pl.DataFrame, expr: pl.Expr) -> list:
    out = df.select(expr)
    assert out.schema == df.schema
    return out.to_series().ext.storage().to_list()


@pytest.mark.parametrize("function", [geo.to_ccw, geo.to_cw])
@pytest.mark.parametrize(
    "df",
    [
        _column(PointXY(), 0, {"x": 1.0, "y": 2.0}),
        _column(MultiPointXY(), 1, _CCW),
    ],
    ids=["point", "multipoint"],
)
def test_points_are_refused(function: object, df: pl.DataFrame) -> None:
    with pytest.raises(TypeError, match="to_ccw expects"):
        df.select(function("geometry"))  # type: ignore[operator]


def test_a_clockwise_linestring_is_reversed() -> None:
    df = _column(LineStringXY(), 1, _CW, _CCW, _FLAT, [], None)

    assert _storage(df, geo.to_ccw("geometry")) == [_CCW, _CCW, _FLAT, [], None]


@pytest.mark.parametrize("holes_cw", [True, False])
def test_a_polygon_gets_its_rings_the_wanted_way(holes_cw: bool) -> None:
    df = _column(PolygonXY(), 2, [_CW, _HOLE_CCW, _HOLE_CW], [_CCW], [], None)
    hole = _HOLE_CW if holes_cw else _HOLE_CCW

    assert _storage(df, geo.to_ccw("geometry", holes_cw=holes_cw)) == [
        [_CCW, hole, hole],
        [_CCW],
        [],
        None,
    ]


def test_a_multilinestring_keeps_its_line_order() -> None:
    df = _column(MultiLineStringXY(), 2, [_CW, _FLAT, _CCW, []], None)

    assert _storage(df, geo.to_ccw("geometry")) == [[_CCW, _FLAT, _CCW, []], None]


def test_a_multipolygon_keeps_its_polygon_order() -> None:
    df = _column(MultiPolygonXY(), 3, [[_CW, _HOLE_CCW], [], [_CCW, _HOLE_CW]], None)

    assert _storage(df, geo.to_ccw("geometry")) == [
        [[_CCW, _HOLE_CW], [], [_CCW, _HOLE_CW]],
        None,
    ]


@pytest.mark.parametrize("holes_cw", [True, False])
def test_the_result_is_ccw(holes_cw: bool) -> None:
    df = _column(
        MultiPolygonXY(),
        3,
        [[_CW, _HOLE_CCW, _HOLE_CW]],
        [[_CCW, _HOLE_CW], [_CW]],
        [[_CCW, _HOLE_CCW]],
    )

    out = df.select(geo.to_ccw("geometry", holes_cw=holes_cw))

    assert out.select(geo.is_ccw("geometry", holes_cw=holes_cw)).to_series().all()


@pytest.mark.parametrize("holes_ccw", [True, False])
def test_to_cw_turns_every_ring_the_other_way(holes_ccw: bool) -> None:
    df = _column(PolygonXY(), 2, [_CCW, _HOLE_CCW, _HOLE_CW], [_CW], None)
    hole = _HOLE_CCW if holes_ccw else _HOLE_CW

    assert _storage(df, geo.to_cw("geometry", holes_ccw=holes_ccw)) == [
        [_CW, hole, hole],
        [_CW],
        None,
    ]


def test_to_cw_keeps_the_part_order() -> None:
    df = _column(MultiLineStringXY(), 2, [_CCW, _FLAT, _CW])

    assert _storage(df, geo.to_cw("geometry")) == [[_CW, _FLAT[::-1], _CW]]


def test_the_crs_is_carried_over() -> None:
    df = _column(PolygonXY(crs="EPSG:28992"), 2, [_CW])

    assert df.select(geo.to_ccw("geometry")).schema == df.schema
    assert df.select(geo.to_cw("geometry")).schema == df.schema
