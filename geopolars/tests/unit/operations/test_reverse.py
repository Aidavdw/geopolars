"""Reversing the order of a geometry's vertices."""

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
    from tests.unit.conftest import Dimension

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})

Vertices = list[dict[str, float]]


def _vertices(*xy: tuple[float, float]) -> Vertices:
    return [{"x": x, "y": y} for x, y in xy]


def _ring(*xy: tuple[float, float]) -> Vertices:
    """A closed ring through `xy`."""
    return _vertices(*xy, xy[0])


_LINE = _vertices((0, 0), (1, 0), (2, 1))
_OTHER_LINE = _vertices((5, 5), (6, 6))
_OUTER = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_HOLE = _ring((1, 1), (2, 1), (2, 2), (1, 2))
_OTHER_HOLE = _ring((3, 3), (3.5, 3), (3.5, 3.5))
_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))


def _column(dtype: GeoArrowType, nesting: int, *rows: object) -> pl.DataFrame:
    """A `geometry` column of `dtype`, its rows `nesting` lists deep around the vertices."""
    storage = _XY
    for _ in range(nesting):
        storage = pl.List(storage)
    series = pl.Series("geometry", list(rows), dtype=storage)
    return pl.DataFrame(series.ext.to(dtype))


def _reversed(df: pl.DataFrame, **kwargs: bool) -> list:
    out = df.select(geo.reverse("geometry", **kwargs))
    assert out.schema == df.schema
    return out.to_series().ext.storage().to_list()


def _reversed_polygon(rings: list[Vertices], *, reverse_holes: bool) -> list[Vertices]:
    if not rings:
        return []
    exterior, *holes = rings
    return [exterior[::-1], *(hole[::-1] if reverse_holes else hole for hole in holes)]


def test_a_point_is_refused() -> None:
    df = _column(PointXY(), 0, {"x": 1.0, "y": 2.0})

    with pytest.raises(TypeError, match="reverse does not accept a point"):
        df.select(geo.reverse("geometry"))


def test_a_linestring_is_reversed() -> None:
    df = _column(LineStringXY(), 1, _LINE, None, [], _OTHER_LINE)

    assert _reversed(df) == [_LINE[::-1], None, [], _OTHER_LINE[::-1]]


@pytest.mark.parametrize("reverse_holes", [False, True])
def test_a_polygon_has_its_exterior_reversed(reverse_holes: bool) -> None:
    polygons = [[_OUTER, _HOLE, _OTHER_HOLE], None, [], [_SQUARE]]
    df = _column(PolygonXY(), 2, *polygons)

    assert _reversed(df, reverse_holes=reverse_holes) == [
        _reversed_polygon(polygon, reverse_holes=reverse_holes)
        if polygon is not None
        else None
        for polygon in polygons
    ]


@pytest.mark.parametrize("nested", [False, True])
def test_a_multipoint_has_its_points_reversed(nested: bool) -> None:
    df = _column(MultiPointXY(), 1, _LINE, None, [])

    assert _reversed(df, nested=nested) == [_LINE[::-1], None, []]


def test_a_multilinestring_is_reversed_as_a_whole() -> None:
    df = _column(MultiLineStringXY(), 2, [_LINE, [], _OTHER_LINE], None, [], [_LINE])

    assert _reversed(df, nested=True) == [
        [_OTHER_LINE[::-1], [], _LINE[::-1]],
        None,
        [],
        [_LINE[::-1]],
    ]


def test_a_multilinestring_keeps_its_line_order_without_nested() -> None:
    df = _column(MultiLineStringXY(), 2, [_LINE, [], _OTHER_LINE], None, [])

    assert _reversed(df, nested=False) == [
        [_LINE[::-1], [], _OTHER_LINE[::-1]],
        None,
        [],
    ]


@pytest.mark.parametrize("reverse_holes", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_a_multipolygon_is_reversed_per_polygon(
    reverse_holes: bool, nested: bool
) -> None:
    polygons = [[_OUTER, _HOLE, _OTHER_HOLE], [], [_SQUARE]]
    df = _column(MultiPolygonXY(), 3, polygons, None, [], [[_SQUARE]])

    expected = [
        _reversed_polygon(polygon, reverse_holes=reverse_holes) for polygon in polygons
    ]
    if nested:
        expected.reverse()
    assert _reversed(df, reverse_holes=reverse_holes, nested=nested) == [
        expected,
        None,
        [],
        [[_SQUARE[::-1]]],
    ]


def test_reverses_every_coordinate(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)

    out = df.select(geo.reverse("polygon", reverse_holes=True))

    assert out.schema == df.schema
    # Every ring reversed, z and m included.
    expected = df.select(
        pl.col("polygon").ext.storage().list.eval(pl.element().list.reverse())
    )
    assert out.select(pl.col("polygon").ext.storage()).equals(expected)


def test_the_crs_is_carried_over() -> None:
    df = _column(LineStringXY(crs="EPSG:28992"), 1, _LINE)

    out = df.select(geo.reverse("geometry"))

    assert out.schema == df.schema
