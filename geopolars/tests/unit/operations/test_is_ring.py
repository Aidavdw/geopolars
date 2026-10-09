"""Whether a linestring is a ring: at least 4 vertices, the last at the first."""

from __future__ import annotations

import polars as pl
import pytest

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    MultiLineStringXY,
)
from tests.unit.conftest import XY, Dimension

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

Line = list[dict[str, float]]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


_TRIANGLE = _line((0, 0), (1, 0), (0, 1), (0, 0))


def _is_ring(*lines: Line | None) -> list[bool | None]:
    storage = pl.Series("line", list(lines), dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))
    return df.select(geo.is_ring("line"))["line"].to_list()


def test_a_closed_line_of_four_vertices_is_a_ring() -> None:
    assert _is_ring(_TRIANGLE) == [True]


def test_an_open_line_is_no_ring() -> None:
    assert _is_ring(_line((0, 0), (1, 0), (0, 1), (1, 1))) == [False]


def test_a_closed_line_of_three_vertices_is_no_ring() -> None:
    """Out and back again encloses nothing."""
    assert _is_ring(_line((0, 0), (1, 0), (0, 0))) == [False]


def test_an_empty_line_is_no_ring_and_missing_stays_missing() -> None:
    assert _is_ring([], None) == [False, None]


def test_the_ends_have_to_be_close_not_equal() -> None:
    """As `pl.Expr.is_close`: within a relative 1e-9."""
    almost = _line((1e6, 0), (2e6, 0), (2e6, 1), (1e6 + 1e-4, 0))
    off = _line((1e6, 0), (2e6, 0), (2e6, 1), (1e6 + 1e-2, 0))

    assert _is_ring(almost, off) == [True, False]


def test_infinite_ends_are_close_only_to_the_same_infinity() -> None:
    inf = float("inf")
    same = _line((inf, 0), (1, 0), (0, 1), (inf, 0))
    opposite = _line((inf, 0), (1, 0), (0, 1), (-inf, 0))

    assert _is_ring(same, opposite) == [True, False]


def test_a_nan_end_closes_nothing() -> None:
    nan = float("nan")
    assert _is_ring(_line((nan, 0), (1, 0), (0, 1), (nan, 0))) == [False]


def test_z_counts_but_m_does_not() -> None:
    """m is a measure along the line rather than a position."""
    z = pl.Series(
        "line",
        [[{"x": 0.0, "y": 0.0, "z": float(i)} for i in (0, 1, 2, 3)]],
    ).ext.to(LineStringXYZ())
    m = pl.Series(
        "line",
        [
            [
                {"x": x, "y": y, "m": float(i)}
                for i, (x, y) in enumerate(
                    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
                )
            ]
        ],
    ).ext.to(LineStringXYM())

    assert pl.DataFrame(z).select(geo.is_ring("line"))["line"].to_list() == [False]
    assert pl.DataFrame(m).select(geo.is_ring("line"))["line"].to_list() == [True]


def test_every_dimension(dimension: Dimension) -> None:
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    line = [
        {axis: {"x": x, "y": y}.get(axis, 7.0) for axis in dimension.coords}
        for x, y in corners
    ]
    storage = pl.Series("line", [line, line[:3]])
    df = pl.DataFrame(storage.ext.to(dimension.linestring_dtype()))

    assert df.select(geo.is_ring("line"))["line"].to_list() == [True, False]


def test_a_multilinestring_gives_one_per_part() -> None:
    open_line = _line((0, 0), (1, 0), (0, 1), (1, 1))
    storage = pl.Series(
        "multi",
        [[_TRIANGLE, open_line], [], [None, _TRIANGLE], None],
        dtype=pl.List(_XY_VERTICES),
    )
    df = pl.DataFrame(storage.ext.to(MultiLineStringXY()))

    out = df.select(geo.is_ring("multi"))

    assert out.schema["multi"] == pl.List(pl.Boolean)
    assert out["multi"].to_list() == [[True, False], [], [None, True], None]


def test_a_sliced_multilinestring_reads_only_its_own_parts() -> None:
    open_line = _line((0, 0), (1, 0), (0, 1), (1, 1))
    storage = pl.Series(
        "multi",
        [[open_line], [_TRIANGLE, open_line], [_TRIANGLE]],
        dtype=pl.List(_XY_VERTICES),
    )
    df = pl.DataFrame(storage.ext.to(MultiLineStringXY())).slice(1, 2)

    assert df.select(geo.is_ring("multi"))["multi"].to_list() == [
        [True, False],
        [True],
    ]


def test_a_sliced_linestring_column() -> None:
    open_line = _line((0, 0), (1, 0), (0, 1), (1, 1))
    storage = pl.Series("line", [open_line, _TRIANGLE], dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY())).slice(1, 1)

    assert df.select(geo.is_ring("line"))["line"].to_list() == [True]


def test_the_namespace_forwards() -> None:
    storage = pl.Series("line", [_TRIANGLE], dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))

    assert df.select(gpl.col("line").geo.is_ring())["line"].to_list() == [True]


@pytest.mark.parametrize("geometry", ["point", "polygon"])
def test_refuses_anything_but_lines(
    geometry: str, coords: pl.DataFrame, ring_coords: pl.DataFrame
) -> None:
    df = coords.select(XY.point()) if geometry == "point" else XY.polygons(ring_coords)

    with pytest.raises(TypeError, match="is_ring expects a linestring"):
        df.select(geo.is_ring(geometry))
