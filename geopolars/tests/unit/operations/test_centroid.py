"""The centroid of a linestring or polygon: the centre of its length or area."""

from __future__ import annotations

import polars as pl
import pytest
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    PointXY,
)

from geopolars import geo
from tests.unit.conftest import XY, Dimension

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))
# An L of three unit squares: its vertices average elsewhere than its area.
_L = _ring((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


def _centroids(df: pl.DataFrame) -> list[tuple[float, float] | None]:
    out = df.select(geo.centroid("polygon")).to_series().ext.storage().to_list()
    return [None if c is None else (c["x"], c["y"]) for c in out]


@pytest.mark.parametrize(
    ("ring", "expected"),
    [
        (_SQUARE, (0.5, 0.5)),
        (_ring((0, 0), (3, 0), (0, 3)), (1.0, 1.0)),
        (_L, (5 / 6, 5 / 6)),
    ],
    ids=["square", "triangle", "L"],
)
def test_the_centroid_is_the_centre_of_the_area(
    ring: Ring, expected: tuple[float, float]
) -> None:
    (got,) = _centroids(_polygons([ring]))

    assert got == pytest.approx(expected)


def test_winding_order_does_not_change_the_centroid() -> None:
    assert _centroids(_polygons([_L[::-1]])) == _centroids(_polygons([_L]))


@pytest.mark.parametrize("reverse_hole", [False, True])
def test_a_hole_is_taken_out_whichever_way_it_winds(reverse_hole: bool) -> None:
    exterior = _ring((0, 0), (4, 0), (4, 4), (0, 4))
    hole = _ring((2, 2), (3, 2), (3, 3), (2, 3))
    # 16 at (2, 2), less 1 at (2.5, 2.5)
    expected = (16 * 2 - 2.5) / 15

    (got,) = _centroids(_polygons([exterior, hole[::-1] if reverse_hole else hole]))

    assert got == pytest.approx((expected, expected))


def test_z_and_m_are_left_out(ring_coords: pl.DataFrame, dimension: Dimension) -> None:
    df = dimension.polygons(ring_coords)
    out = df.select(geo.centroid("polygon"))

    assert out.schema["polygon"] == PointXY()
    assert _centroids(df) == pytest.approx(_centroids(XY.polygons(ring_coords)))


def test_the_crs_is_carried_over() -> None:
    df = _polygons([_SQUARE], crs="EPSG:4326")

    out = df.select(geo.centroid("polygon"))

    assert out.schema["polygon"] == PointXY._with_metadata_of(df.schema["polygon"])


def test_a_missing_or_empty_polygon_has_no_centroid() -> None:
    assert _centroids(_polygons(None, [], [_SQUARE])) == [None, None, (0.5, 0.5)]


def test_a_polygon_that_encloses_nothing_has_no_centroid() -> None:
    assert _centroids(_polygons([_ring((0, 0), (2, 0), (1, 0))])) == [None]


def test_rejects_a_multilinestring_while_resolving_the_schema(
    ring_coords: pl.DataFrame,
) -> None:
    lf = XY.multilinestrings(ring_coords).lazy().select(geo.centroid("multilinestring"))

    with pytest.raises(
        TypeError, match="centroid expects a `geoarrow.linestring` or `geoarrow.polygon`"
    ):
        lf.collect_schema()


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.centroid("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_L], [_SQUARE])

    assert df.select(pl.col("polygon").geo.centroid()).equals(
        df.select(geo.centroid("polygon"))
    )


def test_rows_line_up_across_chunks_and_slices() -> None:
    square = _ring((10, 10), (12, 10), (12, 12), (10, 12))
    df = pl.concat(
        [_polygons([_SQUARE], [_L]), _polygons(None, [square], [_SQUARE])],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _centroids(df.slice(1, 4))

    assert got == pytest.approx([(5 / 6, 5 / 6), None, (11.0, 11.0), (0.5, 0.5)])


# Linestrings

Line = list[tuple[float, ...]] | None


def _lines(
    *lines: Line, dtype: type[GeoArrowType] = LineStringXY, crs: str | None = None
) -> pl.DataFrame:
    """A `line` column of `dtype`, one row per list of coordinate tuples."""
    names = dtype._dimension
    vertex = pl.Struct(dict.fromkeys(names, pl.Float64))
    rows = [
        None if line is None else [dict(zip(names, c, strict=True)) for c in line]
        for line in lines
    ]
    storage = pl.Series("line", rows, dtype=pl.List(vertex))
    return pl.DataFrame(storage.ext.to(dtype(crs=crs)))


def _line_centroids(df: pl.DataFrame) -> list[tuple[float, ...] | None]:
    out = df.select(geo.centroid("line")).to_series().ext.storage().to_list()
    return [None if c is None else tuple(c.values()) for c in out]


# Segments of length 2 and 1: its vertices average to (4/3, 1/3) instead.
_HOOK = [(0, 0), (2, 0), (2, 1)]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ([(0, 0), (2, 4)], (1.0, 2.0)),
        (_HOOK, (4 / 3, 1 / 6)),
        ([(0, 0), (1, 0), (1, 0), (2, 0)], (1.0, 0.0)),
    ],
    ids=["segment", "hook", "repeated vertex"],
)
def test_a_lines_centroid_is_its_midpoints_weighted_by_length(
    line: Line, expected: tuple[float, float]
) -> None:
    (got,) = _line_centroids(_lines(line))

    assert got == pytest.approx(expected)


def test_z_counts_towards_the_length() -> None:
    # Segments of length 2 (straight up) and 1: in 2D the first would weigh nothing.
    line = [(0, 0, 0), (0, 0, 2), (1, 0, 2)]

    (got,) = _line_centroids(_lines(line, dtype=LineStringXYZ))

    assert got == pytest.approx((1 / 6, 0.0, 4 / 3))


def test_m_is_averaged_along_the_length_without_adding_to_it() -> None:
    line = [(x, y, m) for (x, y), m in zip(_HOOK, [0, 10, 10], strict=True)]

    (got,) = _line_centroids(_lines(line, dtype=LineStringXYM))

    # m: (2 * 5 + 1 * 10) / 3
    assert got == pytest.approx((4 / 3, 1 / 6, 20 / 3))


def test_a_line_keeps_its_dimension(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.lines(line_coords).select(geo.centroid("line"))

    assert out.schema["line"] == dimension.point_dtype()


def test_the_crs_of_a_line_is_carried_over() -> None:
    df = _lines(_HOOK, crs="EPSG:4326")

    out = df.select(geo.centroid("line"))

    assert out.schema["line"] == PointXY._with_metadata_of(df.schema["line"])


def test_a_line_without_length_has_no_centroid() -> None:
    df = _lines(None, [], [(1, 1)], [(1, 1), (1, 1)], _HOOK)

    assert _line_centroids(df) == [None, None, None, None, pytest.approx((4 / 3, 1 / 6))]


def test_a_line_with_a_missing_coordinate_has_no_centroid() -> None:
    storage = pl.Series(
        "line",
        [[{"x": 0.0, "y": 0.0}, {"x": None, "y": 1.0}], [{"x": 0.0, "y": 0.0}, None]],
        dtype=pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})),
    )
    df = pl.DataFrame(storage.ext.to(LineStringXY()))

    assert _line_centroids(df) == [None, None]


def test_the_namespace_matches_the_function_for_lines() -> None:
    df = _lines(_HOOK, [(0, 0), (2, 4)])

    assert df.select(pl.col("line").geo.centroid()).equals(
        df.select(geo.centroid("line"))
    )


def test_line_rows_line_up_across_chunks_and_slices() -> None:
    df = pl.concat(
        [_lines([(0, 0), (2, 4)], _HOOK), _lines(None, [(10, 10), (12, 10)], _HOOK)],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _line_centroids(df.slice(1, 4))

    assert got == pytest.approx([(4 / 3, 1 / 6), None, (11.0, 10.0), (4 / 3, 1 / 6)])
