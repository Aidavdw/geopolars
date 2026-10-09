"""The interior rings (holes) of a polygon, as a multilinestring."""

from __future__ import annotations

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import MultiLineStringXY, MultiPolygonXY
from tests.unit.conftest import XY, Dimension

RD = "EPSG:28992"

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


_OUTER = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_HOLE = _ring((1, 1), (2, 1), (2, 2), (1, 2))
_OTHER_HOLE = _ring((3, 3), (3.5, 3), (3.5, 3.5))
_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


def _multipolygons(*multis: list[list[Ring]] | None) -> pl.DataFrame:
    """A `multi` column of `MultiPolygonXY`, one row per list of polygons."""
    storage = pl.Series("multi", list(multis), dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY()))


def _interiors(df: pl.DataFrame) -> list:
    return df.select(geo.interior(df.columns[0])).to_series().ext.storage().to_list()


def test_the_interior_is_every_ring_after_the_exterior() -> None:
    df = _polygons([_OUTER, _HOLE, _OTHER_HOLE], [_SQUARE])

    assert _interiors(df) == [[_HOLE, _OTHER_HOLE], []]


def test_a_missing_or_empty_polygon() -> None:
    assert _interiors(_polygons(None, [])) == [None, []]


def test_keeps_the_dimension(ring_coords: pl.DataFrame, dimension: Dimension) -> None:
    df = dimension.polygons(ring_coords)
    out = df.select(geo.interior("polygon"))

    assert out.schema["polygon"] == dimension.multilinestring_dtype()
    # Every ring of every polygon but the first, z and m included.
    expected = df.select(pl.col("polygon").ext.storage().list.slice(1))
    assert out.select(pl.col("polygon").ext.storage()).equals(expected)


def test_the_crs_is_carried_over() -> None:
    df = _polygons([_SQUARE], crs=RD)

    out = df.select(geo.interior("polygon"))

    assert out.schema["polygon"] == MultiLineStringXY(crs=RD)


def test_a_multipolygon_gets_its_holes_per_polygon() -> None:
    df = _multipolygons([[_OUTER, _HOLE], [_SQUARE], []], None, [])

    assert _interiors(df) == [[[_HOLE], [], []], None, []]


def test_a_multipolygon_keeps_its_dimension_and_crs(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords).select(
        pl.col("multipolygon")
        .ext.storage()
        .ext.to(dimension.multipolygon_dtype(crs=RD))
    )

    out = df.select(geo.interior("multipolygon"))

    assert out.schema["multipolygon"] == pl.List(
        dimension.multilinestring_dtype(crs=RD)
    )
    # `p` holds `a` (with one hole) and `b`; `q` holds only `c`.
    holes = out["multipolygon"].list.eval(pl.element().ext.storage().list.len())
    assert holes.to_list() == [[1, 0], [0]]


def test_rejects_a_linestring_while_resolving_the_schema(
    line_coords: pl.DataFrame,
) -> None:
    lf = XY.lines(line_coords).lazy().select(geo.interior("line"))

    with pytest.raises(
        TypeError, match="interior expects a polygon or multipolygon column"
    ):
        lf.collect_schema()


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.interior("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_OUTER, _HOLE], [_SQUARE])

    assert df.select(pl.col("polygon").geo.interior()).equals(
        df.select(geo.interior("polygon"))
    )


def test_a_missing_polygon_in_a_multipolygon_has_no_interior() -> None:
    df = _multipolygons([[_OUTER, _HOLE], None, []])

    assert _interiors(df) == [[[_HOLE], None, []]]


def test_multipolygon_rows_line_up_across_chunks_and_slices() -> None:
    df = pl.concat(
        [
            _multipolygons([[_SQUARE]], [[_OUTER, _HOLE], [_SQUARE]]),
            _multipolygons(None, [[_OUTER, _HOLE, _OTHER_HOLE]], [[], [_OUTER, _HOLE]]),
        ],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _interiors(df.slice(1, 4))

    assert got == [[[_HOLE], []], None, [[_HOLE, _OTHER_HOLE]], [[], [_HOLE]]]


def test_polygon_rows_line_up_across_chunks_and_slices() -> None:
    df = pl.concat(
        [
            _polygons([_SQUARE], [_OUTER, _HOLE]),
            _polygons(None, [], [_OUTER, _HOLE, _OTHER_HOLE]),
        ],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _interiors(df.slice(1, 4))

    assert got == [[_HOLE], None, [], [_HOLE, _OTHER_HOLE]]
