"""The exterior (outer) ring of a polygon, as a linestring."""

from __future__ import annotations

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import LineStringXY, MultiPolygonXY
from tests.unit.conftest import XY, Dimension

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


_OUTER = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_HOLE = _ring((1, 1), (2, 1), (2, 2), (1, 2))
_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


def _multipolygons(
    *multis: list[list[Ring]] | None, crs: str | None = None
) -> pl.DataFrame:
    """A `multi` column of `MultiPolygonXY`, one row per list of polygons."""
    storage = pl.Series("multi", list(multis), dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY(crs=crs)))


def _rings(df: pl.DataFrame, name: str) -> list:
    return df.select(geo.exterior(name)).to_series().ext.storage().to_list()


def test_the_exterior_is_the_first_ring_without_its_holes() -> None:
    assert _rings(_polygons([_OUTER, _HOLE], [_SQUARE]), "polygon") == [
        _OUTER,
        _SQUARE,
    ]


def test_a_missing_or_empty_polygon() -> None:
    assert _rings(_polygons(None, [], [_SQUARE]), "polygon") == [None, [], _SQUARE]


def test_keeps_the_dimension(ring_coords: pl.DataFrame, dimension: Dimension) -> None:
    df = dimension.polygons(ring_coords)
    out = df.select(geo.exterior("polygon"))

    assert out.schema["polygon"] == dimension.linestring_dtype()
    # Ring 0 of every polygon, z and m included.
    expected = df.select(pl.col("polygon").ext.storage().list.first())
    assert out.select(pl.col("polygon").ext.storage()).equals(expected)


def test_the_crs_is_carried_over() -> None:
    df = _polygons([_SQUARE], crs="EPSG:28992")

    out = df.select(geo.exterior("polygon"))

    assert out.schema["polygon"] == LineStringXY._with_metadata_of(df.schema["polygon"])


def test_a_multipolygon_gets_an_exterior_per_polygon() -> None:
    df = _multipolygons([[_OUTER, _HOLE], [_SQUARE], []], None, [])

    assert _rings(df, "multi") == [[_OUTER, _SQUARE, []], None, []]


def test_a_multipolygon_keeps_its_dimension_and_crs(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)
    df = df.select(
        pl.col("multipolygon")
        .ext.storage()
        .ext.to(dimension.multipolygon_dtype(crs="EPSG:28992"))
    )

    out = df.select(geo.exterior("multipolygon"))

    line = dimension.linestring_dtype._with_metadata_of(df.schema["multipolygon"])
    assert out.schema["multipolygon"] == pl.List(line)
    assert out["multipolygon"].list.len().to_list() == [2, 1]


def test_rejects_a_linestring_while_resolving_the_schema(
    line_coords: pl.DataFrame,
) -> None:
    lf = XY.lines(line_coords).lazy().select(geo.exterior("line"))

    with pytest.raises(
        TypeError, match="exterior expects a polygon or multipolygon column"
    ):
        lf.collect_schema()


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.exterior("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_OUTER, _HOLE], [_SQUARE])

    assert df.select(pl.col("polygon").geo.exterior()).equals(
        df.select(geo.exterior("polygon"))
    )


def test_a_missing_polygon_in_a_multipolygon_has_no_exterior() -> None:
    df = _multipolygons([[_SQUARE], None, []])

    assert _rings(df, "multi") == [[_SQUARE, None, []]]


def test_multipolygon_rows_line_up_across_chunks_and_slices() -> None:
    square = _ring((10, 10), (12, 10), (12, 12), (10, 12))
    df = pl.concat(
        [
            _multipolygons([[_OUTER, _HOLE]], [[_SQUARE], [square]]),
            _multipolygons(None, [[square, _HOLE]], [[], [_OUTER]]),
        ],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _rings(df.slice(1, 4), "multi")

    assert got == [[_SQUARE, square], None, [square], [[], _OUTER]]
