"""The boundary of a geometry: a polygon's rings, or a linestring's endpoints."""

from __future__ import annotations

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
)
from tests.unit.conftest import XY, Dimension

RD = "EPSG:28992"

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Line = list[dict[str, float]]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _ring(*xy: tuple[float, float]) -> Line:
    """A closed ring through `xy`."""
    return _line(*xy, xy[0])


_OUTER = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_HOLE = _ring((1, 1), (2, 1), (2, 2), (1, 2))
_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))


def _polygons(*polygons: list[Line] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


def _multipolygons(*multis: list[list[Line]] | None) -> pl.DataFrame:
    storage = pl.Series("multi", list(multis), dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY()))


def _lines(*lines: Line | None, crs: str | None = None) -> pl.DataFrame:
    storage = pl.Series("line", list(lines), dtype=_XY_VERTICES)
    return pl.DataFrame(storage.ext.to(LineStringXY(crs=crs)))


def _multilines(*multis: list[Line | None] | None) -> pl.DataFrame:
    storage = pl.Series("multi", list(multis), dtype=pl.List(_XY_VERTICES))
    return pl.DataFrame(storage.ext.to(MultiLineStringXY()))


def _boundaries(df: pl.DataFrame) -> list:
    return df.select(geo.boundary(df.columns[0])).to_series().ext.storage().to_list()


# Polygons: their rings


def test_a_polygons_boundary_is_its_rings_exterior_first() -> None:
    df = _polygons([_OUTER, _HOLE], [_SQUARE])

    assert _boundaries(df) == [[_OUTER, _HOLE], [_SQUARE]]


def test_a_missing_or_empty_polygon() -> None:
    assert _boundaries(_polygons(None, [])) == [None, []]


def test_a_polygon_keeps_its_dimension(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)
    out = df.select(geo.boundary("polygon"))

    assert out.schema["polygon"] == dimension.multilinestring_dtype()
    # Nothing but the type changes.
    assert out.select(pl.col("polygon").ext.storage()).equals(
        df.select(pl.col("polygon").ext.storage())
    )


def test_the_crs_of_a_polygon_is_carried_over() -> None:
    df = _polygons([_SQUARE], crs=RD)

    out = df.select(geo.boundary("polygon"))

    assert out.schema["polygon"] == MultiLineStringXY(crs=RD)


def test_a_multipolygon_gets_a_boundary_per_polygon() -> None:
    df = _multipolygons([[_OUTER, _HOLE], [_SQUARE], []], None, [])

    assert _boundaries(df) == [[[_OUTER, _HOLE], [_SQUARE], []], None, []]


def test_a_multipolygon_keeps_its_dimension_and_crs(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords).select(
        pl.col("multipolygon")
        .ext.storage()
        .ext.to(dimension.multipolygon_dtype(crs=RD))
    )

    out = df.select(geo.boundary("multipolygon"))

    assert out.schema["multipolygon"] == pl.List(
        dimension.multilinestring_dtype(crs=RD)
    )
    assert out["multipolygon"].list.len().to_list() == [2, 1]


# Linestrings: their endpoints


def test_a_linestrings_boundary_is_its_first_and_last_vertex() -> None:
    df = _lines(_line((0, 0), (1, 5), (2, 3)), _line((7, 7), (8, 8)))

    assert _boundaries(df) == [_line((0, 0), (2, 3)), _line((7, 7), (8, 8))]


def test_a_closed_or_single_vertex_linestring_has_no_boundary() -> None:
    df = _lines(_SQUARE, _line((3, 4)))

    assert _boundaries(df) == [[], []]


@pytest.mark.parametrize(
    ("off", "closed"), [(1e-12, True), (1e-6, False)], ids=["within", "beyond"]
)
def test_a_line_closes_within_a_relative_tolerance(off: float, closed: bool) -> None:
    line = _line((100, 100), (200, 100), (100 * (1 + off), 100))

    (got,) = _boundaries(_lines(line))

    assert got == ([] if closed else [line[0], line[-1]])


def _line_of(dtype: type[GeoArrowType], *vertices: dict[str, float]) -> pl.DataFrame:
    storage = pl.Series("line", [list(vertices)], dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype()))


def test_a_line_whose_ends_differ_in_z_is_open() -> None:
    start, end = {"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 5.0}
    df = _line_of(LineStringXYZ, start, {"x": 1.0, "y": 1.0, "z": 1.0}, end)

    assert _boundaries(df) == [[start, end]]


def test_m_does_not_keep_a_line_open() -> None:
    start, end = {"x": 0.0, "y": 0.0, "m": 0.0}, {"x": 0.0, "y": 0.0, "m": 9.0}
    df = _line_of(LineStringXYM, start, {"x": 1.0, "y": 1.0, "m": 4.0}, end)

    assert _boundaries(df) == [[]]


def test_a_missing_or_empty_linestring() -> None:
    assert _boundaries(_lines(None, [])) == [None, []]


def test_a_linestring_keeps_its_dimension(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines(line_coords)
    out = df.select(geo.boundary("line"))

    assert out.schema["line"] == dimension.multipoint_dtype()
    storage = pl.col("line").ext.storage()
    expected = df.select(pl.concat_list(storage.list.first(), storage.list.last()))
    assert out.select(storage).equals(expected)


def test_the_crs_of_a_linestring_is_carried_over() -> None:
    df = _lines(_line((0, 0), (1, 1)), crs=RD)

    out = df.select(geo.boundary("line"))

    assert out.schema["line"] == MultiPointXY(crs=RD)


def test_a_multilinestring_gets_a_boundary_per_part() -> None:
    df = _multilines([_line((0, 0), (1, 1), (2, 0)), _SQUARE, None, []], None, [])

    assert _boundaries(df) == [[_line((0, 0), (2, 0)), [], None, []], None, []]


def test_a_multilinestring_keeps_its_dimension_and_crs(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords).select(
        pl.col("multilinestring")
        .ext.storage()
        .ext.to(dimension.multilinestring_dtype(crs=RD))
    )

    out = df.select(geo.boundary("multilinestring"))

    assert out.schema["multilinestring"] == pl.List(dimension.multipoint_dtype(crs=RD))


# Refused


@pytest.mark.parametrize("kind", ["point", "multipoint"])
def test_rejects_points_while_resolving_the_schema(
    kind: str, coords: pl.DataFrame, line_coords: pl.DataFrame
) -> None:
    df = coords.select(XY.point()) if kind == "point" else XY.multipoints(line_coords)
    lf = df.lazy().select(geo.boundary(kind))

    with pytest.raises(
        TypeError, match=r"boundary expects a \(multi\)linestring or \(multi\)polygon"
    ):
        lf.collect_schema()


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.boundary("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_OUTER, _HOLE], [_SQUARE])

    assert df.select(pl.col("polygon").geo.boundary()).equals(
        df.select(geo.boundary("polygon"))
    )
