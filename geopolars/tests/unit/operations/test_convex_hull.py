"""The convex hull of a polygon or multipolygon."""

from __future__ import annotations

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import MultiPolygonXY, PolygonXY, PolygonXYZ, PolygonXYZM
from tests.unit.conftest import XY, Dimension

_XY_RINGS = pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})))

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


# Concave: a notch at (2, 2).
_NOTCHED = _ring((0, 0), (4, 0), (4, 4), (2, 2), (0, 4))
_HULL = _ring((0, 0), (0, 4), (4, 4), (4, 0))
_ALONG_A_LINE = _ring((0, 0), (1, 0), (2, 0))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


def _multipolygons(
    *multis: list[list[Ring]] | None, crs: str | None = None
) -> pl.DataFrame:
    """A `polygon` column of `MultiPolygonXY`, one row per list of polygons."""
    storage = pl.Series("polygon", list(multis), dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY(crs=crs)))


def _hulls(df: pl.DataFrame) -> list:
    return df.select(geo.convex_hull("polygon")).to_series().ext.storage().to_list()


def test_a_concave_polygon_loses_its_notch() -> None:
    assert _hulls(_polygons([_NOTCHED])) == [[_HULL]]


def test_a_missing_or_empty_polygon() -> None:
    assert _hulls(_polygons(None, [], [_NOTCHED])) == [None, [], [_HULL]]


def test_a_polygon_that_encloses_nothing_gives_a_missing_hull() -> None:
    assert _hulls(_polygons([_ALONG_A_LINE], [_NOTCHED])) == [None, [_HULL]]


def test_z_is_kept_but_m_is_not() -> None:
    xy = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4), (0, 0)]
    ring = [
        {"x": x, "y": y, "z": 10.0 * x, "m": float(i)} for i, (x, y) in enumerate(xy)
    ]
    storage = pl.Series("polygon", [[ring]], dtype=PolygonXYZM._geo_storage)
    df = pl.DataFrame(storage.ext.to(PolygonXYZM()))

    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXYZ()
    assert out.to_series().ext.storage().to_list() == [
        [
            [
                {"x": 0, "y": 0, "z": 0},
                {"x": 0, "y": 4, "z": 0},
                {"x": 4, "y": 4, "z": 40},
                {"x": 4, "y": 0, "z": 40},
                {"x": 0, "y": 0, "z": 0},
            ]
        ]
    ]


def test_a_hull_has_no_m(
    dimension: Dimension,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    expected = PolygonXYZ() if dimension.has_z else PolygonXY()

    for df in (
        dimension.polygons(ring_coords),
        dimension.multipolygons(multipolygon_coords),
    ):
        out = df.select(geo.convex_hull(df.columns[0]))

        assert out.schema[df.columns[0]] == expected


def test_the_crs_is_carried_over() -> None:
    out = _polygons([_NOTCHED], crs="EPSG:28992").select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY(crs="EPSG:28992")


def test_rejects_a_linestring_while_resolving_the_schema(
    line_coords: pl.DataFrame,
) -> None:
    lf = XY.lines(line_coords).lazy().select(geo.convex_hull("line"))

    with pytest.raises(
        TypeError, match="convex_hull expects a polygon or multipolygon"
    ):
        lf.collect_schema()


def test_rejects_a_multilinestring_while_resolving_the_schema(
    ring_coords: pl.DataFrame,
) -> None:
    df = XY.multilinestrings(ring_coords)
    lf = df.lazy().select(geo.convex_hull(df.columns[0]))

    with pytest.raises(
        TypeError, match="convex_hull expects a polygon or multipolygon"
    ):
        lf.collect_schema()


def test_a_multipolygon_gets_one_hull_of_the_whole() -> None:
    left = [_ring((0, 0), (1, 0), (1, 1), (0, 1))]
    right = [_ring((3, 0), (4, 0), (4, 1), (3, 1))]

    df = _multipolygons([left, right])
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_ring((0, 0), (0, 1), (4, 1), (4, 0))]]


def test_a_missing_or_empty_multipolygon() -> None:
    assert _hulls(_multipolygons(None, [], [[_NOTCHED]])) == [None, [], [_HULL]]


def test_a_multipolygon_that_encloses_nothing_gives_a_missing_hull() -> None:
    flat = [[_ALONG_A_LINE], [_ring((3, 0), (4, 0), (5, 0))]]

    assert _hulls(_multipolygons(flat, [[_NOTCHED]])) == [None, [_HULL]]


def test_a_multipolygon_hull_carries_the_crs() -> None:
    df = _multipolygons([[_NOTCHED]], crs="EPSG:28992")

    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY(crs="EPSG:28992")


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_NOTCHED], None)

    assert df.select(pl.col("polygon").geo.convex_hull()).equals(
        df.select(geo.convex_hull("polygon"))
    )
