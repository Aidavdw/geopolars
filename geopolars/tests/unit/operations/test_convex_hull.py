"""The convex hull of a linestring or polygon, or their multi-variants."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    MultiLineStringXY,
    MultiPolygonXY,
    PolygonXY,
    PolygonXYZ,
    PolygonXYZM,
)
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


def _line(*xy: tuple[float, float]) -> Ring:
    """An open linestring through `xy`."""
    return [{"x": x, "y": y} for x, y in xy]


def _lines(*lines: Ring | None) -> pl.DataFrame:
    """A `polygon` column of `LineStringXY`, one row per list of vertices.
    Named like the polygon columns, so `_hulls` reads it too."""
    storage = pl.Series("polygon", list(lines), dtype=LineStringXY._geo_storage)
    return pl.DataFrame(storage.ext.to(LineStringXY()))


def _multilines(*multis: list[Ring] | None) -> pl.DataFrame:
    """A `polygon` column of `MultiLineStringXY`, one row per list of linestrings."""
    storage = pl.Series("polygon", list(multis), dtype=MultiLineStringXY._geo_storage)
    return pl.DataFrame(storage.ext.to(MultiLineStringXY()))


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
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    expected = PolygonXYZ() if dimension.has_z else PolygonXY()

    for df in (
        dimension.lines(line_coords),
        dimension.multilinestrings(ring_coords),
        dimension.polygons(ring_coords),
        dimension.multipolygons(multipolygon_coords),
    ):
        out = df.select(geo.convex_hull(df.columns[0]))

        assert out.schema[df.columns[0]] == expected


def test_the_crs_is_carried_over() -> None:
    out = _polygons([_NOTCHED], crs="EPSG:28992").select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY(crs="EPSG:28992")


def test_rejects_a_point_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y").alias("point")
    )

    with pytest.raises(TypeError, match="convex_hull expects a linestring, polygon"):
        lf.select(geo.convex_hull("point")).collect_schema()


def test_rejects_a_multipoint_while_resolving_the_schema(
    line_coords: pl.DataFrame,
) -> None:
    lf = XY.multipoints(line_coords).lazy().select(geo.convex_hull("multipoint"))

    with pytest.raises(TypeError, match="convex_hull expects a linestring, polygon"):
        lf.collect_schema()


def test_a_bent_linestring_is_closed_around() -> None:
    bent = _line((0, 0), (4, 0), (4, 4), (0, 4))

    df = _lines(bent)
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_HULL]]


def test_a_missing_or_empty_linestring() -> None:
    bent = _line((0, 0), (4, 0), (4, 4), (0, 4))

    assert _hulls(_lines(None, [], bent)) == [None, [], [_HULL]]


def test_a_straight_linestring_gives_a_missing_hull() -> None:
    straight = _line((0, 0), (1, 0), (2, 0))
    bent = _line((0, 0), (4, 0), (4, 4), (0, 4))

    assert _hulls(_lines(straight, bent)) == [None, [_HULL]]


def test_a_multilinestring_gets_one_hull_of_the_whole() -> None:
    bottom = _line((0, 0), (4, 0))
    top = _line((0, 4), (4, 4))

    df = _multilines([bottom, top], None, [], [_line((0, 0), (1, 0))])
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_HULL], None, [], None]


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


def test_a_geometry_geos_fails_on_gives_a_missing_hull() -> None:
    # GEOS cannot build a linestring of one vertex.
    df = _lines(_line((1, 1)), _line((0, 0), (4, 0), (4, 4), (0, 4)))

    assert _hulls(df) == [None, [_HULL]]


def test_stop_on_error_raises_what_geos_fails_on() -> None:
    df = _lines(_line((1, 1)), _line((0, 0), (4, 0), (4, 4), (0, 4)))

    with pytest.raises(ComputeError, match="point array"):
        df.select(geo.convex_hull("polygon", stop_on_error=True))


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_NOTCHED], None)

    assert df.select(pl.col("polygon").geo.convex_hull()).equals(
        df.select(geo.convex_hull("polygon"))
    )


def _is_standard(df: pl.DataFrame) -> list[bool | None]:
    return df.select(geo.is_standard_orientation("polygon")).to_series().to_list()


def test_the_exterior_runs_as_geos_gives_it_clockwise() -> None:
    out = _polygons([_NOTCHED]).select(geo.convex_hull("polygon"))

    assert _is_standard(out) == [False]


def test_ccw_outer_runs_the_exterior_counter_clockwise() -> None:
    out = _polygons([_NOTCHED]).select(geo.convex_hull("polygon", ccw_outer=True))

    assert _is_standard(out) == [True]
    assert out.to_series().ext.storage().to_list() == [
        [_ring((0, 0), (4, 0), (4, 4), (0, 4))]
    ]


def test_the_namespace_forwards_ccw_outer() -> None:
    df = _polygons([_NOTCHED])

    assert df.select(pl.col("polygon").geo.convex_hull(ccw_outer=True)).equals(
        df.select(geo.convex_hull("polygon", ccw_outer=True))
    )


def test_the_namespace_forwards_stop_on_error() -> None:
    df = _lines(_line((1, 1)))

    with pytest.raises(ComputeError, match="point array"):
        df.select(pl.col("polygon").geo.convex_hull(stop_on_error=True))
