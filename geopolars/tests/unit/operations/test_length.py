"""How long a linestring is: its segments, measured like `distance`, added up."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    GeoLineString,
    LineStringXY,
    MultiLineStringXY,
)
from tests.unit.conftest import XY, Dimension

WGS84 = "EPSG:4326"
UNIT_SPHERE = "+proj=longlat +R=1 +type=crs"
NEW_YORK = (-74.006, 40.7128)
LONDON = (-0.1278, 51.5074)

Line = list[tuple[float, float]] | None

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})


def _rows(line: Line) -> list[dict[str, float]] | None:
    return None if line is None else [{"x": x, "y": y} for x, y in line]


def _lines(lines: list[Line], crs: str | None = None) -> pl.DataFrame:
    """A `line` column, with a missing line where it says `None`."""
    storage = pl.Series("line", [_rows(line) for line in lines], dtype=pl.List(_XY))
    return pl.DataFrame(storage.ext.to(LineStringXY(crs=crs)))


def _multilines(
    multis: list[list[Line] | None], crs: str | None = None
) -> pl.DataFrame:
    rows = [None if m is None else [_rows(line) for line in m] for m in multis]
    storage = pl.Series("multi", rows, dtype=pl.List(pl.List(_XY)))
    return pl.DataFrame(storage.ext.to(MultiLineStringXY(crs=crs)))


def _lengths(df: pl.DataFrame) -> list:
    (name,) = df.columns
    return df.select(geo.length(name)).to_series().to_list()


def _segment_by_segment(df: pl.DataFrame, crs: str | None) -> list[float | None]:
    """The same lengths out of `distance`: every vertex measured against the
    next one, and added up per line."""
    coords = (
        df.with_row_index("line_id")
        .select("line_id", pl.col("line").ext.storage())
        .explode("line")
        .unnest("line")
    )
    z = "z" if "z" in coords.columns else None
    vertices = coords.select("line_id", point=geo.point("x", "y", z=z, crs=crs))
    following = pl.col("point").shift(-1).over("line_id")
    segments = vertices.select(
        "line_id", segment=geo.distance("point", following).fill_null(0.0)
    )
    return (
        segments.group_by("line_id", maintain_order=True)
        .agg(pl.col("segment").sum())["segment"]
        .to_list()
    )


def test_a_linestring_adds_up_its_segments() -> None:
    df = _lines([[(0.0, 0.0), (3.0, 4.0), (3.0, 5.0)], [(1.0, 1.0), (1.0, 3.0)]])

    assert _lengths(df) == [6.0, 2.0]


def test_z_counts_and_m_does_not(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Line `a` runs (1, 3, 10) -> (-2.5, 4.5, 20) -> (0, 0, 0);
    `b` (7, 9, 30) -> (8, 10, 40). Without a `z`, it is flat."""
    df = dimension.lines(line_coords)

    climb = 1.0 if dimension.has_z else 0.0
    expected = [
        math.hypot(3.5, 1.5, 10.0 * climb) + math.hypot(2.5, 4.5, 20.0 * climb),
        math.hypot(1.0, 1.0, 10.0 * climb),
    ]
    assert _lengths(df) == pytest.approx(expected)


def _line_with_heights(
    line: list[tuple[float, float, float]], crs: str | None
) -> pl.DataFrame:
    vertices = pl.DataFrame(line, schema=["x", "y", "z"], orient="row")
    point = geo.point("x", "y", "z", crs=crs).alias("point")
    return vertices.select(point.implode()).select(
        geo.linestring("point").alias("line")
    )


def test_with_a_crs_straight_up_and_down_is_the_climb() -> None:
    df = _line_with_heights(
        [(*NEW_YORK, 0.0), (*NEW_YORK, 100.0), (*NEW_YORK, 50.0)], crs=WGS84
    )

    assert _lengths(df) == pytest.approx([150.0])


def test_with_a_crs_in_feet_the_length_is_in_feet() -> None:
    # New York Long Island, in US survey feet: 5000 feet on the grid, then straight up.
    line = [
        (984_000.0, 200_000.0, 0.0),
        (987_000.0, 204_000.0, 0.0),
        (987_000.0, 204_000.0, 1000.0),
    ]
    df = _line_with_heights(line, "EPSG:2263")

    assert _lengths(df) == pytest.approx([6000.0], rel=1e-4)


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_with_z_it_is_the_distance_between_consecutive_vertices(
    crs: str | None,
) -> None:
    df = _line_with_heights(
        [(*NEW_YORK, 0.0), (*LONDON, 10_000.0), (4.9041, 52.3676, 3.0)], crs=crs
    )

    assert _lengths(df) == pytest.approx(_segment_by_segment(df, crs))


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_it_is_the_distance_between_consecutive_vertices(crs: str | None) -> None:
    df = _lines(
        [
            [(4.9041, 52.3676), (4.3571, 52.0116), (5.1214, 52.0907)],
            [(-74.006, 40.7128), (-0.1278, 51.5074)],
        ],
        crs=crs,
    )

    assert _lengths(df) == pytest.approx(_segment_by_segment(df, crs))


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_too_few_vertices_have_no_length(crs: str | None) -> None:
    df = _lines([[], [(1.0, 2.0)]], crs=crs)

    assert _lengths(df) == [0.0, 0.0]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_missing_line_has_no_length(crs: str | None) -> None:
    df = _lines([[(0.0, 0.0), (3.0, 4.0)], None], crs=crs)

    assert _lengths(df)[1] is None


def test_with_a_crs_it_is_measured_along_its_ellipsoid() -> None:
    """On a unit sphere, a quarter of the equator is a quarter turn long,
    however many vertices it is drawn with."""
    df = _lines(
        [
            [(0.0, 0.0), (90.0, 0.0)],
            [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (90.0, 0.0)],
        ],
        crs=UNIT_SPHERE,
    )

    assert _lengths(df) == pytest.approx([math.pi / 2, math.pi / 2])


def test_a_projected_crs_is_measured_on_its_own_ellipsoid() -> None:
    route = [(4.9041, 52.3676), (4.3571, 52.0116), (5.1214, 52.0907)]
    lonlat = _lines([route], crs="EPSG:4289")
    rd = lonlat.select(geo.to_crs("line", "EPSG:28992"))

    assert _lengths(rd) == pytest.approx(_lengths(lonlat), abs=1e-3)


@pytest.mark.parametrize("crs", [None, UNIT_SPHERE], ids=["planar", "geodesic"])
def test_a_multilinestring_lists_the_length_of_each_part(crs: str | None) -> None:
    parts: list[Line] = [
        [(0.0, 0.0), (30.0, 0.0)],
        [(0.0, 10.0), (0.0, 20.0), (0.0, 50.0)],
        [],
    ]
    multi = _multilines([parts, parts[1:]], crs=crs)
    separate = _lengths(_lines(parts, crs=crs))

    got = _lengths(multi)

    assert got == [pytest.approx(separate), pytest.approx(separate[1:])]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_multilinestring_without_parts_has_no_lengths(crs: str | None) -> None:
    df = _multilines([[], [[]], None], crs=crs)

    assert _lengths(df) == [[], [0.0], None]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_sliced_multilinestring_keeps_its_own_parts(crs: str | None) -> None:
    df = _multilines(
        [[[(0.0, 0.0), (3.0, 4.0)]], [[(0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0)]]],
        crs=crs,
    )

    (got,) = _lengths(df.slice(1))

    assert got == pytest.approx(_lengths(df)[1])


def test_every_dimension_of_multilinestring_is_measured(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The rings a polygon of 15.5 is made of, read as curves: the 4x4 square
    and two right triangles with legs of 1 and 2.
    Only the last ring changes height: 1 -> 2 -> 3 -> 1."""
    df = dimension.multilinestrings(ring_coords)

    climb = 1.0 if dimension.has_z else 0.0
    last_ring = (
        math.hypot(2.0, climb)
        + math.hypot(2.0, 2.0, climb)
        + math.hypot(2.0, 2.0 * climb)
    )
    got = _lengths(df)
    assert got[0] == pytest.approx([16.0, 2.0 + math.sqrt(2.0)])
    assert got[1] == pytest.approx([last_ring])


def test_metadata_without_a_crs_is_still_planar() -> None:
    spherical = GeoLineString.ext_from_params(
        "geoarrow.linestring", LineStringXY().ext_storage(), '{"edges":"spherical"}'
    )
    df = _lines([[(0.0, 0.0), (3.0, 4.0)]]).select(
        pl.col("line").ext.storage().ext.to(spherical)
    )

    assert _lengths(df) == [5.0]


@pytest.mark.parametrize("geometry", ["point", "polygon", "multipoint"])
def test_other_geometries_are_refused_while_resolving_the_schema(
    geometry: str,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> None:
    df = {
        "point": lambda: coords.select(XY.point()),
        "polygon": lambda: XY.polygons(ring_coords),
        "multipoint": lambda: XY.multipoints(line_coords),
    }[geometry]()

    lf = df.lazy().select(geo.length(geometry))

    with pytest.raises(
        TypeError, match="a length is measured on a `geoarrow.linestring`"
    ):
        lf.collect_schema()


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"line": [1.0]}).select(geo.length("line"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_the_namespace_matches_the_function(crs: str | None) -> None:
    df = _lines([[(0.0, 0.0), (3.0, 4.0)], [(1.0, 1.0), (2.0, 2.0)]], crs=crs)

    assert_series_equal(
        df.select(gpl.col("line").geo.length()).to_series(),
        df.select(geo.length("line")).to_series(),
    )


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_every_form_of_column_gives_the_same_answer(crs: str | None) -> None:
    df = _lines([[(0.0, 0.0), (3.0, 4.0)], [(1.0, 1.0), (2.0, 2.0)]], crs=crs)
    expected = df.select(geo.length("line")).to_series()

    assert_series_equal(df.select(geo.length(pl.col("line"))).to_series(), expected)
    assert_series_equal(pl.select(geo.length(df["line"])).to_series(), expected)
    if crs is None:
        # Moving a line along does not change its length on the plane.
        moved = df.select(geo.length(geo.translate("line", 10.0, -3.0))).to_series()
        assert_series_equal(moved, expected)
