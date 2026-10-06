"""The area a geometry encloses: on the plane without a CRS, along the
ellipsoid with one."""

from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

from geopolars import geo
from geopolars.datatypes import PointXY, PolygonXY
from tests.unit.conftest import XY, Dimension

_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 4.0, "y": 0.0},
    {"x": 4.0, "y": 4.0},
    {"x": 0.0, "y": 4.0},
    {"x": 0.0, "y": 0.0},
]

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)


def _rings(
    rings: list[list[dict[str, float]]], crs: str | None = None
) -> pl.DataFrame:
    """A one-row polygon column, built straight from XY rings."""
    return pl.DataFrame({"rings": [rings]}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


Area = Callable[[str | pl.Expr | pl.Series], pl.Expr]


@pytest.fixture
def area() -> Area:
    return geo.area


def _areas(area: Area, df: pl.DataFrame, name: str = "polygon") -> list[float | None]:
    return df.select(area(name)).to_series().to_list()


def test_a_polygon_is_its_exterior_ring_less_its_holes(
    area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """`a` is a 4x4 square with a triangular hole of 0.5 in it; `b` is a
    triangle with legs of 2."""
    df = dimension.polygons(ring_coords)

    assert _areas(area, df) == [15.5, 2.0]


def test_z_and_m_are_left_out(
    area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The area of the footprint on the xy plane. A sloped roof does not cover
    more ground than a flat one, and `m` is a measure, not an axis."""
    flat = _areas(area, XY.polygons(ring_coords))

    assert _areas(area, dimension.polygons(ring_coords)) == flat


def test_winding_order_does_not_change_the_area(area: Area) -> None:
    """The spec fixes no winding order, so a ring that runs the other way is
    the same ring. Position is what says which one is the hole."""
    assert _areas(area, _rings([_SQUARE])) == _areas(area, _rings([_SQUARE[::-1]]))


def test_a_hole_is_the_second_ring_whichever_way_it_winds(area: Area) -> None:
    hole = [
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 1.0},
        {"x": 2.0, "y": 2.0},
        {"x": 1.0, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]

    assert _areas(area, _rings([_SQUARE, hole])) == [15.0]
    assert _areas(area, _rings([_SQUARE, hole[::-1]])) == [15.0]


def test_a_degenerate_ring_encloses_nothing(area: Area) -> None:
    """Fewer than three distinct vertices cannot bound anything."""
    there_and_back = [
        {"x": 1.0, "y": 1.0},
        {"x": 5.0, "y": 5.0},
        {"x": 1.0, "y": 1.0},
    ]

    assert _areas(area, _rings([there_and_back])) == [0.0]


def test_an_empty_ring_takes_nothing_with_it(area: Area) -> None:
    assert _areas(area, _rings([_SQUARE, []])) == [16.0]


def test_an_empty_polygon_encloses_nothing(area: Area) -> None:
    """No exterior ring at all, which is an area of zero rather than no area."""
    assert _areas(area, _rings([])) == [0.0]


def test_a_missing_polygon_has_no_area(area: Area) -> None:
    df = pl.DataFrame({"polygon": [[_SQUARE], None]}, schema={"polygon": _XY_RINGS})

    df = df.select(pl.col("polygon").ext.to(PolygonXY()))

    assert _areas(area, df) == [16.0, None]


def test_a_linestring_has_an_area_of_zero(
    area: Area, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A curve has no interior, however it runs."""
    assert set(_areas(area, dimension.lines(line_coords), "line")) == {0.0}


def test_a_closed_linestring_still_has_an_area_of_zero(area: Area) -> None:
    """The square's ring, read as a curve. Nothing says a linestring that
    closes bounds the space inside it -- only a polygon does."""
    df = pl.DataFrame(
        {"vertices": [_SQUARE]}, schema={"vertices": _XY_VERTICES}
    ).select(geo.linestring("vertices").alias("line"))

    assert _areas(area, df, "line") == [0.0]


def test_a_multipoint_has_an_area_of_zero(
    area: Area, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)

    assert set(_areas(area, df, "multipoint")) == {0.0}


def test_a_multilinestring_has_an_area_of_zero(
    area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The same vertices a polygon of 15.5 is made of, as curves instead."""
    df = dimension.multilinestrings(ring_coords)

    assert set(_areas(area, df, "multilinestring")) == {0.0}


def test_a_point_has_an_area_of_zero(
    area: Area, coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())

    assert _areas(area, df, "point") == [0.0, 0.0, 0.0]


def test_a_missing_point_has_no_area(area: Area) -> None:
    df = pl.DataFrame(
        {"point": [{"x": 1.0, "y": 2.0}, None]},
        schema={"point": pl.Struct({"x": pl.Float64, "y": pl.Float64})},
    ).select(pl.col("point").ext.to(PointXY()))

    assert _areas(area, df, "point") == [0.0, None]


def test_the_area_survives_coordinates_far_from_the_origin(area: Area) -> None:
    """A metre-scale parcel in a projected CRS. Squaring the coordinates before
    subtracting them -- the shoelace sum written out about the origin -- throws
    away the digits the answer is made of."""
    ox, oy = 5_123_456.789, 4_987_654.321
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    ring = [{"x": ox + x, "y": oy + y} for x, y in unit]

    (got,) = _areas(area, _rings([ring]))

    assert got == pytest.approx(1.0, abs=1e-9)


def test_the_namespace_matches_the_function(
    area: Area, ring_coords: pl.DataFrame
) -> None:
    df = XY.polygons(ring_coords)

    method = getattr(pl.col("polygon").geo, area.__name__)

    assert_series_equal(
        df.select(method()).to_series(),
        df.select(area("polygon")).to_series(),
    )


def test_every_form_of_column_gives_the_same_answer(
    area: Area, ring_coords: pl.DataFrame
) -> None:
    """A name, a `pl.col(...)`, a `Series` and a chained expression all work."""
    df = XY.polygons(ring_coords)
    expected = df.select(area("polygon")).to_series()

    assert_series_equal(df.select(area(pl.col("polygon"))).to_series(), expected)
    assert_series_equal(pl.select(area(df["polygon"])).to_series(), expected)
    translated = geo.translate("polygon", 10.0, -3.0)
    assert_series_equal(df.select(area(translated)).to_series(), expected)


def test_rejects_a_non_geometry_while_resolving_the_schema(area: Area) -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(area("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


WGS84 = "EPSG:4326"

# A degree square on the equator, from GeographicLib's `PolygonArea` docs, in metres².
_DEGREE_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 1.0, "y": 0.0},
    {"x": 1.0, "y": 1.0},
    {"x": 0.0, "y": 1.0},
    {"x": 0.0, "y": 0.0},
]
_DEGREE_SQUARE_AREA = 12_308_778_361.469_452


def test_with_a_crs_it_is_the_area_on_the_ellipsoid_in_square_meters(
    area: Area,
) -> None:
    """Not the 1.0 square degree it is on the plane."""
    (got,) = _areas(area, _rings([_DEGREE_SQUARE], crs=WGS84))

    assert got == pytest.approx(_DEGREE_SQUARE_AREA, rel=1e-9)


def test_with_a_crs_winding_order_does_not_change_the_area(area: Area) -> None:
    """A clockwise ring is not taken to enclose the rest of the planet."""
    forward = _areas(area, _rings([_DEGREE_SQUARE], crs=WGS84))

    assert _areas(area, _rings([_DEGREE_SQUARE[::-1]], crs=WGS84)) == forward


def test_with_a_crs_a_hole_is_taken_out(area: Area) -> None:
    """The square less the same square: whichever way the hole winds."""
    for hole in (_DEGREE_SQUARE, _DEGREE_SQUARE[::-1]):
        (got,) = _areas(area, _rings([_DEGREE_SQUARE, hole], crs=WGS84))

        assert got == pytest.approx(0.0, abs=1e-3)


def test_with_a_crs_the_ellipsoid_is_the_one_the_crs_declares(area: Area) -> None:
    """On a unit sphere, the triangle from the equator to the pole over a
    quarter turn of longitude is an eighth of the sphere."""
    octant = [
        {"x": 0.0, "y": 0.0},
        {"x": 90.0, "y": 0.0},
        {"x": 0.0, "y": 90.0},
        {"x": 0.0, "y": 0.0},
    ]
    sphere = "+proj=longlat +R=1 +type=crs"

    (got,) = _areas(area, _rings([octant], crs=sphere))

    assert got == pytest.approx(4 * math.pi / 8)


def test_a_projected_crs_is_measured_on_its_own_ellipsoid(area: Area) -> None:
    """The Dutch national grid is in metres already, but measured along the
    ellipsoid the area is the same as in the lon/lat it projects from
    (Amersfoort, on Bessel 1841), not the planar area of the grid."""
    ring = [
        {"x": 4.9, "y": 52.0},
        {"x": 5.1, "y": 52.0},
        {"x": 5.1, "y": 52.2},
        {"x": 4.9, "y": 52.2},
        {"x": 4.9, "y": 52.0},
    ]
    lonlat = _rings([ring], crs="EPSG:4289")
    rd = lonlat.select(geo.to_crs("polygon", "EPSG:28992"))

    (expected,) = _areas(area, lonlat)
    (got,) = _areas(area, rd)

    # Projecting bends the geodesic edges a little, which moves the area by
    # a few square metres out of hundreds of millions.
    assert got == pytest.approx(expected, rel=1e-6)


def test_a_projected_crs_in_feet_is_measured_in_square_feet(area: Area) -> None:
    """New York Long Island is in US survey feet, projected from NAD83:
    the same ring is the same area, in square feet rather than square metres."""
    ring = [
        {"x": -74.0, "y": 40.7},
        {"x": -73.9, "y": 40.7},
        {"x": -73.9, "y": 40.8},
        {"x": -74.0, "y": 40.8},
        {"x": -74.0, "y": 40.7},
    ]
    lonlat = _rings([ring], crs="EPSG:4269")
    feet = lonlat.select(geo.to_crs("polygon", "EPSG:2263"))
    meters_per_us_foot = 1200 / 3937

    (square_meters,) = _areas(area, lonlat)
    (got,) = _areas(area, feet)

    assert got == pytest.approx(square_meters / meters_per_us_foot**2, rel=1e-6)


def test_with_a_crs_z_and_m_are_left_out(
    area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    def with_crs(df: pl.DataFrame, of: Dimension) -> pl.DataFrame:
        dtype = of.polygon_dtype(crs=WGS84)
        return df.select(pl.col("polygon").ext.storage().ext.to(dtype))

    flat = _areas(area, with_crs(XY.polygons(ring_coords), XY))

    got = _areas(area, with_crs(dimension.polygons(ring_coords), dimension))

    assert got == flat


def test_with_a_crs_a_missing_polygon_has_no_area(area: Area) -> None:
    df = pl.DataFrame(
        {"polygon": [[_DEGREE_SQUARE], None]}, schema={"polygon": _XY_RINGS}
    )

    df = df.select(pl.col("polygon").ext.to(PolygonXY(crs=WGS84)))

    assert _areas(area, df) == [pytest.approx(_DEGREE_SQUARE_AREA), None]


def test_with_a_crs_an_empty_polygon_encloses_nothing(area: Area) -> None:
    assert _areas(area, _rings([], crs=WGS84)) == [0.0]


def test_with_a_crs_a_point_still_has_an_area_of_zero(area: Area) -> None:
    df = pl.DataFrame(
        {"point": [{"x": 1.0, "y": 2.0}, None]},
        schema={"point": pl.Struct({"x": pl.Float64, "y": pl.Float64})},
    ).select(pl.col("point").ext.to(PointXY(crs=WGS84)))

    assert _areas(area, df, "point") == [0.0, None]


def test_a_crs_not_on_longitude_latitude_is_refused(area: Area) -> None:
    """Earth-centred XYZ has no ellipsoid surface to measure an area on."""
    df = _rings([_DEGREE_SQUARE], crs="EPSG:4978")

    with pytest.raises(ComputeError, match="not defined on longitude/latitude"):
        _areas(area, df)


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_the_result_is_named_after_the_geometry(area: Area, crs: str | None) -> None:
    df = _rings([_DEGREE_SQUARE], crs=crs)

    assert df.select(area("polygon")).schema == pl.Schema({"polygon": pl.Float64})
