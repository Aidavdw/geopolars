"""The planar area a geometry encloses."""

from __future__ import annotations

import polars as pl
import pytest
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


def _rings(rings: list[list[dict[str, float]]]) -> pl.DataFrame:
    """A one-row polygon column, built straight from XY rings."""
    return pl.DataFrame({"rings": [rings]}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings").alias("polygon")
    )


def _areas(df: pl.DataFrame, name: str = "polygon") -> list[float | None]:
    return df.select(geo.area(name)).to_series().to_list()


def test_a_polygon_is_its_exterior_ring_less_its_holes(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """`a` is a 4x4 square with a triangular hole of 0.5 in it; `b` is a
    triangle with legs of 2."""
    df = dimension.polygons(ring_coords)

    assert _areas(df) == [15.5, 2.0]


def test_z_and_m_are_left_out(ring_coords: pl.DataFrame, dimension: Dimension) -> None:
    """The area of the footprint on the xy plane. A sloped roof does not cover
    more ground than a flat one, and `m` is a measure, not an axis."""
    assert _areas(dimension.polygons(ring_coords)) == _areas(XY.polygons(ring_coords))


def test_winding_order_does_not_change_the_area() -> None:
    """The spec fixes no winding order, so a ring that runs the other way is
    the same ring. Position is what says which one is the hole."""
    assert _areas(_rings([_SQUARE])) == _areas(_rings([_SQUARE[::-1]]))


def test_a_hole_is_the_second_ring_whichever_way_it_winds() -> None:
    hole = [
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 1.0},
        {"x": 2.0, "y": 2.0},
        {"x": 1.0, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]

    assert _areas(_rings([_SQUARE, hole])) == [15.0]
    assert _areas(_rings([_SQUARE, hole[::-1]])) == [15.0]


def test_a_degenerate_ring_encloses_nothing() -> None:
    """Fewer than three distinct vertices cannot bound anything."""
    there_and_back = [
        {"x": 1.0, "y": 1.0},
        {"x": 5.0, "y": 5.0},
        {"x": 1.0, "y": 1.0},
    ]

    assert _areas(_rings([there_and_back])) == [0.0]


def test_an_empty_ring_takes_nothing_with_it() -> None:
    assert _areas(_rings([_SQUARE, []])) == [16.0]


def test_an_empty_polygon_encloses_nothing() -> None:
    """No exterior ring at all, which is an area of zero rather than no area."""
    assert _areas(_rings([])) == [0.0]


def test_a_missing_polygon_has_no_area() -> None:
    df = pl.DataFrame({"polygon": [[_SQUARE], None]}, schema={"polygon": _XY_RINGS})

    assert _areas(df.select(pl.col("polygon").ext.to(PolygonXY()))) == [16.0, None]


def test_a_missing_ring_takes_the_whole_area_with_it() -> None:
    df = pl.DataFrame(
        {"polygon": [[_SQUARE, None]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))
    whole = df.select(geo.validate("polygon"))

    assert _areas(whole) == [None]


def test_a_linestring_has_an_area_of_zero(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A curve has no interior, however it runs."""
    assert set(_areas(dimension.lines(line_coords), "line")) == {0.0}


def test_a_closed_linestring_still_has_an_area_of_zero() -> None:
    """The square's ring, read as a curve. Nothing says a linestring that
    closes bounds the space inside it -- only a polygon does."""
    df = pl.DataFrame(
        {"vertices": [_SQUARE]}, schema={"vertices": _XY_VERTICES}
    ).select(geo.linestring("vertices").alias("line"))

    assert _areas(df, "line") == [0.0]


def test_a_multipoint_has_an_area_of_zero(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    assert set(_areas(dimension.multipoints(line_coords), "multipoint")) == {0.0}


def test_a_multilinestring_has_an_area_of_zero(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The same vertices a polygon of 15.5 is made of, as curves instead."""
    df = dimension.multilinestrings(ring_coords)

    assert set(_areas(df, "multilinestring")) == {0.0}


def test_a_point_has_an_area_of_zero(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())

    assert _areas(df, "point") == [0.0, 0.0, 0.0]


def test_a_missing_point_has_no_area() -> None:
    df = pl.DataFrame(
        {"point": [{"x": 1.0, "y": 2.0}, None]},
        schema={"point": pl.Struct({"x": pl.Float64, "y": pl.Float64})},
    ).select(pl.col("point").ext.to(PointXY()))

    assert _areas(df, "point") == [0.0, None]


def test_the_area_survives_coordinates_far_from_the_origin() -> None:
    """A metre-scale parcel in a projected CRS. Squaring the coordinates before
    subtracting them -- the shoelace sum written out about the origin -- throws
    away the digits the answer is made of."""
    ox, oy = 5_123_456.789, 4_987_654.321
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    ring = [{"x": ox + x, "y": oy + y} for x, y in unit]

    (got,) = _areas(_rings([ring]))

    assert got == pytest.approx(1.0, abs=1e-9)


def test_the_namespace_matches_the_function(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)

    assert_series_equal(
        df.select(pl.col("polygon").geo.area()).to_series(),
        df.select(geo.area("polygon")).to_series(),
    )


def test_every_form_of_column_gives_the_same_answer(ring_coords: pl.DataFrame) -> None:
    """A name, a `pl.col(...)`, a `Series` and a chained expression all work."""
    df = XY.polygons(ring_coords)
    expected = df.select(geo.area("polygon")).to_series()

    assert_series_equal(df.select(geo.area(pl.col("polygon"))).to_series(), expected)
    assert_series_equal(pl.select(geo.area(df["polygon"])).to_series(), expected)
    translated = geo.translate("polygon", 10.0, -3.0)
    assert_series_equal(df.select(geo.area(translated)).to_series(), expected)


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.area("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
