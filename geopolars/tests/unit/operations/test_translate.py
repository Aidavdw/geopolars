"""Unit tests for `geo.translate` by an offset per row.
The tests with one offset for the whole column are in `namespaces/test_geo.py`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, XYM, XYZ, Dimension, coordinates, ring_coordinates


def test_a_column_of_one_offset_is_the_same_as_its_numbers(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The kernel that takes an offset per row, against the one that takes one offset."""
    dz = 3.0 if dimension.has_z else 0.0
    df = dimension.polygons(ring_coords).with_columns(
        dx=pl.lit(1.5), dy=pl.lit(-2.0), dz=pl.lit(dz)
    )

    assert_frame_equal(
        df.select(geo.translate("polygon", ("dx", "dy", "dz" if dz else 0.0))),
        df.select(geo.translate("polygon", (1.5, -2.0, dz))),
        check_exact=True,
    )


def test_each_geometry_moves_by_its_own_offset(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    offsets = [(1.0, -2.0, 3.0), (-10.0, 20.0, -30.0)]
    df = dimension.polygons(ring_coords).with_columns(
        dx=pl.Series([dx for dx, _, _ in offsets]),
        dy=pl.Series([dy for _, dy, _ in offsets]),
        dz=pl.Series([dz for _, _, dz in offsets]),
    )
    dz = "dz" if dimension.has_z else 0.0
    out = df.select(geo.translate("polygon", ("dx", "dy", dz)))

    for row, (dx, dy, dz_) in enumerate(offsets):
        alone = df.slice(row, 1).select(
            geo.translate("polygon", (dx, dy, dz_ if dimension.has_z else 0.0))
        )
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


def test_a_point_column_is_the_same_as_its_coordinates(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A point's `x`, `y` and `z` are `dx`, `dy` and `dz`; its `m` is not an offset."""
    df = dimension.polygons(ring_coords).with_columns(
        dx=pl.Series([1.0, -10.0]), dy=pl.Series([-2.0, 20.0]), dz=pl.lit(3.0)
    )
    z = "dz" if dimension.has_z else None
    m = "dz" if dimension.has_m else None
    df = df.with_columns(offset=geo.point("dx", "dy", z=z, m=m))

    assert_frame_equal(
        df.select(geo.translate("polygon", "offset")),
        df.select(geo.translate("polygon", ("dx", "dy", z or 0.0))),
        check_exact=True,
    )


def test_a_single_point_moves_every_geometry(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)
    offset = geo.point(pl.lit(1.0), pl.lit(2.0))

    assert_frame_equal(
        df.select(geo.translate("polygon", offset)),
        df.select(geo.translate("polygon", (1.0, 2.0))),
        check_exact=True,
    )


def test_numbers_and_expressions_mix(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point(), "x")
    out = df.select(geo.translate("point", (1.0, pl.col("x") * 2, pl.lit(5))))

    expected = coordinates(df).with_columns(
        x=pl.col("x") + 1, y=pl.col("y") + 2 * pl.col("x"), z=pl.col("z") + 5
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_m_stays_as_it_is(ring_coords: pl.DataFrame) -> None:
    df = XYM.polygons(ring_coords)
    out = df.select(geo.translate("polygon", (pl.lit(1.0), pl.lit(2.0))))

    expected = ring_coordinates(df).with_columns(pl.col("x") + 1, pl.col("y") + 2)
    assert_frame_equal(ring_coordinates(out), expected, check_exact=True)


def test_a_slice_moves_by_its_own_offsets(ring_coords: pl.DataFrame) -> None:
    """A slice keeps the coordinates of the rows around it, as a streaming morsel does."""
    df = XY.polygons(ring_coords).with_columns(dx=pl.Series([1.0, 2.0]), dy=pl.lit(0.5))
    moved = geo.translate("polygon", ("dx", "dy"))

    assert_frame_equal(
        df.slice(1, 1).select(moved), df.select(moved).slice(1, 1), check_exact=True
    )


def test_a_missing_offset_gives_a_missing_geometry(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords).with_columns(dx=pl.Series([None, 1.0]))

    for offset in [("dx", 0.0), geo.point("dx", pl.lit(0.0))]:
        out = df.select(geo.translate("polygon", offset))
        assert out["polygon"].is_null().to_list() == [True, False]


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
def test_a_z_offset_column_needs_a_z_at_plan_time(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Even one that is 0 everywhere: that is only known once it is computed."""
    lf = coords.lazy().select(
        dimension.point(), dz=pl.lit(0.0), offset=geo.point("x", "y", z=pl.lit(0.0))
    )

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        lf.select(geo.translate("point", (1.0, 1.0, "dz"))).collect_schema()
    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        lf.select(geo.translate("point", "offset")).collect_schema()


@pytest.mark.parametrize(
    "offset",
    [(1.0,), (1.0, 2.0, 3.0, 4.0), (math.nan, 1.0), ("dx", math.inf)],
    ids=["1", "4", "nan", "inf"],
)
def test_bad_offsets_are_refused(offset: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="`offset` has to be"):
        geo.translate("point", offset)  # type: ignore[arg-type]


def test_an_offset_column_has_to_hold_points(line_coords: pl.DataFrame) -> None:
    lf = XY.lines(line_coords).lazy().with_columns(dx=pl.lit(1.0))

    with pytest.raises(TypeError, match="`offset` has to be a `geoarrow.point`"):
        lf.select(geo.translate("line", "line")).collect_schema()
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.select(geo.translate("line", "dx")).collect_schema()


def test_an_offset_column_in_another_crs_is_refused(coords: pl.DataFrame) -> None:
    lf = coords.lazy().select(
        geo.point("x", "y", crs="EPSG:28992").alias("point"),
        offset=geo.point("x", "y", crs="EPSG:4326"),
    )

    with pytest.raises(ComputeError, match="use `to_crs`"):
        lf.select(geo.translate("point", "offset")).collect_schema()


def test_refuses_a_column_that_is_not_a_geometry() -> None:
    lf = pl.LazyFrame({"point": [1.0, 2.0], "dx": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.select(geo.translate("point", ("dx", 0.0))).collect_schema()


def test_the_namespace_forwards(ring_coords: pl.DataFrame) -> None:
    df = XYZ.polygons(ring_coords).with_columns(dx=pl.Series([1.0, 2.0]))
    df = df.with_columns(offset=geo.point("dx", pl.lit(1.0), z="dx"))

    assert_frame_equal(
        df.select(gpl.col("polygon").geo.translate("offset")),
        df.select(geo.translate("polygon", ("dx", 1.0, pl.col("dx")))),
    )
