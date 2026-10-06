"""Getters for the coordinates a geometry is stored as."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_series_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, Dimension

Getter = Callable[[str], pl.Expr]

AXES = pytest.mark.parametrize(
    ("axis", "getter"), [("x", geo.x), ("y", geo.y)], ids=["x", "y"]
)


@AXES
def test_reads_a_points_axis(
    axis: str, getter: Getter, coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(getter("point")).to_series()

    assert out.dtype == pl.Float64
    assert_series_equal(out, coords[axis].alias("point"))


@AXES
def test_a_missing_point_has_no_coordinate(axis: str, getter: Getter) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).select(
        pl.when(pl.col("x") > 1.0).then(XY.point())
    )

    expected = {"x": [None, 2.0], "y": [None, 4.0]}[axis]
    assert df.select(getter("point")).to_series().to_list() == expected


@AXES
def test_drops_the_crs(axis: str, getter: Getter) -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )

    assert df.select(getter("point")).schema["point"] == pl.Float64


@AXES
@pytest.mark.parametrize("geometry", ["line", "polygon", "multipoint"])
def test_refuses_anything_but_a_point_while_resolving_the_schema(
    axis: str,
    getter: Getter,
    geometry: str,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> None:
    df = {
        "line": lambda: XY.lines(line_coords),
        "polygon": lambda: XY.polygons(ring_coords),
        "multipoint": lambda: XY.multipoints(line_coords),
    }[geometry]()

    lf = df.lazy().select(getter(geometry))

    with pytest.raises(TypeError, match=f"`{axis}` is read off a `geoarrow.point`"):
        lf.collect_schema()


@AXES
def test_refuses_a_non_geometry_while_resolving_the_schema(
    axis: str, getter: Getter
) -> None:
    lf = pl.LazyFrame({"point": [1.0]}).select(getter("point"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


@AXES
def test_the_namespace_matches_the_function(
    axis: str, getter: Getter, coords: pl.DataFrame
) -> None:
    df = coords.select(XY.point())
    method = getattr(gpl.col("point").geo, axis)

    assert_series_equal(
        df.select(method()).to_series(),
        df.select(getter("point")).to_series(),
    )
