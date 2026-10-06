"""Getters for the coordinates a geometry is stored as."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import LineStringXY, PolygonXY
from tests.unit.conftest import XY, Dimension

Getter = Callable[[str], pl.Expr]

AXES = pytest.mark.parametrize(
    ("axis", "getter"), [("x", geo.x), ("y", geo.y)], ids=["x", "y"]
)


def _per(vertices: pl.DataFrame, axis: str, *groups: str) -> list:
    """One axis straight off a vertex frame, as a list per group,
    nested one level deeper for every further group."""
    out = vertices.group_by(*groups, maintain_order=True).agg(axis)
    for depth in range(len(groups) - 1, 0, -1):
        out = out.group_by(*groups[:depth], maintain_order=True).agg(axis)
    return out[axis].to_list()


@AXES
def test_reads_a_points_axis(
    axis: str, getter: Getter, coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(getter("point")).to_series()

    assert out.dtype == pl.Float64
    assert_series_equal(out, coords[axis].alias("point"))


@AXES
def test_a_linestring_gives_a_list(
    axis: str, getter: Getter, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.lines(line_coords).select(getter("line")).to_series()

    assert out.dtype == pl.List(pl.Float64)
    assert out.to_list() == _per(line_coords, axis, "line")


@AXES
def test_a_multipoint_gives_a_list(
    axis: str, getter: Getter, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    out = df.select(getter("multipoint")).to_series()

    assert out.dtype == pl.List(pl.Float64)
    assert out.to_list() == _per(line_coords, axis, "line")


@AXES
def test_a_polygon_gives_a_list_per_ring(
    axis: str, getter: Getter, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Closing coordinates included: these are the coordinates as stored."""
    out = dimension.polygons(ring_coords).select(getter("polygon")).to_series()

    assert out.dtype == pl.List(pl.List(pl.Float64))
    assert out.to_list() == _per(ring_coords, axis, "polygon", "ring")


@AXES
def test_a_multilinestring_gives_a_list_per_linestring(
    axis: str, getter: Getter, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)
    out = df.select(getter("multilinestring")).to_series()

    assert out.dtype == pl.List(pl.List(pl.Float64))
    assert out.to_list() == _per(ring_coords, axis, "polygon", "ring")


@AXES
def test_a_multipolygon_gives_a_list_per_polygon_per_ring(
    axis: str, getter: Getter, multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)
    out = df.select(getter("multipolygon")).to_series()

    assert out.dtype == pl.List(pl.List(pl.List(pl.Float64)))
    assert out.to_list() == _per(
        multipolygon_coords, axis, "multipolygon", "polygon", "ring"
    )


@AXES
def test_a_missing_point_has_no_coordinate(axis: str, getter: Getter) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).select(
        pl.when(pl.col("x") > 1.0).then(XY.point())
    )

    expected = {"x": [None, 2.0], "y": [None, 4.0]}[axis]
    assert df.select(getter("point")).to_series().to_list() == expected


@AXES
def test_missing_and_empty_geometries_stay_so(axis: str, getter: Getter) -> None:
    xy = pl.Struct({"x": pl.Float64, "y": pl.Float64})
    lines = pl.Series("line", [None, [], [{"x": 1.0, "y": 2.0}]], pl.List(xy))
    polygons = pl.Series("polygon", [None, [], [[]]], pl.List(pl.List(xy)))
    df = pl.DataFrame([lines.ext.to(LineStringXY()), polygons.ext.to(PolygonXY())])

    out = df.select(getter("line"), getter("polygon"))

    assert out["line"].to_list() == [None, [], [{"x": 1.0, "y": 2.0}[axis]]]
    assert out["polygon"].to_list() == [None, [], [[]]]


@AXES
@pytest.mark.parametrize("geometry", ["point", "line"])
def test_drops_the_crs(axis: str, getter: Getter, geometry: str) -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )
    if geometry == "line":
        df = df.select(geo.linestring(pl.col("point").implode()).alias("line"))

    dtype = df.select(getter(geometry)).schema[geometry]
    assert dtype == {"point": pl.Float64, "line": pl.List(pl.Float64)}[geometry]


@AXES
def test_refuses_a_non_geometry_while_resolving_the_schema(
    axis: str, getter: Getter
) -> None:
    lf = pl.LazyFrame({"point": [1.0]}).select(getter("point"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


@AXES
def test_the_namespace_matches_the_function(
    axis: str, getter: Getter, line_coords: pl.DataFrame
) -> None:
    df = XY.lines(line_coords)
    method = getattr(gpl.col("line").geo, axis)

    assert_series_equal(
        df.select(method()).to_series(),
        df.select(getter("line")).to_series(),
    )
