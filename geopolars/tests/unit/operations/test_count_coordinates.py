"""How many coordinates a geometry is made of."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringType,
    MultiLineStringType,
    MultiPointType,
    MultiPolygonType,
    PolygonType,
)
from tests.unit.conftest import XY, Dimension


def _nested(dimension: Dimension, layers: int) -> pl.DataType:
    """The storage of a geometry `layers` lists deep."""
    dtype: pl.DataType = pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))
    for _ in range(layers):
        dtype = pl.List(dtype)
    return dtype


def test_a_point_counts_one(coords: pl.DataFrame, dimension: Dimension) -> None:
    out = coords.select(dimension.point()).select(geo.count_coordinates("point"))

    assert out["point"].to_list() == [1] * coords.height


def test_an_empty_point_counts_zero_and_a_missing_one_null(
    dimension: Dimension,
) -> None:
    """GeoArrow spells the empty point as NaN `x` and `y`."""
    nan = math.nan
    df = pl.DataFrame({axis: [nan, 1.0, None] for axis in dimension.coords}).select(
        dimension.point()
    )

    out = df.select(geo.count_coordinates("point"))

    assert out["point"].to_list() == [0, 1, None]


def test_a_linestring_counts_its_vertices(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.lines(line_coords).select(geo.count_coordinates("line"))

    assert out["line"].to_list() == [3, 2]


def test_a_multipoint_counts_its_points(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    out = df.select(geo.count_coordinates("multipoint"))

    assert out["multipoint"].to_list() == [3, 2]


def test_a_polygon_skips_the_closing_coordinate_of_each_ring(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """`a` has rings of 5 and 4 stored coordinates, `b` one of 4."""
    out = dimension.polygons(ring_coords).select(geo.count_coordinates("polygon"))

    assert out["polygon"].to_list() == [4 + 3, 3]


def test_a_multilinestring_counts_every_vertex(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The same vertices as the polygons, but a linestring is not a ring."""
    df = dimension.multilinestrings(ring_coords)
    out = df.select(geo.count_coordinates("multilinestring"))

    assert out["multilinestring"].to_list() == [5 + 4, 4]


def test_a_multipolygon_counts_over_all_its_polygons(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)
    out = df.select(geo.count_coordinates("multipolygon"))

    assert out["multipolygon"].to_list() == [(4 + 3) + 3, 4]


@pytest.mark.parametrize(
    ("geometry", "layers", "rows"),
    [
        (LineStringType, 1, [[], None]),
        (MultiPointType, 1, [[], None]),
        (PolygonType, 2, [[], [[]], [[], []], None]),
        (MultiLineStringType, 2, [[], [[]], [[], []], None]),
        (MultiPolygonType, 3, [[], [[]], [[[]]], [[[]], [[], []]], None]),
    ],
    ids=["linestring", "multipoint", "polygon", "multilinestring", "multipolygon"],
)
def test_no_coordinates_counts_zero_and_missing_is_null(
    geometry: type[GeoArrowType], layers: int, rows: list, dimension: Dimension
) -> None:
    # Relabelled rather than built: a constructor refuses a ring without vertices,
    # but one can still arrive around it (e.g. from a file).
    dtype = geometry.of_dimension(dimension.coords)()
    df = pl.DataFrame(
        {"storage": rows}, schema={"storage": _nested(dimension, layers)}
    ).select(pl.col("storage").ext.to(dtype).alias("geometry"))

    out = df.select(geo.count_coordinates("geometry"))

    assert out["geometry"].to_list() == [0] * (len(rows) - 1) + [None]


@pytest.mark.parametrize("geometry", ["point", "line", "polygon"])
def test_gives_a_u32_and_drops_the_crs(geometry: str) -> None:
    df = pl.DataFrame({"x": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )
    if geometry != "point":
        df = df.select(geo.line_string(pl.col("point").implode()).alias("line"))
    if geometry == "polygon":
        df = df.select(geo.polygon(pl.col("line").implode()).alias("polygon"))

    lf = df.lazy().select(geo.count_coordinates(geometry))

    assert lf.collect_schema() == pl.Schema({geometry: pl.UInt32})
    assert lf.collect()[geometry].dtype == pl.UInt32


def test_refuses_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"point": [1.0]}).select(geo.count_coordinates("point"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)

    assert_series_equal(
        df.select(gpl.col("polygon").geo.count_coordinates()).to_series(),
        df.select(geo.count_coordinates("polygon")).to_series(),
    )
