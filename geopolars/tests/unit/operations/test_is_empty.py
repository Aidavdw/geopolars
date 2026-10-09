"""Whether a geometry holds anything at all."""

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
    LineStringXY,
    MultiLineStringType,
    MultiPointType,
    MultiPointXY,
    MultiPolygonType,
    MultiPolygonXY,
    PointXY,
    PolygonType,
    PolygonXY,
)
from tests.unit.conftest import XY, Dimension


def _coordinate(dimension: Dimension) -> pl.Struct:
    return pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))


def _nested(dimension: Dimension, layers: int) -> pl.DataType:
    """The storage of a geometry `layers` lists deep."""
    dtype: pl.DataType = _coordinate(dimension)
    for _ in range(layers):
        dtype = pl.List(dtype)
    return dtype


def test_a_point_is_not_empty(coords: pl.DataFrame, dimension: Dimension) -> None:
    out = coords.select(dimension.point()).select(geo.is_empty("point"))

    assert out["point"].to_list() == [False] * coords.height


def test_a_linestring_is_not_empty(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.lines(line_coords).select(geo.is_empty("line"))

    assert out["line"].to_list() == [False, False]


def test_a_multipoint_is_not_empty(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.multipoints(line_coords).select(geo.is_empty("multipoint"))

    assert out["multipoint"].to_list() == [False, False]


def test_a_polygon_is_not_empty(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.polygons(ring_coords).select(geo.is_empty("polygon"))

    assert out["polygon"].to_list() == [False, False]


def test_a_multilinestring_is_not_empty(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)
    out = df.select(geo.is_empty("multilinestring"))

    assert out["multilinestring"].to_list() == [False, False]


def test_a_multipolygon_is_not_empty(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)
    out = df.select(geo.is_empty("multipolygon"))

    assert out["multipolygon"].to_list() == [False, False]


def test_a_nan_point_is_empty(dimension: Dimension) -> None:
    """GeoArrow spells the empty point as NaN `x` and `y`."""
    nan = math.nan
    df = pl.DataFrame(
        {axis: [nan, nan, 1.0, None] for axis in dimension.coords}
        | {"y": [nan, 2.0, nan, None]}
    ).select(dimension.point())

    out = df.select(geo.is_empty("point"))

    assert out["point"].to_list() == [True, False, False, None]


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
def test_no_coordinates_is_empty_and_missing_is_null(
    geometry: type[GeoArrowType], layers: int, rows: list, dimension: Dimension
) -> None:
    # Relabelled rather than built: a constructor refuses a ring without vertices,
    # but one can still arrive around it (e.g. from a file).
    dtype = geometry.of_dimension(dimension.coords)()
    df = pl.DataFrame(
        {"storage": rows}, schema={"storage": _nested(dimension, layers)}
    ).select(pl.col("storage").ext.to(dtype).alias("geometry"))

    out = df.select(geo.is_empty("geometry"))

    assert out["geometry"].to_list() == [True] * (len(rows) - 1) + [None]


def test_one_part_with_coordinates_is_enough() -> None:
    """An empty part next to one with coordinates doesn't make the whole empty."""
    point = {"x": 1.0, "y": 2.0}
    df = pl.DataFrame(
        {"storage": [[[], [point]], [[point], []]]},
        schema={"storage": _nested(XY, 2)},
    ).select(geo.multi_line_string("storage").alias("multilinestring"))

    out = df.select(geo.is_empty("multilinestring"))

    assert out["multilinestring"].to_list() == [False, False]


@pytest.mark.parametrize(
    ("constructor", "name"),
    [(geo.line_string, "line"), (geo.multi_point, "multipoint")],
    ids=["linestring", "multipoint"],
)
def test_only_empty_points_is_empty(
    constructor: object, name: str, dimension: Dimension
) -> None:
    """One non-empty point is enough to make the whole non-empty."""
    empty = dict.fromkeys(dimension.coords, math.nan)
    point = dict.fromkeys(dimension.coords, 1.0)
    df = pl.DataFrame(
        {"storage": [[empty], [empty, empty], [empty, point], [point, empty]]},
        schema={"storage": _nested(dimension, 1)},
    ).select(constructor("storage").alias(name))  # type: ignore[operator]

    out = df.select(geo.is_empty(name))

    assert out[name].to_list() == [True, True, False, False]


@pytest.mark.parametrize(
    ("wkt", "dtype"),
    [
        ("POINT EMPTY", PointXY),
        ("LINESTRING EMPTY", LineStringXY),
        ("POLYGON EMPTY", PolygonXY),
        ("MULTIPOINT EMPTY", MultiPointXY),
        ("MULTIPOLYGON EMPTY", MultiPolygonXY),
    ],
)
def test_an_empty_wkt_reads_back_as_empty(wkt: str, dtype: type) -> None:
    wkts = pl.Series([wkt, wkt.replace("EMPTY", _filled(dtype))], dtype=pl.String)

    out = pl.select(geo.is_empty(geo.from_wkt(wkts, dtype))).to_series()

    assert out.to_list() == [True, False]


def _filled(dtype: type) -> str:
    """The body of a WKT geometry of `dtype` that is not empty."""
    return {
        PointXY: "(1 2)",
        LineStringXY: "(1 2, 3 4)",
        PolygonXY: "((0 0, 1 0, 1 1, 0 0))",
        MultiPointXY: "((1 2))",
        MultiPolygonXY: "(((0 0, 1 0, 1 1, 0 0)))",
    }[dtype]


@pytest.mark.parametrize("geometry", ["point", "line"])
def test_gives_a_boolean_and_drops_the_crs(geometry: str) -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )
    if geometry == "line":
        df = df.select(geo.line_string(pl.col("point").implode()).alias("line"))

    assert df.lazy().select(geo.is_empty(geometry)).collect_schema() == pl.Schema(
        {geometry: pl.Boolean}
    )


def test_refuses_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"point": [1.0]}).select(geo.is_empty("point"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords)

    assert_series_equal(
        df.select(gpl.col("line").geo.is_empty()).to_series(),
        df.select(geo.is_empty("line")).to_series(),
    )
