"""Adding or dropping the `z` and `m` coordinates of a geometry."""

from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import LineStringXY, PolygonXY
from tests.unit.conftest import DIMENSIONS, XY, XYM, XYZ, XYZM, Dimension

Op = Callable[[str], pl.Expr]

# Each op, with whether it sets `z` and `m` (`None` leaves it as it is).
OPS = pytest.mark.parametrize(
    ("op", "z", "m"),
    [
        (geo.force_2d, False, None),
        (geo.force_3d, True, None),
        (geo.without_m, None, False),
        (geo.force_m, None, True),
    ],
    ids=["force_2d", "force_3d", "without_m", "force_m"],
)


def _result_dimension(
    dimension: Dimension, z: bool | None, m: bool | None
) -> Dimension:
    has_z = dimension.has_z if z is None else z
    has_m = dimension.has_m if m is None else m
    return next(d for d in DIMENSIONS if (d.has_z, d.has_m) == (has_z, has_m))


def _expected_coordinates(
    vertices: pl.DataFrame, dimension: Dimension, result: Dimension
) -> pl.DataFrame:
    """The vertices on the axes of `result`: kept ones as they are, added ones `0.0`."""
    return vertices.select(
        pl.col(axis) if axis in dimension.coords else pl.lit(0.0).alias(axis)
        for axis in result.coords
    )


def _flat(df: pl.DataFrame, name: str, nesting: int) -> pl.DataFrame:
    flat = pl.col(name).ext.storage()
    for _ in range(nesting):
        flat = flat.explode(empty_as_null=False)
    return df.select(flat).to_series().struct.unnest()


@OPS
def test_a_point(
    op: Op, z: bool | None, m: bool | None, coords: pl.DataFrame, dimension: Dimension
) -> None:
    result = _result_dimension(dimension, z, m)
    out = coords.select(dimension.point()).select(op("point"))

    assert out.schema["point"] == result.point_dtype()
    assert_frame_equal(
        _flat(out, "point", 0), _expected_coordinates(coords, dimension, result)
    )


@OPS
@pytest.mark.parametrize(
    ("geometry", "dtype", "vertices", "nesting"),
    [
        ("line", "linestring_dtype", "line_coords", 1),
        ("multipoint", "multipoint_dtype", "line_coords", 1),
        ("polygon", "polygon_dtype", "ring_coords", 2),
        ("multilinestring", "multilinestring_dtype", "ring_coords", 2),
        ("multipolygon", "multipolygon_dtype", "multipolygon_coords", 3),
    ],
)
def test_a_nested_geometry(
    op: Op,
    z: bool | None,
    m: bool | None,
    geometry: str,
    dtype: str,
    vertices: str,
    nesting: int,
    dimension: Dimension,
    request: pytest.FixtureRequest,
) -> None:
    coords = request.getfixturevalue(vertices)
    build = {
        "line": dimension.lines,
        "multipoint": dimension.multipoints,
        "polygon": dimension.polygons,
        "multilinestring": dimension.multilinestrings,
        "multipolygon": dimension.multipolygons,
    }[geometry]
    df = build(coords)
    result = _result_dimension(dimension, z, m)

    out = df.select(op(geometry))

    assert out.schema[geometry] == getattr(result, dtype)()
    assert_frame_equal(
        _flat(out, geometry, nesting), _expected_coordinates(coords, dimension, result)
    )
    # The nesting itself is untouched.
    assert_series_equal(
        out.select(geo.x(geometry)).to_series(), df.select(geo.x(geometry)).to_series()
    )


@OPS
def test_already_there_returns_the_input(
    op: Op, z: bool | None, m: bool | None, line_coords: pl.DataFrame
) -> None:
    dimension = next(
        d for d in DIMENSIONS if _result_dimension(d, z, m) == d and d != XY
    )
    df = dimension.lines(line_coords)

    assert_frame_equal(df.select(op("line")), df)


@pytest.mark.parametrize("op", [geo.force_3d, geo.force_m])
def test_an_empty_point_stays_empty(op: Op) -> None:
    df = pl.DataFrame({"x": [math.nan, 1.0], "y": [math.nan, 2.0]}).select(XY.point())

    out = df.select(geo.is_empty(op("point"))).to_series()

    assert out.to_list() == [True, False]
    added = _flat(df.select(op("point")), "point", 0).drop("x", "y").to_series()
    assert math.isnan(added[0])
    assert added[1] == 0.0


@OPS
def test_missing_and_empty_geometries_stay_so(
    op: Op, z: bool | None, m: bool | None
) -> None:
    xy = pl.Struct({"x": pl.Float64, "y": pl.Float64})
    points = pl.Series("point", [None, {"x": 1.0, "y": 2.0}], xy)
    lines = pl.Series("line", [None, []], pl.List(xy))
    polygons = pl.Series("polygon", [None, [], [[]]], pl.List(pl.List(xy)))
    point = points.ext.to(XY.point_dtype()).to_frame().select(op("point"))
    line = lines.ext.to(LineStringXY()).to_frame().select(op("line"))
    polygon = polygons.ext.to(PolygonXY()).to_frame().select(op("polygon"))

    assert point.to_series().is_null().to_list() == [True, False]
    assert line.to_series().ext.storage().to_list() == [None, []]
    assert polygon.to_series().ext.storage().to_list() == [None, [], [[]]]


@OPS
def test_keeps_the_crs(op: Op, z: bool | None, m: bool | None) -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0], "m": [3.0]}).select(
        geo.point("x", "y", m="m", crs="EPSG:4326").alias("point")
    )

    dtype = df.select(op("point")).schema["point"]

    assert dtype._crs() == "EPSG:4326"


def test_toggles_z_and_m_independently(line_coords: pl.DataFrame) -> None:
    df = XYZM.lines(line_coords)

    assert df.select(geo.force_2d("line")).schema["line"] == XYM.linestring_dtype()
    assert df.select(geo.without_m("line")).schema["line"] == XYZ.linestring_dtype()
    df = XYM.lines(line_coords)
    assert df.select(geo.force_3d("line")).schema["line"] == XYZM.linestring_dtype()


@OPS
def test_the_namespace_matches_the_function(
    op: Op, z: bool | None, m: bool | None, line_coords: pl.DataFrame
) -> None:
    df = XYZ.lines(line_coords)
    method = getattr(gpl.col("line").geo, op.__name__)

    assert_frame_equal(df.select(method()), df.select(op("line")))


def test_refuses_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"point": [1.0]}).select(geo.force_3d("point"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
