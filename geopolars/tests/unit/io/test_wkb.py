"""Converting between `geoarrow.wkb` and the native geometries.
Verificationg against Shapely (GEOS)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal, assert_series_equal

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PointXYZ,
    PolygonType,
    Wkb,
)
from tests.unit.conftest import XY, Dimension
from tests.unit.io._geometries import (
    GEOMETRIES,
    column,
    kind_dtype,
    shapely_wkb,
)

if TYPE_CHECKING:
    from pathlib import Path


def _binary(values: list[bytes | None]) -> pl.Series:
    return pl.Series("g", values, dtype=pl.Binary)


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_writes_what_shapely_writes(kind: str, dimension: Dimension) -> None:
    out = pl.select(geo.to_wkb(column(kind, dimension))).to_series()

    assert out.dtype == Wkb()
    assert out.ext.storage().to_list() == [shapely_wkb(kind, dimension), None]


@pytest.mark.parametrize("byte_order", [0, 1], ids=["big-endian", "little-endian"])
@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_reads_what_shapely_writes(
    kind: str, dimension: Dimension, byte_order: int
) -> None:
    wkb = _binary([shapely_wkb(kind, dimension, byte_order=byte_order), None])
    out = pl.select(geo.from_wkb(wkb, kind_dtype(kind, dimension))).to_series()

    assert_series_equal(out, column(kind, dimension))


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_round_trip_keeps_the_crs(kind: str, dimension: Dimension) -> None:
    geometries = column(kind, dimension, crs="EPSG:4326")
    wkb = pl.select(geo.to_wkb(geometries)).to_series()
    assert wkb.dtype == Wkb(crs="EPSG:4326")

    back = pl.select(geo.from_wkb(wkb, kind_dtype(kind, dimension))).to_series()
    assert back.dtype == geometries.dtype
    assert_series_equal(back, geometries)


def test_round_trip_of_built_geometries(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Geometries out of the constructors, chunked and grouped like real data."""
    df = dimension.multipolygons(multipolygon_coords)
    back = df.select(
        geo.to_wkb("multipolygon").geo.from_wkb(dimension.multipolygon_dtype)
    )
    assert_frame_equal(back, df)


def test_namespace() -> None:
    geometries = column("linestring", XY)
    back = geometries.to_frame().select(
        pl.col("g").geo.to_wkb().geo.from_wkb(LineStringXY)
    )
    assert_series_equal(back.to_series(), geometries)


def test_a_slice_is_encoded_from_its_own_offsets(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    lines = dimension.lines(line_coords)["line"]
    out = pl.select(geo.to_wkb(lines.slice(1, 1))).to_series()
    back = pl.select(geo.from_wkb(out, dimension.linestring_dtype)).to_series()
    assert_series_equal(back, lines.slice(1, 1))


@pytest.mark.parametrize(
    ("wkt", "dtype", "expected"),
    [
        ("POINT (1 2)", MultiPointXY, [{"x": 1.0, "y": 2.0}]),
        (
            "LINESTRING (1 2, 3 4)",
            MultiLineStringXY,
            [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]],
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 0))",
            MultiPolygonXY,
            [[[{"x": x, "y": y} for x, y in [(0, 0), (1, 0), (1, 1), (0, 0)]]]],
        ),
    ],
)
def test_a_single_geometry_is_promoted_into_its_multi_geometry(
    wkt: str, dtype: type[GeoArrowType], expected: Any
) -> None:
    out = pl.select(
        geo.from_wkb(_binary([shapely.to_wkb(shapely.from_wkt(wkt))]), dtype)
    )
    assert out.to_series().ext.storage().to_list() == [expected]


def test_polygons_and_multipolygons_mix() -> None:
    """The common case in GeoParquet files."""
    polygon = shapely.from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 0))")
    multipolygon = shapely.from_wkt(
        "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 0)), ((5 5, 6 5, 6 6, 5 5)))"
    )
    wkb = _binary(list(shapely.to_wkb([polygon, multipolygon])))
    out = pl.select(geo.from_wkb(wkb, MultiPolygonXY)).to_series()

    encoded = pl.select(geo.to_wkb(out)).to_series().ext.storage().to_list()
    assert list(shapely.from_wkb(encoded)) == [
        shapely.multipolygons([polygon]),
        multipolygon,
    ]


def test_another_geometry_is_an_error() -> None:
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 0))"))])
    with pytest.raises(pl.exceptions.ComputeError, match="cannot decode a WKB Polygon"):
        pl.select(geo.from_wkb(wkb, LineStringXY))


def test_a_multi_geometry_does_not_demote() -> None:
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("MULTIPOINT ((1 2))"))])
    with pytest.raises(
        pl.exceptions.ComputeError, match="cannot decode a WKB MultiPoint"
    ):
        pl.select(geo.from_wkb(wkb, PointXY))


def test_another_dimension_is_an_error() -> None:
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("POINT Z (1 2 3)"))])
    with pytest.raises(pl.exceptions.ComputeError, match="xyz coordinates"):
        pl.select(geo.from_wkb(wkb, PointXY))
    out = pl.select(geo.from_wkb(wkb, PointXYZ)).to_series()
    assert out.ext.storage().to_list() == [{"x": 1.0, "y": 2.0, "z": 3.0}]


def test_invalid_wkb_is_an_error() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="invalid WKB"):
        pl.select(geo.from_wkb(_binary([b"\x01\x02"]), PointXY))


def test_nulls_and_empties() -> None:
    """A null stays null, an empty geometry stays empty (not null)."""
    empties = ["LINESTRING EMPTY", "LINESTRING (1 2, 3 4)"]
    wkb = _binary([*shapely.to_wkb([shapely.from_wkt(w) for w in empties]), None])
    out = pl.select(geo.from_wkb(wkb, LineStringXY)).to_series()
    assert out.ext.storage().to_list() == [
        [],
        [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        None,
    ]

    encoded = pl.select(geo.to_wkb(out)).to_series().ext.storage().to_list()
    assert encoded[2] is None
    assert [shapely.to_wkt(g) for g in shapely.from_wkb(encoded[:2])] == empties


def test_an_empty_point_is_nan() -> None:
    """GeoArrow, like WKB, spells an empty point as NaN coordinates."""
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("POINT EMPTY"))])
    out = pl.select(geo.from_wkb(wkb, PointXY)).to_series()
    coordinates = out.ext.storage().struct.unnest()
    assert coordinates["x"].is_nan().all()
    assert coordinates["y"].is_nan().all()
    assert out.null_count() == 0

    # Promoted, it is an empty multipoint rather than a multipoint of one NaN point.
    multi = pl.select(geo.from_wkb(wkb, MultiPointXY)).to_series()
    assert multi.ext.storage().to_list() == [[]]


def test_plain_binary_takes_a_crs() -> None:
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("POINT (1 2)"))])
    out = pl.select(geo.from_wkb(wkb, PointXY, crs="EPSG:4326")).to_series()
    assert out.dtype == PointXY(crs="EPSG:4326")


def test_a_conflicting_crs_is_an_error() -> None:
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("POINT (1 2)"))]).ext.to(
        Wkb(crs="EPSG:4326")
    )
    with pytest.raises(pl.exceptions.ComputeError, match="EPSG:4326"):
        pl.select(geo.from_wkb(wkb, PointXY, crs="EPSG:28992"))


def test_the_type_has_to_be_concrete() -> None:
    with pytest.raises(TypeError, match="concrete geometry type"):
        geo.from_wkb("g", PolygonType)


def test_from_wkb_wants_binary() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="binary column"):
        pl.select(geo.from_wkb(pl.Series("g", ["POINT (1 2)"]), PointXY))


def test_to_wkb_wants_a_geometry() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="expected a `geoarrow.point`"):
        pl.select(geo.to_wkb(_binary([b""])))


def test_operations_do_not_take_wkb() -> None:
    wkb = pl.select(geo.to_wkb(column("polygon", XY)))
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        wkb.select(geo.area("g"))


def test_wkb_survives_a_parquet_round_trip(tmp_path: Path) -> None:
    wkb = _binary([shapely.to_wkb(shapely.from_wkt("POINT (1 2)")), None]).ext.to(
        Wkb(crs="EPSG:4326")
    )
    path = tmp_path / "wkb.parquet"
    wkb.to_frame().write_parquet(path)
    back = pl.read_parquet(path)

    assert back.schema["g"] == Wkb(crs="EPSG:4326")
    assert_frame_equal(back, wkb.to_frame())
