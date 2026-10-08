"""Converting between `geoarrow.wkt` and the native geometries.
Verified against Shapely (GEOS).

Our text is not spelled like Shapely's (`POINT(1 2)` against `POINT (1 2)`),
so what we write is compared by the geometry Shapely reads from it.
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
    Wkt,
)
from tests.unit.conftest import XY, Dimension
from tests.unit.io._geometries import (
    GEOMETRIES,
    column,
    kind_dtype,
    reference_wkt,
    shapely_wkb,
)

if TYPE_CHECKING:
    from pathlib import Path


def _string(values: list[str | None]) -> pl.Series:
    return pl.Series("g", values, dtype=pl.String)


def _geometries(dtype: GeoArrowType, values: list[Any]) -> pl.Series:
    return pl.Series("g", values, dtype=dtype.ext_storage()).ext.to(dtype)


def _as_shapely_wkb(wkt: str) -> bytes:
    """What Shapely makes of `wkt`, comparable to `shapely_wkb`."""
    return shapely.to_wkb(  # type: ignore[no-any-return]
        shapely.from_wkt(wkt), flavor="iso", output_dimension=4
    )


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_writes_what_shapely_reads(kind: str, dimension: Dimension) -> None:
    out = pl.select(geo.to_wkt(column(kind, dimension))).to_series()

    assert out.dtype == Wkt()
    wkt, null = out.ext.storage().to_list()
    assert null is None
    assert _as_shapely_wkb(wkt) == shapely_wkb(kind, dimension)


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_reads_what_shapely_writes(kind: str, dimension: Dimension) -> None:
    geometry = shapely.from_wkt(reference_wkt(kind, dimension))
    wkt = shapely.to_wkt(geometry, rounding_precision=-1, output_dimension=4)
    out = pl.select(geo.from_wkt(_string([wkt, None]), kind_dtype(kind, dimension)))

    assert_series_equal(out.to_series(), column(kind, dimension))


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_round_trip_keeps_the_crs(kind: str, dimension: Dimension) -> None:
    geometries = column(kind, dimension, crs="EPSG:4326")
    wkt = pl.select(geo.to_wkt(geometries)).to_series()
    assert wkt.dtype == Wkt(crs="EPSG:4326")

    back = pl.select(geo.from_wkt(wkt, kind_dtype(kind, dimension))).to_series()
    assert back.dtype == geometries.dtype
    assert_series_equal(back, geometries)


def test_round_trip_of_built_geometries(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Geometries out of the constructors, chunked and grouped like real data."""
    df = dimension.multipolygons(multipolygon_coords)
    back = df.select(
        geo.to_wkt("multipolygon").geo.from_wkt(dimension.multipolygon_dtype)
    )
    assert_frame_equal(back, df)


def test_coordinates_round_trip_exactly() -> None:
    xs = [0.1 + 0.2, 1e-300, -1.7976931348623157e308, 123456789.123456789]
    points = _geometries(PointXY(), [{"x": x, "y": -x} for x in xs])
    back = pl.select(geo.to_wkt(points).geo.from_wkt(PointXY)).to_series()
    assert_series_equal(back, points)


def test_namespace() -> None:
    geometries = column("linestring", XY)
    back = geometries.to_frame().select(
        pl.col("g").geo.to_wkt().geo.from_wkt(LineStringXY)
    )
    assert_series_equal(back.to_series(), geometries)


def test_a_slice_is_encoded_from_its_own_offsets(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    lines = dimension.lines(line_coords)["line"]
    out = pl.select(geo.to_wkt(lines.slice(1, 1))).to_series()
    back = pl.select(geo.from_wkt(out, dimension.linestring_dtype)).to_series()
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
    out = pl.select(geo.from_wkt(_string([wkt]), dtype))
    assert out.to_series().ext.storage().to_list() == [expected]


def test_polygons_and_multipolygons_mix() -> None:
    polygon = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    multipolygon = "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 0)), ((5 5, 6 5, 6 6, 5 5)))"
    wkt = _string([polygon, multipolygon])
    out = pl.select(geo.from_wkt(wkt, MultiPolygonXY)).to_series()

    encoded = pl.select(geo.to_wkt(out)).to_series().ext.storage().to_list()
    assert list(shapely.from_wkt(encoded)) == [
        shapely.multipolygons([shapely.from_wkt(polygon)]),
        shapely.from_wkt(multipolygon),
    ]


def test_another_geometry_is_an_error() -> None:
    wkt = _string(["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    with pytest.raises(pl.exceptions.ComputeError, match="cannot decode a WKT Polygon"):
        pl.select(geo.from_wkt(wkt, LineStringXY))


def test_a_geometry_collection_is_an_error() -> None:
    wkt = _string(["GEOMETRYCOLLECTION (POINT (1 2))"])
    with pytest.raises(
        pl.exceptions.ComputeError, match="cannot decode a WKT GeometryCollection"
    ):
        pl.select(geo.from_wkt(wkt, PointXY))


def test_a_multi_geometry_does_not_demote() -> None:
    with pytest.raises(
        pl.exceptions.ComputeError, match="cannot decode a WKT MultiPoint"
    ):
        pl.select(geo.from_wkt(_string(["MULTIPOINT ((1 2))"]), PointXY))


def test_another_dimension_is_an_error() -> None:
    wkt = _string(["POINT Z (1 2 3)"])
    with pytest.raises(pl.exceptions.ComputeError, match="xyz coordinates"):
        pl.select(geo.from_wkt(wkt, PointXY))
    out = pl.select(geo.from_wkt(wkt, PointXYZ)).to_series()
    assert out.ext.storage().to_list() == [{"x": 1.0, "y": 2.0, "z": 3.0}]


def test_invalid_wkt_is_an_error() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="invalid WKT"):
        pl.select(geo.from_wkt(_string(["POINT (1"]), PointXY))


def test_nulls_and_empties() -> None:
    """A null stays null, an empty geometry stays empty (not null)."""
    empties = ["LINESTRING EMPTY", "LINESTRING (1 2, 3 4)"]
    out = pl.select(geo.from_wkt(_string([*empties, None]), LineStringXY)).to_series()
    assert out.ext.storage().to_list() == [
        [],
        [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        None,
    ]

    encoded = pl.select(geo.to_wkt(out)).to_series().ext.storage().to_list()
    assert encoded[2] is None
    assert [shapely.to_wkt(g) for g in shapely.from_wkt(encoded[:2])] == empties


def test_an_empty_point_is_nan() -> None:
    """GeoArrow spells an empty point as NaN coordinates, WKT as `POINT EMPTY`."""
    out = pl.select(geo.from_wkt(_string(["POINT EMPTY"]), PointXY)).to_series()
    coordinates = out.ext.storage().struct.unnest()
    assert coordinates["x"].is_nan().all()
    assert coordinates["y"].is_nan().all()
    assert out.null_count() == 0

    encoded = pl.select(geo.to_wkt(out)).to_series().ext.storage().to_list()
    assert shapely.from_wkt(encoded[0]).is_empty

    # Promoted, it is an empty multipoint rather than a multipoint of one NaN point.
    multi = pl.select(geo.from_wkt(_string(["POINT EMPTY"]), MultiPointXY))
    assert multi.to_series().ext.storage().to_list() == [[]]


def test_a_nan_point_in_a_multipoint_is_written() -> None:
    """WKT cannot spell an empty point inside a multipoint, so it stays NaN."""
    nan = float("nan")
    multi = _geometries(MultiPointXY(), [[{"x": nan, "y": nan}, {"x": 1.0, "y": 2.0}]])
    encoded = pl.select(geo.to_wkt(multi)).to_series().ext.storage().to_list()
    assert len(shapely.get_parts(shapely.from_wkt(encoded[0]))) == 2


def test_plain_string_takes_a_crs() -> None:
    out = pl.select(geo.from_wkt(_string(["POINT (1 2)"]), PointXY, crs="EPSG:4326"))
    assert out.to_series().dtype == PointXY(crs="EPSG:4326")


def test_a_conflicting_crs_is_an_error() -> None:
    wkt = _string(["POINT (1 2)"]).ext.to(Wkt(crs="EPSG:4326"))
    with pytest.raises(pl.exceptions.ComputeError, match="EPSG:4326"):
        pl.select(geo.from_wkt(wkt, PointXY, crs="EPSG:28992"))


def test_the_type_has_to_be_concrete() -> None:
    with pytest.raises(TypeError, match="concrete geometry type"):
        geo.from_wkt("g", PolygonType)


def test_from_wkt_wants_strings() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="string column"):
        pl.select(geo.from_wkt(pl.Series("g", [b"POINT (1 2)"]), PointXY))


def test_wkb_is_not_wkt() -> None:
    wkb = pl.select(geo.to_wkb(column("point", XY))).to_series()
    with pytest.raises(pl.exceptions.ComputeError, match="`geoarrow.wkt` or string"):
        pl.select(geo.from_wkt(wkb, PointXY))


def test_to_wkt_wants_a_geometry() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="expected a `geoarrow.point`"):
        pl.select(geo.to_wkt(_string(["POINT (1 2)"])))


def test_operations_do_not_take_wkt() -> None:
    wkt = pl.select(geo.to_wkt(column("polygon", XY)))
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        wkt.select(geo.area("g"))


def test_wkt_survives_a_parquet_round_trip(tmp_path: Path) -> None:
    wkt = _string(["POINT (1 2)", None]).ext.to(Wkt(crs="EPSG:4326"))
    path = tmp_path / "wkt.parquet"
    wkt.to_frame().write_parquet(path)
    back = pl.read_parquet(path)

    assert back.schema["g"] == Wkt(crs="EPSG:4326")
    assert_frame_equal(back, wkt.to_frame())
