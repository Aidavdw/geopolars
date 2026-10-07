"""Writing GeoParquet"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars.datatypes import (
    GeoPoint,
    MultiLineStringXYZ,
    PointXY,
    PointXYM,
    PointXYZ,
    PolygonXYZM,
    Wkb,
    Wkt,
)
from tests.unit.io.test_geoparquet import DATA, SPEC_GEOMETRIES

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import PolarsDataType

CRS84 = "OGC:CRS84"
XY = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]


def _point(metadata: str) -> PolarsDataType:
    """A `PointXY` with metadata the constructor would not write."""
    return GeoPoint.ext_from_params("geoarrow.point", PointXY().ext_storage(), metadata)


def _geo_of(path: Path) -> dict[str, Any]:
    return json.loads(pl.read_parquet_metadata(path)["geo"])


def _written(tmp_path: Path, dtype: PolarsDataType) -> dict[str, Any]:
    """The `geo` metadata written for a `geometry` column of `dtype`."""
    path = tmp_path / "written.parquet"
    gpl.write_parquet(pl.DataFrame(schema={"geometry": dtype}), path)
    return _geo_of(path)["columns"]["geometry"]


def test_writes_a_native_column(tmp_path: Path) -> None:
    path = tmp_path / "points.parquet"
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs="EPSG:4326"))
    )
    gpl.write_parquet(df, path)
    geo = _geo_of(path)

    assert geo["version"] == "1.1.0"
    assert geo["primary_column"] == "geometry"
    column = geo["columns"]["geometry"]
    assert column["encoding"] == "point"
    assert column["geometry_types"] == ["Point"]
    # Converted by PROJ, as GeoParquet only takes PROJJSON.
    assert column["crs"]["id"] == {"authority": "EPSG", "code": 4326}
    assert "edges" not in column
    assert "epoch" not in column


def test_a_z_coordinate_is_in_the_geometry_type(tmp_path: Path) -> None:
    column = _written(tmp_path, MultiLineStringXYZ())
    assert column["encoding"] == "multilinestring"
    assert column["geometry_types"] == ["MultiLineString Z"]


def test_no_crs_is_written_as_null(tmp_path: Path) -> None:
    """Leaving the key out would claim the data is in OGC:CRS84."""
    for dtype in [PointXY(), _point('{"crs":null}')]:
        column = _written(tmp_path, dtype)
        assert "crs" in column
        assert column["crs"] is None


def test_crs84_is_left_out(tmp_path: Path) -> None:
    """because it is the default"""
    assert "crs" not in _written(tmp_path, PointXY(crs=CRS84))


def test_writes_spherical_edges(tmp_path: Path) -> None:
    column = _written(tmp_path, _point('{"crs":"OGC:CRS84","edges":"spherical"}'))
    assert column["edges"] == "spherical"


def test_writes_wkb_without_geometry_types(tmp_path: Path) -> None:
    """Knowing them would mean reading every value. `[]` says they are unknown."""
    column = _written(tmp_path, Wkb())
    assert column["encoding"] == "WKB"
    assert column["geometry_types"] == []


@pytest.mark.parametrize(
    ("dtype", "error", "match"),
    [
        (PointXYM(), pl.exceptions.SchemaError, "M values"),
        (PolygonXYZM(), pl.exceptions.SchemaError, "M values"),
        (Wkt(), pl.exceptions.SchemaError, "WKT"),
        (
            _point('{"crs":"4326","crs_type":"srid"}'),
            pl.exceptions.ComputeError,
            "SRID",
        ),
        (_point('{"crs":"not a crs"}'), pl.exceptions.ComputeError, "PROJ"),
        (_point('{"edges":"karney"}'), pl.exceptions.ComputeError, "karney"),
        (_point('{"edges":"geodesic-ish"}'), pl.exceptions.ComputeError, "edges"),
    ],
)
def test_refuses_what_geoparquet_cannot_hold(
    tmp_path: Path, dtype: PolarsDataType, error: type[Exception], match: str
) -> None:
    path = tmp_path / "refused.parquet"
    with pytest.raises(error, match=match):
        gpl.write_parquet(pl.DataFrame(schema={"geometry": dtype}), path)
    assert not path.exists()


def test_describes_every_geometry_column(tmp_path: Path) -> None:
    path = tmp_path / "columns.parquet"
    schema = {"id": pl.Int64, "a": PointXY(), "name": pl.String, "b": Wkb()}
    gpl.write_parquet(pl.DataFrame(schema=schema), path)
    geo = _geo_of(path)

    assert geo["primary_column"] == "a"
    assert list(geo["columns"]) == ["a", "b"]


def test_a_frame_without_geometry_is_plain_parquet(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    gpl.write_parquet(pl.DataFrame({"id": [1]}), path, metadata={"a": "b"})
    metadata = pl.read_parquet_metadata(path)
    assert metadata["a"] == "b"
    assert "geo" not in metadata


def _frames() -> list[pl.DataFrame]:
    xyz = [{"x": 1.0, "y": 2.0, "z": 3.0}, None]
    wkb = shapely.to_wkb(shapely.points([[1.0, 2.0], [3.0, 4.0]]))
    projjson = json.dumps({"type": "GeographicCRS", "name": "WGS 84"})
    spherical = _point('{"crs":"OGC:CRS84","edges":"spherical"}')
    return [
        pl.DataFrame({"geometry": XY}).select(pl.col("geometry").ext.to(dtype))
        for dtype in [PointXY(crs=CRS84), PointXY(), PointXY(crs=projjson), spherical]
    ] + [
        pl.DataFrame({"geometry": xyz}).select(pl.col("geometry").ext.to(PointXYZ())),
        pl.DataFrame({"geometry": wkb}, schema={"geometry": pl.Binary}).select(
            pl.col("geometry").ext.to(Wkb(crs=CRS84))
        ),
        pl.DataFrame({"geometry": [*wkb, None]}, schema={"geometry": pl.Binary}).select(
            pl.col("geometry").ext.to(Wkb())
        ),
    ]


@pytest.mark.parametrize("df", _frames())
def test_round_trips(tmp_path: Path, df: pl.DataFrame) -> None:
    path = tmp_path / "round_trip.parquet"
    gpl.write_parquet(df, path)
    assert_frame_equal(gpl.read_parquet(path), df)


def test_a_crs_comes_back_as_projjson(tmp_path: Path) -> None:
    """GeoParquet only holds PROJJSON, so that is what a file gives back."""
    path = tmp_path / "epsg.parquet"
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs="EPSG:4326"))
    )
    gpl.write_parquet(df, path)
    back = gpl.read_parquet(path).schema["geometry"]

    assert back != df.schema["geometry"]
    crs = json.loads(back.ext_metadata())["crs"]
    assert crs["id"] == {"authority": "EPSG", "code": 4326}


@pytest.mark.parametrize("name", SPEC_GEOMETRIES)
@pytest.mark.parametrize("encoding", ["native", "wkb"])
def test_the_spec_test_data_round_trips(
    tmp_path: Path, name: str, encoding: str
) -> None:
    df = gpl.read_parquet(DATA / f"data-{name}-encoding_{encoding}.parquet")
    path = tmp_path / "rewritten.parquet"
    gpl.write_parquet(df, path)
    assert_frame_equal(gpl.read_parquet(path), df)


def test_sink_writes_the_same_metadata(tmp_path: Path) -> None:
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs=CRS84))
    )
    written, sunk = tmp_path / "written.parquet", tmp_path / "sunk.parquet"
    gpl.write_parquet(df, written)

    lf = gpl.sink_parquet(df.lazy(), sunk, lazy=True)
    assert isinstance(lf, pl.LazyFrame)
    assert not sunk.exists()
    lf.collect()

    assert _geo_of(sunk) == _geo_of(written)
    assert_frame_equal(gpl.read_parquet(sunk), df)


def test_sink_refuses_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "refused.parquet"
    lf = pl.LazyFrame(schema={"geometry": PointXYM()})
    with pytest.raises(pl.exceptions.SchemaError, match="M values"):
        gpl.sink_parquet(lf, path)
    assert not path.exists()


def test_keeps_the_users_own_metadata(tmp_path: Path) -> None:
    df = pl.DataFrame(schema={"geometry": PointXY()})
    by_dict, by_callback = tmp_path / "dict.parquet", tmp_path / "callback.parquet"
    gpl.write_parquet(df, by_dict, metadata={"owner": "me"})
    gpl.write_parquet(df, by_callback, metadata=lambda _: {"owner": "me"})

    for path in [by_dict, by_callback]:
        metadata = pl.read_parquet_metadata(path)
        assert metadata["owner"] == "me"
        assert "geo" in metadata


def test_refuses_the_users_own_geo_key(tmp_path: Path) -> None:
    df = pl.DataFrame(schema={"geometry": PointXY()})
    with pytest.raises(ValueError, match="`geo`"):
        gpl.write_parquet(df, tmp_path / "dict.parquet", metadata={"geo": "{}"})
    # A callback only runs while writing, and Polars wraps what it raises.
    with pytest.raises(pl.exceptions.ComputeError, match="`geo`"):
        gpl.write_parquet(
            df, tmp_path / "callback.parquet", metadata=lambda _: {"geo": "{}"}
        )
