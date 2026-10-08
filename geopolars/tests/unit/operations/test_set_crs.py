"""Declaring the CRS a geometry's coordinates are in, without moving them."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import GeoArrowType, PointXY
from tests.unit.conftest import Dimension

RD = "EPSG:28992"
WGS84 = "EPSG:4326"


def _geometries(
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> list[tuple[pl.DataFrame, type[GeoArrowType]]]:
    """One column of every geometry, none declaring a CRS."""
    return [
        (coords.select(dimension.point()), dimension.point_dtype),
        (dimension.lines(line_coords), dimension.linestring_dtype),
        (dimension.polygons(ring_coords), dimension.polygon_dtype),
        (dimension.multipoints(line_coords), dimension.multipoint_dtype),
        (dimension.multilinestrings(ring_coords), dimension.multilinestring_dtype),
        (dimension.multipolygons(multipolygon_coords), dimension.multipolygon_dtype),
    ]


def _points(crs: str | None = None) -> pl.DataFrame:
    return pl.DataFrame({"x": [4.9, 5.1], "y": [52.4, 52.1]}).select(
        geo.point("x", "y", crs=crs).alias("point")
    )


def test_every_geometry_gets_the_crs_and_keeps_its_coordinates(
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    geometries = _geometries(
        dimension, coords, line_coords, ring_coords, multipolygon_coords
    )
    for df, dtype in geometries:
        (name,) = df.columns

        out = df.lazy().select(geo.set_crs(name, RD))

        assert out.collect_schema()[name] == dtype(crs=RD)
        assert_frame_equal(
            out.select(pl.col(name).ext.storage()).collect(),
            df.select(pl.col(name).ext.storage()),
        )


def test_a_declared_crs_is_refused_while_the_plan_is_built() -> None:
    lf = _points(crs=WGS84).lazy().select(geo.set_crs("point", RD))

    with pytest.raises(ValueError, match="to_crs.*force=True"):
        lf.collect_schema()


def test_force_overwrites_the_label_but_not_the_coordinates() -> None:
    points = _points(crs=WGS84)

    out = points.select(geo.set_crs("point", RD, force=True))

    assert out.schema["point"] == PointXY(crs=RD)
    assert_frame_equal(
        out.select(pl.col("point").ext.storage()),
        points.select(pl.col("point").ext.storage()),
    )


def test_force_on_a_geometry_without_a_crs_just_sets_it() -> None:
    out = _points().select(geo.set_crs("point", RD, force=True))

    assert out.schema["point"] == PointXY(crs=RD)


def test_force_keeps_every_other_metadata_key() -> None:
    metadata = '{"edges":"spherical","crs":"EPSG:4326","crs_type":"authority_code"}'
    dtype = PointXY.ext_from_params(
        PointXY._extension_name, PointXY._geo_storage, metadata
    )
    points = _points().select(pl.col("point").ext.storage().ext.to(dtype))

    out = points.select(geo.set_crs("point", RD, force=True))

    assert (
        out.schema["point"].ext_metadata() == '{"edges":"spherical","crs":"EPSG:28992"}'
    )


def test_the_crs_can_then_be_reprojected_from() -> None:
    out = _points().select(geo.to_crs(geo.set_crs("point", WGS84), RD))

    assert out.schema["point"] == PointXY(crs=RD)


def test_a_non_geometry_is_refused() -> None:
    lf = pl.LazyFrame({"x": [1.0]}).select(geo.set_crs("x", RD))

    with pytest.raises(TypeError):
        lf.collect_schema()
