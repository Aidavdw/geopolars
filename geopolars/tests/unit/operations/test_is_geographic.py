"""Whether a geometry's CRS is geographic."""

from __future__ import annotations

import polars as pl
import pytest

from geopolars import geo
from tests.unit.conftest import Dimension

RD = "EPSG:28992"
WGS84 = "EPSG:4326"


def _points(crs: str | None = WGS84) -> pl.LazyFrame:
    return pl.LazyFrame({"x": [4.9, 5.1, 5.3], "y": [52.4, 52.1, 52.0]}).select(
        geo.point("x", "y", crs=crs).alias("point")
    )


def _is_geographic(lf: pl.LazyFrame, **kwargs: bool) -> bool:
    return lf.select(geo.is_geographic("point", **kwargs)).collect().item()


@pytest.mark.parametrize("crs", [WGS84, "EPSG:4807"], ids=["degrees", "grads"])
def test_a_geographic_crs_is_geographic(crs: str) -> None:
    assert _is_geographic(_points(crs))


@pytest.mark.parametrize("crs", [None, RD], ids=["no crs", "projected"])
def test_no_or_a_projected_crs_is_not_geographic(crs: str | None) -> None:
    assert not _is_geographic(_points(crs))


def test_it_is_one_value_for_the_column() -> None:
    out = _points().select(geo.is_geographic("point")).collect()

    assert out.schema == pl.Schema({"point": pl.Boolean})
    assert out.height == 1


def test_it_broadcasts_over_the_rows() -> None:
    out = _points().with_columns(geo.is_geographic("point").alias("geographic"))

    assert out.collect()["geographic"].to_list() == [True, True, True]


def test_every_geometry_can_be_asked(
    dimension: Dimension,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    geometries = [
        dimension.lines(line_coords),
        dimension.polygons(ring_coords),
        dimension.multipoints(line_coords),
        dimension.multilinestrings(ring_coords),
        dimension.multipolygons(multipolygon_coords),
    ]
    for df in geometries:
        (name,) = df.columns
        lf = df.lazy().select(geo.set_crs(name, WGS84))

        assert lf.select(geo.is_geographic(name)).collect().item()


def test_an_unreadable_crs_is_refused_at_plan_time() -> None:
    lf = _points(None).select(geo.set_crs("point", "not a crs"))

    with pytest.raises(pl.exceptions.ComputeError, match="not a crs"):
        lf.select(geo.is_geographic("point")).collect_schema()


def test_an_unreadable_crs_is_not_geographic_when_ignoring_errors() -> None:
    lf = _points(None).select(geo.set_crs("point", "not a crs"))

    assert not _is_geographic(lf, ignore_errors=True)


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_a_box_is_refused_at_plan_time(ignore_errors: bool) -> None:
    boxes = pl.LazyFrame({"a": [0.0], "b": [0.0], "c": [1.0], "d": [1.0]}).select(
        geo.box("a", "b", "c", "d", crs=WGS84).alias("box")
    )

    with pytest.raises(TypeError, match="got: BoxXY"):
        boxes.select(
            geo.is_geographic("box", ignore_errors=ignore_errors)
        ).collect_schema()
