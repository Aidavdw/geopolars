"""The convex hull of a polygon."""

from __future__ import annotations

import math

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import PolygonXY, PolygonXYZM
from tests.unit.conftest import XY

_XY_RINGS = pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})))

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


# Concave: a notch at (2, 2).
_NOTCHED = _ring((0, 0), (4, 0), (4, 4), (2, 2), (0, 4))
_HULL = _ring((0, 0), (0, 4), (4, 4), (4, 0))
_ALONG_A_LINE = _ring((0, 0), (1, 0), (2, 0))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings", crs=crs).alias("polygon")
    )


def _hulls(df: pl.DataFrame) -> list:
    return df.select(geo.convex_hull("polygon")).to_series().ext.storage().to_list()


def test_a_concave_polygon_loses_its_notch() -> None:
    assert _hulls(_polygons([_NOTCHED])) == [[_HULL]]


def test_a_missing_or_empty_polygon() -> None:
    assert _hulls(_polygons(None, [], [_NOTCHED])) == [None, [], [_HULL]]


def test_a_polygon_that_encloses_nothing_gives_a_missing_hull() -> None:
    assert _hulls(_polygons([_ALONG_A_LINE], [_NOTCHED])) == [None, [_HULL]]


def test_z_is_kept_but_m_is_not() -> None:
    xy = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4), (0, 0)]
    ring = [
        {"x": x, "y": y, "z": 10.0 * x, "m": float(i)} for i, (x, y) in enumerate(xy)
    ]
    storage = pl.Series("polygon", [[ring]], dtype=PolygonXYZM._geo_storage)
    df = pl.DataFrame(storage.ext.to(PolygonXYZM()))

    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXYZM()
    [[hull]] = out.to_series().ext.storage().to_list()
    assert [(c["x"], c["y"], c["z"]) for c in hull] == [
        (0, 0, 0),
        (0, 4, 0),
        (4, 4, 40),
        (4, 0, 40),
        (0, 0, 0),
    ]
    assert all(math.isnan(c["m"]) for c in hull)


def test_the_crs_is_carried_over() -> None:
    out = _polygons([_NOTCHED], crs="EPSG:28992").select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY(crs="EPSG:28992")


def test_rejects_a_linestring_while_resolving_the_schema(
    line_coords: pl.DataFrame,
) -> None:
    lf = XY.lines(line_coords).lazy().select(geo.convex_hull("line"))

    with pytest.raises(TypeError, match="convex_hull expects a polygon column"):
        lf.collect_schema()


def test_rejects_a_multipolygon_while_resolving_the_schema(
    multipolygon_coords: pl.DataFrame,
) -> None:
    df = XY.multipolygons(multipolygon_coords)
    lf = df.lazy().select(geo.convex_hull(df.columns[0]))

    with pytest.raises(TypeError, match="convex_hull expects a polygon column"):
        lf.collect_schema()


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_NOTCHED], None)

    assert df.select(pl.col("polygon").geo.convex_hull()).equals(
        df.select(geo.convex_hull("polygon"))
    )
