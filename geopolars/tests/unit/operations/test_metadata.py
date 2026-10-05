"""Operations on geometries that declare a CRS.

Every operation that hands back a geometry in the same place has to hand back
its metadata too.
Add every new operation that does to `KEEPS_METADATA`.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import GeoArrowType
from tests.unit.conftest import Dimension

RD = "EPSG:28992"

KEEPS_METADATA: dict[str, Callable[[str], pl.Expr]] = {
    "mean_coordinate": geo.mean_coordinate,
    "translate": lambda name: geo.translate(name, 1.0, -1.0),
}


@pytest.fixture(params=list(KEEPS_METADATA), ids=str)
def operation(request: pytest.FixtureRequest) -> Callable[[str], pl.Expr]:
    return KEEPS_METADATA[request.param]


def _with_crs(df: pl.DataFrame, dtype: type[GeoArrowType]) -> pl.DataFrame:
    name = df.columns[0]
    return df.select(pl.col(name).ext.storage().ext.to(dtype(crs=RD)))


def _geometries(
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> list[pl.DataFrame]:
    """One column of every geometry, all declaring `RD`."""
    plain = [
        (coords.select(dimension.point()), dimension.point_dtype),
        (dimension.lines(line_coords), dimension.linestring_dtype),
        (dimension.polygons(ring_coords), dimension.polygon_dtype),
        (dimension.multipoints(line_coords), dimension.multipoint_dtype),
        (dimension.multilinestrings(ring_coords), dimension.multilinestring_dtype),
    ]
    return [_with_crs(df, dtype) for df, dtype in plain]


def test_every_geometry_keeps_its_crs(
    operation: Callable[[str], pl.Expr],
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> None:
    for df in _geometries(dimension, coords, line_coords, ring_coords):
        (name,) = df.columns

        out = df.lazy().select(operation(name))

        assert out.collect_schema()[name].ext_metadata() == f'{{"crs":"{RD}"}}'
        out.collect()


def test_the_same_as_without_a_crs(
    operation: Callable[[str], pl.Expr],
    dimension: Dimension,
    line_coords: pl.DataFrame,
) -> None:
    """Declaring a CRS changes the dtype, never the coordinates."""
    plain = dimension.lines(line_coords)
    declared = _with_crs(plain, dimension.linestring_dtype)

    got = declared.select(operation("line").ext.storage())
    expected = plain.select(operation("line").ext.storage())

    assert got.equals(expected)
