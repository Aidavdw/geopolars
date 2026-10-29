"""Kernels that walk a whole level, on a chunk sliced from a larger one.

Such a chunk still holds every list and coordinate of the larger one below its rows.
A debug build asserts that it was trimmed before the kernel saw it.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PolygonXY,
)

WGS84 = "EPSG:4326"
NAN = float("nan")


def _ring(dx: float, *, with_nan: bool = False) -> list[dict[str, float]]:
    """A closed square, `dx` to the right. With `with_nan=True`, one corner's x is NaN."""
    xy = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    ring = [{"x": x + dx, "y": float(y)} for x, y in xy]
    if with_nan:
        ring[2]["x"] = NAN
    return ring


def _geometry(dtype: type[GeoArrowType], dx: float, *, with_nan: bool = False) -> object:
    ring = _ring(dx, with_nan=with_nan)
    return {
        LineStringXY: ring,
        MultiPointXY: ring[:3],
        PolygonXY: [ring],
        MultiLineStringXY: [ring, ring],
        MultiPolygonXY: [[ring], [ring]],
    }[dtype]


def _frame(dtype: type[GeoArrowType], rows: list[object]) -> pl.DataFrame:
    storage = pl.Series("g", rows, dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype(crs=WGS84)))


Kernel = Callable[[str], pl.Expr]

KERNELS: dict[str, tuple[Kernel, set[type[GeoArrowType]]]] = {
    # The name, the kernel, and the types it refuses.
    "translate": (lambda g: geo.translate(g, (1.0, 2.0)), set()),
    "filter_coordinates set_empty": (
        lambda g: geo.filter_coordinates(g, action="set_empty"),
        set(),
    ),
    "filter_coordinates drop": (lambda g: geo.filter_coordinates(g), set()),
    "filter_coordinates set_null": (
        lambda g: geo.filter_coordinates(g, action="set_null"),
        set(),
    ),
    "fix_out_of_bounds": (lambda g: geo.fix_out_of_bounds(g), set()),
    "has_repeats_in_succession": (lambda g: geo.has_repeats_in_succession(g), set()),
    "remove_repeats_in_succession": (
        lambda g: geo.remove_repeats_in_succession(g),
        set(),
    ),
    "reverse": (lambda g: geo.reverse(g), set()),
    "to_ccw": (lambda g: geo.to_ccw(g), {MultiPointXY}),
    "close_rings": (lambda g: geo.close_rings(g), {MultiPointXY}),
    "is_valid_geoarrow": (
        lambda g: geo.is_valid_geoarrow(g, check_finite_coordinates=True),
        set(),
    ),
    "is_valid_ogc": (lambda g: geo.is_valid_ogc(g), set()),
    "to_crs": (lambda g: geo.to_crs(g, "EPSG:3857"), set()),
}


@pytest.mark.parametrize(
    "dtype",
    [LineStringXY, MultiPointXY, PolygonXY, MultiLineStringXY, MultiPolygonXY],
    ids=["linestring", "multipoint", "polygon", "multilinestring", "multipolygon"],
)
@pytest.mark.parametrize("kernel", list(KERNELS))
def test_a_sliced_chunk_gives_what_its_rows_alone_give(
    dtype: type[GeoArrowType], kernel: str
) -> None:
    expr, refused = KERNELS[kernel]
    if dtype in refused:
        pytest.skip(f"{kernel} refuses {dtype.__name__}")
    # One out of the bounds of WGS 84, a null, and one with a NaN.
    rows = [
        _geometry(dtype, 200.0),
        None,
        _geometry(dtype, 0.0, with_nan=True),
        _geometry(dtype, 1.0),
    ]
    padding = [_geometry(dtype, -5.0 - i) for i in range(3)]
    whole = _frame(dtype, padding + rows + padding)
    sliced = whole.slice(len(padding), len(rows))

    out = sliced.select(expr("g"))

    assert_frame_equal(out, _frame(dtype, rows).select(expr("g")))
