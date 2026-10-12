"""Operations on a list of geometries, run on each geometry through `list.eval`."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.unit.conftest import Dimension

#: One of each dispatcher: `on_geometry`, `on_geometry_pair` and `on_box`.
_OPERATIONS: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "area": lambda geometry: geometry.geo.area(),
    "rotate_about_centroid": lambda geometry: geometry.geo.rotate_about_centroid(90.0),
    "to_polygon": lambda geometry: geometry.geo.bounds().geo.to_polygon(),
}


def _nested(flat: pl.DataFrame) -> pl.DataFrame:
    """The rows of `flat` as lists: all of them, then just the first."""
    return pl.concat(
        [flat.select(pl.all().implode()), flat.head(1).select(pl.all().implode())]
    )


@pytest.mark.parametrize("operation", _OPERATIONS.values(), ids=_OPERATIONS.keys())
def test_list_eval_runs_the_operation_on_each_geometry(
    ring_coords: pl.DataFrame,
    dimension: Dimension,
    operation: Callable[[pl.Expr], pl.Expr],
) -> None:
    """A list of polygons, rather than a multipolygon: each one is operated on
    by itself, and the results keep the lists they came in."""
    polygons = dimension.polygons(ring_coords)

    out = _nested(polygons).select(pl.col("polygon").list.eval(operation(pl.element())))

    expected = _nested(polygons.select(operation(pl.col("polygon"))))
    assert_frame_equal(out, expected)
