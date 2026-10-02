"""Behaviour of the `geoarrow.point` extension dtype.

All four dimensions have a different dtype, but the same extension.
These tests focus on what separates them:
what a schema reports, what survives a round trip, what renders,
and what Polars refuses to mix.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SchemaError, StructFieldNotFoundError
from polars.testing import assert_frame_equal

from tests.unit.conftest import DIMENSIONS, XY, XYM, XYZ, Dimension, coordinates


def test_dtype_survives_a_lazy_round_trip(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = coords.lazy().select(dimension.point())
    df = lf.collect()

    assert lf.collect_schema()["point"] == dimension.point_dtype()
    assert df.schema["point"] == dimension.point_dtype()
    assert_frame_equal(coordinates(df), coords.select(dimension.coords))


def test_dtype_renders_its_dimension(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates a point carries.
    (otherwise all four dimensions would be indistinguishable on screen)"""
    df = coords.select(dimension.point())
    tag = "".join(dimension.coords)

    assert f"point[{tag}]" in str(df)
    assert repr(df.schema["point"]) == dimension.point_dtype.__name__


def test_points_of_different_dimensions_do_not_stack(coords: pl.DataFrame) -> None:
    xy = coords.select(XY.point())
    xyz = coords.select(XYZ.point())

    with pytest.raises(SchemaError):
        pl.concat([xy, xyz])


def test_points_of_the_same_dimension_stack(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Matching dimensions must be able to concatenate,
    and stay a point column while doing it."""
    df = coords.select(dimension.point())
    stacked = pl.concat([df, df])

    assert stacked.schema["point"] == dimension.point_dtype()
    assert stacked.height == 2 * df.height


def test_storage_is_the_plain_coordinate_struct(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Individual coordinates must stay accessible."""
    df = coords.select(dimension.point())
    storage = coordinates(df)

    assert storage.columns == list(dimension.coords)
    assert_frame_equal(storage, coords.select(dimension.coords))


def test_every_dimension_is_a_different_dtype() -> None:
    """XYM and XYZ carry as many coordinates as
    each other and are still not the same type."""
    dtypes = [dim.point_dtype() for dim in DIMENSIONS]

    assert len(set(dtypes)) == len(DIMENSIONS)
    assert XYM.point_dtype() != XYZ.point_dtype()


def test_coordinates_are_not_reachable_as_struct_fields(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The extension type hides its storage:
    a point is a point, not a struct that happens to have an `x`."""
    df = coords.select(dimension.point())

    with pytest.raises(StructFieldNotFoundError):
        df.select(pl.col("point").struct.field("x"))
