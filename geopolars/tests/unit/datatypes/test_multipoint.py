"""Behaviour of the `geoarrow.multipoint` extension dtype."""


# A multipoint shares its storage with a linestring,
# so a lot of focus here is on making sure that the two stay separate.

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

from tests.unit.conftest import XY, XYZ, Dimension, multipoint_coordinates


def test_dtype_survives_a_lazy_round_trip(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = (
        line_coords.lazy()
        .group_by("line", maintain_order=True)
        .agg(dimension.point())
        .select(dimension.multipoint())
    )
    df = lf.collect()

    assert lf.collect_schema()["multipoint"] == dimension.multipoint_dtype()
    assert df.schema["multipoint"] == dimension.multipoint_dtype()
    assert_frame_equal(multipoint_coordinates(df), line_coords.select(dimension.coords))


def test_dtype_renders_its_dimension(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates the points carry."""
    df = dimension.multipoints(line_coords)
    tag = "".join(dimension.coords)

    assert f"multipoint[{tag}]" in str(df)
    assert repr(df.schema["multipoint"]) == dimension.multipoint_dtype.__name__


def test_multipoints_of_different_dimensions_do_not_stack(
    line_coords: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        pl.concat([XY.multipoints(line_coords), XYZ.multipoints(line_coords)])


def test_multipoints_of_the_same_dimension_stack(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    stacked = pl.concat([df, df])

    assert stacked.schema["multipoint"] == dimension.multipoint_dtype()
    assert stacked.height == 2 * df.height


def test_a_multipoint_is_not_a_point(
    coords: pl.DataFrame, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    points = coords.select(dimension.point())
    multipoints = dimension.multipoints(line_coords)

    assert multipoints.schema["multipoint"] != points.schema["point"]
    with pytest.raises(SchemaError):
        pl.concat([multipoints, points.rename({"point": "multipoint"})], how="vertical")


def test_a_multipoint_is_not_a_linestring(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """These two have byte-identical storage,
    but the spec still calls them different geometries."""
    multipoints = dimension.multipoints(line_coords)
    lines = dimension.lines(line_coords)

    assert (
        multipoints.schema["multipoint"].ext_storage()
        == lines.schema["line"].ext_storage()
    )
    assert multipoints.schema["multipoint"] != lines.schema["line"]
    with pytest.raises(SchemaError):
        pl.concat([multipoints, lines.rename({"line": "multipoint"})], how="vertical")


def test_storage_is_a_list_of_coordinate_structs(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    storage = df.select(pl.col("multipoint").ext.storage())

    assert storage.schema["multipoint"] == pl.List(
        pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))
    )
    assert_frame_equal(multipoint_coordinates(df), line_coords.select(dimension.coords))


def test_the_list_keeps_the_multipoints_apart(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The five points are two multipoints, of three and of two."""
    df = dimension.multipoints(line_coords)

    assert df.height == 2
    assert df.select(pl.col("multipoint").ext.storage().list.len())[
        "multipoint"
    ].to_list() == [3, 2]
