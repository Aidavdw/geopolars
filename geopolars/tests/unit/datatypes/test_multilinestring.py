"""Behaviour of the `geoarrow.multilinestring` extension dtype."""


# A multilinestring shares its storage with a polygon,
# so a lot of focus here is on making sure that the two stay separate.

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

from tests.unit.conftest import XY, XYZ, Dimension, multilinestring_coordinates


def test_dtype_survives_a_lazy_round_trip(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = (
        ring_coords.lazy()
        .group_by("polygon", "ring", maintain_order=True)
        .agg(dimension.point())
        .select("polygon", dimension.linestring())
        .group_by("polygon", maintain_order=True)
        .agg("line")
        .select(dimension.multilinestring())
    )
    df = lf.collect()

    assert lf.collect_schema()["multilinestring"] == dimension.multilinestring_dtype()
    assert df.schema["multilinestring"] == dimension.multilinestring_dtype()
    assert_frame_equal(
        multilinestring_coordinates(df), ring_coords.select(dimension.coords)
    )


def test_dtype_renders_its_dimension(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates the vertices carry."""
    df = dimension.multilinestrings(ring_coords)
    tag = "".join(dimension.coords)

    assert f"multilinestring[{tag}]" in str(df)
    assert (
        repr(df.schema["multilinestring"]) == dimension.multilinestring_dtype.__name__
    )


def test_multilinestrings_of_different_dimensions_do_not_stack(
    ring_coords: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        pl.concat([XY.multilinestrings(ring_coords), XYZ.multilinestrings(ring_coords)])


def test_multilinestrings_of_the_same_dimension_stack(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)
    stacked = pl.concat([df, df])

    assert stacked.schema["multilinestring"] == dimension.multilinestring_dtype()
    assert stacked.height == 2 * df.height


def test_a_multilinestring_is_not_a_linestring(
    ring_coords: pl.DataFrame, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """One linestring is not a collection of them, whatever the dimension."""
    multilines = dimension.multilinestrings(ring_coords)
    lines = dimension.lines(line_coords)

    assert multilines.schema["multilinestring"] != lines.schema["line"]
    with pytest.raises(SchemaError):
        pl.concat(
            [multilines, lines.rename({"line": "multilinestring"})], how="vertical"
        )


def test_a_multilinestring_is_not_a_polygon(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """These two have byte-identical storage,
    but the spec still calls them different geometries."""
    multilines = dimension.multilinestrings(ring_coords)
    polygons = dimension.polygons(ring_coords)

    assert (
        multilines.schema["multilinestring"].ext_storage()
        == polygons.schema["polygon"].ext_storage()
    )
    assert multilines.schema["multilinestring"] != polygons.schema["polygon"]
    with pytest.raises(SchemaError):
        pl.concat(
            [multilines, polygons.rename({"polygon": "multilinestring"})],
            how="vertical",
        )


def test_a_multilinestring_is_not_a_multipoint(
    ring_coords: pl.DataFrame, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A multipoint nests one layer less: its parts are points, not lines."""
    multilines = dimension.multilinestrings(ring_coords)
    multipoints = dimension.multipoints(line_coords)

    assert multilines.schema["multilinestring"] != multipoints.schema["multipoint"]


def test_storage_is_a_list_of_lists_of_coordinate_structs(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)
    storage = df.select(pl.col("multilinestring").ext.storage())

    assert storage.schema["multilinestring"] == pl.List(
        pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
    )
    assert_frame_equal(
        multilinestring_coordinates(df), ring_coords.select(dimension.coords)
    )


def test_the_lists_keep_the_linestrings_apart(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The extra nesting is what separates the parts from their vertices:
    thirteen vertices are three linestrings, which are two multilinestrings."""
    df = dimension.multilinestrings(ring_coords)
    lines = pl.col("multilinestring").ext.storage()

    assert df.height == 2
    assert df.select(lines.list.len())["multilinestring"].to_list() == [2, 1]
    assert df.select(lines.explode(empty_as_null=False).list.len())[
        "multilinestring"
    ].to_list() == [5, 4, 4]
