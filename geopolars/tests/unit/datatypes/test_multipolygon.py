"""Behaviour of the `geoarrow.multipolygon` extension dtype."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

from tests.unit.conftest import XY, XYZ, Dimension, multipolygon_coordinates


def test_dtype_survives_a_lazy_round_trip(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = dimension.multipolygons(multipolygon_coords).lazy()
    df = lf.collect()

    assert lf.collect_schema()["multipolygon"] == dimension.multipolygon_dtype()
    assert df.schema["multipolygon"] == dimension.multipolygon_dtype()
    assert_frame_equal(
        multipolygon_coordinates(df), multipolygon_coords.select(dimension.coords)
    )


def test_dtype_renders_its_dimension(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates the vertices carry."""
    df = dimension.multipolygons(multipolygon_coords)
    tag = "".join(dimension.coords)

    assert f"multipolygon[{tag}]" in str(df)
    assert repr(df.schema["multipolygon"]) == dimension.multipolygon_dtype.__name__


def test_multipolygons_of_different_dimensions_do_not_stack(
    multipolygon_coords: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        pl.concat(
            [
                XY.multipolygons(multipolygon_coords),
                XYZ.multipolygons(multipolygon_coords),
            ]
        )


def test_multipolygons_of_the_same_dimension_stack(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)
    stacked = pl.concat([df, df])

    assert stacked.schema["multipolygon"] == dimension.multipolygon_dtype()
    assert stacked.height == 2 * df.height


def test_a_multipolygon_is_not_a_polygon(
    multipolygon_coords: pl.DataFrame, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """One polygon is not a collection of them, whatever the dimension."""
    multipolygons = dimension.multipolygons(multipolygon_coords)
    polygons = dimension.polygons(ring_coords)

    assert multipolygons.schema["multipolygon"] != polygons.schema["polygon"]
    with pytest.raises(SchemaError):
        pl.concat(
            [multipolygons, polygons.rename({"polygon": "multipolygon"})],
            how="vertical",
        )


def test_a_multipolygon_is_not_a_multilinestring(
    multipolygon_coords: pl.DataFrame, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A multilinestring nests one layer less: its parts are lines, not polygons."""
    multipolygons = dimension.multipolygons(multipolygon_coords)
    multilines = dimension.multilinestrings(ring_coords)

    assert multipolygons.schema["multipolygon"] != multilines.schema["multilinestring"]


def test_storage_is_a_list_of_lists_of_lists_of_coordinate_structs(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)
    storage = df.select(pl.col("multipolygon").ext.storage())

    assert storage.schema["multipolygon"] == pl.List(
        pl.List(pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))))
    )
    assert_frame_equal(
        multipolygon_coordinates(df), multipolygon_coords.select(dimension.coords)
    )


def test_the_lists_keep_the_polygons_and_rings_apart(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Eighteen vertices are four rings, which are three polygons,
    which are two multipolygons."""
    df = dimension.multipolygons(multipolygon_coords)
    polygons = pl.col("multipolygon").ext.storage()
    rings = polygons.explode(empty_as_null=False)

    assert df.height == 2
    assert df.select(polygons.list.len())["multipolygon"].to_list() == [2, 1]
    assert df.select(rings.list.len())["multipolygon"].to_list() == [2, 1, 1]
    assert df.select(rings.explode(empty_as_null=False).list.len())[
        "multipolygon"
    ].to_list() == [5, 4, 4, 5]
