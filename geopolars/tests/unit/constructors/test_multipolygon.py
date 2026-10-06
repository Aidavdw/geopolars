"""Building `geoarrow.multipolygon` columns with `geo.multipolygon`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    GeoPolygon,
    MultiPolygonXY,
    PolygonXY,
)
from tests.unit.conftest import Dimension, multipolygon_coordinates

# A list of bare coordinate structs: what a ring stores.
_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

# A list of those: what a polygon stores, before it is one.
_XY_RINGS = pl.List(_XY_VERTICES)

# A list of those: what a multipolygon stores, before it is one.
_XY_POLYGONS = pl.List(_XY_RINGS)

# One coordinate column per axis: one list per multipolygon, of one per polygon,
# of one per ring.
_XY_COORDS = {
    "lon": pl.List(pl.List(pl.List(pl.Float64))),
    "lat": pl.List(pl.List(pl.List(pl.Float64))),
}

# A single unit square, the smallest multipolygon that encloses something.
_SQUARE = [
    [
        [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
            {"x": 0.0, "y": 1.0},
            {"x": 0.0, "y": 0.0},
        ]
    ]
]


def test_the_polygons_decide_the_dtype(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A multipolygon has the dimension of the polygons it is built from."""
    df = dimension.multipolygons(multipolygon_coords)

    assert df.schema["multipolygon"] == dimension.multipolygon_dtype()


def test_accepts_bare_lists_of_rings() -> None:
    """The parts do not have to be polygons already:
    a list of the ring lists a polygon wraps is the same storage."""
    df = pl.DataFrame(
        {"polygons": [_SQUARE]}, schema={"polygons": _XY_POLYGONS}
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    assert df.schema["multipolygon"] == MultiPolygonXY()
    assert_frame_equal(
        multipolygon_coordinates(df),
        pl.DataFrame({"x": [0.0, 1.0, 1.0, 0.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0, 0.0]}),
    )


def test_an_empty_list_is_an_empty_multipolygon() -> None:
    """See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"polygons": [[]]}, schema={"polygons": _XY_POLYGONS}).select(
        geo.multipolygon("polygons").alias("multipolygon")
    )

    assert df.schema["multipolygon"] == MultiPolygonXY()
    assert df["multipolygon"].is_null().to_list() == [False]
    assert df.select(pl.col("multipolygon").ext.storage().list.len())[
        "multipolygon"
    ].to_list() == [0]


def test_a_single_polygon_is_a_valid_multipolygon() -> None:
    df = pl.DataFrame(
        {"polygons": [_SQUARE]}, schema={"polygons": _XY_POLYGONS}
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    assert df.schema["multipolygon"] == MultiPolygonXY()
    assert df.schema["multipolygon"] != PolygonXY()


def test_a_missing_polygon_invalidates_the_whole_multipolygon() -> None:
    """GeoArrow allows nulls only at the outermost level.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"polygons": [[*_SQUARE, None], _SQUARE, None]},
        schema={"polygons": _XY_POLYGONS},
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    assert df["multipolygon"].is_null().to_list() == [True, False, True]


def test_a_missing_ring_invalidates_the_whole_multipolygon() -> None:
    df = pl.DataFrame(
        {"polygons": [[[*_SQUARE[0], None]], _SQUARE]},
        schema={"polygons": _XY_POLYGONS},
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    assert df["multipolygon"].is_null().to_list() == [True, False]


def test_a_missing_vertex_invalidates_the_whole_multipolygon() -> None:
    df = pl.DataFrame(
        {"polygons": [[[[{"x": 1.0, "y": 2.0}, None]]], _SQUARE]},
        schema={"polygons": _XY_POLYGONS},
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    assert df["multipolygon"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_invalidates_the_whole_multipolygon() -> None:
    df = pl.DataFrame(
        {"polygons": [[[[{"x": 1.0, "y": None}]]], _SQUARE]},
        schema={"polygons": _XY_POLYGONS},
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    assert df["multipolygon"].is_null().to_list() == [True, False]


def test_metadata_is_carried_over_from_the_polygons() -> None:
    metadata = '{"edges":"spherical"}'
    spherical = GeoPolygon.ext_from_params(
        "geoarrow.polygon", PolygonXY().ext_storage(), metadata
    )
    df = (
        pl.DataFrame({"rings": _SQUARE}, schema={"rings": _XY_RINGS})
        .select(polygon=pl.col("rings").ext.to(spherical))
        .select(pl.col("polygon").implode())
        .select(geo.multipolygon("polygon").alias("multipolygon"))
    )

    assert df.schema["multipolygon"].ext_metadata() == metadata
    assert df.schema["multipolygon"] == MultiPolygonXY.ext_from_params(
        "geoarrow.multipolygon", MultiPolygonXY().ext_storage(), metadata
    )


def test_rejects_polygons_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"polygons": [[[[1.0, 2.0]]]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.multipolygon("polygons"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A polygon column is a part per row, not a multipolygon per row."""
    df = pl.DataFrame({"rings": _SQUARE}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings").alias("polygon")
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.multipolygon("polygon"))


def test_rejects_a_list_of_linestrings() -> None:
    """A multipolygon is built out of polygons, and a linestring is not one.
    That gather is a multilinestring."""
    df = (
        pl.DataFrame({"vertices": _SQUARE[0]}, schema={"vertices": _XY_VERTICES})
        .select(geo.linestring("vertices").alias("line"))
        .select(pl.col("line").implode())
    )

    with pytest.raises(ComputeError, match="expected a list of `geoarrow.polygon`s"):
        df.select(geo.multipolygon("line"))


def test_rejects_a_list_of_multilinestrings() -> None:
    """A multilinestring has the storage of a polygon, but is not one."""
    df = (
        pl.DataFrame({"lines": _SQUARE}, schema={"lines": _XY_RINGS})
        .select(geo.multilinestring("lines").alias("multilinestring"))
        .select(pl.col("multilinestring").implode())
    )

    with pytest.raises(ComputeError, match="expected a list of `geoarrow.polygon`s"):
        df.select(geo.multipolygon("multilinestring"))


def test_rejects_bad_polygons_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so parts that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"country": ["nl"]}).select(geo.multipolygon("country"))

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a multipolygon column."""
    df = pl.DataFrame(schema={"polygons": _XY_POLYGONS}).select(
        geo.multipolygon("polygons").alias("multipolygon")
    )

    assert df.height == 0
    assert df.schema["multipolygon"] == MultiPolygonXY()


def test_coordinate_columns_build_the_same_multipolygons(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    assert_frame_equal(
        dimension.multipolygons_from_coords(multipolygon_coords),
        dimension.multipolygons(multipolygon_coords),
    )


def test_coordinate_columns_keep_their_parts_in_order() -> None:
    """The nesting is the polygon and ring structure, read in the order given."""
    df = pl.DataFrame(
        {
            "lon": [[[[0.0, 1.0], [2.0]], [[3.0, 4.0]]]],
            "lat": [[[[5.0, 6.0], [7.0]], [[8.0, 9.0]]]],
        }
    ).select(geo.multipolygon("lon", "lat").alias("multipolygon"))
    polygons = pl.col("multipolygon").ext.storage()

    assert df.select(polygons.list.len())["multipolygon"].to_list() == [2]
    assert df.select(polygons.explode(empty_as_null=False).list.len())[
        "multipolygon"
    ].to_list() == [2, 1]
    assert_frame_equal(
        multipolygon_coordinates(df),
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "y": [5.0, 6.0, 7.0, 8.0, 9.0],
            }
        ),
    )


def test_coordinate_columns_are_cast_to_f64() -> None:
    """Coordinates are doubles.
    integer columns are widened rather than refused."""
    df = pl.DataFrame({"lon": [[[[1, 3]]]], "lat": [[[[2, 4]]]]}).select(
        geo.multipolygon("lon", "lat").alias("multipolygon")
    )

    assert df.schema["multipolygon"] == MultiPolygonXY()
    assert_frame_equal(
        multipolygon_coordinates(df),
        pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]}),
    )


def test_a_missing_coordinate_invalidates_the_multipolygon_of_coords() -> None:
    df = pl.DataFrame(
        {"lon": [[[[1.0]]], [[[1.0]]]], "lat": [[[[None]]], [[[2.0]]]]},
        schema=_XY_COORDS,
    ).select(geo.multipolygon("lon", "lat").alias("multipolygon"))

    assert df["multipolygon"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # A ring's vertex counts disagree.
        ([[[[1.0, 3.0]]]], [[[[2.0]]]]),
        # A polygon's ring counts disagree.
        ([[[[1.0], [3.0]]]], [[[[2.0]]]]),
        # The multipolygons disagree about how many polygons they have.
        ([[[[1.0]], [[3.0]]]], [[[[2.0]]]]),
    ],
)
def test_rejects_coordinate_columns_that_nest_differently(
    lon: list[list[list[list[float]]]], lat: list[list[list[list[float]]]]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geo.multipolygon("lon", "lat"))


def test_rejects_coordinate_columns_nested_only_twice() -> None:
    """Two lists per geometry is a polygon's shape;
    a multipolygon needs the polygons around them too."""
    lf = pl.LazyFrame({"lon": [[[1.0]]], "lat": [[[2.0]]]}).select(
        geo.multipolygon("lon", "lat")
    )

    with pytest.raises(
        ComputeError, match="lists of lists of lists of f64, one per multipolygon"
    ):
        lf.collect_schema()


def test_rejects_a_measure_without_a_y_coordinate() -> None:
    with pytest.raises(TypeError, match="without a y coordinate"):
        geo.multipolygon("lon", m="dist")


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geo.multipolygon("lon", "lat").alias("multipolygon")
    )

    assert df.height == 0
    assert df.schema["multipolygon"] == MultiPolygonXY()


def test_one_argument_dispatches_to_the_polygon_form() -> None:
    df = pl.DataFrame({"polygons": [_SQUARE]}, schema={"polygons": _XY_POLYGONS})

    assert_frame_equal(
        df.select(geo.multipolygon("polygons")),
        df.select(geo.multipolygon_from_polygons("polygons")),
    )


def test_coordinate_columns_dispatch_to_the_column_form() -> None:
    df = pl.DataFrame(
        {"lon": [[[[0.0, 1.0]]]], "lat": [[[[2.0, 3.0]]]]}, schema=_XY_COORDS
    )

    assert_frame_equal(
        df.select(geo.multipolygon("lon", "lat")),
        df.select(geo.multipolygon_from_columns("lon", "lat")),
    )
