"""The mean of the coordinates a geometry is made of."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    PointType,
    PointXY,
)
from tests.unit.conftest import XY, XYZM, Dimension, coordinates

_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 4.0, "y": 0.0},
    {"x": 4.0, "y": 4.0},
    {"x": 0.0, "y": 4.0},
    {"x": 0.0, "y": 0.0},
]

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)


def _mean_per(vertices: pl.DataFrame, group: str, dimension: Dimension) -> pl.DataFrame:
    """The mean of each coordinate, per geometry, straight off a vertex frame."""
    return (
        vertices.group_by(group, maintain_order=True)
        .agg(pl.col(*dimension.coords).mean())
        .select(*dimension.coords)
    )


def _opened(rings: pl.DataFrame) -> pl.DataFrame:
    """A vertex frame with the closing vertex of every ring dropped."""
    within = pl.int_range(pl.len()).over("polygon", "ring")
    return rings.filter(within < pl.len().over("polygon", "ring") - 1)


def test_a_point_is_its_own_mean_coordinate(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.mean_coordinate("point"))

    assert out.schema["point"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out), coordinates(df))


def test_a_linestring_averages_its_vertices(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines(line_coords)
    out = df.select(geo.mean_coordinate("line"))

    assert_frame_equal(
        coordinates(out, "line"), _mean_per(line_coords, "line", dimension)
    )


def test_a_polygon_averages_the_vertices_of_all_its_rings(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Holes count as much as the exterior ring: every coordinate counts once."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.mean_coordinate("polygon"))

    assert_frame_equal(
        coordinates(out, "polygon"),
        _mean_per(_opened(ring_coords), "polygon", dimension),
    )


def test_the_coordinate_that_closes_a_ring_is_left_out() -> None:
    """A square's four corners average to its middle. Counting the repeat that
    closes the ring would pull the mean towards wherever it starts."""
    df = pl.DataFrame({"rings": [[_SQUARE]]}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings").alias("polygon")
    )
    out = df.select(geo.mean_coordinate("polygon"))

    assert_frame_equal(
        coordinates(out, "polygon"), pl.DataFrame({"x": [2.0], "y": [2.0]})
    )


def test_a_closed_linestring_keeps_every_vertex() -> None:
    """A linestring is not a ring, even when it closes: nothing says its last
    vertex is there to close it, so all five are averaged."""
    df = pl.DataFrame(
        {"vertices": [_SQUARE]}, schema={"vertices": _XY_VERTICES}
    ).select(geo.linestring("vertices").alias("line"))
    out = df.select(geo.mean_coordinate("line"))

    assert_frame_equal(coordinates(out, "line"), pl.DataFrame({"x": [1.6], "y": [1.6]}))


def test_an_empty_ring_takes_nothing_with_it() -> None:
    df = pl.DataFrame(
        {"rings": [[_SQUARE, []], [[], _SQUARE]]}, schema={"rings": _XY_RINGS}
    ).select(geo.polygon("rings").alias("polygon"))
    out = df.select(geo.mean_coordinate("polygon"))

    # The square's four corners, whichever side of it the empty ring is on.
    assert_frame_equal(
        coordinates(out, "polygon"),
        pl.DataFrame({"x": [2.0, 2.0], "y": [2.0, 2.0]}),
    )


def test_a_multipoint_averages_its_points(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    out = df.select(geo.mean_coordinate("multipoint"))

    assert_frame_equal(
        coordinates(out, "multipoint"), _mean_per(line_coords, "line", dimension)
    )


def test_a_multipoint_and_a_linestring_of_the_same_points_agree(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    multipoints = dimension.multipoints(line_coords).select(
        geo.mean_coordinate("multipoint")
    )
    lines = dimension.lines(line_coords).select(geo.mean_coordinate("line"))

    assert_frame_equal(
        coordinates(multipoints, "multipoint"), coordinates(lines, "line")
    )


def test_a_closed_multipoint_keeps_every_point() -> None:
    """A multipoint is not a ring: a repeated point is a point that is there
    twice, and it counts twice."""
    df = pl.DataFrame({"points": [_SQUARE]}, schema={"points": _XY_VERTICES}).select(
        geo.multipoint("points").alias("multipoint")
    )
    out = df.select(geo.mean_coordinate("multipoint"))

    assert_frame_equal(
        coordinates(out, "multipoint"), pl.DataFrame({"x": [1.6], "y": [1.6]})
    )


def test_an_empty_or_missing_multipoint_has_no_mean_coordinate(
    dimension: Dimension,
) -> None:
    df = pl.DataFrame(
        {"points": [[], None]},
        schema={
            "points": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geo.multipoint("points").alias("multipoint"))

    out = df.select(geo.mean_coordinate("multipoint"))

    assert out["multipoint"].is_null().to_list() == [True, True]


def test_a_multilinestring_averages_the_vertices_of_all_its_parts(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)
    out = df.select(geo.mean_coordinate("multilinestring"))

    assert_frame_equal(
        coordinates(out, "multilinestring"),
        _mean_per(ring_coords, "polygon", dimension),
    )


def test_a_multilinestrings_parts_are_not_rings(ring_coords: pl.DataFrame) -> None:
    """The same vertices, so the same storage,
    but a polygon drops the coordinate that closes each ring
    and a multilinestring keeps it."""
    multilines = XY.multilinestrings(ring_coords).select(
        geo.mean_coordinate("multilinestring")
    )
    polygons = XY.polygons(ring_coords).select(geo.mean_coordinate("polygon"))

    assert (
        coordinates(multilines, "multilinestring").rows()
        != coordinates(polygons, "polygon").rows()
    )


def test_a_closed_multilinestring_keeps_every_vertex() -> None:
    """A part that happens to close is still a linestring:
    all five of the square's vertices are averaged"""
    df = pl.DataFrame({"lines": [[_SQUARE]]}, schema={"lines": _XY_RINGS}).select(
        geo.multilinestring("lines").alias("multilinestring")
    )
    out = df.select(geo.mean_coordinate("multilinestring"))

    assert_frame_equal(
        coordinates(out, "multilinestring"), pl.DataFrame({"x": [1.6], "y": [1.6]})
    )


def test_an_empty_or_missing_multilinestring_has_no_mean_coordinate(
    dimension: Dimension,
) -> None:
    df = pl.DataFrame(
        {"lines": [[], [[]], None]},
        schema={
            "lines": pl.List(
                pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
            )
        },
    ).select(geo.multilinestring("lines").alias("multilinestring"))

    out = df.select(geo.mean_coordinate("multilinestring"))

    assert out["multilinestring"].is_null().to_list() == [True, True, True]


def test_a_multipolygon_averages_the_vertices_of_all_its_polygons(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Every ring of every polygon counts, holes included,
    each without the coordinate that closes it."""
    df = dimension.multipolygons(multipolygon_coords)
    out = df.select(geo.mean_coordinate("multipolygon"))

    assert_frame_equal(
        coordinates(out, "multipolygon"),
        _mean_per(_opened(multipolygon_coords), "multipolygon", dimension),
    )


def test_a_multipolygon_of_one_polygon_agrees_with_the_polygon() -> None:
    polygon = pl.DataFrame({"rings": [[_SQUARE]]}, schema={"rings": _XY_RINGS}).select(
        geo.polygon("rings").alias("polygon")
    )
    multipolygon = polygon.select(
        geo.multipolygon(pl.col("polygon").implode()).alias("polygon")
    )

    assert_frame_equal(
        multipolygon.select(geo.mean_coordinate("polygon").ext.storage()),
        polygon.select(geo.mean_coordinate("polygon").ext.storage()),
    )


def test_an_empty_or_missing_multipolygon_has_no_mean_coordinate(
    dimension: Dimension,
) -> None:
    df = pl.DataFrame(
        {"polygons": [[], [[]], [[[]]], None]},
        schema={
            "polygons": pl.List(
                pl.List(pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))))
            )
        },
    ).select(geo.multipolygon("polygons").alias("multipolygon"))

    out = df.select(geo.mean_coordinate("multipolygon"))

    assert out["multipolygon"].is_null().to_list() == [True, True, True, True]


@pytest.mark.parametrize(
    "builder",
    ["point", "line", "polygon", "multipoint", "multilinestring", "multipolygon"],
)
def test_the_result_is_a_point_of_the_same_dimension(
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
    dimension: Dimension,
    builder: str,
) -> None:
    """Whatever went in, a point of the same dimension comes out."""
    if builder == "point":
        df = coords.select(dimension.point())
    elif builder == "line":
        df = dimension.lines(line_coords)
    elif builder == "multipoint":
        df = dimension.multipoints(line_coords)
    elif builder == "multilinestring":
        df = dimension.multilinestrings(ring_coords)
    elif builder == "multipolygon":
        df = dimension.multipolygons(multipolygon_coords)
    else:
        df = dimension.polygons(ring_coords)
    name = df.columns[0]

    out = df.select(geo.mean_coordinate(name))

    assert out.schema[name] == PointType.of_dimension(dimension.coords)()


def test_m_is_averaged_like_any_other_coordinate(line_coords: pl.DataFrame) -> None:
    """Unlike `translate`, which leaves a measure where it was, there is nothing
    to carry through here: the mean coordinate of a trajectory carries the mean
    of the measures its vertices hold."""
    df = XYZM.lines(line_coords)
    out = df.select(geo.mean_coordinate("line"))

    assert_frame_equal(
        coordinates(out, "line").select("m"),
        _mean_per(line_coords, "line", XYZM).select("m"),
    )


def test_an_empty_or_missing_linestring_has_no_mean_coordinate(
    dimension: Dimension,
) -> None:
    """A geometry with no coordinates has no mean coordinate, and a null
    geometry stays null: GeoArrow has no point with null coordinates, only a
    null point."""
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geo.linestring("vertices").alias("line"))

    out = df.select(geo.mean_coordinate("line"))

    assert out["line"].is_null().to_list() == [True, True]


def test_an_empty_or_missing_polygon_has_no_mean_coordinate(
    dimension: Dimension,
) -> None:
    """A polygon with no rings, or with nothing but empty ones, has no
    coordinates to average either."""
    df = pl.DataFrame(
        {"rings": [[], [[]], None]},
        schema={
            "rings": pl.List(
                pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
            )
        },
    ).select(geo.polygon("rings").alias("polygon"))

    out = df.select(geo.mean_coordinate("polygon"))

    assert out["polygon"].is_null().to_list() == [True, True, True]


def test_a_missing_point_has_no_mean_coordinate() -> None:
    df = pl.DataFrame({"x": [None, 1.0], "y": [2.0, 2.0]}).select(
        geo.point("x", "y").alias("point")
    )

    out = df.select(geo.mean_coordinate("point"))

    assert out["point"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_takes_the_whole_mean_with_it() -> None:
    """The constructors do not let a vertex go missing, so this goes around
    them. Averaging what is left would hand back the mean coordinate of a
    geometry that is not there, which is what `validate` is there to prevent."""
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": None}]]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))
    whole = df.select(geo.validate("line"))

    out = whole.select(geo.mean_coordinate("line"))

    assert out["line"].is_null().to_list() == [True]


def test_every_form_of_column_gives_the_same_answer(line_coords: pl.DataFrame) -> None:
    """A name, a `pl.col(...)` and a `Series` are the same column."""
    df = XY.lines(line_coords)
    expected = df.select(geo.mean_coordinate("line"))

    assert_frame_equal(df.select(geo.mean_coordinate(pl.col("line"))), expected)
    assert_frame_equal(pl.select(geo.mean_coordinate(df["line"])), expected)


def test_it_runs_on_the_streaming_engine(line_coords: pl.DataFrame) -> None:
    """The point of building this out of ordinary expressions."""
    lf = XY.lines(line_coords).lazy().select(geo.mean_coordinate("line"))

    assert_frame_equal(lf.collect(engine="streaming"), lf.collect())


def test_rejects_a_plain_float_column() -> None:
    df = pl.DataFrame({"line": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.select(geo.mean_coordinate("line"))


def test_rejects_a_bare_coordinate_struct() -> None:
    """The storage of a point is not a point: the dtype is what says it is one."""
    df = pl.DataFrame({"point": [{"x": 1.0, "y": 2.0}]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.select(geo.mean_coordinate("point"))


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    """The geometry is read off the dtype, so a bad column is a schema error and
    not something that waits until the data is there."""
    lf = pl.LazyFrame({"lon": [1.0]}).select(geo.mean_coordinate("lon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_crs_is_carried_through(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The mean of coordinates in a CRS is in that same CRS."""
    plain = dimension.lines(line_coords)
    rd = dimension.linestring_dtype(crs="EPSG:28992")
    df = plain.select(pl.col("line").ext.storage().ext.to(rd))

    out = df.select(geo.mean_coordinate("line"))

    # Compared against a dtype whose metadata Rust wrote.
    assert out.schema["line"] == dimension.point_dtype(crs="EPSG:28992")
    expected = plain.select(geo.mean_coordinate("line"))
    assert_frame_equal(coordinates(out, "line"), coordinates(expected, "line"))


def test_metadata_is_carried_through_verbatim() -> None:
    """Keys we don't know of either, spelled as they came in."""
    metadata = '{ "edges": "spherical", "x-vendor": [1, 2] }'
    spherical = LineStringXY._with_metadata_of(
        PointType.ext_from_params("geoarrow.point", PointXY().ext_storage(), metadata)
    )
    df = pl.DataFrame({"line": [_SQUARE]}, schema={"line": _XY_VERTICES}).select(
        pl.col("line").ext.to(spherical)
    )

    out = df.select(geo.mean_coordinate("line"))

    assert out.schema["line"].ext_metadata() == metadata


def test_a_point_keeps_its_crs() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        point=geo.point("x", "y", crs="EPSG:4326")
    )

    assert df.select(geo.mean_coordinate("point")).schema == df.schema


def test_takes_an_expression_as_well_as_a_column(line_coords: pl.DataFrame) -> None:
    """The geometry is read off the resolved dtype, so it need not be a column."""
    df = XY.lines(line_coords)
    translated = gpl.col("line").geo.translate(1.0, 1.0)

    assert_frame_equal(
        df.select(geo.mean_coordinate(translated)),
        df.select(geo.mean_coordinate("line")).select(geo.translate("line", 1.0, 1.0)),
    )


def test_rejects_a_series_that_is_not_a_geometry() -> None:
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        pl.select(geo.mean_coordinate(pl.Series("line", [1.0])))


def test_rejects_a_column_that_is_not_there() -> None:
    df = pl.DataFrame({"a": [1.0]})

    with pytest.raises(ColumnNotFoundError, match="line"):
        df.select(geo.mean_coordinate("line"))
