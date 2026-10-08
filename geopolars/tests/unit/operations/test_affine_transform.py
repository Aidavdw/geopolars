"""Unit tests for `geo.affine_transform`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import (
    XY,
    XYM,
    XYZ,
    XYZM,
    Dimension,
    coordinates,
    ring_coordinates,
)


def test_six_numbers_transform_in_the_plane(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """[a, b, d, e, xoff, yoff], in shapely's order. `z` and `m` stay as they are."""
    df = coords.select(dimension.point())
    out = df.select(geo.affine_transform("point", [1, 2, 3, 4, 5, 6]))

    assert out.schema["point"] == dimension.point_dtype()
    expected = coordinates(df).with_columns(
        x=pl.col("x") + 2 * pl.col("y") + 5, y=3 * pl.col("x") + 4 * pl.col("y") + 6
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["PointXYZ", "PointXYZM"])
def test_twelve_numbers_transform_in_space(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """[a, b, c, d, e, f, g, h, i, xoff, yoff, zoff], in shapely's order.
    `m` stays as it is."""
    df = coords.select(dimension.point())
    matrix = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    out = df.select(geo.affine_transform("point", matrix))

    x, y, z = pl.col("x"), pl.col("y"), pl.col("z")
    expected = coordinates(df).with_columns(
        x=x + 2 * y + 3 * z + 10,
        y=4 * x + 5 * y + 6 * z + 11,
        z=7 * x + 8 * y + 9 * z + 12,
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize(
    "matrix", [[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]
)
def test_the_identity_changes_nothing(
    coords: pl.DataFrame, dimension: Dimension, matrix: list[float]
) -> None:
    df = coords.select(dimension.point())

    assert_frame_equal(
        df.select(geo.affine_transform("point", matrix)), df, check_exact=True
    )


def test_twelve_numbers_that_leave_z_alone_work_in_the_plane(
    coords: pl.DataFrame,
) -> None:
    df = coords.select(XYM.point())

    assert_frame_equal(
        df.select(geo.affine_transform("point", [1, 2, 0, 3, 4, 0, 0, 0, 1, 5, 6, 0])),
        df.select(geo.affine_transform("point", [1, 2, 3, 4, 5, 6])),
    )


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
@pytest.mark.parametrize(
    "matrix",
    [
        [1, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # x' reads z
        [1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0],  # y' reads z
        [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],  # z' takes x
        [1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],  # z' stretches z
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3],  # z' moves
    ],
    ids=["c", "f", "g", "i", "zoff"],
)
def test_a_matrix_that_uses_z_needs_a_z_at_plan_time(
    coords: pl.DataFrame, dimension: Dimension, matrix: list[float]
) -> None:
    lf = coords.select(dimension.point()).lazy()

    with pytest.raises(TypeError, match="cannot apply a matrix that uses z"):
        lf.select(geo.affine_transform("point", matrix)).collect_schema()


@pytest.mark.parametrize(
    ("named", "matrix"),
    [
        (lambda c: geo.translate(c, 1, 2, 3), [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3]),
        (lambda c: geo.rotate(c, 90), [0, -1, 1, 0, 0, 0]),
        (lambda c: geo.rotate(c, 90, axis="x"), [1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0]),
        (lambda c: geo.skew(c, xs=45), [1, 1, 0, 1, 0, 0]),
        (lambda c: geo.scale(c, 2, 3, 4), [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]),
        (
            lambda c: geo.scale(c, 2, 2, origin=(1.0, 1.0)),
            [2, 0, 0, 2, -1, -1],
        ),
    ],
    ids=["translate", "rotate", "rotate about x", "skew", "scale", "scale about"],
)
def test_the_named_transforms_are_each_one_of_these(
    coords: pl.DataFrame, named: object, matrix: list[float]
) -> None:
    df = coords.select(XYZ.point())

    assert_frame_equal(
        df.select(named("point")),  # type: ignore[operator]
        df.select(geo.affine_transform("point", matrix)),
        check_exact=True,
    )


@pytest.mark.parametrize(
    "matrix",
    [[], [1, 0, 0, 1], [1] * 9, [1] * 13, [1, 0, 0, 1, math.nan, 0], [math.inf] * 12],
    ids=["empty", "4", "9", "13", "nan", "inf"],
)
def test_bad_matrices_are_refused(matrix: list[float]) -> None:
    with pytest.raises(ValueError, match="`matrix` has to be 6 or 12 finite numbers"):
        geo.affine_transform("point", matrix)


def test_refuses_a_column_that_is_not_a_geometry() -> None:
    df = pl.DataFrame({"point": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.lazy().select(
            geo.affine_transform("point", [1, 0, 0, 1, 0, 0])
        ).collect_schema()


def test_every_vertex_of_every_ring_is_transformed(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon is transformed as a whole, and keeps its rings as they were."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.affine_transform("polygon", [0, 1, 1, 0, 0, 0]))

    expected = ring_coordinates(df).with_columns(x=pl.col("y"), y=pl.col("x"))
    assert_frame_equal(ring_coordinates(out), expected, check_exact=True)
    assert_frame_equal(
        out.select(pl.col("polygon").ext.storage().list.eval(pl.element().list.len())),
        df.select(pl.col("polygon").ext.storage().list.eval(pl.element().list.len())),
    )


def test_keeps_empty_and_missing_linestrings(dimension: Dimension) -> None:
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geo.line_string("vertices").alias("line"))

    assert_frame_equal(df.select(geo.affine_transform("line", [2, 0, 0, 2, 1, 1])), df)


def test_the_namespace_forwards(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())
    matrix = (1.5, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 2.0, 3.0)

    assert_frame_equal(
        df.select(gpl.col("point").geo.affine_transform(matrix)),
        df.select(geo.affine_transform("point", matrix)),
    )
