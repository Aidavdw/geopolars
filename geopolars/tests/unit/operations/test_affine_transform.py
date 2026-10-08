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


def _origins(x: float, y: float, z: float | None = None) -> pl.Expr:
    """The same point for every row, as a column."""
    return geo.point(pl.lit(x), pl.lit(y), z=None if z is None else pl.lit(z))


def test_about_an_origin(coords: pl.DataFrame, dimension: Dimension) -> None:
    """Relative to (1, 2): x' = 2·(x - 1) + (y - 2) + 5 + 1, y' = 3·(y - 2) + 6 + 2."""
    df = coords.select(dimension.point())
    out = df.select(geo.affine_transform("point", [2, 1, 0, 3, 5, 6], (1.0, 2.0)))

    x, y = pl.col("x"), pl.col("y")
    expected = coordinates(df).with_columns(x=2 * x + y + 2, y=3 * y + 2)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["PointXYZ", "PointXYZM"])
def test_about_an_origin_in_space(coords: pl.DataFrame, dimension: Dimension) -> None:
    """Relative to z = 10: z' = 2·(z - 10) + 1 + 10."""
    df = coords.select(dimension.point())
    matrix = [1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1]
    out = df.select(geo.affine_transform("point", matrix, (0.0, 0.0, 10.0)))

    expected = coordinates(df).with_columns(z=2 * pl.col("z") - 9)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_the_origin_moves_by_the_offsets_alone() -> None:
    df = pl.DataFrame({"x": [5.0], "y": [7.0]}).select(XY.point())
    out = df.select(geo.affine_transform("point", [0.3, 2, -4, 1.5, 1, 2], (5.0, 7.0)))

    assert_frame_equal(coordinates(out), pl.DataFrame({"x": [6.0], "y": [9.0]}))


@pytest.mark.parametrize(
    "matrix",
    [[0.5, -2, 3, 1.5, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]],
    ids=["6", "12"],
)
def test_a_column_of_one_origin_is_the_same_as_its_numbers(
    ring_coords: pl.DataFrame, matrix: list[float]
) -> None:
    """The kernel that takes an origin per row, against the one that takes one origin.
    Without offsets, both add up the same numbers in the same order."""
    df = XYZ.polygons(ring_coords).with_columns(origin=_origins(1.5, -2.0, 3.0))

    assert_frame_equal(
        df.select(geo.affine_transform("polygon", matrix, "origin")),
        df.select(geo.affine_transform("polygon", matrix, (1.5, -2.0, 3.0))),
        check_exact=True,
    )


def test_each_geometry_is_transformed_about_its_own_origin(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """With offsets, the two kernels add them up in another order."""
    df = dimension.polygons(ring_coords).with_columns(
        origin=geo.mean_coordinate("polygon")
    )
    matrix = [0.5, -2, 3, 1.5, 10, -20]
    out = df.select(geo.affine_transform("polygon", matrix, "origin"))

    for row, origin in enumerate(df["origin"].ext.storage().to_list()):
        alone = df.slice(row, 1).select(
            geo.affine_transform("polygon", matrix, (origin["x"], origin["y"]))
        )
        assert_frame_equal(ring_coordinates(out.slice(row, 1)), ring_coordinates(alone))


@pytest.mark.parametrize(
    ("named", "matrix"),
    [
        (lambda c, o: geo.rotate(c, 90, origin=o), [0, -1, 1, 0, 0, 0]),
        (lambda c, o: geo.skew(c, xs=45, origin=o), [1, 1, 0, 1, 0, 0]),
        (
            lambda c, o: geo.scale(c, 2, 3, 4, origin=o),
            [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
        ),
    ],
    ids=["rotate", "skew", "scale"],
)
def test_the_named_transforms_about_a_column_are_each_one_of_these(
    ring_coords: pl.DataFrame, named: object, matrix: list[float]
) -> None:
    df = XYZ.polygons(ring_coords).with_columns(origin=geo.mean_coordinate("polygon"))

    assert_frame_equal(
        df.select(named("polygon", "origin")),  # type: ignore[operator]
        df.select(geo.affine_transform("polygon", matrix, "origin")),
        check_exact=True,
    )


def test_a_missing_origin_gives_a_missing_geometry(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords).with_columns(
        origin=pl.when(pl.int_range(pl.len()) == 0).then(geo.mean_coordinate("line"))
    )
    out = df.select(geo.affine_transform("line", [2, 0, 0, 2, 1, 1], "origin"))

    assert out["line"].is_null().to_list() == [False, True]


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
def test_a_matrix_that_uses_z_about_a_column_needs_a_z(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    lf = coords.lazy().select(dimension.point(), origin=_origins(1.0, 2.0))
    matrix = [1, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    with pytest.raises(TypeError, match="cannot apply a matrix that uses z"):
        lf.select(geo.affine_transform("point", matrix, "origin")).collect_schema()


def test_an_origin_column_has_to_hold_points(line_coords: pl.DataFrame) -> None:
    lf = XY.lines(line_coords).lazy()

    with pytest.raises(TypeError, match="`origin` has to be a `geoarrow.point`"):
        lf.select(
            geo.affine_transform("line", [1, 0, 0, 1, 0, 0], "line")
        ).collect_schema()


def test_a_bad_origin_is_refused() -> None:
    with pytest.raises(ValueError, match="`origin`"):
        geo.affine_transform("point", [1, 0, 0, 1, 0, 0], (1.0,))  # type: ignore[arg-type]


def test_the_namespace_forwards_an_origin_column(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords).with_columns(origin=geo.mean_coordinate("line"))
    matrix = [0.5, -2, 3, 1.5, 10, -20]

    assert_frame_equal(
        df.select(gpl.col("line").geo.affine_transform(matrix, "origin")),
        df.select(geo.affine_transform("line", matrix, "origin")),
    )
