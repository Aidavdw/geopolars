"""Unit tests for `geo.rotate`."""

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
    line_coordinates,
    ring_coordinates,
)


def test_a_quarter_turn_is_exact(coords: pl.DataFrame, dimension: Dimension) -> None:
    """(x, y) becomes (-y, x) exactly, rather than off by a rounding error of cos(90°).
    Turning in the plane leaves `z` and `m` as they are."""
    df = coords.select(dimension.point())
    out = df.select(geo.rotate("point", 90))

    assert out.schema["point"] == dimension.point_dtype()
    expected = coordinates(df).with_columns(x=-pl.col("y"), y=pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize(("amount", "unit"), [(90, "deg"), (0.5, "pi"), (450, "deg")])
def test_a_quarter_turn_leaves_no_rounding_error(amount: float, unit: str) -> None:
    """With cos(90°) computed, (1, 0) would land on (6e-17, 1)."""
    df = pl.DataFrame({"x": [1.0], "y": [0.0]}).select(XY.point())
    out = df.select(geo.rotate("point", amount, unit=unit))  # type: ignore[arg-type]

    expected = pl.DataFrame({"x": [0.0], "y": [1.0]})
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_negative_amount_turns_clockwise(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())
    out = df.select(geo.rotate("point", -90))

    expected = coordinates(df).select(x=pl.col("y"), y=-pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_full_turn_changes_nothing(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())

    assert_frame_equal(df.select(geo.rotate("point", 360)), df, check_exact=True)


def test_any_angle_turns_by_its_cosine_and_sine() -> None:
    df = pl.DataFrame({"x": [2.0], "y": [0.0]}).select(XY.point())
    out = df.select(geo.rotate("point", 30))

    expected = pl.DataFrame({"x": [2 * math.cos(math.pi / 6)], "y": [1.0]})
    assert_frame_equal(coordinates(out), expected)


@pytest.mark.parametrize(
    ("degrees", "pis"),
    [(90, 0.5), (180, 1), (-45, -0.25), (30, 1 / 6)],
)
def test_degrees_and_multiples_of_pi_agree(
    coords: pl.DataFrame, degrees: float, pis: float
) -> None:
    df = coords.select(XY.point())

    assert_frame_equal(
        df.select(geo.rotate("point", degrees)),
        df.select(geo.rotate("point", pis, unit="pi")),
    )


def test_turning_about_x_moves_y_into_z(coords: pl.DataFrame) -> None:
    df = coords.select(XYZM.point())
    out = df.select(geo.rotate("point", 90, axis="x"))

    expected = coordinates(df).with_columns(y=-pl.col("z"), z=pl.col("y"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_turning_about_y_moves_z_into_x(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())
    out = df.select(geo.rotate("point", 90, axis="y"))

    expected = coordinates(df).with_columns(x=pl.col("z"), z=-pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_turning_out_of_the_plane_needs_a_z_at_plan_time(
    coords: pl.DataFrame, dimension: Dimension, axis: str
) -> None:
    lf = coords.select(dimension.point()).lazy()

    with pytest.raises(TypeError, match=f"cannot rotate about the {axis} axis"):
        lf.select(geo.rotate("point", 90, axis=axis)).collect_schema()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unit": "rad"}, "`unit`"),
        ({"unit": "π"}, "`unit`"),
        ({"axis": "w"}, "`axis`"),
        ({"amount": math.nan}, "`amount`"),
        ({"amount": math.inf}, "`amount`"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], match: str) -> None:
    arguments: dict[str, object] = {"amount": 90.0, **kwargs}

    with pytest.raises(ValueError, match=match):
        geo.rotate("point", **arguments)  # type: ignore[arg-type]


def test_every_vertex_of_every_ring_turns(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon turns as a whole, and keeps its rings as they were."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.rotate("polygon", 180))

    expected = ring_coordinates(df).with_columns(x=-pl.col("x"), y=-pl.col("y"))
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

    assert_frame_equal(df.select(geo.rotate("line", 45)), df)


def test_the_namespace_forwards(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords)

    assert_frame_equal(
        line_coordinates(df.select(gpl.col("line").geo.rotate(0.25, "pi", "x"))),
        line_coordinates(df.select(geo.rotate("line", 45, axis="x"))),
    )
