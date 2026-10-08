"""Unit tests for `geo.skew`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, Dimension, coordinates, ring_coordinates


def test_xs_leans_along_x(coords: pl.DataFrame, dimension: Dimension) -> None:
    """At 45°, every position moves along `x` by exactly its `y`.
    `z` and `m` stay as they are."""
    df = coords.select(dimension.point())
    out = df.select(geo.skew("point", xs=45))

    assert out.schema["point"] == dimension.point_dtype()
    expected = coordinates(df).with_columns(x=pl.col("x") + pl.col("y"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_ys_leans_along_y(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.skew("point", ys=-45))

    expected = coordinates(df).with_columns(y=pl.col("y") - pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_both_angles_apply_to_the_positions_as_they_were(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())
    out = df.select(geo.skew("point", xs=45, ys=45))

    expected = coordinates(df).select(
        x=pl.col("x") + pl.col("y"), y=pl.col("y") + pl.col("x")
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_no_angle_changes_nothing(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())

    assert_frame_equal(df.select(geo.skew("point")), df, check_exact=True)


def test_any_angle_leans_by_its_tangent() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(XY.point())
    out = df.select(geo.skew("point", xs=30, ys=10))

    expected = pl.DataFrame(
        {
            "x": [1.0 + math.tan(math.radians(30)) * 2.0],
            "y": [2.0 + math.tan(math.radians(10)) * 1.0],
        }
    )
    assert_frame_equal(coordinates(out), expected)


@pytest.mark.parametrize(
    ("degrees", "pis"),
    [(45, 0.25), (-45, -0.25), (135, 0.75), (30, 1 / 6)],
)
def test_degrees_and_multiples_of_pi_agree(
    coords: pl.DataFrame, degrees: float, pis: float
) -> None:
    df = coords.select(XY.point())

    assert_frame_equal(
        df.select(geo.skew("point", xs=degrees, ys=degrees)),
        df.select(geo.skew("point", xs=pis, ys=pis, unit="pi")),
    )


def test_an_eighth_turn_leaves_no_rounding_error() -> None:
    """With tan(45°) computed, the factor would be 0.9999999999999999."""
    df = pl.DataFrame({"x": [0.0], "y": [3.0]}).select(XY.point())
    out = df.select(geo.skew("point", xs=45))

    expected = pl.DataFrame({"x": [3.0], "y": [3.0]})
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_skewing_about_an_origin(coords: pl.DataFrame, dimension: Dimension) -> None:
    """About (2, 1), `x` moves by how far `y` is from 1."""
    df = coords.select(dimension.point())
    out = df.select(geo.skew("point", xs=45, origin=(2.0, 1.0)))

    expected = coordinates(df).with_columns(x=pl.col("x") + pl.col("y") - 1)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_the_origin_itself_stays_where_it_is() -> None:
    df = pl.DataFrame({"x": [5.0], "y": [7.0]}).select(XY.point())

    assert_frame_equal(
        df.select(geo.skew("point", xs=20, ys=-35, origin=(5.0, 7.0))), df
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"xs": 90}, "`xs` cannot be a quarter turn"),
        ({"ys": -90}, "`ys` cannot be a quarter turn"),
        ({"xs": 270}, "`xs` cannot be a quarter turn"),
        ({"xs": 0.5, "unit": "pi"}, "`xs` cannot be a quarter turn"),
        ({"xs": math.nan}, "`xs`"),
        ({"ys": math.inf}, "`ys`"),
        ({"unit": "rad"}, "`unit`"),
        ({"origin": (1.0,)}, "`origin`"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        geo.skew("point", **kwargs)  # type: ignore[arg-type]


def test_refuses_a_column_that_is_not_a_geometry() -> None:
    df = pl.DataFrame({"point": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.lazy().select(geo.skew("point", xs=10)).collect_schema()


def test_every_vertex_of_every_ring_leans(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon is sheared as a whole, and keeps its rings as they were."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.skew("polygon", ys=45))

    expected = ring_coordinates(df).with_columns(y=pl.col("y") + pl.col("x"))
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

    assert_frame_equal(df.select(geo.skew("line", xs=30)), df)


def test_the_namespace_forwards(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())

    assert_frame_equal(
        df.select(gpl.col("point").geo.skew(0.1, -0.2, "pi", (1.0, 2.0))),
        df.select(geo.skew("point", xs=18, ys=-36, origin=(1.0, 2.0))),
    )
