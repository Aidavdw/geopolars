"""Unit tests for `geo.scale`."""

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


def test_stretches_x_and_y(coords: pl.DataFrame, dimension: Dimension) -> None:
    """`z` and `m` stay as they are without a `zfact`."""
    df = coords.select(dimension.point())
    out = df.select(geo.scale("point", xfact=2.0, yfact=0.5))

    assert out.schema["point"] == dimension.point_dtype()
    expected = coordinates(df).with_columns(pl.col("x") * 2, pl.col("y") * 0.5)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["PointXYZ", "PointXYZM"])
def test_zfact_stretches_z_but_not_m(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.scale("point", zfact=3.0))

    expected = coordinates(df).with_columns(pl.col("z") * 3)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
def test_zfact_needs_a_z_at_plan_time(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    lf = coords.select(dimension.point()).lazy()

    with pytest.raises(TypeError, match="cannot scale by zfact"):
        lf.select(geo.scale("point", zfact=2.0)).collect_schema()


def test_no_factor_changes_nothing(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())

    assert_frame_equal(df.select(geo.scale("point")), df, check_exact=True)


def test_a_negative_factor_mirrors(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())
    out = df.select(geo.scale("point", xfact=-1.0))

    expected = coordinates(df).with_columns(-pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_zero_factor_flattens_onto_the_origin(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())
    out = df.select(geo.scale("point", yfact=0.0, origin=(0.0, 5.0)))

    expected = coordinates(df).with_columns(y=pl.lit(5.0))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_scaling_about_an_origin(coords: pl.DataFrame, dimension: Dimension) -> None:
    """About (1, 2), x' = 1 + 3·(x - 1) and y' = 2 + 3·(y - 2)."""
    df = coords.select(dimension.point())
    out = df.select(geo.scale("point", xfact=3.0, yfact=3.0, origin=(1.0, 2.0)))

    expected = coordinates(df).with_columns(
        x=3 * pl.col("x") - 2, y=3 * pl.col("y") - 4
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_scaling_z_about_an_origin(coords: pl.DataFrame) -> None:
    """About z = 10, z' = 10 + 2·(z - 10)."""
    df = coords.select(XYZ.point())
    out = df.select(geo.scale("point", zfact=2.0, origin=(0.0, 0.0, 10.0)))

    expected = coordinates(df).with_columns(z=2 * pl.col("z") - 10)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_the_origin_itself_stays_where_it_is() -> None:
    df = pl.DataFrame({"x": [5.0], "y": [7.0]}).select(XY.point())
    out = df.select(geo.scale("point", xfact=0.3, yfact=-4.0, origin=(5.0, 7.0)))

    assert_frame_equal(out, df)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"xfact": math.nan}, "`xfact`"),
        ({"yfact": math.inf}, "`yfact`"),
        ({"zfact": -math.inf}, "`zfact`"),
        ({"origin": (1.0, 2.0, 3.0, 4.0)}, "`origin`"),
        ({"origin": (math.nan, 2.0)}, "`origin`"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        geo.scale("point", **kwargs)  # type: ignore[arg-type]


def test_refuses_a_column_that_is_not_a_geometry() -> None:
    df = pl.DataFrame({"point": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.lazy().select(geo.scale("point", xfact=2.0)).collect_schema()


def test_every_vertex_of_every_ring_stretches(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon is scaled as a whole, and keeps its rings as they were."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.scale("polygon", xfact=2.0, yfact=-1.0))

    expected = ring_coordinates(df).with_columns(pl.col("x") * 2, -pl.col("y"))
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

    assert_frame_equal(df.select(geo.scale("line", xfact=2.0)), df)


def test_the_namespace_forwards(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())

    assert_frame_equal(
        df.select(gpl.col("point").geo.scale(2.0, 3.0, 4.0, (1.0, 2.0, 3.0))),
        df.select(
            geo.scale("point", xfact=2.0, yfact=3.0, zfact=4.0, origin=(1.0, 2.0, 3.0))
        ),
    )


def _origins(x: float, y: float, z: float | None = None) -> pl.Expr:
    """The same point for every row, as a column."""
    return geo.point(pl.lit(x), pl.lit(y), z=None if z is None else pl.lit(z))


def test_a_column_of_one_origin_is_the_same_as_its_numbers(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The kernel that takes an origin per row, against the one that takes one origin."""
    zfact = 4.0 if dimension.has_z else 1.0
    df = dimension.polygons(ring_coords).with_columns(origin=_origins(1.5, -2.0, 3.0))

    assert_frame_equal(
        df.select(geo.scale("polygon", 2.0, 0.5, zfact, origin="origin")),
        df.select(geo.scale("polygon", 2.0, 0.5, zfact, origin=(1.5, -2.0, 3.0))),
        check_exact=True,
    )


def test_each_geometry_scales_about_its_own_origin(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords).with_columns(
        origin=geo.mean_coordinate("polygon")
    )
    out = df.select(geo.scale("polygon", 3.0, -0.5, origin="origin"))

    for row, origin in enumerate(df["origin"].ext.storage().to_list()):
        alone = df.slice(row, 1).select(
            geo.scale("polygon", 3.0, -0.5, origin=(origin["x"], origin["y"]))
        )
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


def test_scaling_about_its_own_middle_keeps_the_middle(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)
    middle = geo.mean_coordinate("polygon")
    scaled = df.select(geo.scale("polygon", 3.0, 0.25, origin=middle))

    assert_frame_equal(
        coordinates(scaled.select(middle), "polygon"),
        coordinates(df.select(middle), "polygon"),
    )


def test_points_scale_about_a_single_origin(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())

    assert_frame_equal(
        df.select(geo.scale("point", 2.0, 3.0, 4.0, origin=_origins(1.0, 2.0, 3.0))),
        df.select(geo.scale("point", 2.0, 3.0, 4.0, origin=(1.0, 2.0, 3.0))),
        check_exact=True,
    )


def test_a_missing_origin_gives_a_missing_geometry(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords).with_columns(
        origin=pl.when(pl.int_range(pl.len()) == 0).then(geo.mean_coordinate("line"))
    )
    out = df.select(geo.scale("line", 2.0, origin="origin"))

    assert out["line"].is_null().to_list() == [False, True]


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
def test_zfact_about_a_column_needs_a_z_at_plan_time(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    lf = coords.lazy().select(dimension.point(), origin=_origins(1.0, 2.0, 3.0))

    with pytest.raises(TypeError, match="cannot scale by zfact"):
        lf.select(geo.scale("point", zfact=2.0, origin="origin")).collect_schema()


def test_an_origin_column_has_to_hold_points(line_coords: pl.DataFrame) -> None:
    lf = XY.lines(line_coords).lazy()

    with pytest.raises(TypeError, match="`origin` has to be a `geoarrow.point`"):
        lf.select(geo.scale("line", 2.0, origin="line")).collect_schema()


def test_the_namespace_forwards_an_origin_column(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords).with_columns(origin=geo.mean_coordinate("line"))

    assert_frame_equal(
        df.select(gpl.col("line").geo.scale(2.0, 3.0, 4.0, "origin")),
        df.select(geo.scale("line", 2.0, 3.0, 4.0, origin="origin")),
    )
