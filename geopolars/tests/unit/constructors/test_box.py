"""Building `geoarrow.box` columns with `geopolars.geo.box`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import BoxType, BoxXY, BoxXYM, BoxXYZ, BoxXYZM

_BOUNDS = pl.DataFrame(
    {
        "xmin": [0.0, 1.0],
        "ymin": [2.0, 3.0],
        "zmin": [4.0, 5.0],
        "mmin": [6.0, 7.0],
        "xmax": [10.0, 11.0],
        "ymax": [12.0, 13.0],
        "zmax": [14.0, 15.0],
        "mmax": [16.0, 17.0],
    }
)


def _storage(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.col("box").ext.storage()).unnest("box")


@pytest.mark.parametrize(
    ("optional", "expected"),
    [
        ((), BoxXY),
        (("zmin", "zmax"), BoxXYZ),
        (("mmin", "mmax"), BoxXYM),
        (("zmin", "zmax", "mmin", "mmax"), BoxXYZM),
    ],
    ids=["xy", "xyz", "xym", "xyzm"],
)
def test_optional_bounds_pick_the_dtype(
    optional: tuple[str, ...], expected: type[BoxType]
) -> None:
    optional_bounds = {name: name for name in optional}
    df = _BOUNDS.select(
        geo.box("xmin", "ymin", "xmax", "ymax", **optional_bounds).alias("box")
    )

    assert df.schema["box"] == expected()
    fields = list(expected().ext_storage().to_schema())
    assert_frame_equal(_storage(df), _BOUNDS.select(fields))


@pytest.mark.parametrize("given", ["zmin", "zmax", "mmin", "mmax"])
def test_half_a_pair_of_bounds_is_refused(given: str) -> None:
    with pytest.raises(ValueError, match=f"pass both {given[0]}min and {given[0]}max"):
        geo.box("xmin", "ymin", "xmax", "ymax", **{given: given})


@pytest.mark.parametrize("form", ["column name", "expression", "series"])
def test_accepts_every_into_expr_column_form(form: str) -> None:
    names = ["xmin", "ymin", "xmax", "ymax"]
    if form == "column name":
        args = names
    elif form == "expression":
        args = [pl.col(name) for name in names]
    else:
        args = [_BOUNDS[name] for name in names]

    df = _BOUNDS.select(geo.box(*args).alias("box"))

    assert df.schema["box"] == BoxXY()
    assert_frame_equal(_storage(df), _BOUNDS.select(names))


def test_integer_bounds_are_accepted() -> None:
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]}).select(
        geo.box("a", "b", "c", "d").alias("box")
    )

    assert df.schema["box"] == BoxXY()
    assert _storage(df).row(0) == (1.0, 2.0, 3.0, 4.0)


def test_a_missing_bound_invalidates_the_whole_box() -> None:
    """See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {
            "xmin": [None, 0.0, 0.0],
            "ymin": [0.0, 0.0, 0.0],
            "xmax": [1.0, 1.0, 1.0],
            "ymax": [1.0, 1.0, 1.0],
            "zmin": [0.0, 0.0, 0.0],
            "zmax": [1.0, None, 1.0],
        }
    ).select(
        geo.box("xmin", "ymin", "xmax", "ymax", zmin="zmin", zmax="zmax").alias("box")
    )

    assert df["box"].is_null().to_list() == [True, True, False]


def test_bounds_are_taken_as_given() -> None:
    """`xmin > xmax` crosses the antimeridian, and `inf` to `-inf` is an empty range:
    neither is reordered or refused."""
    df = pl.DataFrame(
        {
            "xmin": [170.0, math.inf],
            "ymin": [-10.0, math.inf],
            "xmax": [-170.0, -math.inf],
            "ymax": [10.0, -math.inf],
        }
    ).select(geo.box("xmin", "ymin", "xmax", "ymax").alias("box"))

    assert _storage(df).rows() == [
        (170.0, -10.0, -170.0, 10.0),
        (math.inf, math.inf, -math.inf, -math.inf),
    ]


def test_crs_is_carried() -> None:
    df = _BOUNDS.select(
        geo.box("xmin", "ymin", "xmax", "ymax", crs="EPSG:4326").alias("box")
    )

    assert df.schema["box"] == BoxXY(crs="EPSG:4326")


def test_empty_frame_keeps_its_dtype() -> None:
    df = _BOUNDS.clear().select(geo.box("xmin", "ymin", "xmax", "ymax").alias("box"))

    assert df.schema["box"] == BoxXY()
