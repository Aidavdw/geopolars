"""Behaviour of the `geoarrow.box` extension dtype."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError, SchemaError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import BoxXY, BoxXYM, BoxXYZ, BoxXYZM, GeoBox

_BOXES = [BoxXY, BoxXYZ, BoxXYM, BoxXYZM]
_IDS = ["xy", "xyz", "xym", "xyzm"]


def _bounds(dtype: type[GeoBox]) -> pl.DataFrame:
    names = dtype().ext_storage().to_schema()
    return pl.DataFrame(
        {name: [float(i), float(i) + 10] for i, name in enumerate(names)}
    )


def _boxes(dtype: type[GeoBox], crs: str | None = None) -> pl.LazyFrame:
    bounds = _bounds(dtype)
    return bounds.lazy().select(box=pl.struct(bounds.columns).ext.to(dtype(crs=crs)))


@pytest.mark.parametrize("dtype", _BOXES, ids=_IDS)
def test_dtype_survives_a_lazy_round_trip(dtype: type[GeoBox]) -> None:
    lf = _boxes(dtype)
    df = lf.collect()

    assert lf.collect_schema()["box"] == dtype()
    assert df.schema["box"] == dtype()
    assert_frame_equal(
        df.select(pl.col("box").ext.storage()).unnest("box"), _bounds(dtype)
    )


@pytest.mark.parametrize(
    ("dtype", "fields"),
    [
        (BoxXY, ["xmin", "ymin", "xmax", "ymax"]),
        (BoxXYZ, ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]),
        (BoxXYM, ["xmin", "ymin", "mmin", "xmax", "ymax", "mmax"]),
        (BoxXYZM, ["xmin", "ymin", "zmin", "mmin", "xmax", "ymax", "zmax", "mmax"]),
    ],
    ids=_IDS,
)
def test_storage_is_the_spec_struct_of_bounds(
    dtype: type[GeoBox], fields: list[str]
) -> None:
    assert dtype().ext_storage() == pl.Struct(dict.fromkeys(fields, pl.Float64))


@pytest.mark.parametrize("dtype", _BOXES, ids=_IDS)
def test_dtype_renders_its_dimension(dtype: type[GeoBox]) -> None:
    df = _boxes(dtype).collect()
    tag = "".join(dtype._dimension)

    assert f"box[{tag}]" in str(df)
    assert repr(df.schema["box"]) == dtype.__name__


def test_every_dimension_is_a_different_dtype() -> None:
    assert len({dtype() for dtype in _BOXES}) == len(_BOXES)


def test_boxes_of_different_dimensions_do_not_stack() -> None:
    with pytest.raises(SchemaError):
        pl.concat([_boxes(BoxXY).collect(), _boxes(BoxXYZ).collect()])


def test_boxes_of_the_same_dimension_stack() -> None:
    df = _boxes(BoxXYM).collect()
    stacked = pl.concat([df, df])

    assert stacked.schema["box"] == BoxXYM()
    assert stacked.height == 2 * df.height


def test_crs_survives_a_lazy_round_trip() -> None:
    lf = _boxes(BoxXY, crs="EPSG:4326")

    assert lf.collect().schema["box"] == BoxXY(crs="EPSG:4326")
    assert lf.collect_schema()["box"]._declares_crs()


@pytest.mark.parametrize(
    "storage",
    [
        # A point's coordinates are not bounds.
        pl.Struct({"x": pl.Float64, "y": pl.Float64}),
        # Every minimum comes before every maximum.
        pl.Struct(dict.fromkeys(["xmin", "xmax", "ymin", "ymax"], pl.Float64)),
        pl.Struct(dict.fromkeys(["xmin", "ymin", "xmax", "ymax"], pl.Float32)),
    ],
    ids=["coordinates", "wrong order", "f32"],
)
def test_unsupported_storage_is_refused(storage: pl.DataType) -> None:
    with pytest.raises(ValueError, match="unsupported 'geoarrow.box' storage"):
        GeoBox.ext_from_params("geoarrow.box", storage, None)


def test_a_box_is_not_a_geometry_to_a_native_operation() -> None:
    """Refused while the plan is built, by the Python dispatch."""
    with pytest.raises(TypeError, match="got: BoxXY"):
        _boxes(BoxXY).select(geo.is_empty("box")).collect_schema()


def test_a_box_is_not_a_geometry_to_a_plugin() -> None:
    """Refused while the plan is built, by the Rust `describe`."""
    with pytest.raises(ComputeError, match=r"got: ext\[box\[xy\]\]"):
        _boxes(BoxXY).select(geo.translate("box", dx=1, dy=1)).collect_schema()
