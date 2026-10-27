"""Closing the open rings (and linestrings) of a geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    MultiPolygonXYZM,
    PointXY,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from geopolars.datatypes import GeoArrowType

Vertices = list[dict[str, float]]


def _vertices(axes: str, *values: tuple[float, ...]) -> Vertices:
    return [dict(zip(axes, vertex, strict=True)) for vertex in values]


def _closed(ring: Vertices) -> Vertices:
    return [*ring, ring[0]]


def _column(
    dtype: GeoArrowType, nesting: int, *rows: object, axes: str = "xy"
) -> pl.DataFrame:
    """A `geometry` column of `dtype`, its rows `nesting` lists deep around the vertices."""
    storage: pl.DataType = pl.Struct(dict.fromkeys(axes, pl.Float64))
    for _ in range(nesting):
        storage = pl.List(storage)
    series = pl.Series("geometry", list(rows), dtype=storage)
    return pl.DataFrame(series.ext.to(dtype))


def _storage(df: pl.DataFrame, expr: pl.Expr) -> list:
    out = df.select(expr)
    assert out.schema == df.schema
    return out.to_series().ext.storage().to_list()


_OPEN = _vertices("xy", (0, 0), (4, 0), (4, 4), (0, 4))
_HOLE_OPEN = _vertices("xy", (1, 1), (2, 1), (2, 2))
_CLOSED = _closed(_vertices("xy", (5, 5), (6, 5), (6, 6)))


@pytest.mark.parametrize(
    "df",
    [
        _column(PointXY(), 0, {"x": 1.0, "y": 2.0}),
        _column(MultiPointXY(), 1, _OPEN),
    ],
    ids=["point", "multipoint"],
)
def test_points_are_refused(df: pl.DataFrame) -> None:
    with pytest.raises(TypeError, match="close_rings expects"):
        df.select(geo.close_rings("geometry"))


def test_a_linestring_is_closed() -> None:
    df = _column(LineStringXY(), 1, _OPEN, _CLOSED, [], None)

    assert _storage(df, geo.close_rings("geometry")) == [
        _closed(_OPEN),
        _CLOSED,
        [],
        None,
    ]


def test_every_ring_of_a_polygon_is_closed() -> None:
    df = _column(
        PolygonXY(), 2, [_OPEN, _HOLE_OPEN, _CLOSED], [_CLOSED], [], [[]], None
    )

    assert _storage(df, geo.close_rings("geometry")) == [
        [_closed(_OPEN), _closed(_HOLE_OPEN), _CLOSED],
        [_CLOSED],
        [],
        [[]],
        None,
    ]


def test_every_part_of_a_multilinestring_is_closed() -> None:
    df = _column(MultiLineStringXY(), 2, [_OPEN, _CLOSED, []], None)

    assert _storage(df, geo.close_rings("geometry")) == [
        [_closed(_OPEN), _CLOSED, []],
        None,
    ]


def test_every_ring_of_a_multipolygon_is_closed() -> None:
    df = _column(MultiPolygonXY(), 3, [[_OPEN, _HOLE_OPEN], [], [_CLOSED]], None)

    assert _storage(df, geo.close_rings("geometry")) == [
        [[_closed(_OPEN), _closed(_HOLE_OPEN)], [], [_CLOSED]],
        None,
    ]


def test_a_closed_column_is_returned_as_it_is() -> None:
    df = _column(PolygonXY(), 2, [_CLOSED], [], None)

    assert _storage(df, geo.close_rings("geometry")) == [[_CLOSED], [], None]


def test_the_result_is_valid_geoarrow() -> None:
    df = _column(MultiPolygonXY(), 3, [[_OPEN, _HOLE_OPEN]], [[_CLOSED], [_OPEN]])

    out = df.select(geo.close_rings("geometry"))

    assert out.select(geo.is_valid_geoarrow("geometry")).to_series().all()


def test_the_crs_is_carried_over() -> None:
    df = _column(PolygonXY(crs="EPSG:28992"), 2, [_OPEN])

    assert df.select(geo.close_rings("geometry")).schema == df.schema


def test_a_sliced_column() -> None:
    df = _column(PolygonXY(), 2, [_OPEN], [_OPEN], [_CLOSED]).slice(1, 2)

    assert _storage(df, geo.close_rings("geometry")) == [[_closed(_OPEN)], [_CLOSED]]


_Z_OPEN = _vertices("xyz", (0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 0, 1))


@pytest.mark.parametrize(
    ("snap_z", "closed"),
    [
        (False, _closed(_Z_OPEN)),
        (True, [*_Z_OPEN[:-1], {"x": 0.0, "y": 0.0, "z": 0.0}]),
    ],
)
def test_ends_differing_in_z_are_appended_or_snapped(
    snap_z: bool, closed: Vertices
) -> None:
    df = _column(PolygonXYZ(), 2, [_Z_OPEN], axes="xyz")

    assert _storage(df, geo.close_rings("geometry", snap_z=snap_z)) == [[closed]]


def test_ends_differing_in_x_and_y_are_appended_despite_snap_z() -> None:
    ring = _vertices("xyz", (0, 0, 0), (4, 0, 0), (4, 4, 1))
    df = _column(PolygonXYZ(), 2, [ring], axes="xyz")

    assert _storage(df, geo.close_rings("geometry", snap_z=True)) == [[_closed(ring)]]


_M_OPEN = _vertices("xym", (0, 0, 0), (4, 0, 1), (4, 4, 2), (0, 0, 3))


@pytest.mark.parametrize(
    ("snap_m", "closed"),
    [
        (False, _M_OPEN),
        (True, [*_M_OPEN[:-1], {"x": 0.0, "y": 0.0, "m": 0.0}]),
    ],
)
def test_ends_differing_in_m_are_kept_or_snapped(
    snap_m: bool, closed: Vertices
) -> None:
    df = _column(PolygonXYM(), 2, [_M_OPEN], axes="xym")

    assert _storage(df, geo.close_rings("geometry", snap_m=snap_m)) == [[closed]]


_ZM_OPEN = _vertices("xyzm", (0, 0, 0, 0), (4, 0, 0, 1), (4, 4, 0, 2), (0, 0, 1, 3))


def test_snap_z_and_snap_m_snap_both() -> None:
    df = _column(PolygonXYZM(), 2, [_ZM_OPEN], axes="xyzm")
    snapped = [*_ZM_OPEN[:-1], {"x": 0.0, "y": 0.0, "z": 0.0, "m": 0.0}]

    out = geo.close_rings("geometry", snap_z=True, snap_m=True)
    assert _storage(df, out) == [[snapped]]


def test_snap_m_alone_appends_where_z_differs() -> None:
    df = _column(PolygonXYZM(), 2, [_ZM_OPEN], axes="xyzm")

    out = geo.close_rings("geometry", snap_m=True)
    assert _storage(df, out) == [[_closed(_ZM_OPEN)]]


# One ring per row, each long enough to be a ring once closed.
_ZM_RINGS = [
    _vertices("xyzm", (0, 0, 0, 0), (4, 0, 0, 1), (4, 4, 0, 2)),
    _ZM_OPEN,
    _vertices("xyzm", (0, 0, 0, 0), (4, 0, 0, 1), (4, 4, 0, 2), (0, 0, 0, 3)),
    _vertices("xyzm", (0, 0, 0, 0), (4, 0, 0, 1), (4, 4, 0, 2), (0, 0, 1, 0)),
    _closed(_vertices("xyzm", (0, 0, 0, 0), (4, 0, 0, 1), (4, 4, 0, 2))),
]


@pytest.mark.parametrize("snap_z", [False, True])
@pytest.mark.parametrize("snap_m", [False, True])
@pytest.mark.parametrize(
    ("df", "constructor"),
    [
        (
            _column(PolygonXYZM(), 2, *([ring] for ring in _ZM_RINGS), axes="xyzm"),
            geo.polygons_from_rings_in_single_column,
        ),
        (
            _column(
                MultiPolygonXYZM(), 3, *([[ring]] for ring in _ZM_RINGS), axes="xyzm"
            ),
            geo.multi_polygons_from_polygons,
        ),
    ],
    ids=["polygon", "multipolygon"],
)
def test_the_constructors_close_as_close_rings_does(
    df: pl.DataFrame, constructor: Callable[..., pl.Expr], snap_z: bool, snap_m: bool
) -> None:
    closing = geo.CloseRings(snap_z=snap_z, snap_m=snap_m)
    built = constructor(pl.col("geometry").ext.storage(), close=closing)

    assert _storage(df, built) == _storage(
        df, geo.close_rings("geometry", snap_z=snap_z, snap_m=snap_m)
    )
