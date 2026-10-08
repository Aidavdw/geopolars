from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import polars as pl
import pytest

from geopolars import geo
from geopolars.datatypes import (
    LineStringType,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    LineStringXYZM,
    MultiLineStringType,
    MultiLineStringXY,
    MultiLineStringXYM,
    MultiLineStringXYZ,
    MultiLineStringXYZM,
    MultiPointType,
    MultiPointXY,
    MultiPointXYM,
    MultiPointXYZ,
    MultiPointXYZM,
    MultiPolygonType,
    MultiPolygonXY,
    MultiPolygonXYM,
    MultiPolygonXYZ,
    MultiPolygonXYZM,
    PointType,
    PointXY,
    PointXYM,
    PointXYZ,
    PointXYZM,
    PolygonType,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)

#: Every coordinate column the fixtures carry. Aggregating these is what turns
#: a vertex frame into the nested coordinate columns the constructors take.
COORDS = pl.col("x", "y", "z", "m")


class Dimension(NamedTuple):
    """One supported dimension: its dtypes and the coordinates it carries.

    The coordinate names are written out here rather than read back off the
    dtype, so that the tests state what GeoArrow requires instead of agreeing
    with whatever the implementation happens to do.
    """

    point_dtype: type[PointType]
    linestring_dtype: type[LineStringType]
    polygon_dtype: type[PolygonType]
    multipoint_dtype: type[MultiPointType]
    multilinestring_dtype: type[MultiLineStringType]
    multipolygon_dtype: type[MultiPolygonType]
    coords: tuple[str, ...]

    @property
    def has_z(self) -> bool:
        return "z" in self.coords

    @property
    def has_m(self) -> bool:
        return "m" in self.coords

    def of_coords(self, constructor: Callable[..., pl.Expr]) -> pl.Expr:
        """Call a constructor on the columns named x/y/z/m this dimension has."""
        return constructor(
            "x",
            "y",
            z="z" if self.has_z else None,
            m="m" if self.has_m else None,
        )

    def point(self) -> pl.Expr:
        """Build a point of this dimension from columns named x/y/z/m."""
        return self.of_coords(geo.point).alias("point")

    def linestring(self) -> pl.Expr:
        """Build a linestring of this dimension from a `point` list column."""
        return geo.linestring("point").alias("line")

    def polygon(self) -> pl.Expr:
        """Build a polygon of this dimension from a `line` list column."""
        return geo.polygon("line").alias("polygon")

    def multipoint(self) -> pl.Expr:
        """Build a multipoint of this dimension from a `point` list column."""
        return geo.multipoint("point").alias("multipoint")

    def multilinestring(self) -> pl.Expr:
        """Build a multilinestring of this dimension from a `line` list column."""
        return geo.multilinestring("line").alias("multilinestring")

    def multipolygon(self) -> pl.Expr:
        """Build a multipolygon of this dimension from a `polygon` list column."""
        return geo.multipolygon("polygon").alias("multipolygon")

    def lines(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """One linestring of this dimension per `line` in a vertex frame."""
        return (
            vertices.group_by("line", maintain_order=True)
            .agg(self.point())
            .select(self.linestring())
        )

    def polygons(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """One polygon of this dimension per `polygon` in a vertex frame,
        with one ring per `ring` within it."""
        return (
            vertices.group_by("polygon", "ring", maintain_order=True)
            .agg(self.point())
            .select("polygon", self.linestring())
            .group_by("polygon", maintain_order=True)
            .agg("line")
            .select(self.polygon())
        )

    def multipoints(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """One multipoint of this dimension per `line` in a vertex frame.
        Grouped exactly like `lines`, off the same fixture."""
        return (
            vertices.group_by("line", maintain_order=True)
            .agg(self.point())
            .select(self.multipoint())
        )

    def multilinestrings(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """One multilinestring of this dimension per `polygon` in a vertex frame,
        with one linestring per `ring` within it.
        Grouped exactly like `polygons`, off the same fixture."""
        return (
            vertices.group_by("polygon", "ring", maintain_order=True)
            .agg(self.point())
            .select("polygon", self.linestring())
            .group_by("polygon", maintain_order=True)
            .agg("line")
            .select(self.multilinestring())
        )

    def multipolygons(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """One multipolygon of this dimension per `multipolygon` in a vertex
        frame, with one polygon per `polygon` and one ring per `ring` within it."""
        return (
            vertices.group_by("multipolygon", "polygon", "ring", maintain_order=True)
            .agg(self.point())
            .select("multipolygon", "polygon", self.linestring())
            .group_by("multipolygon", "polygon", maintain_order=True)
            .agg("line")
            .select("multipolygon", self.polygon())
            .group_by("multipolygon", maintain_order=True)
            .agg("polygon")
            .select(self.multipolygon())
        )

    def lines_from_coords(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """The same lines as `lines`, out of one coordinate column per axis."""
        return (
            vertices.group_by("line", maintain_order=True)
            .agg(COORDS)
            .select(self.of_coords(geo.linestring).alias("line"))
        )

    def polygons_from_coords(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """The same polygons as `polygons`, out of one coordinate column per
        axis. Grouping twice is what nests them one list deeper."""
        return (
            vertices.group_by("polygon", "ring", maintain_order=True)
            .agg(COORDS)
            .group_by("polygon", maintain_order=True)
            .agg(COORDS)
            .select(self.of_coords(geo.polygon).alias("polygon"))
        )

    def multipoints_from_coords(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """The same multipoints as `multipoints`, out of one column per axis."""
        return (
            vertices.group_by("line", maintain_order=True)
            .agg(COORDS)
            .select(self.of_coords(geo.multipoint).alias("multipoint"))
        )

    def multilinestrings_from_coords(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """The same multilinestrings as `multilinestrings`, out of one
        coordinate column per axis."""
        return (
            vertices.group_by("polygon", "ring", maintain_order=True)
            .agg(COORDS)
            .group_by("polygon", maintain_order=True)
            .agg(COORDS)
            .select(self.of_coords(geo.multilinestring).alias("multilinestring"))
        )

    def multipolygons_from_coords(self, vertices: pl.DataFrame) -> pl.DataFrame:
        """The same multipolygons as `multipolygons`, out of one coordinate
        column per axis."""
        return (
            vertices.group_by("multipolygon", "polygon", "ring", maintain_order=True)
            .agg(COORDS)
            .group_by("multipolygon", "polygon", maintain_order=True)
            .agg(COORDS)
            .group_by("multipolygon", maintain_order=True)
            .agg(COORDS)
            .select(self.of_coords(geo.multipolygon).alias("multipolygon"))
        )


XY = Dimension(
    PointXY,
    LineStringXY,
    PolygonXY,
    MultiPointXY,
    MultiLineStringXY,
    MultiPolygonXY,
    ("x", "y"),
)
XYZ = Dimension(
    PointXYZ,
    LineStringXYZ,
    PolygonXYZ,
    MultiPointXYZ,
    MultiLineStringXYZ,
    MultiPolygonXYZ,
    ("x", "y", "z"),
)
XYM = Dimension(
    PointXYM,
    LineStringXYM,
    PolygonXYM,
    MultiPointXYM,
    MultiLineStringXYM,
    MultiPolygonXYM,
    ("x", "y", "m"),
)
XYZM = Dimension(
    PointXYZM,
    LineStringXYZM,
    PolygonXYZM,
    MultiPointXYZM,
    MultiLineStringXYZM,
    MultiPolygonXYZM,
    ("x", "y", "z", "m"),
)

DIMENSIONS = [XY, XYZ, XYM, XYZM]


@pytest.fixture(params=DIMENSIONS, ids=lambda dim: dim.point_dtype.__name__)
def dimension(request: pytest.FixtureRequest) -> Dimension:
    """Parametrize a test over every supported dimension."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def coords() -> pl.DataFrame:
    """Loose coordinate columns, enough to build a point of any dimension.

    `m` is deliberately unlike the positional coordinates: it is a measure, and
    operations that move a geometry must leave it alone.
    """
    return pl.DataFrame(
        {
            "x": [1.0, -2.5, 0.0],
            "y": [3.0, 4.5, 0.0],
            "z": [10.0, 20.0, 0.0],
            "m": [100.0, 200.0, 300.0],
        }
    )


@pytest.fixture
def line_coords() -> pl.DataFrame:
    """Loose coordinate columns, usable for building lines.
    `line` says which line a vertex belongs to.
    The two are of different lengths.
    """
    return pl.DataFrame(
        {
            "line": ["a", "a", "a", "b", "b"],
            "x": [1.0, -2.5, 0.0, 7.0, 8.0],
            "y": [3.0, 4.5, 0.0, 9.0, 10.0],
            "z": [10.0, 20.0, 0.0, 30.0, 40.0],
            "m": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def ring_coords() -> pl.DataFrame:
    """Loose coordinate columns, usable for building polygons.

    `polygon` says which polygon a vertex belongs to, `ring` which ring of it.
    Ring 0 is the exterior ring; ring 1 of `a` is a hole in it. Every ring is
    closed, as the spec requires: its last vertex repeats its first.
    """
    return pl.DataFrame(
        {
            "polygon": ["a"] * 9 + ["b"] * 4,
            "ring": [0] * 5 + [1] * 4 + [0] * 4,
            "x": [0.0, 4.0, 4.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 10.0, 12.0, 10.0, 10.0],
            "y": [0.0, 0.0, 4.0, 4.0, 0.0, 1.0, 1.0, 2.0, 1.0, 10.0, 10.0, 12.0, 10.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 1.0],
            "m": [
                100.0,
                101.0,
                102.0,
                103.0,
                100.0,
                200.0,
                201.0,
                202.0,
                200.0,
                300.0,
                301.0,
                302.0,
                300.0,
            ],
        }
    )


@pytest.fixture
def multipolygon_coords(ring_coords: pl.DataFrame) -> pl.DataFrame:
    """The polygons of `ring_coords`, plus a third, `c`, grouped into multipolygons.

    `multipolygon` says which multipolygon a vertex belongs to:
    `p` holds polygons `a` (15.5, with its hole) and `b` (2.0),
    `q` holds only `c`, a 3x3 square (9.0).
    """
    square = pl.DataFrame(
        {
            "polygon": ["c"] * 5,
            "ring": [0] * 5,
            "x": [20.0, 23.0, 23.0, 20.0, 20.0],
            "y": [20.0, 20.0, 23.0, 23.0, 20.0],
            "z": [7.0] * 5,
            "m": [400.0, 401.0, 402.0, 403.0, 400.0],
        }
    )
    vertices = pl.concat([ring_coords, square])
    multipolygon = pl.when(pl.col("polygon") == "c").then(pl.lit("q"))
    return vertices.select(
        multipolygon.otherwise(pl.lit("p")).alias("multipolygon"), *vertices.columns
    )


def coordinates(df: pl.DataFrame, name: str = "point") -> pl.DataFrame:
    """The coordinates behind a point column, as plain float columns.

    `.ext.storage()` is the public way back to the values an extension column
    wraps, so tests assert on coordinates rather than on opaque point objects.
    """
    return df.select(pl.col(name).ext.storage()).to_series().struct.unnest()


def line_coordinates(df: pl.DataFrame, name: str = "line") -> pl.DataFrame:
    return (
        df.select(pl.col(name).ext.storage().explode(empty_as_null=False))
        .to_series()
        .struct.unnest()
    )


def multipoint_coordinates(df: pl.DataFrame, name: str = "multipoint") -> pl.DataFrame:
    return line_coordinates(df, name)


def ring_coordinates(df: pl.DataFrame, name: str = "polygon") -> pl.DataFrame:
    """The vertices behind a polygon column, every ring after the other."""
    return (
        df.select(
            pl.col(name)
            .ext.storage()
            .explode(empty_as_null=False)
            .explode(empty_as_null=False)
        )
        .to_series()
        .struct.unnest()
    )


def multilinestring_coordinates(
    df: pl.DataFrame, name: str = "multilinestring"
) -> pl.DataFrame:
    return ring_coordinates(df, name)


def multipolygon_coordinates(
    df: pl.DataFrame, name: str = "multipolygon"
) -> pl.DataFrame:
    """The vertices behind a multipolygon column, every ring of every polygon
    after the other."""
    return (
        df.select(
            pl.col(name)
            .ext.storage()
            .explode(empty_as_null=False)
            .explode(empty_as_null=False)
            .explode(empty_as_null=False)
        )
        .to_series()
        .struct.unnest()
    )
