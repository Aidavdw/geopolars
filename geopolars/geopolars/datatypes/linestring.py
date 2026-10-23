"""The `geoarrow.linestring` extension type."""

from __future__ import annotations

from typing import ClassVar

from geopolars._docs import doc
from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class LineStringType(GeoArrowType):
    """Base class for `geoarrow.linestring`."""

    _extension_name: ClassVar[str] = "geoarrow.linestring"
    _display: ClassVar[str] = "linestring"
    _nesting: ClassVar[int] = 1


class LineStringXY(LineStringType):
    """A 2D linestring: `geoarrow.linestring` over `List<Struct<x: f64, y: f64>>`.

    One list of vertices per linestring. That list may be empty: unlike a point,
    an empty linestring is a geometry this layout can hold.
    """

    _dimension: ClassVar[Dimension] = XY


class LineStringXYZ(LineStringType):
    """A 3D linestring: `geoarrow.linestring` over `List<Struct<x, y, z: f64>>`.

    `z` is the elevation of a vertex.
    """

    _dimension: ClassVar[Dimension] = XYZ


@doc()
class LineStringXYM(LineStringType):
    """A 2D linestring with a measure: `List<Struct<x, y, m: f64>>`.

    {linestring_measure}
    """

    _dimension: ClassVar[Dimension] = XYM


@doc()
class LineStringXYZM(LineStringType):
    """A 3D linestring with a measure: `List<Struct<x, y, z, m: f64>>`.

    {linestring_measure}
    """

    _dimension: ClassVar[Dimension] = XYZM
