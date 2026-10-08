"""The `geoarrow.multilinestring` extension type."""

# Same layout as a geoarrow.polygon, but its parts are linestrings rather than
# rings, so nothing here closes.
# TODO: and that the child name of the outer list should be "linestrings"

from __future__ import annotations

from typing import ClassVar

from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class MultiLineStringType(GeoArrowType):
    """Base class for `geoarrow.multilinestring`."""

    _extension_name: ClassVar[str] = "geoarrow.multilinestring"
    _display: ClassVar[str] = "multilinestring"
    _nesting: ClassVar[int] = 2


class MultiLineStringXY(MultiLineStringType):
    """A 2D multilinestring: `List<List<Struct<x: f64, y: f64>>>`."""

    _dimension: ClassVar[Dimension] = XY


class MultiLineStringXYZ(MultiLineStringType):
    """A 3D multilinestring: `List<List<Struct<x, y, z: f64>>>`."""

    _dimension: ClassVar[Dimension] = XYZ


class MultiLineStringXYM(MultiLineStringType):
    """A 2D multilinestring with a measure: `List<List<Struct<x, y, m: f64>>>`."""

    _dimension: ClassVar[Dimension] = XYM


class MultiLineStringXYZM(MultiLineStringType):
    """A 3D multilinestring with a measure: `List<List<Struct<x, y, z, m: f64>>>`."""

    _dimension: ClassVar[Dimension] = XYZM
