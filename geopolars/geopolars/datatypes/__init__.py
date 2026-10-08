"""GeoArrow extension data types.

GeoArrow uses one extension name per geometry, covering every dimension.
XY vs XYZ is a property of the coordinates it stores,
so one registered name and one dispatching class covers them all.
"""

from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.box import BoxType, BoxXY, BoxXYM, BoxXYZ, BoxXYZM
from geopolars.datatypes.linestring import (
    LineStringType,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    LineStringXYZM,
)
from geopolars.datatypes.multilinestring import (
    MultiLineStringType,
    MultiLineStringXY,
    MultiLineStringXYM,
    MultiLineStringXYZ,
    MultiLineStringXYZM,
)
from geopolars.datatypes.multipoint import (
    MultiPointType,
    MultiPointXY,
    MultiPointXYM,
    MultiPointXYZ,
    MultiPointXYZM,
)
from geopolars.datatypes.multipolygon import (
    MultiPolygonType,
    MultiPolygonXY,
    MultiPolygonXYM,
    MultiPolygonXYZ,
    MultiPolygonXYZM,
)
from geopolars.datatypes.point import (
    PointType,
    PointXY,
    PointXYM,
    PointXYZ,
    PointXYZM,
)
from geopolars.datatypes.polygon import (
    PolygonType,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)
from geopolars.datatypes.registry import GEOMETRY_DTYPES
from geopolars.datatypes.wkb import Wkb
from geopolars.datatypes.wkt import Wkt

__all__ = [
    "GEOMETRY_DTYPES",
    "BoxType",
    "BoxXY",
    "BoxXYM",
    "BoxXYZ",
    "BoxXYZM",
    "GeoArrowType",
    "LineStringType",
    "LineStringXY",
    "LineStringXYM",
    "LineStringXYZ",
    "LineStringXYZM",
    "MultiLineStringType",
    "MultiLineStringXY",
    "MultiLineStringXYM",
    "MultiLineStringXYZ",
    "MultiLineStringXYZM",
    "MultiPointType",
    "MultiPointXY",
    "MultiPointXYM",
    "MultiPointXYZ",
    "MultiPointXYZM",
    "MultiPolygonType",
    "MultiPolygonXY",
    "MultiPolygonXYM",
    "MultiPolygonXYZ",
    "MultiPolygonXYZM",
    "PointType",
    "PointXY",
    "PointXYM",
    "PointXYZ",
    "PointXYZM",
    "PolygonType",
    "PolygonXY",
    "PolygonXYM",
    "PolygonXYZ",
    "PolygonXYZM",
    "Wkb",
    "Wkt",
]
