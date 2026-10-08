"""Registering the GeoArrow types on both sides of the boundary.
Importing this module registers the types in *both* the Polars registry and the cdylib.
They *have to* agree on names.
"""

from __future__ import annotations

import polars as pl

# Importing the compiled module runs `PyInit_geopolars`,
# which is what registers these types on the Rust side.
from geopolars import geopolars as _rust  # noqa: F401
from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.box import GeoBox
from geopolars.datatypes.encoded import EncodedGeometry
from geopolars.datatypes.linestring import GeoLineString
from geopolars.datatypes.multilinestring import GeoMultiLineString
from geopolars.datatypes.multipoint import GeoMultiPoint
from geopolars.datatypes.multipolygon import GeoMultiPolygon
from geopolars.datatypes.point import GeoPoint
from geopolars.datatypes.polygon import GeoPolygon
from geopolars.datatypes.wkb import Wkb
from geopolars.datatypes.wkt import Wkt

# Mirrors `Kind::ALL`
GEOMETRIES: tuple[type[GeoArrowType], ...] = (
    GeoPoint,
    GeoLineString,
    GeoPolygon,
    GeoMultiPoint,
    GeoMultiLineString,
    GeoMultiPolygon,
)

for _geometry in GEOMETRIES:
    pl.register_extension_type(_geometry._extension_name, _geometry)

pl.register_extension_type(GeoBox._extension_name, GeoBox)

# Mirrors `Encoding::ALL`
ENCODINGS: tuple[type[EncodedGeometry], ...] = (Wkb, Wkt)

for _encoding in ENCODINGS:
    pl.register_extension_type(_encoding._extension_name, _encoding)
