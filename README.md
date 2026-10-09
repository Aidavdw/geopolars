# GeoPolars

In this branch, we are rewriting large chunks of GeoPolars to work much more closely with the rest of Polars.
This is all very much still work-in-progress, and as such this branch is HIGHLY unstable.
We'll need a little bit more time to make things nice, but we're making a lot of progress very quickly,
so stay tuned!

## Features

- Geo data types as first-class Polars citizens,
  represented using the Arrow Extension format as [GeoArrow](https://geoarrow.org/),
  enabling zero-cost conversion with native polars data types.
- CRS-aware by default, backed by PROJ
- Support for 3D (z) and measured quantities (m) for most operations
- (De)serialisation with GeoParquet

### Developer Documentation

If you are interested in the design paradigm and architecture used for this plugin,
or are curious about developing GeoPolars further, read [DEVELOPERS.md](docs/DEVELOPERS.md).

### Two APIs

We present two equivalent APIs.
All functionality is is available through both.

```python
# functional
from geopolars import geo

geo.translate("route", (1.0, 2.0))

# expressions using a namespace
import geopolars as gpl

gpl.col("route").geo.translate((1.0, 2.0))
```

Both are fully type-checked, but with an asterisk.
Plain `pl.col("route").geo...` works at runtime,
but a checker cannot see namespaces that `register_expr_namespace`
patches onto `pl.Expr`.
To avoid that, prefer `gpl.col` to keep static typing.

### Geometry Operations

All operations can be applied to all of the geometries.

#### Distance

Does not need to reproject: Uses the CRS geodetic.

#### Area

#### Centroid

Gets the centre of mass of the geometry.
This operation takes the CRS into consideration.
For MultiPoint, this is the same as its mean coordinate.

#### Mean Coordinate

This takes the average of all the coordinates that show up in this geometry.
It does not have to be CRS-aware.
