# GeoPolars

In this branch, we are rewriting large chunks of GeoPolars to work much more closely with the rest of Polars.
This is all very much still work-in-progress, and as such this branch is HIGHLY unstable.
We'll need a little bit more time to make things nice, but we're making a lot of progress very quickly,
so stay tuned!

## Features

GeoPolars has Geo data types as first-class Polars citizens,
  represented using the Arrow Extension format as [GeoArrow](https://geoarrow.org/).
  This has several advantages:

- zero-cost conversion with native polars data types.
- Allows using regular polars operations directly on these new data types
- Full compatibility with streaming engine
- Mismatches, illegal operations etc. can be caught at plan time or expression creation time,
  rather than at runtime.

In addition, GeoPolars aims to be a modern GeoSpatial toolset, offering:

- CRS-aware by default, backed by PROJ
- Support for 3D (z) and measured quantities (m) for most operations
- (De)serialisation with GeoParquet

### Developer Documentation

If you are interested in the design paradigm and architecture used for this plugin,
See the `Docs` directory for more information about the internals of this plugin.

- [A high-level overview of the architecture of the plugin](./docs/Architecture.md)
- [A document that describes what happens when you call an expression, at plan and execution time](./docs/Typical expression flow.md).
  This is accompanied by [a diagram](./docs/expression-flow.dot).
- [What to consider when implementing new or editing existing expressions](./docs/Expression implementation guide.md)
- [An overview of which expression is implemented at which *tier*](./docs/Tier per expression.md)

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

To look into:
how does geopandas/shapely/ link to geos?

- We can't ship GEOS as statci lib, can we do it as so/dll?
- delete my scan parquet wrapper
- Show the metadata/crs in the long name
- crs must be present in the python data type, not just in the metadata of a column
