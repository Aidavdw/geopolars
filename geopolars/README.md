# GeoPolars

If you are interested in developing GeoPolars further, read [DEVELOPERS.md](DEVELOPERS.md).

## Two APIs

We present two equivalent APIs.
All functionality is is available through both.

```python
# functional
from geopolars import geo

geo.translate("route", dx=1.0, dy=2.0)

# expressions using a namespace
import geopolars as gpl

gpl.col("route").geo.translate(1.0, 2.0)
```

Both are fully type-checked, but with an asterisk.
Plain `pl.col("route").geo...` works at runtime,
but a checker cannot see namespaces that `register_expr_namespace`
patches onto `pl.Expr`.
To avoid that, prefer `gpl.col` to keep static typing.

## Geometry Operations

All operations can be applied to all of the geometries.

### Distance

Does not need to reproject: Uses the CRS geodetic.

### Area

### Centroid

Gets the centre of mass of the geometry.
This operation takes the CRS into consideration.
For MultiPoint, this is the same as its mean coordinate.

### Mean Coordinate

This takes the average of all the coordinates that show up in this geometry.
It does not have to be CRS-aware.
