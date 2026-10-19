# geopolars-core

Views over the Polars storage of GeoArrow geometries,
and the arrays kernels build their output in.
The kernels in [`geopolars-compute`](../geopolars-compute/README.md) take these directly.

## Views

The views are zero-copy: they share the Polars buffers (`Buffer`, `OffsetsBuffer`, `Bitmap`),
which only bumps a refcount once per chunk.
They are split up by validity, because the GeoArrow spec says
validity is only ever expressed at the outer level.
A kernel on a nested type like Polygon therefore never has to check validity below the rows.

The coordinate views, from 'least valid' to 'most valid', are:

- `Coords`, which considers the validity of coordinates.
  Use this only for Points.
- `ValidCoords`, which assumes every coordinate is valid.
  Use this for nested geometry types like Polygon.
  **Do not** use this for Points.

The list levels above the coordinates are split the same way:

- `List`, which considers the validity of lists.
  Use this for the outer level of a geometry column.
- `ValidList`, which assumes every list is valid.
  Use this for the inner levels of a nested geometry (the linestrings of a multilinestring).

Kernels take these held together per geometry type,
such as `LineStringArray` or `PolygonArray`.
They build their output with the functions in `output` (`f64_array`, `polygon_array`, ...).

The single-geometry types (`Point`, `LineString`, ...) implement geo-traits,
for the `wkb` and `wkt` writers.

`AxisArrays` considers the validity of each coordinate field separately
('x' may be valid, but 'y' may not be).
Use it only to check that something is valid GeoArrow.
