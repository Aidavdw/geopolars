# geoarrow_abstr

Kernels and bindings to kernels for GeoArrow.
Instead of supplying an arrow type, this crate defines traits,
which can be implemented for your specific arrow implementation.
Kernels use these trait types directly.

The crate holds:

- the views and output traits, at the crate root.
- the metrics that kernels measure with (`Planar`, `GeodesicMetric`).
- kernels for simple operations implemented directly on the arrow type behind the `native` feature.
- kernels forwarding to GEOS in `geos`, behind the `geos` feature.

## Views

The views are zero-allocation references to the underlying storage.
They are split up by validity, because the GeoArrow spec says
validity is only ever expressed at the outer level.
A kernel on a nested type like Polygon therefore never has to check validity below the rows.

The coordinate traits, from 'least valid' to 'most valid', are:

- `GeoArrowCoordsView`, which considers the validity of coordinates.
  Use this only for Points.
- `GeoArrowValidCoordsView`, which assumes every coordinate is valid.
  Use this for nested geometry types like Polygon.
  **Do not** use this for Points.

The list levels above the coordinates are split the same way:

- `GeoArrowListView`, which considers the validity of lists.
  Use this for the outer level of a geometry column.
- `GeoArrowValidListView`, which assumes every list is valid.
  Use this for the inner levels of a nested geometry (the linestrings of a multilinestring).

Kernels take these held together per geometry type,
such as `LineStringArrayView` or `PolygonArrayView`.
They build their output through a trait as well
(`GeoArrowF64Array`, `GeoArrowBoolArray`, `GeoArrowPolygonOutput`, ...),
so the implementation decides what array comes out.

## Writing kernels

### Valid input

A kernel can always assume its input is valid according to the GeoArrow spec.
Only the outer level may have nulls.
All other levels, including the coordinate values, contain none.
There is therefore no need to branch on what to do with invalid data.
Making sure of this is up to whoever builds the views.
Every kernel in turn has to give valid output.

This does **not** shield you from NaNs!
A valid Polygon cannot have missing coordinates, but it might still have e.g. a NaN x-coordinate.
They show up in two ways.

- Individual NaN values (e.g. `x = 32.1, y = NaN`).
- *Empty* points (`x = NaN, y = NaN`).

How to handle these is up to the kernel.

### Branching inside a loop

Try to keep branching outside of kernel loops.
If you can already do it outside, do it outside.
Sometimes you conditionally need to do something extra
(e.g. also check for overlaps in a validation loop).
Write that as a generic function instead, and dispatch to its specific versions.
The `with_const_flags!` macro is very useful here.
Be careful you don't blow it up with combinatorics though.

### Fallible functions

How to handle errors depends on where they are raised.

- An error before looping over the rows would make *all rows* invalid.
  Return it, so the user sees it.
- An error in a single row does not affect the other rows.
  Switch on an `ignore_per_row_errors` flag.
  Errors are the exception, and we cannot pre-filter for them.
  So here it is fine to check the `bool` in the hot path, rather than make it a const generic.
  If `ignore_per_row_errors=false`, which is the default, return the error.
  This gives the user a way to inspect errors if they want.
  If `ignore_per_row_errors=true`, mark the row as missing instead (not as an empty geometry!).
  The output traits take the rows that failed for this.
- An accumulator-like function combines many rows into far fewer results.
  Ask the caller how to handle errors here, with an `IGNORE_ERRORS` const `bool`.
  The default should be to return the error.
