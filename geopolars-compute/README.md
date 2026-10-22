# geopolars-compute

Kernels for GeoArrow.
They take the views of [`geopolars-core`](../geopolars-core/README.md),
and return Polars arrays.

The crate holds:

- the metrics that kernels measure with (`Planar`, `GeodesicMetric`).
- kernels for simple operations implemented directly on the arrow type in `native`, behind the `native` feature.
- kernels forwarding to GEOS in `geos`, behind the `geos` feature.
- kernels forwarding to rsgeo (the Rust `geo` crate) in `rsgeo`, behind the `rsgeo` feature.
- kernels forwarding to PROJ in `proj`.
- `is_valid`'s kernels in `validity_checking`, which run the checks of `native` and `rsgeo` in one pass per row.
  These need both features.
- the pass that moves every missing value below a row out to the row in `inner_nulls`.

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
  The `geopolars-core` output functions take the rows that failed for this.
- An accumulator-like function combines many rows into far fewer results.
  Ask the caller how to handle errors here, with an `IGNORE_ERRORS` const `bool`.
  The default should be to return the error.
