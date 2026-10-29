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
These are built from a chunk of the Polars storage with `new`.
A kernel returning geometries returns these types too.
`into_list_array` turns them back into Polars storage (`into_struct_array` for `PointArray`).
A kernel returning a value per row builds it with the functions in `output` (`f64_array`, ...).

The single-geometry types (`ScalarPoint`, `ScalarLineString`, ...) implement geo-traits,
for the `wkb` and `wkt` writers.

`AxisArrays` considers the validity of each coordinate field separately
('x' may be valid, but 'y' may not be).
Use it only to check that something is valid GeoArrow.
`UncheckedStorage` holds a chunk as it comes from Polars, with the validity of every level.
Use it only to move the null values below the rows out to the rows
(`inner_nulls` in `geopolars-compute`).

## Trimming

A chunk can be a slice of a larger array, such as a streaming morsel or the result of `slice`.
Then only its rows are sliced.
The levels below them still hold every list and coordinate of the larger array.
A kernel that walks a whole level would also process everything outside the chunk.

So `new` trims: it slices every level of the view to what its rows use.
This shares the buffers, but copies the offsets of a level when they do not start at zero.
`UncheckedStorage::new` trims as well.
A `PointArray` needs no trimming, as slicing a struct array slices its fields.
A view a kernel builds has to be trimmed too: build its levels fresh, or reuse those of a trimmed view.
Then a kernel can walk any level of a view it is given, unless that view is `Untrimmed`.

A kernel that only reads the items of its rows does not need this.
To skip the cost of trimming, it can take an `Untrimmed` view instead, built with `new_untrimmed`.
It reaches the levels below the rows only through the offsets of the rows.
Only do this where the trim is a noticeable part of the kernel's cost.
