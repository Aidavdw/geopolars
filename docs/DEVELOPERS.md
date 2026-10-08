# GeoPolars Developer Guide

## Architecture

### A Polars plugin

GeoPolars is a [Polars plugin](https://docs.pola.rs/user-guide/plugins/):
a Python package with a compiled Rust library (as `cdylib`).
The two halves each have their own job:

- **Python** (`geopolars/`) is what users interact with.
  It defines the dtypes, builds the expressions,
  and decides at plan time which implementation each geometry gets.
- **Rust** (`src/`) holds the kernels that plain Polars expressions can't express.
  Polars loads the library itself and looks up each plugin function by itself.

The Python side points Polars at the library through `LIB` in `_utils.py`,
which is the package directory.

### GeoArrow data types

Geometries are stored as [GeoArrow](https://geoarrow.org/),
using [Polars' Arrow extension types](https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Extension.html).
An extension type consists of a name (`geoarrow.point`, `geoarrow.polygon`, ...),
an ordinary Polars storage dtype, and optional metadata (where the CRS lives).
The storage is a struct of separate `xy[/z][/m]` `f64` fields (referred to as *coordinates*),
wrapped in a `List` per level of nesting (a point has 0, a linestring 1, a polygon 2).

| What | Python | Rust |
| - | - | - |
| List of geometries | `GEOMETRY_DTYPES` in `datatypes/registry.py` | `Kind::ALL` in `src/geoarrow/kind.rs` |
| Common base type | `GeoArrowType` in `datatypes/base.py` | `Geo` in `src/geoarrow/geo.rs` |
| One geometry | `datatypes/point.py`, `polygon.py`, ... | a `Kind` variant |
| Dimension (`xy`, `xyzm`) | `datatypes/dimension.py` | `src/geoarrow/dimension.rs` |
| CRS metadata | goes through `extension_metadata` (Rust) | `src/geoarrow/crs.rs` |

Two things work differently on the two sides:

- In Python, each geometry *and dimension* is its own class (`PointXY`, `PolygonXYZM`),
  so users and type checkers can tell them apart.
  In Rust, there is a single `Geo { kind, dim }` and code matches on its enums,
  the same way Polars itself dispatches on dtypes.
- Metadata JSON is only ever handled in Rust to keep a single source of truth.
  Python calls `extension_metadata(crs=...)` rather than writing the JSON itself.
  The reason for this is twofold:
  It stops operations on the metadata from drifting (lowering maintenance burden),
  and it ensures that metadata processed by both sides can be compared.
  Dtypes compare their metadata as strings,
  so if the two sides serialised the same CRS differently
  you'd get two dtypes that don't compare equal.
  In Rust, metadata is held parsed as an `ExtensionMetadata`, behind an `Arc`,
  so it compares by content and is written back out as compact JSON.

`geoarrow.box` (`datatypes/box.py`, `src/geoarrow/bbox.rs`), which represents a bounding box,
is registered on both sides too, but it is not one of these geometries:
its storage is a flat struct of bounds (`xmin, ymin, ..., xmax, ymax, ...`),
so, like WKB and WKT, it is not a `Kind`, and it is not accepted by most operations.

Rust expressions never name a concrete geometry.
They call `geoarrow::describe(dtype)` to get a `GeoColumn` (kind and dimension),
then either ignore it (`coords::map_coords`) or match exhaustively on `Kind`.
That way, adding a geometry only touches `src/geoarrow/` and doesn't affect `src/expr/`.

#### Extension registries

There are two extension-type registries, one in the host `polars` wheel and one inside our `cdylib`,
so **both Python and Rust have to register every type, and the names have to agree**.
Importing `geopolars` handles both:
`datatypes/registry.py` imports the compiled module,
which runs `PyInit_geopolars` and registers the Rust side,
then calls `pl.register_extension_type` for the Python side.

### Expressions

Expressions are the operations on these types.
Each operation is written once, as a function in the **functional API** (`geopolars/geo/`),
which is available as `from geopolars import geo` and then `geo.area("parcel")`.
The **namespace API** (`geopolars/expr/geo.py`, `gpl.col("parcel").geo.area()`)
is a thin wrapper in which every method just forwards to the functional version.
New functionality is added in `geo/`, plus a one-line method in the namespace.

Because of this, dependencies only go one way.
Nothing in `datatypes` imports from `geo`, and nothing in `geo` imports from `expr`.

```text
expr (namespaces) →  geo (functional) →  datatypes.
```

On the Rust side, `src/expr/` mirrors `geo/` module by module (`area.rs`, `affine.rs`, `crs.rs`, ...),
and `src/geoarrow/` is the shared glue: dtypes, dispatch, and reading the Arrow storage in a kernel.

#### Type checker

`register_expr_namespace` patches `.geo` onto `pl.Expr` at runtime, which a checker can't see.
To fix this, we make `gpl.col` / `as_plugin` return a `PluginExpr` subclass that declares it.
This is the purpose of `expr/plugin.py`.

## Testing

Prefer testing from the python side, as this saves a lot of compilation time.
That way, you are also much more sure you are actually testing behaviour,
rather than internal implementation details :)

Rust tests can still be necessary if a certain functionality is only reachable from rust directly.

## Three-tier implementation

### 1. Native

Native. This functionality can be expressed directly using existing Polars nodes.
An example: The mean coordinate.

We can directly access the individual coordinate columns,
and use Polars' mean implementation on it.
If this is possible, this is preferred because you will never re-invent the wheel.
Such operations will always be the most efficient,
and the planner will be able to apply further optimisations.

You might want to change exactly what you want to put here using `pipe_with_dtype`
(through `on_geometry`) to change what you wish to lower it to based on the metadata.
For that, you can dispatch on the input's geometry through `on_geometry` in `geo/_dispatch.py`.
This wraps `Expr.pipe_with_dtype`, so the callback runs once at plan time
with the concrete geometry dtype and returns a plain expression for that geometry only.
The dtype carries the column's metadata (such as the CRS).
A callback that builds a geometry has to pass it on with `_with_metadata_of`,
or the result silently loses its CRS (`tests/unit/operations/test_metadata.py` checks this).
`on_geometry_pair` does the same for operations between two geometries.
Wrong input (not a geometry, or a geometry we don't support) therefore fails while the plan is built,
rather than partway through execution.
The callback can return native Polars expressions or a custom kernel call into Rust,
imported using `register_plugin_function`.
The tiers below cover that choice.

### 2. Plugin Expression

This functionality has a dedicated Rust kernel that lives in the plugin.
This makes them very fast, but they cannot be subdivided at plan-time.
These are ideally element-wise if possible.

### 3. Forward to external

A subcategory of the above:
the Rust kernel hands our data to an external library,
putting it into that library's own data type.
This usually means at least one allocation and copy pass
for every item processed this way, so prefer tier 2 where it is feasible.
In return we don't have to re-implement (and maintain) well-tested algorithms.

Examples are:

- [PROJ](https://proj.org/) (through `rsgeo::proj`), to reproject coordinates.
  Each coordinate is passed to it as a tuple, one by one.
- [geographiclib](https://geographiclib.sourceforge.io/), to measure along an ellipsoid.
- The [geo ecosystem](https://georust.org/)
  (which we refer to as *rsgeo* in this crate to avoid confusion with our own `geo`),
  which has very fast implementations for many geo algorithms,
  but whose data encoding is not compatible with ours.
  In addition, rsgeo has a couple of limitations:
  - it only supports 2D operations
  - no M (measure) values
  - no CRS-awareness on a data level (this has to be carried separately)
  - no curved geometries

### Tier per operation

Which tier each operation uses, per geometry type.
`1` is native, `2` a plugin expression, `3` forwarded to an external library,
and `-` means the operation refuses that geometry.
All dimensions of a geometry share a column.
Geometries usually carry a CRS, so the tier given is for a geometry with a CRS;
the comment says when one without a CRS takes a different tier.

| Operation | Point | | LineString | | Polygon | | MultiPoint | | MultiLineString | | MultiPolygon | |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| `area` | 2 | `0.0`; no CRS: 1 | 2 | `0.0`; no CRS: 1 | 3 | PROJ + ggl; no CRS: 1 (shoelace) | 2 | `0.0`; no CRS: 1 | 2 | `0.0`; no CRS: 1 | 3 | as list of Polygon |
| `length` | - | | 3 | PROJ + ggl; no CRS: 1 | - | | - | | 3 | as list of LineString | - | |
| `distance` | 3 | PROJ + ggl; no CRS: 1 | - | | - | | - | | - | | - | |
| `mean_coordinate` | 1 | returns the point | 1 | | 1 | skips ring's closing coord | 1 | | 1 | | 1 | skips ring's closing coord |
| `translate` | 2 | impl differs for const or col | 2 | | 2 | | 2 | | 2 | | 2 | |
| `rotate` | 2 | | 2 | | 2 | | 2 | | 2 | | 2 | |
| `skew` | 2 | `origin` may be a point column | 2 | | 2 | | 2 | | 2 | | 2 | |
| `scale` | 2 | impl differs for const or col | 2 | | 2 | | 2 | | 2 | | 2 | |
| `affine_transform` | 2 | `origin` may be a point column | 2 | | 2 | | 2 | | 2 | | 2 | |
| `bounds` | 1 | repeats its coordinate | 1 | | 1 | | 1 | `list[box]`, one per part | 1 | `list[box]`, one per part | 1 | `list[box]`, one per part |
| `set_crs` | 1 | relabels only | 1 | | 1 | | 1 | | 1 | | 1 | |
| `to_crs` | 3 | PROJ | 3 | PROJ | 3 | PROJ | 3 | PROJ | 3 | PROJ | 3 | PROJ |
| `is_geographic` | 1 | scalar, calls PROJ at plan | 1 | | 1 | | 1 | | 1 | | 1 | |
| `x` | 1 | `f64`, drops the CRS | 1 | `list[f64]` | 1 | `list[list[f64]]` | 1 | `list[f64]` | 1 | `list[list[f64]]` | 1 | `list[list[list[f64]]]` |
| `y` | 1 | `f64`, drops the CRS | 1 | `list[f64]` | 1 | `list[list[f64]]` | 1 | `list[f64]` | 1 | `list[list[f64]]` | 1 | `list[list[list[f64]]]` |
| `z` | 1 | like `x`; only with a `z` | 1 | | 1 | | 1 | | 1 | | 1 | |
| `m` | 1 | like `x`; only with an `m` | 1 | | 1 | | 1 | | 1 | | 1 | |
| `count_coordinates` | 1 | `0` if empty | 1 | | 1 | skips ring's closing coord | 1 | | 1 | | 1 | skips ring's closing coord |
| `is_empty` | 1 | NaN `x` and `y` | 1 | no verts or only empty | 1 | only empty rings | 1 | no verts, or only empty | 1 | only empty parts | 1 | only empty polygons |
| `wrap_longitude` | 1 | needs geographic CRS | 1 | | 1 | | 1 | | 1 | | 1 | |
| `to_wkb` | 2 | `wkb` crate | 2 | | 2 | | 2 | | 2 | | 2 | |
| `from_wkb` | 2 | `wkb` crate | 2 | | 2 | | 2 | promotes a point | 2 | promotes a linestring | 2 | promotes a polygon |
| `to_wkt` | 2 | `wkt` crate | 2 | | 2 | | 2 | | 2 | | 2 | |
| `from_wkt` | 2 | `wkt` crate | 2 | | 2 | | 2 | promotes a point | 2 | promotes a linestring | 2 | promotes a polygon |

In this table:

- `ggl` means 'geographiclib'

## Typical flow

This section describes what happens when you add a typical expression
(in this case, `geo.area`) to a query.
This should give you a good idea of how the plugin works.
This is a written description of `docs/expression-flow.dot`
(which you can render with `dot -Tsvg docs/expression-flow.dot -o flow.svg`).

At two points, the paths diverge depending on which tier the operation is implemented at.
The numbering here is the same as before.
Blue steps run in our Python package, orange ones in our library.
Every red edge crosses the FFI.
What happens inside Polars itself (in-between) is left out.

### Adding an expression

( The top half of `docs/expression-flow.dot` )

This code mostly lives in the Python part of GeoPolars.

There are two ways in:

- The functional API (A2), with a column name, an expression or a Series.
- The namespace (`ExprGeoNameSpace.area`) which simply forwards (A1) to the functional API directly.

From there, the operation goes one of two ways,
depending on whether it needs to be aware of type information (A3).
The input's dtype isn't known yet, so nothing can be checked against it.
This matters for tier 1 functions,
but also for some other functions which switch their approach based on the geometry or metadata.
An example of such a function would be `area`, which is tier 3 for a polygon with a CRS
(which is stored in metadata), but tier 1 for one without.
These need to choose at plan-time, so they can decide when this extra information is available.
They do this with a callback, which is called in P4.

For some expressions, you would always use a tier 2 or tier 3 implementation (A4).
These can use the simple path.

#### With a plan-time callback

`on_geometry` turns the input into an expression (A5),
where a name becomes `pl.col`, a Series `pl.lit`.
It then hands the operation's callback (unique for every operation) to `pipe_with_dtype` (through its own wrapper, `resolved` ) (A6).
This yields an ordinary `pl.Expr` that carries the callback that can run at plan time.
It picks up again at P1.

#### The simple path

These do not need to know the exact data type here.
The rust side will do the switching inside of their tier 2 or tier 3 kernel.

For tier 2, the arguments are packed into `kwargs` (A7),
which the rust side can later deserialize into a struct like `AffineMatrix`.
Tier 3 works basically the same way here, with kwargs for what the external library needs (A9),
(e.g. the target CRS of `to_crs`).

( For some complex ops, a dtype the operation is asked to produce can't cross as a Python object,
so `from_wkb` flattens it into kind, dimension and CRS (`_decode_kwargs`).)

`register_plugin_function` then names the kernel by its symbol in `LIB` (A8).
That records the name, so the appropriate rust function can be called later in P8.

The forwarding kernel is named like any other (A10), and picks up again at P13.

### Planning an expression

( The bottom half of `docs/expression-flow.dot` )

Type level checking is implemented on this level as much as possible,
so that they can raise while the plan is being built, rather than during execution.

When the query is collected, Polars resolves the input's dtype and calls back into our code.
A Tier 2 or 3 expression that was directly chosen without a `pipe_with_dtype`
(from A8 or A10) does not use a callback, and goes straight to P8 or P13.

The other path is one that requires a callback.
A Tier 1 expression (or one that resolves based on metadata) from A6 starts at P1.

#### Callbacks with geometry and metadata

Polars hands the dtype back (across the FFI) with the python callback.
This will be one of our registered new types, such as `PolygonType` (P1).
It can then choose a concrete subclass (e.g. `PolygonXY`),
based on the dimension using its implementation for `ext_from_params`.
Metadata is left as-is for now.
Storage we don't support, such as interleaved coordinates, raises here.
`_geometry_of` then refuses anything that isn't one of our geometries,
such as a `Wkb` column or a plain struct (P2).

If the choice of implementation depends on the metadata, the callback calls into Rust (P3):
`_declares_crs` calls `declares_crs` in our library directly through pyo3, not through Polars.
Right now we only check whether a CRS exists, but we might expand on that.
We could return the entire CRS back over to Python here.

Finally, the callback chooses the implementation (P4).
`on_geometry` gives whatever it returns the input's name.

#### Tier 1: native

The callback builds the result out of the storage in plain Polars expressions (P5).
`.ext.storage()` drops the extension type,
and the class's `_nesting`, `_rings` and `_dimension` say how many `List` layers there are,
and which coordinate fields.
A geometry result gets its dtype back with `.ext.to(...)` (P6).
Along with it, input's metadata is carried as an unparsed string using `_with_metadata_of`,
Some plain results (`f64`) need nothing, so the CRS can be discarded then.
Polars can then read the dtype off the expression as its name, storage and metadata.

After this, our code is done with the expression.
The plan only holds Polars nodes, which can be optimised like any other.

#### Tier 2: plugin expression

We can get here either directly from a registered kernel that was lowered immediately (A8),
or because a callback at planning time lowered to this (P4).

Which function is run depends on what was registered as its `register_plugin_function` call (P7).
Polars then calls `_polars_plugin_field_<name>` in our library over the C ABI (P8),
passing the input fields as Arrow C schemas and the pickled kwargs.
Importing a field looks its name up in our library's own registry.
`GeoFactory` can now parse the field.
This reads:

- The `Kind` from the name
- the `GeoDimension` from the storage
- the `ExtensionMetadata` from its metadata JSON.

That `GeoFactory` parses that into a `Geo`,
or an `Unsupported` if the storage or metadata didn't parse (the factory cannot return an error).
`describe` turns the dtype into a `GeoColumn` (P9),
and refuses an `Unsupported` or anything that isn't a geometry with a `SchemaMismatch`
(`from_wkb`, whose input is encoded, uses `describe_encoded` instead).
Whatever output type function (in Rust) this uses determines the output field type (P10).

- `same_geometry` hands back the input's field,
- `to_wkb` gives a `Wkb` carrying the same metadata.
- and any other function we can call to return that type.

The field goes back over the C ABI.
A dtype registered from Python, such as a geometry or `Wkb`,
is rebuilt through our `ext_from_params` in Python as in P1 (P11),
and a plain one (`f64`) goes straight back.
The plan now holds a node for our kernel, with a known output dtype.

Ready to run!

#### Tier 3: forward to external

Steps P12 to P16 are the same as P7 to P11, but now for a forwarding function.

## Roadmap

### Core IO

- [x] WKB encoding and decoding
- [x] WKT encoding and decoding
- [x] GeoParquet read
- [x] GeoParquet write
- [x] GeoParquet bbox covering (written on request, read as a box)

Later goals:

- [ ] GeoJson read
- [ ] PostGIS read
- [ ] GDAL vector files
- [ ] Interop with arrow

### Per-geometry properties and ops

- [x] Area
- [x] Length
- [x] getters for coordinates (z/m)
- [ ] CRS: opaque SRIDs refused for reprojection?
- [x] is_empty for all geo
- [ ] validity checking functions
- [ ] Handling closing of loop consistently for polygon
- [ ] mean coordinate
- [ ] centroid
- [ ] convex hull
- [ ] simplify/decimate
- [ ] equality

affine transforms:

- [x] translate
- [x] rotate
- [x] scale
- [x] skew
- [x] transform (with a matrix): `affine_transform`
- [x] shift (by a value in a different column)
- [x] rotate around coordinate from another column
- [x] scale around coordinate from another column
- [x] skew around a coordinate from another column
- [ ] rotate around own centroid (can use the above)
- [ ] scale around own centroid (can use the above)
- [ ] skew around own centroid (can use the above)

transfer transforms

- [ ] interpolate
- [ ] project

geometry editing

- [ ] Linestring joining
- [ ] polygon construction from linestrings
- [ ] Adding additional holes to polygons from linestrings

### Binary predicates

- [ ] intersect
- [ ] contains
- [ ] within
- [ ] covers
- [ ] covered_by
- [ ] touches
- [ ] crosses
- [ ] overlaps
- [ ] disjoint
- [ ] distance_to

creating new geometries

- [ ] intersection
- [ ] union
- [ ] difference
