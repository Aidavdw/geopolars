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
| List of geometries | `GEOMETRIES` in `datatypes/registry.py` | `Kind::ALL` in `src/geoarrow/kind.rs` |
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

| Operation | Point | | LineString | | Polygon | | MultiPoint | | MultiLineString | |
| - | - | - | - | - | - | - | - | - | - | - |
| `area` | 2 | `0.0`; no CRS: 1 | 2 | `0.0`; no CRS: 1 | 3 | PROJ + geographiclib; no CRS: 1 (shoelace) | 2 | `0.0`; no CRS: 1 | 2 | `0.0`; no CRS: 1 |
| `length` | - | | 3 | PROJ + geographiclib; no CRS: 1 | - | | - | | 3 | calls LineString impl as list; no CRS: 1 |
| `distance` | 3 | point to point only; PROJ + geographiclib; no CRS: 1 | - | | - | | - | | - | |
| `mean_coordinate` | 1 | returns the point itself | 1 | | 1 | skips each ring's closing coordinate | 1 | | 1 | |
| `translate` | 2 | | 2 | | 2 | | 2 | | 2 | |
| `to_crs` | 3 | PROJ | 3 | PROJ | 3 | PROJ | 3 | PROJ | 3 | PROJ |
