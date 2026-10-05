# GeoPolars Developer Guide

## Three-tier implementation

1. Native. This functionality can be expressed directly using existing Polars nodes.
   Example: The mean coordinate. We can directly access the individual coordinate columns,
   and use Polars' mean implementation on it.
   If this is possible, this is preferred because you will never re-invent the wheel.
   Such operations will always be the most efficient,
   and the planner will be able to apply further optimisations.
   You might have to use `map_with_dtype` here to change what you wish to lower it to based on the
   metadata of the columns.

2. Plugin Expression. This functionality has a dedicated Rust kernel that lives in the plugin.
   This makes them very fast, but they cannot be subdivided at plan-time.
   These are ideally element-wise if possible.

3. Forward to Geo
   The [geo ecosystem](https://georust.org/)
   (which we refer to as *geors* in this crate to avoid confusion)
   gives very fast implementations for many geo algorithms,
   but their data encoding is not fully compatible with ours. We can use their algorithms,
   but we incur at least one allocation and copy pass for every item processed this way
   due to data conversion.
   Example: area calculation.
   In addition, geors has a couple of limitations:
   - it only supports 2D operations
   - No M (measure) values.
   - No CRS-awareness on a data level (this has to be carried separately)
   - no curved geometries
