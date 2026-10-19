//! Glue for the GeoArrow data types, and dispatching to it.
//!
//! Expressions never name a concrete geometry:
//! They ask [`describe`] what a column holds and dispatch on the [`Kind`] and [`Dimension`] it reports,
//! so adding a geometry does not touch `crate::expr`.

pub mod bbox;
pub mod coord;
pub mod crs;
pub mod decode;
pub mod encoded;
mod geo;
mod kind;

pub use geo::Geo;
pub use geopolars_core::GeoDimension;
pub use kind::Kind;

use std::any::Any;
use std::sync::Arc;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use polars_core::datatypes::extension::{register_extension_type, ExtensionTypeInstance};

use geopolars_core::{
    downcast, Coords, Geometry, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ValidCoords, ValidList,
};

use bbox::BoxFactory;
use crs::ExtensionMetadata;
use encoded::{EncodedFactory, Encoding};
use geo::GeoFactory;

/// Populate this library's extension-type registry.
/// This exposes the types from this plugin's library.
/// This cannot run lazily.
///
/// Registering also needs to happen in the host `polars` wheel,
/// which is done with `pl.register_extension_type`.
/// Both are needed, and they must agree on names.
pub fn register() -> PolarsResult<()> {
    for kind in Kind::ALL {
        register_extension_type(kind.name(), Some(Arc::new(GeoFactory(kind))))?;
    }
    for encoding in Encoding::ALL {
        register_extension_type(encoding.name(), Some(Arc::new(EncodedFactory(encoding))))?;
    }
    register_extension_type(bbox::NAME, Some(Arc::new(BoxFactory)))?;
    Ok(())
}

/// A view of a geometry column, containing only what an expression over it needs.
pub struct GeoColumn<'a> {
    /// The [`ExtensionTypeInstance`] the column came in, so the result can be
    /// put back under it without losing metadata.
    pub typ: &'a ExtensionTypeInstance,
    pub kind: Kind,
    pub dim: GeoDimension,
    pub metadata: &'a Arc<ExtensionMetadata>,
}

/// Parses a geometry column's dtype (as a polars [DataType]) into a [GeoColumn].
pub fn describe(dtype: &DataType) -> PolarsResult<GeoColumn<'_>> {
    let DataType::Extension(typ, storage) = dtype else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    let Some(kind) = Kind::from_name(&typ.name()) else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    // `GeoFactory` already parsed everything, unless it had to give up.
    if let Some(geo) = (typ.0.as_ref() as &dyn Any).downcast_ref::<Geo>() {
        return Ok(GeoColumn {
            typ,
            kind: geo.kind(),
            dim: geo.dim(),
            metadata: geo.metadata(),
        });
    }

    polars_ensure!(
        coord::dimension_of_storage(storage, kind.nesting()).is_some(),
        // An interleaved encoding, most likely.
        SchemaMismatch: "expected a `{}` column with separated \
        x/y[/z][/m] coordinates, got: {}", kind.name(), dtype
    );
    ExtensionMetadata::parse(typ.serialize_metadata().as_deref())?;
    polars_bail!(
        SchemaMismatch: "`{}` is not registered with this library, got: {}", kind.name(), dtype
    )
}

/// Errors unless `a` and `b` declare the same CRS, or neither declares one.
/// Should be used as a backup, as this is runtime, not plan time.
pub fn ensure_same_crs(a: &GeoColumn, b: &GeoColumn) -> PolarsResult<()> {
    match (a.metadata.declares_crs(), b.metadata.declares_crs()) {
        (false, false) => Ok(()),
        (true, true) => {
            let (crs_a, crs_b) = (a.metadata.crs()?, b.metadata.crs()?);
            polars_ensure!(
                crs_a == crs_b,
                SchemaMismatch: "the geometries declare different CRSs, {crs_a} and {crs_b}; \
                use `to_crs` to bring one onto the other first"
            );
            Ok(())
        }
        (true, false) | (false, true) => polars_bail!(
            SchemaMismatch: "only one of the geometries declares a CRS; declare the same one on both"
        ),
    }
}

/// Calls `f` with `row(i)` for every row `i` of `chunk`, or `None` where it is null.
fn rows<'a>(
    chunk: &dyn Array,
    f: &mut impl FnMut(Option<Geometry<'a>>) -> PolarsResult<()>,
    row: impl Fn(usize) -> Geometry<'a>,
) -> PolarsResult<()> {
    (0..chunk.len()).try_for_each(|i| f((!chunk.is_null(i)).then(|| row(i))))
}

/// Calls `f` with every row of `chunk`, a chunk of a `kind` column's storage:
/// the geometry, or `None` for a null.
/// For the `wkb` and `wkt` writers, which take geometries through geo-traits.
pub fn for_each_geometry(
    chunk: &dyn Array,
    kind: Kind,
    mut f: impl FnMut(Option<Geometry<'_>>) -> PolarsResult<()>,
) -> PolarsResult<()> {
    match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            rows(chunk, &mut f, |i| Geometry::Point(array.get(i)))
        }
        Kind::LineString => {
            let rows_list = downcast::<ListArray<i64>>(chunk, "a list of coordinates")?;
            let array = LineStringArray {
                rows: List::new(rows_list),
                coords: ValidCoords::new(rows_list.values().as_ref())?,
            };
            rows(chunk, &mut f, |i| Geometry::LineString(array.get(i)))
        }
        Kind::MultiPoint => {
            let rows_list = downcast::<ListArray<i64>>(chunk, "a list of coordinates")?;
            let array = MultiPointArray {
                rows: List::new(rows_list),
                coords: ValidCoords::new(rows_list.values().as_ref())?,
            };
            rows(chunk, &mut f, |i| Geometry::MultiPoint(array.get(i)))
        }
        Kind::Polygon => {
            let rows_list = downcast::<ListArray<i64>>(chunk, "a list of rings")?;
            let rings =
                downcast::<ListArray<i64>>(rows_list.values().as_ref(), "a list of coordinates")?;
            let array = PolygonArray {
                rows: List::new(rows_list),
                rings: ValidList::new(rings),
                coords: ValidCoords::new(rings.values().as_ref())?,
            };
            rows(chunk, &mut f, |i| Geometry::Polygon(array.get(i)))
        }
        Kind::MultiLineString => {
            let rows_list = downcast::<ListArray<i64>>(chunk, "a list of linestrings")?;
            let lines =
                downcast::<ListArray<i64>>(rows_list.values().as_ref(), "a list of coordinates")?;
            let array = MultiLineStringArray {
                rows: List::new(rows_list),
                lines: ValidList::new(lines),
                coords: ValidCoords::new(lines.values().as_ref())?,
            };
            rows(chunk, &mut f, |i| Geometry::MultiLineString(array.get(i)))
        }
        Kind::MultiPolygon => {
            let rows_list = downcast::<ListArray<i64>>(chunk, "a list of polygons")?;
            let polygons =
                downcast::<ListArray<i64>>(rows_list.values().as_ref(), "a list of rings")?;
            let rings =
                downcast::<ListArray<i64>>(polygons.values().as_ref(), "a list of coordinates")?;
            let array = MultiPolygonArray {
                rows: List::new(rows_list),
                polygons: ValidList::new(polygons),
                rings: ValidList::new(rings),
                coords: ValidCoords::new(rings.values().as_ref())?,
            };
            rows(chunk, &mut f, |i| Geometry::MultiPolygon(array.get(i)))
        }
    }
}
