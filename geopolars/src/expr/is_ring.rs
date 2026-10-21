//! Whether a linestring is a ring (see [`geopolars_compute::native::line_strings_are_rings`]).
//!
//! A multilinestring gets a list, one per part.

use geopolars_compute::native::{line_strings_are_rings, multi_line_strings_are_rings};
use geopolars_compute::with_const_flags;
use geopolars_core::{LineStringArray, MultiLineStringArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::{describe, Kind};

/// For [`is_ring`]:
/// a `bool` per linestring, or a list of them (one per part) for a multilinestring.
fn kernel_output_type_is_ring(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = match describe(field.dtype())?.kind {
        Kind::LineString => DataType::Boolean,
        Kind::MultiLineString => DataType::List(Box::new(DataType::Boolean)),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            polars_bail!(
                SchemaMismatch: "is_ring expects a `{}` or `{}` column, got: `{}`",
                Kind::LineString.name(), Kind::MultiLineString.name(), kind.name()
            )
        }
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// See `is_ring`.
#[polars_expr(output_type_func=kernel_output_type_is_ring)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let field = kernel_output_type_is_ring(&[inputs[0].field().into_owned()])?;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    if *field.dtype() == DataType::Boolean {
        let chunks = chunks.map(|rows| {
            let array = LineStringArray::new(rows)?;
            let z = array.coords.dim().has_z();
            PolarsResult::Ok(with_const_flags!(line_strings_are_rings(&array); z))
        });
        return Ok(BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series());
    }
    let chunks = chunks.map(|rows| {
        let array = MultiLineStringArray::new(rows)?;
        let z = array.coords.dim().has_z();
        PolarsResult::Ok(with_const_flags!(multi_line_strings_are_rings(&array); z))
    });
    Ok(ListChunked::try_from_chunk_iter(name, chunks)?.into_series())
}
