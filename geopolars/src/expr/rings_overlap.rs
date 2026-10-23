//! Whether two rings overlap
//! (see [`geopolars_compute::geos::rings_overlap_pairwise`]).

use geopolars_compute::geos::{rings_overlap_against_one, rings_overlap_pairwise};
use geopolars_compute::with_const_flags;
use geopolars_core::LineStringArray;
use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, ensure_same_crs, Kind};

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RingsOverlapKwargs {
    /// Whether rings that only touch overlap too.
    include_touching: bool,
    ignore_per_row_errors: bool,
}

/// For [`rings_overlap`]: a `bool`, named after the first ring.
///
/// Both have to be linestrings, in the same CRS (or neither declaring one).
fn kernel_output_type_rings_overlap(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "rings_overlap takes two rings, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    for (field, geo) in [(a, &geo_a), (b, &geo_b)] {
        polars_ensure!(
            geo.kind == Kind::LineString,
            SchemaMismatch: "rings_overlap expects two `{}` columns, got: {}",
            Kind::LineString.name(), field.dtype()
        );
    }
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

fn compute_err(e: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "rings_overlap {e}")
}

/// The rings of `rings`, as a single chunk.
fn single_chunk(rings: &Series) -> PolarsResult<ListChunked> {
    Ok(rings.ext()?.storage().list()?.rechunk().into_owned())
}

/// [rings_overlap_pairwise], into a polars error.
fn pairwise<const TOUCHING: bool>(
    a: &LineStringArray,
    b: &LineStringArray,
    ignore_per_row_errors: bool,
) -> PolarsResult<BooleanArray> {
    rings_overlap_pairwise::<TOUCHING>(a, b, ignore_per_row_errors).map_err(compute_err)
}

/// [rings_overlap_against_one] for every chunk of `many`.
fn every_against_one<const TOUCHING: bool>(
    one: &Series,
    many: &Series,
    ignore_per_row_errors: bool,
) -> PolarsResult<Vec<BooleanArray>> {
    let one = single_chunk(one)?;
    let one = LineStringArray::new(one.downcast_as_array())?;
    let many = many.ext()?.storage().list()?;
    if !one.rows.is_valid(0) {
        return Ok(many
            .downcast_iter()
            .map(|many| BooleanArray::new_null(ArrowDataType::Boolean, many.len()))
            .collect());
    }
    let one = one.get(0);
    many.downcast_iter()
        .map(|many| {
            let many = LineStringArray::new(many)?;
            rings_overlap_against_one::<TOUCHING>(&one, &many, ignore_per_row_errors)
                .map_err(compute_err)
        })
        .collect()
}

/// See `rings_overlap`.
#[polars_expr(output_type_func=kernel_output_type_rings_overlap)]
fn rings_overlap(inputs: &[Series], kwargs: RingsOverlapKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (touching, ignore_errors) = (kwargs.include_touching, kwargs.ignore_per_row_errors);
    let chunks = match (a.len(), b.len()) {
        (n, m) if n == m => {
            let (a, b) = (single_chunk(a)?, single_chunk(b)?);
            let (a, b) = (
                LineStringArray::new(a.downcast_as_array())?,
                LineStringArray::new(b.downcast_as_array())?,
            );
            vec![with_const_flags!(pairwise(&a, &b, ignore_errors); touching)?]
        }
        (1, _) => with_const_flags!(every_against_one(a, b, ignore_errors); touching)?,
        (_, 1) => with_const_flags!(every_against_one(b, a, ignore_errors); touching)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} rings against {m}"),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
