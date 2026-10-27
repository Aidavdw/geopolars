//! Whether two geometries are almost the same
//! (see [`geopolars_compute::native::points_are_close`] and the kernels of the other kinds).

use geopolars_compute::native::{
    line_strings_are_close, multi_line_strings_are_close, multi_points_are_close,
    multi_polygons_are_close, points_are_close, polygons_are_close,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, Tolerance,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, ensure_same_crs, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsCloseKwargs {
    abs_tol: f64,
    rel_tol: f64,
    nans_equal: bool,
    unique_z: bool,
    unique_m: bool,
}

impl Default for IsCloseKwargs {
    fn default() -> Self {
        Self {
            abs_tol: 0.0,
            rel_tol: 1e-9,
            nans_equal: false,
            unique_z: true,
            unique_m: false,
        }
    }
}

/// For [`is_close`]: a `bool`, named after the first geometry.
///
/// Both have to be the same kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_close(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_close takes two geometries, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    polars_ensure!(
        geo_a.kind == geo_b.kind && geo_a.dim == geo_b.dim,
        SchemaMismatch: "is_close expects two geometries of the same type, got: {} and {}",
        a.dtype(), b.dtype()
    );
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// The kernel of `kind` on one chunk of `a`'s storage and one of `b`'s.
/// `b` is as long as `a`, or a single row.
fn chunk_is_close(
    a: &dyn Array,
    b: &dyn Array,
    kind: Kind,
    kwargs: &IsCloseKwargs,
) -> PolarsResult<BooleanArray> {
    let tolerance = Tolerance {
        abs_tol: kwargs.abs_tol,
        rel_tol: kwargs.rel_tol,
    };
    let (unique_z, unique_m, nans_equal) = (kwargs.unique_z, kwargs.unique_m, kwargs.nans_equal);
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_are_close(
            &PointArray::new(Coords::new(a)?),
            &PointArray::new(Coords::new(b)?),
            tolerance,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPoint => multi_points_are_close(
            &MultiPointArray::new(rows(a)?)?,
            &MultiPointArray::new(rows(b)?)?,
            tolerance,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::LineString => line_strings_are_close(
            &LineStringArray::new(rows(a)?)?,
            &LineStringArray::new(rows(b)?)?,
            tolerance,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiLineString => multi_line_strings_are_close(
            &MultiLineStringArray::new(rows(a)?)?,
            &MultiLineStringArray::new(rows(b)?)?,
            tolerance,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::Polygon => polygons_are_close(
            &PolygonArray::new(rows(a)?)?,
            &PolygonArray::new(rows(b)?)?,
            tolerance,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPolygon => multi_polygons_are_close(
            &MultiPolygonArray::new(rows(a)?)?,
            &MultiPolygonArray::new(rows(b)?)?,
            tolerance,
            unique_z,
            unique_m,
            nans_equal,
        ),
    })
}

/// The chunks of the storages of `a` and `b`, which are the same length, split at the same rows.
fn aligned_chunks(a: &Series, b: &Series) -> PolarsResult<Vec<(ArrayRef, ArrayRef)>> {
    let (a, b) = (a.ext()?.storage(), b.ext()?.storage());
    let zipped =
        |a: &[ArrayRef], b: &[ArrayRef]| a.iter().cloned().zip(b.iter().cloned()).collect();
    Ok(match a.dtype() {
        DataType::Struct(_) => {
            let (a, b) = align_chunks_binary(a.struct_()?, b.struct_()?);
            zipped(a.chunks(), b.chunks())
        }
        _ => {
            let (a, b) = align_chunks_binary(a.list()?, b.list()?);
            zipped(a.chunks(), b.chunks())
        }
    })
}

/// [chunk_is_close] for every chunk of `many`, against the single row of `one`.
/// Closeness goes both ways, so this serves a single geometry on either side.
fn every_against_one(
    one: &Series,
    many: &Series,
    kind: Kind,
    kwargs: &IsCloseKwargs,
) -> PolarsResult<Vec<BooleanArray>> {
    let one = one.ext()?.storage().rechunk();
    let one = one.chunks()[0].as_ref();
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| chunk_is_close(many.as_ref(), one, kind, kwargs))
        .collect()
}

/// See `is_close`.
#[polars_expr(output_type_func=kernel_output_type_is_close)]
fn is_close(inputs: &[Series], kwargs: IsCloseKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let kind = describe(a.dtype())?.kind;
    let chunks = match (a.len(), b.len()) {
        (n, m) if n == m => aligned_chunks(a, b)?
            .iter()
            .map(|(a, b)| chunk_is_close(a.as_ref(), b.as_ref(), kind, &kwargs))
            .collect::<PolarsResult<Vec<_>>>()?,
        (1, _) => every_against_one(a, b, kind, &kwargs)?,
        (_, 1) => every_against_one(b, a, kind, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} geometries against {m}"),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
