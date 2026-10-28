//! Emptying, dropping or nulling the coordinates that are not finite, or empty,
//! through `geopolars_compute::native::*_with_filtered_coordinates`.
//! See `filter_coordinates` in Python for the rules.

use geopolars_compute::native::{
    line_strings_with_filtered_coordinates, multi_line_strings_with_filtered_coordinates,
    multi_points_with_filtered_coordinates, multi_polygons_with_filtered_coordinates,
    points_with_filtered_coordinates, polygons_with_filtered_coordinates, NonFinite,
    SelectedAction, TooShort,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::{kernel_output_type_same_geometry, trimmed};
use super::remove_repeated_points::TooShortDef;
use crate::geoarrow::{describe, Kind};

#[derive(Deserialize)]
#[serde(remote = "SelectedAction", rename_all = "snake_case")]
enum SelectedActionDef {
    SetEmpty,
    Drop,
    SetNull,
}

/// See python
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonFiniteKwargs {
    pub(super) select_non_finite: bool,
    pub(super) select_empty: bool,
    pub(super) ignore_z: bool,
    pub(super) ignore_m: bool,
    #[serde(with = "SelectedActionDef")]
    pub(super) action: SelectedAction,
    #[serde(with = "TooShortDef")]
    pub(super) too_short: TooShort,
}

impl From<NonFiniteKwargs> for NonFinite {
    fn from(kwargs: NonFiniteKwargs) -> Self {
        let NonFiniteKwargs {
            select_non_finite,
            select_empty,
            ignore_z,
            ignore_m,
            action,
            too_short,
        } = kwargs;
        NonFinite {
            select_non_finite,
            select_empty,
            ignore_z,
            ignore_m,
            action,
            too_short,
        }
    }
}

/// [`filter_coordinates`] for one chunk of the storage of a `kind`.
fn chunk_with_filtered_coordinates(
    chunk: &dyn Array,
    kind: Kind,
    filter: &NonFinite,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => {
            points_with_filtered_coordinates(&PointArray::new(Coords::new(chunk)?), filter)
                .into_struct_array()
                .boxed()
        }
        Kind::LineString => {
            line_strings_with_filtered_coordinates(&LineStringArray::new(rows()?)?, filter)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            multi_points_with_filtered_coordinates(&MultiPointArray::new(rows()?)?, filter)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => polygons_with_filtered_coordinates(&PolygonArray::new(rows()?)?, filter)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => multi_line_strings_with_filtered_coordinates(
            &MultiLineStringArray::new(rows()?)?,
            filter,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPolygon => {
            multi_polygons_with_filtered_coordinates(&MultiPolygonArray::new(rows()?)?, filter)
                .into_list_array()
                .boxed()
        }
    })
}

/// See `filter_coordinates`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn filter_coordinates(inputs: &[Series], kwargs: NonFiniteKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let filter = NonFinite::from(kwargs);
    let storage = trimmed(inputs[0].ext()?.storage());
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_with_filtered_coordinates(chunk.as_ref(), kind, &filter))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
