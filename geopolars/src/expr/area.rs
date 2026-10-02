//! The planar area a geometry encloses, computed by `rsgeo`.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rsgeo::Area;

use crate::geoarrow::{describe, to_rsgeo, Kind};

// TODO: Move this up into a shareable thing?
/// `output_type_func` for an operation that gives one `f64` per geometry.
fn float_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Float64))
}

/// The planar area of every geometry. Only a polygon encloses anything.
/// Everything else is `0.0`, and a missing geometry has no area.
#[polars_expr(output_type_func=float_output)]
fn area_rsgeo(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();

    let out = match geo.kind {
        Kind::Polygon => Float64Chunked::from_iter_options(
            name,
            to_rsgeo::polygons(storage)?.map(|polygon| polygon.map(|p| p.unsigned_area())),
        ),
        Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => {
            Float64Chunked::from_iter_options(
                name,
                storage
                    .is_not_null()
                    .iter()
                    .map(|present| (present == Some(true)).then_some(0.0)),
            )
        }
    };
    Ok(out.into_series())
}
