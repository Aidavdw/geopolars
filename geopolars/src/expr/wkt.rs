//! Converting between `geoarrow.wkt` and our native geometries.
//!
//! The `wkt` crate does the encoding and decoding itself.

use ::wkt::to_wkt::write_geometry;
use ::wkt::Wkt;
use polars::prelude::*;
use polars_arrow::array::MutableBinaryViewArray;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::decode::{target, Builder, DecodeKwargs};
use crate::geoarrow::describe;
use crate::geoarrow::encoded::{describe_encoded, Encoded, Encoding};
use crate::geoarrow::geotraits::for_each_geometry;

/// `output_type_func` for [`to_wkt`]: WKT carrying the geometry's metadata.
fn wkt_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = describe(field.dtype())?;
    let wkt = Encoded::new(Encoding::Wkt, geo.metadata.clone());
    Ok(Field::new(field.name().clone(), wkt.dtype()))
}

/// See `to_wkt`.
#[polars_expr(output_type_func=wkt_type)]
fn to_wkt(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();

    let mut out = MutableBinaryViewArray::<str>::with_capacity(storage.len());
    let mut buf = String::new();
    for chunk in storage.chunks() {
        for_each_geometry(chunk.as_ref(), geo.kind, |geometry| {
            let Some(geometry) = geometry else {
                out.push_null();
                return Ok(());
            };
            buf.clear();
            write_geometry(&mut buf, &geometry)
                .map_err(|e| polars_err!(ComputeError: "failed to write WKT: {e}"))?;
            out.push_value(&buf);
            Ok(())
        })?;
    }

    let text = StringChunked::with_chunk(inputs[0].name().clone(), out.freeze()).into_series();
    let wkt = Encoded::new(Encoding::Wkt, geo.metadata.clone());
    Ok(text.into_extension(wkt.instance()))
}

fn from_wkt_type(input_fields: &[Field], kwargs: DecodeKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = target(describe_encoded(field.dtype(), Encoding::Wkt)?, &kwargs)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// See `from_wkt`.
#[polars_expr(output_type_func_with_kwargs=from_wkt_type)]
fn from_wkt(inputs: &[Series], kwargs: DecodeKwargs) -> PolarsResult<Series> {
    let geo = target(describe_encoded(inputs[0].dtype(), Encoding::Wkt)?, &kwargs)?;
    let text = match inputs[0].dtype() {
        DataType::Extension(..) => inputs[0].ext()?.storage().clone(),
        _ => inputs[0].clone(),
    };

    let mut builder = Builder::new(&geo, Encoding::Wkt.label(), text.len());
    for value in text.str()?.iter() {
        match value {
            None => builder.null(),
            Some(text) => {
                let wkt = text
                    .parse::<Wkt<f64>>()
                    .map_err(|e| polars_err!(ComputeError: "invalid WKT: {e}"))?;
                builder.geometry(&wkt)?;
            }
        }
    }

    Ok(builder
        .finish(inputs[0].name().clone())?
        .into_extension(geo.instance()))
}
