//! Converting between `geoarrow.wkb` and our native geometries.
//!
//! The `wkb` crate does the encoding and decoding itself.

use ::wkb::reader::{read_wkb, Wkb as WkbGeometry};
use ::wkb::writer::{
    write_line_string, write_multi_line_string, write_multi_point, write_multi_polygon,
    write_point, write_polygon, WriteOptions,
};
use geo_traits::{
    CoordTrait, Dimensions, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, MutableBinaryViewArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::crs::MetadataKwargs;
use crate::geoarrow::geotraits::{
    LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
};
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::wkb::{describe_wkb, Wkb};
use crate::geoarrow::{describe, Geo, GeoDimension, Kind};

/// `output_type_func` for [`to_wkb`]: WKB carrying the geometry's metadata.
fn wkb_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = describe(field.dtype())?;
    Ok(Field::new(
        field.name().clone(),
        Wkb::new(geo.metadata.clone()).dtype(),
    ))
}

/// Writes one WKB value per row of `chunk` into `out`, nulls staying null.
fn encode_rows(
    chunk: &dyn Array,
    out: &mut MutableBinaryViewArray<[u8]>,
    mut write: impl FnMut(usize, &mut Vec<u8>) -> ::wkb::error::WkbResult<()>,
) -> PolarsResult<()> {
    let mut buf = Vec::new();
    for i in 0..chunk.len() {
        if chunk.is_null(i) {
            out.push_null();
            continue;
        }
        buf.clear();
        write(i, &mut buf).map_err(|e| polars_err!(ComputeError: "failed to write WKB: {e}"))?;
        out.push_value(&buf);
    }
    Ok(())
}

fn encode_chunk(
    chunk: &dyn Array,
    kind: Kind,
    out: &mut MutableBinaryViewArray<[u8]>,
) -> PolarsResult<()> {
    let options = WriteOptions::default();

    match kind {
        Kind::Point => {
            let coords = CoordsView::new(chunk)?;
            encode_rows(chunk, out, |i, buf| {
                write_point(buf, &Point { coords: &coords, i }, &options)
            })
        }
        Kind::LineString | Kind::MultiPoint => {
            let rows = downcast::<ListArray<i64>>(chunk, "a list of coordinates")?;
            let coords = CoordsView::new(rows.values().as_ref())?;
            encode_rows(chunk, out, |i, buf| {
                let (start, end) = rows.offsets().start_end(i);
                let coords = &coords;
                match kind {
                    Kind::LineString => {
                        write_line_string(buf, &LineString { coords, start, end }, &options)
                    }
                    _ => write_multi_point(buf, &MultiPoint { coords, start, end }, &options),
                }
            })
        }
        Kind::Polygon | Kind::MultiLineString => {
            let rows = downcast::<ListArray<i64>>(chunk, "a list of rings or linestrings")?;
            let lines =
                downcast::<ListArray<i64>>(rows.values().as_ref(), "a list of coordinates")?;
            let coords = CoordsView::new(lines.values().as_ref())?;
            encode_rows(chunk, out, |i, buf| {
                let (start, end) = rows.offsets().start_end(i);
                let coords = &coords;
                match kind {
                    Kind::Polygon => write_polygon(
                        buf,
                        &Polygon {
                            rings: lines,
                            coords,
                            start,
                            end,
                        },
                        &options,
                    ),
                    _ => write_multi_line_string(
                        buf,
                        &MultiLineString {
                            lines,
                            coords,
                            start,
                            end,
                        },
                        &options,
                    ),
                }
            })
        }
        Kind::MultiPolygon => {
            let rows = downcast::<ListArray<i64>>(chunk, "a list of polygons")?;
            let polygons = downcast::<ListArray<i64>>(rows.values().as_ref(), "a list of rings")?;
            let rings =
                downcast::<ListArray<i64>>(polygons.values().as_ref(), "a list of coordinates")?;
            let coords = CoordsView::new(rings.values().as_ref())?;
            encode_rows(chunk, out, |i, buf| {
                let (start, end) = rows.offsets().start_end(i);
                let multipolygon = MultiPolygon {
                    polygons,
                    rings,
                    coords: &coords,
                    start,
                    end,
                };
                write_multi_polygon(buf, &multipolygon, &options)
            })
        }
    }
}

/// See `to_wkb`.
#[polars_expr(output_type_func=wkb_type)]
fn to_wkb(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();

    let mut out = MutableBinaryViewArray::<[u8]>::with_capacity(storage.len());
    for chunk in storage.chunks() {
        encode_chunk(chunk.as_ref(), geo.kind, &mut out)?;
    }

    let binary = BinaryChunked::with_chunk(inputs[0].name().clone(), out.freeze()).into_series();
    Ok(binary.into_extension(Wkb::new(geo.metadata.clone()).instance()))
}

/// The geometry WKB is decoded into.
/// Mirrors what `from_wkb` passes on the Python side.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FromWkbKwargs {
    /// As spelled by [`Kind::display`].
    kind: String,
    /// As spelled by [`GeoDimension::tag`].
    dimension: String,
    /// See [`MetadataKwargs::crs`].
    crs: Option<String>,
}

/// The geometry a column of WKB decodes into.
fn target(dtype: &DataType, kwargs: &FromWkbKwargs) -> PolarsResult<Geo> {
    let Some(kind) = Kind::from_display(&kwargs.kind) else {
        polars_bail!(InvalidOperation: "unknown geometry: {}", kwargs.kind);
    };
    let Some(dim) = GeoDimension::from_tag(&kwargs.dimension) else {
        polars_bail!(InvalidOperation: "unknown dimension: {}", kwargs.dimension);
    };
    let metadata = MetadataKwargs {
        crs: kwargs.crs.clone(),
    }
    .apply(describe_wkb(dtype)?)?;
    Ok(Geo::new(kind, dim, metadata))
}

fn from_wkb_type(input_fields: &[Field], kwargs: FromWkbKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = target(field.dtype(), &kwargs)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// Builds the storage of one geometry column, one decoded geometry at a time.
///
/// `offsets[level]` holds the offsets of list level `level`, outermost first,
/// each pointing into the level below it (the coordinates, for the innermost one).
struct Builder {
    kind: Kind,
    dim: GeoDimension,
    /// One buffer per coordinate, in [`GeoDimension::field_names`] order.
    coords: Vec<Vec<f64>>,
    offsets: Vec<Vec<i64>>,
    /// Only the outermost level can be null.
    validity: MutableBitmap,
}

impl Builder {
    fn new(kind: Kind, dim: GeoDimension, capacity: usize) -> Self {
        Self {
            kind,
            dim,
            coords: vec![Vec::with_capacity(capacity); dim.field_names().len()],
            offsets: (0..kind.nesting()).map(|_| vec![0]).collect(),
            validity: MutableBitmap::with_capacity(capacity),
        }
    }

    fn coord(&mut self, coord: &impl CoordTrait<T = f64>) {
        for (n, values) in self.coords.iter_mut().enumerate() {
            values.push(coord.nth_or_panic(n));
        }
    }

    /// GeoArrow (like WKB) spells an empty point as all-NaN coordinates.
    fn empty_coord(&mut self) {
        for values in &mut self.coords {
            values.push(f64::NAN);
        }
    }

    /// Ends the current entry of list level `level`.
    fn close(&mut self, level: usize) {
        let len = match self.offsets.get(level + 1) {
            Some(below) => below.len() - 1,
            None => self.coords[0].len(),
        };
        self.offsets[level].push(len as i64);
    }

    fn point(&mut self, point: &impl PointTrait<T = f64>) {
        match point.coord() {
            Some(coord) => self.coord(&coord),
            None => self.empty_coord(),
        }
    }

    fn line(&mut self, level: usize, line: &impl LineStringTrait<T = f64>) {
        for coord in line.coords() {
            self.coord(&coord);
        }
        self.close(level);
    }

    fn polygon(&mut self, level: usize, polygon: &impl PolygonTrait<T = f64>) {
        if let Some(exterior) = polygon.exterior() {
            self.line(level + 1, &exterior);
        }
        for interior in polygon.interiors() {
            self.line(level + 1, &interior);
        }
        self.close(level);
    }

    fn null(&mut self) {
        match self.kind {
            Kind::Point => self.empty_coord(),
            _ => self.close(0),
        }
        self.validity.push(false);
    }

    /// Decodes one geometry. A single geometry is promoted into the multi geometry
    /// of the same type, as GeoParquet files commonly mix those two.
    fn geometry(&mut self, wkb: &WkbGeometry) -> PolarsResult<()> {
        polars_ensure!(
            wkb.dim() == Dimensions::from(self.dim),
            ComputeError: "cannot decode WKB with {} coordinates into a `{}` column with {} coordinates",
            dimension_name(wkb.dim()), self.kind.name(), self.dim.tag()
        );

        match (self.kind, wkb.as_type()) {
            (Kind::Point, GeometryType::Point(point)) => self.point(point),
            (Kind::LineString, GeometryType::LineString(line)) => self.line(0, line),
            (Kind::Polygon, GeometryType::Polygon(polygon)) => self.polygon(0, polygon),
            (Kind::MultiPoint, GeometryType::MultiPoint(points)) => {
                for point in points.points() {
                    self.point(&point);
                }
                self.close(0);
            }
            (Kind::MultiPoint, GeometryType::Point(point)) => {
                // An empty point promotes into an empty multipoint, not into one NaN point.
                if let Some(coord) = point.coord() {
                    self.coord(&coord);
                }
                self.close(0);
            }
            (Kind::MultiLineString, GeometryType::MultiLineString(lines)) => {
                for line in lines.line_strings() {
                    self.line(1, &line);
                }
                self.close(0);
            }
            (Kind::MultiLineString, GeometryType::LineString(line)) => {
                self.line(1, line);
                self.close(0);
            }
            (Kind::MultiPolygon, GeometryType::MultiPolygon(polygons)) => {
                for polygon in polygons.polygons() {
                    self.polygon(1, &polygon);
                }
                self.close(0);
            }
            (Kind::MultiPolygon, GeometryType::Polygon(polygon)) => {
                self.polygon(1, polygon);
                self.close(0);
            }
            _ => polars_bail!(
                ComputeError: "cannot decode a WKB {:?} into a `{}` column",
                wkb.geometry_type(), self.kind.name()
            ),
        }
        self.validity.push(true);
        Ok(())
    }

    fn finish(self, name: PlSmallStr) -> PolarsResult<Series> {
        let validity: Bitmap = self.validity.into();
        let validity = (validity.unset_bits() > 0).then_some(validity);
        let len = self.coords[0].len();

        let coords_dtype = self.dim.coordinates().to_arrow(CompatLevel::newest());
        let values = self
            .coords
            .into_iter()
            .map(|values| PrimitiveArray::from_vec(values).boxed())
            .collect();
        let mut array: Box<dyn Array> = if self.offsets.is_empty() {
            StructArray::new(coords_dtype, len, values, validity.clone()).boxed()
        } else {
            StructArray::new(coords_dtype, len, values, None).boxed()
        };

        let levels = self.offsets.len();
        for (level, offsets) in self.offsets.into_iter().enumerate().rev() {
            let dtype = ListArray::<i64>::default_datatype(array.dtype().clone());
            let offsets = Offsets::try_from(offsets)?.into();
            let validity = (level == 0 && levels > 0)
                .then(|| validity.clone())
                .flatten();
            array = ListArray::<i64>::new(dtype, offsets, array, validity).boxed();
        }
        Series::from_arrow(name, array)
    }
}

/// How a WKB dimension is spelled in an error message: `xyz`.
fn dimension_name(dim: Dimensions) -> String {
    match dim {
        Dimensions::Xy => "xy".into(),
        Dimensions::Xyz => "xyz".into(),
        Dimensions::Xym => "xym".into(),
        Dimensions::Xyzm => "xyzm".into(),
        Dimensions::Unknown(n) => format!("{n}"),
    }
}

/// See `from_wkb`.
#[polars_expr(output_type_func_with_kwargs=from_wkb_type)]
fn from_wkb(inputs: &[Series], kwargs: FromWkbKwargs) -> PolarsResult<Series> {
    let geo = target(inputs[0].dtype(), &kwargs)?;
    let binary = match inputs[0].dtype() {
        DataType::Extension(..) => inputs[0].ext()?.storage().clone(),
        _ => inputs[0].clone(),
    };

    let mut builder = Builder::new(geo.kind(), geo.dim(), binary.len());
    for value in binary
        .binary()?
        .downcast_iter()
        .flat_map(|chunk| chunk.iter())
    {
        match value {
            None => builder.null(),
            Some(bytes) => {
                let wkb =
                    read_wkb(bytes).map_err(|e| polars_err!(ComputeError: "invalid WKB: {e}"))?;
                builder.geometry(&wkb)?;
            }
        }
    }

    Ok(builder
        .finish(inputs[0].name().clone())?
        .into_extension(geo.instance()))
}
