//! The centroid of a geometry, in one pass over the storage:
//! - a linestring's is the midpoint of every segment, weighted by its length,
//! - a polygon's is the shoelace formula, over the planar area,
//! - a multi-geometry's is that of its parts together: the running sum just goes on over them.
//!
//! With generics, a total 8 (lines) + 8 (polygons) = 16 copies of functions exist.

// Should add_ring and add_polygon not also be generic?

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::MutableBitmap;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::storage::{downcast, ValidCoordsView};
use crate::geoarrow::{describe, Geo, GeoDimension, Kind};

/// The point a `dtype` column's centroids are, and the kind of geometry they are of.
/// The centroid keeps the geometry's dimension.
fn centroid_of(dtype: &DataType) -> PolarsResult<(Geo, Kind)> {
    let geo = describe(dtype)?;
    if let Kind::Point | Kind::MultiPoint = geo.kind {
        polars_bail!(
            SchemaMismatch: "centroid expects a (multi)linestring or (multi)polygon column, got: {}",
            dtype
        )
    }
    Ok((
        Geo::new(Kind::Point, geo.dim, geo.metadata.clone()),
        geo.kind,
    ))
}

/// For [`centroid`].
fn kernel_output_type_centroid(input_fields: &[Field]) -> PolarsResult<Field> {
    let (point, _) = centroid_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
}

/// Running average for a weight.
/// You can keep adding to this!
struct Moment {
    /// Σ weight * centre, in x, y, z, m.
    /// If geometry doesn't have one axis, it stays `0.0`.
    sum: [f64; 4],
    /// Σ weight
    /// All share the same
    weight: f64,
}

impl Moment {
    const ZERO: Self = Self {
        sum: [0.0; 4],
        weight: 0.0,
    };

    /// Σ weight * centre / Σ weight.
    fn average(&self) -> Option<[f64; 4]> {
        (self.weight > 0.0).then(|| self.sum.map(|s| s / self.weight))
    }
}

#[inline(always)]
fn vertex<const Z: bool, const M: bool>(axes: &ValidCoordsView, i: usize) -> [f64; 4] {
    let [x, y, z] = axes.position::<Z>(i);
    [x, y, z, if M { axes.m(i) } else { 0.0 }]
}

/// Adds the line through `vertices`: Σ len (a + b) / 2 and Σ len over its segments `ab`,
/// where `len` is measured in xy, or xyz with `Z`,
/// and every axis (incl. m with `M`) is averaged by it.
#[inline(always)]
fn add_line<const Z: bool, const M: bool>(
    moment: &mut Moment,
    axes: &ValidCoordsView,
    mut vertices: Range<usize>,
) {
    let Some(first) = vertices.next() else {
        return;
    };
    let mut previous = vertex::<Z, M>(axes, first);
    for i in vertices {
        let here = vertex::<Z, M>(axes, i);
        // Without `Z`, both z are `0.0`.
        let len = (0..3)
            .map(|k| (here[k] - previous[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        for k in 0..4 {
            moment.sum[k] += 0.5 * len * (previous[k] + here[k]);
        }
        moment.weight += len;
        previous = here;
    }
}

/// Moment += that of a (closed) ring.
/// shoelace formula:
/// Over the ring, a = Σ cross, mx = Σ (x_i + x_{i+1}) cross and my = Σ (y_i + y_{i+1}) cross,
/// with cross = x_i y_{i+1} - x_{i+1} y_i: its centroid is (mx, my) / 3a,
/// so weighted by its area a / 2, (mx, my) / 6.
/// z (if `Z=true`) and m (if `M=true`) are averaged over the area too:
/// over the triangles fanning out from the first vertex `p0`,
/// each weighted by its area, fan / 2 (with fan the cross of `p_i - p0` and `p_{i+1} - p0`),
/// at the mean of its corners, (z_0 + z_i + z_{i+1}) / 3: Σ fan (z_0 + z_i + z_{i+1}) / 6.
/// On a flat ring, that is the height of its plane above the centroid.
#[inline(always)]
fn add_ring<const Z: bool, const M: bool>(
    moment: &mut Moment,
    axes: &ValidCoordsView,
    vertices: Range<usize>,
    sign: f64,
) {
    if vertices.is_empty() {
        return;
    }
    let p0 = vertex::<Z, M>(axes, vertices.start);
    let (mut a, mut sum) = (0.0, [0.0; 4]);
    // The repeated last vertex is only ever the `i + 1`.
    for i in vertices.start..vertices.end - 1 {
        let (p, q) = (vertex::<Z, M>(axes, i), vertex::<Z, M>(axes, i + 1));
        let cross = p[0] * q[1] - q[0] * p[1];
        a += cross;
        sum[0] += (p[0] + q[0]) * cross;
        sum[1] += (p[1] + q[1]) * cross;
        if Z || M {
            let fan = (p[0] - p0[0]) * (q[1] - p0[1]) - (q[0] - p0[0]) * (p[1] - p0[1]);
            for k in 2..4 {
                sum[k] += fan * (p0[k] + p[k] + q[k]);
            }
        }
    }
    let sign = sign * a.signum();
    for (total, s) in moment.sum.iter_mut().zip(sum) {
        *total += sign * s / 6.0;
    }
    moment.weight += sign * a / 2.0;
}

/// moment += (extorior - holes)
#[inline(always)]
fn add_polygon<const Z: bool, const M: bool>(
    moment: &mut Moment,
    rings: &ListArray<i64>,
    axes: &ValidCoordsView,
    ring_range: Range<usize>,
) {
    let mut rings = ring_range.map(|ring| {
        let (start, end) = rings.offsets().start_end(ring);
        start..end
    });
    let Some(exterior) = rings.next() else {
        return;
    };
    add_ring::<Z, M>(moment, axes, exterior, 1.0);
    for hole in rings {
        add_ring::<Z, M>(moment, axes, hole, -1.0);
    }
}

/// The centroid of every row of `rows`, as a coordinate struct of `dtype` with the fields of `dim`.
/// `add_part` adds geometry `i` one list down from the rows:
/// once per row, or with `MULTI`, once for each of its parts.
fn centroids<const MULTI: bool>(
    rows: &ListArray<i64>,
    dtype: ArrowDataType,
    dim: GeoDimension,
    add_part: impl Fn(&mut Moment, usize),
) -> Box<dyn Array> {
    // Which of a moment's x, y, z, m are fields of `dtype`.
    let axes: Vec<usize> = [
        Some(0),
        Some(1),
        dim.has_z().then_some(2),
        dim.has_m().then_some(3),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut fields = vec![Vec::with_capacity(rows.len()); axes.len()];
    let mut valid = MutableBitmap::with_capacity(rows.len());
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not measured.
        let centroid = if rows.is_null(i) {
            None
        } else {
            let mut moment = Moment::ZERO;
            if MULTI {
                let (start, end) = rows.offsets().start_end(i);
                for part in start..end {
                    add_part(&mut moment, part);
                }
            } else {
                add_part(&mut moment, i);
            }
            moment.average()
        };
        let values = centroid.unwrap_or([0.0; 4]);
        for (field, &k) in fields.iter_mut().zip(&axes) {
            field.push(values[k]);
        }
        valid.push(centroid.is_some());
    }
    let fields = fields
        .into_iter()
        .map(|field| PrimitiveArray::from_vec(field).boxed())
        .collect();
    StructArray::new(dtype, rows.len(), fields, valid.into()).boxed()
}

/// [`centroids`] of (multi)linestrings: `lines` is `rows`, or with `MULTI` its parts.
fn line_centroids<const Z: bool, const M: bool, const MULTI: bool>(
    rows: &ListArray<i64>,
    lines: &ListArray<i64>,
    axes: &ValidCoordsView,
    dim: GeoDimension,
) -> Box<dyn Array> {
    let dtype = lines.values().dtype().clone();
    centroids::<MULTI>(rows, dtype, dim, |moment, line| {
        let (start, end) = lines.offsets().start_end(line);
        add_line::<Z, M>(moment, axes, start..end)
    })
}

/// [`centroids`] of (multi)polygons: `polygons` is `rows`, or with `MULTI` its parts.
fn polygon_centroids<const Z: bool, const M: bool, const MULTI: bool>(
    rows: &ListArray<i64>,
    polygons: &ListArray<i64>,
    rings: &ListArray<i64>,
    axes: &ValidCoordsView,
    dim: GeoDimension,
) -> Box<dyn Array> {
    let dtype = rings.values().dtype().clone();
    centroids::<MULTI>(rows, dtype, dim, |moment, polygon| {
        let (start, end) = polygons.offsets().start_end(polygon);
        add_polygon::<Z, M>(moment, rings, axes, start..end)
    })
}

/// See python `centroid`.
#[polars_expr(output_type_func=kernel_output_type_centroid)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let (point, kind) = centroid_of(inputs[0].dtype())?;
    let multi = matches!(kind, Kind::MultiLineString | Kind::MultiPolygon);
    let polygons = matches!(kind, Kind::Polygon | Kind::MultiPolygon);
    let storage = inputs[0].ext()?.storage();

    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| {
            let parts = if multi {
                downcast(rows.values().as_ref(), "a list of parts")?
            } else {
                rows
            };
            Ok(if polygons {
                let rings: &ListArray<i64> = downcast(parts.values().as_ref(), "a list of rings")?;
                let (axes, dim) = ValidCoordsView::new(rings.values().as_ref())?;
                with_const_flags!(
                    polygon_centroids(rows, parts, rings, &axes, dim);
                    dim.has_z(), dim.has_m(), multi
                )
            } else {
                let (axes, dim) = ValidCoordsView::new(parts.values().as_ref())?;
                with_const_flags!(
                    line_centroids(rows, parts, &axes, dim);
                    dim.has_z(), dim.has_m(), multi
                )
            })
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `point` describes.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &point.dtype())
    })
}
