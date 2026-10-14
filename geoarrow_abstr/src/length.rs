//! How long a linestring is, as a [Metric] measures it.

use std::ops::Range;

use crate::{
    GeoArrowF64Array, GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, Metric,
};

/// The length of the line through `vertices`: fewer than two have a length of `0.0`.
/// With `IGNORE_NAN`, a vertex with a NaN is skipped, so the line runs on to the next one.
#[inline(always)]
fn line_length<const Z: bool, const IGNORE_NAN: bool, M: Metric>(
    metric: &M,
    axes: &impl GeoArrowValidCoordsView,
    vertices: Range<usize>,
) -> Result<f64, M::Error> {
    let mut vertices = vertices
        .map(|i| axes.position::<Z>(i))
        .filter(|&[x, y, z]| !IGNORE_NAN || !(x.is_nan() || y.is_nan() || (Z && z.is_nan())));
    let Some(first) = vertices.next() else {
        return Ok(0.0);
    };
    let mut previous = metric.vertex(first)?;
    let mut total = 0.0;
    for position in vertices {
        let here = metric.vertex(position)?;
        total += metric.segment::<Z>(previous, here);
        previous = here;
    }
    Ok(total)
}

/// The length of every row of a chunk: a linestring,
/// or with `MULTI`, a multilinestring as the sum of its parts.
pub fn lengths<
    const Z: bool,
    const MULTI: bool,
    const IGNORE_NAN: bool,
    M: Metric,
    O: GeoArrowF64Array,
>(
    metric: &M,
    rows: &impl GeoArrowListView<Validity = O::Validity>,
    lines: &impl GeoArrowValidListView,
    axes: &impl GeoArrowValidCoordsView,
) -> Result<O, M::Error> {
    let line = |i: usize| line_length::<Z, IGNORE_NAN, _>(metric, axes, lines.range(i));
    let values = (0..rows.len())
        .map(|i| {
            // A missing row's offsets point anywhere, so it is not measured.
            if !rows.is_valid(i) {
                Ok(0.0)
            } else if MULTI {
                rows.range(i).map(line).sum()
            } else {
                line(i)
            }
        })
        .collect::<Result<Vec<f64>, M::Error>>()?;
    Ok(O::from_values(values, rows.validity()))
}
