//! How long a linestring is, as a [Metric] measures it.

use crate::{
    GeoArrowF64Array, GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, LineString,
    LineStringArray, Metric, MultiLineStringArray,
};

/// The length of `line`: fewer than two vertices have a length of `0.0`.
/// With `IGNORE_NAN`, a vertex with a NaN is skipped, so the line runs on to the next one.
#[inline(always)]
fn line_length<const Z: bool, const IGNORE_NAN: bool, M: Metric>(
    metric: &M,
    line: &LineString<impl GeoArrowValidCoordsView>,
) -> Result<f64, M::Error> {
    let mut vertices = line
        .positions::<Z>()
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

/// `length` of every row of a chunk, with the rows' own validity.
#[inline(always)]
fn per_row<O: GeoArrowF64Array, E>(
    rows: &impl GeoArrowListView<Validity = O::Validity>,
    length: impl Fn(usize) -> Result<f64, E>,
) -> Result<O, E> {
    let values = (0..rows.len())
        // A missing row's offsets point anywhere, so it is not measured.
        .map(|i| if rows.is_valid(i) { length(i) } else { Ok(0.0) })
        .collect::<Result<Vec<f64>, E>>()?;
    Ok(O::from_values(values, rows.validity()))
}

/// The length of every linestring of a chunk.
pub fn line_string_lengths<
    const Z: bool,
    const IGNORE_NAN: bool,
    M: Metric,
    O: GeoArrowF64Array,
>(
    metric: &M,
    array: &LineStringArray<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidCoordsView,
    >,
) -> Result<O, M::Error> {
    per_row(&array.rows, |i| {
        line_length::<Z, IGNORE_NAN, _>(metric, &array.get(i))
    })
}

/// The length of every multilinestring of a chunk, as the sum of its parts.
pub fn multi_line_string_lengths<
    const Z: bool,
    const IGNORE_NAN: bool,
    M: Metric,
    O: GeoArrowF64Array,
>(
    metric: &M,
    array: &MultiLineStringArray<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidListView,
        impl GeoArrowValidCoordsView,
    >,
) -> Result<O, M::Error> {
    per_row(&array.rows, |i| {
        array
            .get(i)
            .line_strings()
            .map(|line| line_length::<Z, IGNORE_NAN, _>(metric, &line))
            .sum()
    })
}
