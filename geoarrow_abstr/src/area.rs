//! The area a polygon encloses, as a [Metric] measures it.

use crate::arrays::per_row;
use crate::{
    GeoArrowF64Array, GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, Metric,
    MultiPolygonArray, Polygon, PolygonArray,
};

/// The area of `polygon`: its exterior ring, less the holes inside it.
/// No rings at all is an empty polygon, which encloses nothing.
#[inline(always)]
fn polygon_area<M: Metric>(
    metric: &M,
    polygon: &Polygon<impl GeoArrowValidListView, impl GeoArrowValidCoordsView>,
) -> Result<f64, M::Error> {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return Ok(0.0);
    };
    let exterior = metric.ring_area(&exterior)?;
    let holes = rings
        .map(|hole| metric.ring_area(&hole))
        .sum::<Result<f64, _>>()?;
    Ok(exterior - holes)
}

/// The area of every polygon of a chunk.
pub fn polygon_areas<M: Metric, O: GeoArrowF64Array>(
    metric: &M,
    array: &PolygonArray<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidListView,
        impl GeoArrowValidCoordsView,
    >,
) -> Result<O, M::Error> {
    per_row(&array.rows, |i| polygon_area(metric, &array.get(i)))
}

/// The area of every multipolygon of a chunk, as the sum of its parts.
pub fn multi_polygon_areas<M: Metric, O: GeoArrowF64Array>(
    metric: &M,
    array: &MultiPolygonArray<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidListView,
        impl GeoArrowValidListView,
        impl GeoArrowValidCoordsView,
    >,
) -> Result<O, M::Error> {
    per_row(&array.rows, |i| {
        array
            .get(i)
            .polygons()
            .map(|polygon| polygon_area(metric, &polygon))
            .sum()
    })
}
