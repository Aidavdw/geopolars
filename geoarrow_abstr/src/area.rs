//! The area a polygon encloses, as a [Metric] measures it.

use std::convert::Infallible;

use geographiclib_rs::{PolygonArea, Winding};

use crate::arrays::per_row;
use crate::{
    GeoArrowF64Array, GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView,
    GeodesicMetric, LineString, Metric, MultiPolygonArray, Planar, Polygon, PolygonArray,
    ReprojectError,
};

pub trait AreaMetric: Metric {
    /// The area the closed `ring` encloses, whichever way it winds. Its heights do not count.
    fn ring_area(
        &self,
        ring: &LineString<impl GeoArrowValidCoordsView>,
    ) -> Result<f64, Self::Error>;
}

impl AreaMetric for Planar {
    fn ring_area(
        &self,
        ring: &LineString<impl GeoArrowValidCoordsView>,
    ) -> Result<f64, Infallible> {
        // By the shoelace formula, relative to the ring's first vertex,
        // so coordinates far from the origin keep their precision.
        // The repeated closing vertex adds nothing.
        let mut positions = ring.positions::<false>();
        let Some([x0, y0, _]) = positions.next() else {
            return Ok(0.0);
        };
        let (mut x_previous, mut y_previous) = (0.0, 0.0);
        let mut twice_signed = 0.0;
        for [x, y, _] in positions {
            let (x, y) = (x - x0, y - y0);
            twice_signed += x_previous * y - x * y_previous;
            (x_previous, y_previous) = (x, y);
        }
        Ok(twice_signed.abs() / 2.0)
    }
}

impl AreaMetric for GeodesicMetric {
    fn ring_area(
        &self,
        ring: &LineString<impl GeoArrowValidCoordsView>,
    ) -> Result<f64, ReprojectError> {
        // On the ellipsoid, in the CRS's unit squared, like [`between`](GeodesicMetric::between).
        // Based on `geographiclib_rs`.
        let mut area = PolygonArea::new(&self.ellipsoid, Winding::CounterClockwise);
        for [x, y, _] in ring.positions::<false>() {
            let (lon, lat) = self.lonlat((x, y))?;
            // The repeated closing vertex adds an edge of length zero, which encloses nothing.
            area.add_point(lat, lon);
        }
        // Signed, then made positive.
        // unsigned would take a clockwise ring
        // to enclose everything outside of it.
        let (_perimeter, area, _vertices) = area.compute(true);
        Ok(area.abs())
    }
}

/// The area of `polygon`: its exterior ring, less the holes inside it.
/// No rings at all is an empty polygon, which encloses nothing.
#[inline(always)]
fn polygon_area<M: AreaMetric>(
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
pub fn polygon_areas<M: AreaMetric, O: GeoArrowF64Array>(
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
pub fn multi_polygon_areas<M: AreaMetric, O: GeoArrowF64Array>(
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
