//! To use `rsgeo` algorithms, data needs to be in their data format.
//!
//! Downsides:
//! - **Conversion is an extra copy + allocation**.
//!   The polars data can be read directly as an Arrow array,
//!   so no extra series allocation on our side fortunately.
//! - Row must fully fit in memory (minor)
//! - 2D only, no 3D:
//!   `z` and `m` are dropped.
//!
//!   So, if possible we prefer our own impls :)

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use rsgeo::{Coord, LineString, Polygon};

/// Downcast one level of the storage to the array it has to be.
/// [`describe`](super::describe) already checked the dtype, so this failing is a bug.
fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> PolarsResult<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(
        || polars_err!(ComputeError: "expected {what} in geometry storage, got: {:?}", array.dtype()),
    )
}

/// The `x` and `y` of one chunk's coordinates.
/// Reads directly from the Arrow array, no allocation.
struct Coords<'a> {
    coords: &'a StructArray,
    x: &'a PrimitiveArray<f64>,
    y: &'a PrimitiveArray<f64>,
}

impl<'a> Coords<'a> {
    fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        // `coord::dimension_of` fixes the field order, so x and y always come first.
        let x = downcast(coords.values()[0].as_ref(), "f64 `x` coordinates")?;
        let y = downcast(coords.values()[1].as_ref(), "f64 `y` coordinates")?;
        Ok(Self { coords, x, y })
    }

    /// The coordinates `start..end`.
    /// `None` if any is missing: GeoArrow only allows nulls at the outermost level,
    /// so a geometry with a missing coordinate is missing as a whole
    /// (the same thing `validate` says).
    fn line_string(&self, start: usize, end: usize) -> Option<LineString<f64>> {
        (start..end)
            .map(|i| {
                let present = self.coords.is_valid(i) && self.x.is_valid(i) && self.y.is_valid(i);
                present.then(|| Coord {
                    x: self.x.value(i),
                    y: self.y.value(i),
                })
            })
            .collect::<Option<Vec<_>>>()
            .map(LineString)
    }
}

/// One chunk of a polygon column: a list of rings, each a list of coordinates.
struct Polygons<'a> {
    polygons: &'a ListArray<i64>,
    rings: &'a ListArray<i64>,
    coords: Coords<'a>,
}

impl<'a> Polygons<'a> {
    fn new(polygons: &'a ListArray<i64>) -> PolarsResult<Self> {
        let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
        let coords = Coords::new(rings.values().as_ref())?;
        Ok(Self {
            polygons,
            rings,
            coords,
        })
    }

    fn ring(&self, i: usize) -> Option<LineString<f64>> {
        if self.rings.is_null(i) {
            return None;
        }
        let (start, end) = self.rings.offsets().start_end(i);
        self.coords.line_string(start, end)
    }

    /// The polygon in row `i`: its first ring is the exterior, the rest are holes.
    fn polygon(&self, i: usize) -> Option<Polygon<f64>> {
        if self.polygons.is_null(i) {
            return None;
        }
        let (start, end) = self.polygons.offsets().start_end(i);
        let mut rings = (start..end)
            .map(|ring| self.ring(ring))
            .collect::<Option<Vec<_>>>()?
            .into_iter();
        // No rings at all is an empty polygon, not a missing one.
        let exterior = rings.next().unwrap_or_else(|| LineString(Vec::new()));
        Some(Polygon::new(exterior, rings.collect()))
    }
}

/// Every row of a `geoarrow.polygon`'s storage as an `rsgeo` polygon,
/// `None` where it is missing.
pub fn polygons(storage: &Series) -> PolarsResult<impl Iterator<Item = Option<Polygon<f64>>> + '_> {
    // Downcast every chunk before reading any row so we can error early if the layout is wrong.
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(Polygons::new)
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(chunks
        .into_iter()
        .flat_map(|chunk| (0..chunk.polygons.len()).map(move |i| chunk.polygon(i))))
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach: Arrow-level layout the
    //! Python constructors never produce on purpose.

    use super::super::GeoDimension;
    use super::*;

    fn ring(coords: &[(f64, f64)]) -> Series {
        let x = Series::new("x".into(), coords.iter().map(|c| c.0).collect::<Vec<_>>());
        let y = Series::new("y".into(), coords.iter().map(|c| c.1).collect::<Vec<_>>());
        StructChunked::from_series("".into(), coords.len(), [x, y].iter())
            .unwrap()
            .into_series()
    }

    fn polygon(rings: &[&[(f64, f64)]]) -> Series {
        if rings.is_empty() {
            // Nothing to infer the ring dtype from.
            let ring = DataType::List(Box::new(GeoDimension::XY.coordinates()));
            return Series::new_empty("".into(), &ring);
        }
        rings
            .iter()
            .map(|r| Some(ring(r)))
            .collect::<ListChunked>()
            .into_series()
    }

    fn column(polygons: Vec<Option<Series>>) -> Series {
        polygons.into_iter().collect::<ListChunked>().into_series()
    }

    const SQUARE: &[(f64, f64)] = &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)];
    const HOLE: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 1.0)];
    const TRIANGLE: &[(f64, f64)] = &[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)];

    fn sample() -> Series {
        column(vec![
            Some(polygon(&[SQUARE, HOLE])),
            None,
            Some(polygon(&[TRIANGLE])),
            Some(polygon(&[])),
        ])
    }

    fn read(storage: &Series) -> Vec<Option<Polygon<f64>>> {
        polygons(storage).unwrap().collect()
    }

    #[test]
    fn reads_rings_holes_and_missing_rows() {
        let got = read(&sample());

        let square = LineString::from(SQUARE.to_vec());
        let hole = LineString::from(HOLE.to_vec());
        assert_eq!(got[0], Some(Polygon::new(square, vec![hole])));
        assert_eq!(got[1], None);
        assert_eq!(got[2], Some(Polygon::new(TRIANGLE.to_vec().into(), vec![])));
        assert_eq!(got[3], Some(Polygon::new(LineString(vec![]), vec![])));
    }

    /// A slice keeps the full child arrays and starts its offsets past zero.
    #[test]
    fn a_slice_reads_the_rows_it_shows() {
        let all = read(&sample());

        assert_eq!(read(&sample().slice(1, 3)), all[1..4]);
        assert_eq!(read(&sample().slice(2, 1)), all[2..3]);
    }

    #[test]
    fn every_chunk_is_read_in_order() {
        let mut chunked = sample().slice(0, 2);
        chunked.append(&sample().slice(2, 2)).unwrap();
        assert_eq!(chunked.n_chunks(), 2);

        assert_eq!(read(&chunked), read(&sample()));
    }

    #[test]
    fn a_missing_coordinate_makes_the_polygon_missing() {
        let x = Series::new("x".into(), [Some(0.0), None, Some(0.0), Some(0.0)]);
        let y = Series::new("y".into(), [0.0, 1.0, 1.0, 0.0]);
        let broken = StructChunked::from_series("".into(), 4, [x, y].iter())
            .unwrap()
            .into_series();
        let rings = [Some(broken)].into_iter().collect::<ListChunked>();

        let got = read(&column(vec![Some(rings.into_series())]));

        assert_eq!(got, vec![None]);
    }
}
