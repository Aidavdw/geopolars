//! Shared functionality for expressions is often implemented over coordinates,
//! the shared building block of geometries.

use polars::prelude::*;

use crate::geoarrow::describe;

/// `output_type_func` for an operation that hands back the geometry it got.
pub fn same_geometry(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(field.clone())
}

/// Apply a coordinate-wise kernel through however many `List` layers the
/// geometry nests its coordinates under.
///
/// This works on any geometry, because it works on the coordinates directly.
/// Geometry-specific data (e.g. group association) is never touched.
pub fn map_coords(
    storage: &Series,
    nesting: u8,
    f: &dyn Fn(&Series) -> PolarsResult<Series>,
) -> PolarsResult<Series> {
    // A streaming morsel still holds every coordinate of the
    // column it was sliced from, and `apply_to_inner` hands `f` all of them.
    // Trimming (zero-copy, every layer at once) leaves `f` only this slice's coordinates.
    let trimmed = storage.trim_lists_to_normalized_offsets();
    map_nested(trimmed.as_ref().unwrap_or(storage), nesting, f)
}

/// Where each row of `storage` is among the coordinates `map_coords` hands its kernel:
/// row `r` is coordinates `bounds[r]..bounds[r + 1]`.
///
/// For a kernel that needs something per geometry (such as the point to rotate it about)
/// at every one of its coordinates.
pub fn coordinate_bounds(storage: &Series, nesting: u8) -> PolarsResult<Vec<i64>> {
    // The same coordinates as `map_coords` hands over: trimmed, and in one chunk,
    // as `apply_to_inner` rechunks every layer.
    let storage = storage.rechunk();
    let mut layer = storage
        .trim_lists_to_normalized_offsets()
        .unwrap_or(storage)
        .rechunk();
    let mut bounds: Vec<i64> = (0..=layer.len() as i64).collect();
    for _ in 0..nesting {
        let list = layer.list()?.rechunk();
        let offsets = list.downcast_as_array().offsets();
        for at in &mut bounds {
            *at = offsets[*at as usize];
        }
        layer = list.get_inner();
    }

    let (first, last) = (bounds[0], bounds[bounds.len() - 1]);
    polars_ensure!(
        first == 0 && last as usize == layer.len(),
        ComputeError: "the rows span coordinates {first}..{last}, out of {}", layer.len()
    );
    Ok(bounds)
}

fn map_nested(
    storage: &Series,
    nesting: u8,
    f: &dyn Fn(&Series) -> PolarsResult<Series>,
) -> PolarsResult<Series> {
    match nesting {
        0 => f(storage),
        n => Ok(storage
            .list()?
            .apply_to_inner(&|inner| map_nested(&inner, n - 1, f))?
            .into_series()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! What the Python suite cannot see: how many coordinates `f` is handed.
    //! A slice gives the right answer either way, just after much more work.

    use std::cell::Cell;

    use super::*;

    /// How many coordinates `map_coords` hands `f` for the middle geometry of three,
    /// sliced out the way a streaming morsel is.
    fn coordinates_seen(storage: Series, nesting: u8) -> usize {
        let seen = Cell::new(0);
        let out = map_coords(&storage.slice(1, 1), nesting, &|coords| {
            seen.set(seen.get() + coords.len());
            Ok(coords.clone())
        })
        .unwrap();
        assert_eq!(out, storage.slice(1, 1));
        seen.get()
    }

    pub(crate) fn coords(n: usize) -> Series {
        let x = Series::new("x".into(), vec![1.0; n]);
        let y = Series::new("y".into(), vec![2.0; n]);
        StructChunked::from_series("".into(), n, [x, y].iter())
            .unwrap()
            .into_series()
    }

    /// Each of `lengths` as one list over consecutive values of `inner`.
    pub(crate) fn lists(inner: Series, lengths: &[i64]) -> Series {
        let offsets = std::iter::once(0)
            .chain(lengths.iter().scan(0, |end, len| {
                *end += len;
                Some(*end)
            }))
            .collect::<Vec<i64>>();
        let arr = LargeListArray::new(
            LargeListArray::default_datatype(inner.dtype().to_arrow(CompatLevel::newest())),
            offsets.try_into().unwrap(),
            inner.rechunk().chunks()[0].clone(),
            None,
        );
        Series::from_arrow("".into(), arr.boxed()).unwrap()
    }

    #[test]
    fn every_row_of_a_slice_knows_its_coordinates() {
        let lines = lists(coords(9), &[2, 3, 4]);
        assert_eq!(coordinate_bounds(&lines.slice(1, 2), 1).unwrap(), [0, 3, 7]);

        // Three polygons of two rings each, of 2 + 3, 4 + 5 and 6 + 7 coordinates.
        let polygons = lists(lists(coords(27), &[2, 3, 4, 5, 6, 7]), &[2, 2, 2]);
        assert_eq!(coordinate_bounds(&polygons.slice(1, 1), 2).unwrap(), [0, 9]);
        assert_eq!(coordinate_bounds(&polygons, 2).unwrap(), [0, 5, 14, 27]);
    }

    #[test]
    fn a_slice_of_linestrings_maps_only_its_own_coordinates() {
        assert_eq!(coordinates_seen(lists(coords(9), &[2, 3, 4]), 1), 3);
    }

    #[test]
    fn a_slice_of_polygons_maps_only_its_own_coordinates() {
        // Three polygons of two rings each, of 2 + 3, 4 + 5 and 6 + 7 coordinates.
        let rings = lists(coords(27), &[2, 3, 4, 5, 6, 7]);
        assert_eq!(coordinates_seen(lists(rings, &[2, 2, 2]), 2), 9);
    }
}
