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
