//! Shared functionality for expressions is often implemented over coordinates,
//! the shared building block of geometries.

use polars::prelude::*;

use crate::geoarrow::describe;

/// For an operation that hands back the geometry it got.
pub fn kernel_output_type_same_geometry(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(field.clone())
}

/// `storage`, holding only the coordinates its rows use.
/// A streaming morsel still holds every coordinate of the column it was sliced from,
/// which the kernels would otherwise all process. Trimming is zero-copy.
pub fn trimmed(storage: &Series) -> Series {
    storage
        .trim_lists_to_normalized_offsets()
        .unwrap_or_else(|| storage.clone())
}

#[cfg(test)]
pub(crate) mod tests {
    //! Helpers to build storage, for the tests of other expressions.

    use super::*;

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
}
