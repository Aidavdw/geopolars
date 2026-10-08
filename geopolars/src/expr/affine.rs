//! Geometric transformations that preserve lines and parallelism,
//! but not necessarily Euclidean distances and angles.
//!
//! Translating, rotating, scaling and skewing are all one of these,
//! so they share one kernel, and differ only in the matrix they hand it.

use polars::prelude::*;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::{map_coords, same_geometry};
use crate::geoarrow::{describe, GeoDimension};

/// An affine transformation of the coordinates that are positions (`x`, `y` and `z`),
/// as the top three rows of its augmented matrix:
///
/// ```text
/// | x' |   | a  b  c  xoff |   | x |
/// | y' | = | d  e  f  yoff | · | y |
/// | z' |   | g  h  i  zoff |   | z |
///                              | 1 |
/// ```
///
/// Stored row by row in one flat array, `[a, b, c, xoff, d, e, f, yoff, g, h, i, zoff]`.
///
/// A geometry without a `z` reads it as 0 and has no `z'` to write,
/// so `c`, `f` and the whole bottom row do nothing to it.
/// `m` is a measure, not a position, so it is never transformed.
#[derive(Deserialize)]
struct AffineMatrix {
    // Only the coordinates the row actually uses are read:
    // a kernel this simple spends its time moving data, not multiplying it,
    // and most rows (of a translation, a scaling, a rotation in the plane) use only one or two.
    // A coordinate a row multiplies by 0 therefore has no say in it, not even when it is NaN.
    coefficients: [f64; 12],
}

impl AffineMatrix {
    /// The row that makes output coordinate `axis` (0 is `x`, 1 is `y`, 2 is `z`),
    /// with its offset last.
    fn row(&self, axis: usize) -> [f64; 4] {
        let (rows, _) = self.coefficients.as_chunks::<4>();
        rows[axis]
    }

    /// Output coordinate `axis` (0 is `x`, 1 is `y`, 2 is `z`) for every position,
    /// or `None` if the matrix leaves that coordinate as it is.
    fn apply(&self, axis: usize, x: &[f64], y: &[f64], z: Option<&[f64]>) -> Option<Vec<f64>> {
        let row = self.row(axis);
        let offset = row[3];
        let mut unchanged = [0.0; 4];
        unchanged[axis] = 1.0;
        if row == unchanged {
            return None;
        }

        let terms: Vec<(f64, &[f64])> = [Some(x), Some(y), z]
            .into_iter()
            .zip(row)
            .filter(|&(_, coefficient)| coefficient != 0.0)
            .filter_map(|(values, coefficient)| Some((coefficient, values?)))
            .collect();
        Some(match terms.as_slice() {
            [] => vec![offset; x.len()],
            [(a, u)] => u.iter().map(|u| a * u + offset).collect(),
            [(a, u), (b, v)] => u
                .iter()
                .zip(*v)
                .map(|(u, v)| a * u + b * v + offset)
                .collect(),
            [(a, u), (b, v), (c, w)] => u
                .iter()
                .zip(*v)
                .zip(*w)
                .map(|((u, v), w)| a * u + b * v + c * w + offset)
                .collect(),
            _ => unreachable!("a row has three coefficients before its offset"),
        })
    }
}

/// Transform one flat array of coordinates.
fn transform(coords: &Series, dim: GeoDimension, matrix: &AffineMatrix) -> PolarsResult<Series> {
    let fields = coords.struct_()?;
    let position = |name: &str| -> PolarsResult<Float64Chunked> {
        Ok(fields.field_by_name(name)?.f64()?.rechunk().into_owned())
    };
    let x = position("x")?;
    let y = position("y")?;
    let z = dim.has_z().then(|| position("z")).transpose()?;

    let (x, y) = (x.downcast_as_array(), y.downcast_as_array());
    let z = z.as_ref().map(|z| z.downcast_as_array());
    // Every output coordinate is made from all of the input ones,
    // so it is missing wherever any of those is.
    let validity = combine_validities_and(x.validity(), y.validity());
    let validity = match z {
        Some(z) => combine_validities_and(validity.as_ref(), z.validity()),
        None => validity,
    };

    let z_values = z.map(|z| z.values().as_slice());
    let transformed = |axis: usize, name: &str| -> PolarsResult<Series> {
        Ok(match matrix.apply(axis, x.values(), y.values(), z_values) {
            None => fields.field_by_name(name)?,
            Some(values) => {
                Float64Chunked::from_vec_validity(name.into(), values, validity.clone())
                    .into_series()
            }
        })
    };
    // In the order the spec gives the fields in, so the result has the input's dtype.
    let mut out = vec![transformed(0, "x")?, transformed(1, "y")?];
    if dim.has_z() {
        out.push(transformed(2, "z")?);
    }
    if dim.has_m() {
        out.push(fields.field_by_name("m")?);
    }

    let mut out = StructChunked::from_series(coords.name().clone(), coords.len(), out.iter())?;
    // Building from fields alone drops the validity that says a coordinate is missing entirely.
    out.zip_outer_validity(fields);
    Ok(out.into_series())
}

/// The kernel behind affine ops like `translate` (see `geo/affine.py`).
#[polars_expr(output_type_func=same_geometry)]
fn affine(inputs: &[Series], kwargs: AffineMatrix) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let out = map_coords(inputs[0].ext()?.storage(), geo.kind.nesting(), &|coords| {
        transform(coords, geo.dim, &kwargs)
    })?;

    Ok(out.into_extension(geo.typ.clone()))
}

#[cfg(test)]
mod tests {
    //! What the Python suite cannot reach yet:
    //! a matrix that mixes coordinates, rather than only offsetting them.

    use super::*;

    #[test]
    fn each_row_mixes_the_coordinates_in_order() {
        let matrix = AffineMatrix {
            coefficients: [
                1.0, 2.0, 3.0, 4.0, // x' = x + 2y + 3z + 4
                5.0, 6.0, 7.0, 8.0, // y' = 5x + 6y + 7z + 8
                9.0, 10.0, 11.0, 12.0, // z' = 9x + 10y + 11z + 12
            ],
        };
        let (x, y, z) = ([1.0, 0.0], [0.0, 1.0], [2.0, 3.0]);

        assert_eq!(matrix.apply(0, &x, &y, Some(&z)), Some(vec![11.0, 15.0]));
        assert_eq!(matrix.apply(1, &x, &y, Some(&z)), Some(vec![27.0, 35.0]));
        assert_eq!(matrix.apply(2, &x, &y, Some(&z)), Some(vec![43.0, 55.0]));
    }

    #[test]
    fn without_z_the_z_column_does_nothing() {
        let matrix = AffineMatrix {
            coefficients: [
                0.0, -1.0, 100.0, 1.0, // x' = -y + 1, plus 100z if there were a z
                1.0, 0.0, 100.0, 2.0, // y' = x + 2, plus 100z if there were a z
                0.0, 0.0, 1.0, 0.0,
            ],
        };
        let (x, y) = ([1.0, 3.0], [2.0, 4.0]);

        assert_eq!(matrix.apply(0, &x, &y, None), Some(vec![-1.0, -3.0]));
        assert_eq!(matrix.apply(1, &x, &y, None), Some(vec![3.0, 5.0]));
    }

    #[test]
    fn a_coordinate_the_matrix_leaves_alone_is_not_rebuilt() {
        let matrix = AffineMatrix {
            coefficients: [
                1.0, 0.0, 0.0, 5.0, // x' = x + 5
                0.0, 1.0, 0.0, 0.0, // y' = y
                0.0, 0.0, 0.0, 7.0, // z' = 7
            ],
        };
        let (x, y, z) = ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]);

        assert_eq!(matrix.apply(0, &x, &y, Some(&z)), Some(vec![6.0, 7.0]));
        assert_eq!(matrix.apply(1, &x, &y, Some(&z)), None);
        assert_eq!(matrix.apply(2, &x, &y, Some(&z)), Some(vec![7.0, 7.0]));
    }
}
