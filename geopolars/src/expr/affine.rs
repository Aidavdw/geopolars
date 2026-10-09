//! Geometric transformations that preserve lines and parallelism,
//! but not necessarily Euclidean distances and angles.
//!
//! Translating, rotating, scaling and skewing are all one of these,
//! so they share one kernel, and differ only in the matrix they hand it.

use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::{coordinate_bounds, map_coords, same_geometry};
use crate::geoarrow::{describe, GeoDimension, Kind};

// TODO: (Blocked until z-centroid is implemented): allow rotation around x/y for polygon
// TODO: (Blocked until z-centroid is implemented): allow scaling around z for polygon(?)

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

    /// Whether the row that makes `axis` takes that coordinate as it is,
    /// before its offset is added.
    fn keeps(&self, axis: usize) -> bool {
        let mut unchanged = [0.0; 3];
        unchanged[axis] = 1.0;
        self.row(axis)[..3] == unchanged
    }

    /// Output coordinate `axis` (0 is `x`, 1 is `y`, 2 is `z`) for every position,
    /// plus `shift` (one value per position) if there is one,
    /// or `None` if that leaves the coordinate as it is.
    fn apply(
        &self,
        axis: usize,
        x: &[f64],
        y: &[f64],
        z: Option<&[f64]>,
        shift: Option<&[f64]>,
    ) -> Option<Vec<f64>> {
        let row = self.row(axis);
        let offset = row[3];
        if self.keeps(axis) && offset == 0.0 && shift.is_none() {
            return None;
        }

        let terms: Vec<(f64, &[f64])> = [Some(x), Some(y), z]
            .into_iter()
            .zip(row)
            .filter(|&(_, coefficient)| coefficient != 0.0)
            .filter_map(|(values, coefficient)| Some((coefficient, values?)))
            // Last, so a shift equal to a constant offset adds up exactly the same.
            .chain(shift.map(|shift| (1.0, shift)))
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
            [(a, u), (b, v), (c, w), (d, t)] => u
                .iter()
                .zip(*v)
                .zip(*w)
                .zip(*t)
                .map(|(((u, v), w), t)| a * u + b * v + c * w + d * t + offset)
                .collect(),
            _ => unreachable!("a row has three coefficients and a shift before its offset"),
        })
    }
}

/// Transform one flat array of coordinates,
/// shifting output coordinate `axis` by `shifts[axis]` (one value per position) as well.
fn transform(
    coords: &Series,
    dim: GeoDimension,
    matrix: &AffineMatrix,
    shifts: &[Option<Vec<f64>>; 3],
) -> PolarsResult<Series> {
    for shift in shifts.iter().flatten() {
        polars_ensure!(
            shift.len() == coords.len(),
            ComputeError: "{} shifts for {} coordinates", shift.len(), coords.len()
        );
    }
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
        let shift = shifts[axis].as_deref();
        Ok(
            match matrix.apply(axis, x.values(), y.values(), z_values, shift) {
                None => fields.field_by_name(name)?,
                Some(values) => {
                    Float64Chunked::from_vec_validity(name.into(), values, validity.clone())
                        .into_series()
                }
            },
        )
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
        transform(coords, geo.dim, &kwargs, &[None, None, None])
    })?;

    Ok(out.into_extension(geo.typ.clone()))
}

/// `output_type_func` for an operation by the points of a second column,
/// one per geometry: it hands back the geometry it got.
fn same_geometry_by_points(input_fields: &[Field]) -> PolarsResult<Field> {
    let [geometry, points] = input_fields else {
        polars_bail!(ComputeError: "expected a geometry and its points, got {} inputs", input_fields.len());
    };
    let metadata = describe(geometry.dtype())?.metadata;
    let points = describe(points.dtype())?;
    polars_ensure!(
        points.kind == Kind::Point,
        SchemaMismatch: "expected a `{}` column of points to transform by, got: {}",
        Kind::Point.name(), points.typ
    );
    // Points without a CRS are plain positions, like ones given as numbers.
    if metadata.declares_crs() && points.metadata.declares_crs() {
        let (crs, points_crs) = (metadata.crs()?, points.metadata.crs()?);
        polars_ensure!(
            crs == points_crs,
            SchemaMismatch: "the geometry is in {crs}, but its points are in {points_crs}; \
            use `to_crs` to bring one onto the other first"
        );
    }
    Ok(geometry.clone())
}

/// The kernel behind affine ops about a point per row (see `geo/affine.py`):
/// the matrix applied about each geometry's own `origin`, `M.(p − o) + o`.
///
/// That is `L.p + offset + (o − L.o)`, with `L` the matrix without its offsets:
/// the constant kernel, with a shift that is the same for every position of a geometry.
#[polars_expr(output_type_func=same_geometry_by_points)]
fn affine_about(inputs: &[Series], kwargs: AffineMatrix) -> PolarsResult<Series> {
    shifted_per_row(inputs, &kwargs, &|axis, origin| {
        // A row that keeps its coordinate moves it by its offset alone, about any origin.
        if kwargs.keeps(axis) {
            return None;
        }
        let [ox, oy, oz] = origin;
        let offset = kwargs.row(axis)[3];
        let moved = kwargs.apply(axis, ox?, oy?, oz, None)?;
        // `o − L·o`, for each geometry.
        Some(match origin[axis] {
            Some(at) => at
                .iter()
                .zip(moved)
                .map(|(at, moved)| at - (moved - offset))
                .collect(),
            // An origin without a `z` is at `z` = 0, like one given as two numbers.
            None => moved.iter().map(|moved| -(moved - offset)).collect(),
        })
    })
}

/// The kernel behind affine ops that move every geometry by a vector of its own
/// (see `geo/affine.py`): `M.p + v`, with `v` a point per row.
#[polars_expr(output_type_func=same_geometry_by_points)]
fn affine_then_shift(inputs: &[Series], kwargs: AffineMatrix) -> PolarsResult<Series> {
    // A vector without a `z` leaves it where the matrix puts it.
    shifted_per_row(inputs, &kwargs, &|axis, by| by[axis].map(<[f64]>::to_vec))
}

/// The `x`, `y` and `z` of a column of points, each `None` if the points do not have it.
type Coordinates<'a> = [Option<&'a [f64]>; 3];

/// Transform the geometries of `inputs[0]` by `matrix`,
/// and shift each one along `axis` by `per_geometry(axis, [x, y, z])`
/// of its own point in `inputs[1]` (a `None` for a coordinate that point does not have),
/// or not at all where that is `None`.
fn shifted_per_row(
    inputs: &[Series],
    matrix: &AffineMatrix,
    per_geometry: &dyn Fn(usize, Coordinates<'_>) -> Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let points = inputs[1].ext()?.storage();
    let points = match (storage.len(), points.len()) {
        (n, m) if n == m => points.rechunk(),
        (n, 1) => points.new_from_index(0, n),
        (n, m) => polars_bail!(ShapeMismatch: "cannot transform {n} geometries by {m} points"),
    };

    let fields = points.struct_()?;
    let position = |name: &str| -> PolarsResult<Option<Float64Chunked>> {
        Ok(match fields.field_by_name(name) {
            Ok(field) => Some(field.f64()?.rechunk().into_owned()),
            Err(_) => None,
        })
    };
    let at = [position("x")?, position("y")?, position("z")?];
    // A geometry is missing wherever its point, or any part of it, is.
    let mut missing = fields.rechunk_validity();
    for at in at.iter().flatten() {
        missing = combine_validities_and(missing.as_ref(), at.downcast_as_array().validity());
    }
    let at = [0, 1, 2].map(|axis| {
        at[axis]
            .as_ref()
            .map(|at| at.downcast_as_array().values().as_slice())
    });

    let bounds = (geo.kind.nesting() > 0)
        .then(|| coordinate_bounds(storage, geo.kind.nesting()))
        .transpose()?;
    let axes = if geo.dim.has_z() { 3 } else { 2 };
    let shift = |axis: usize| -> Option<Vec<f64>> {
        let per_geometry = per_geometry(axis, at)?;
        Some(match &bounds {
            None => per_geometry,
            // Each geometry's own value, at every one of its coordinates.
            Some(bounds) => {
                let mut shift = Vec::with_capacity(*bounds.last().unwrap_or(&0) as usize);
                for (part, value) in bounds.windows(2).zip(per_geometry) {
                    shift.resize(part[1] as usize, value);
                }
                shift
            }
        })
    };
    let shifts = [0, 1, 2].map(|axis| if axis < axes { shift(axis) } else { None });

    let out = map_coords(storage, geo.kind.nesting(), &|coords| {
        transform(coords, geo.dim, matrix, &shifts)
    })?;
    Ok(with_missing(out, missing).into_extension(geo.typ.clone()))
}

/// `geometries`, missing as well wherever `validity` says so.
fn with_missing(geometries: Series, validity: Option<Bitmap>) -> Series {
    let Some(validity) = validity else {
        return geometries;
    };
    let geometries = geometries.rechunk();
    let chunk = &geometries.chunks()[0];
    let validity = combine_validities_and(chunk.validity(), Some(&validity));
    // SAFETY: only the validity changes, not the dtype.
    unsafe {
        Series::from_chunks_and_dtype_unchecked(
            geometries.name().clone(),
            vec![chunk.with_validity(validity)],
            geometries.dtype(),
        )
    }
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

        assert_eq!(
            matrix.apply(0, &x, &y, Some(&z), None),
            Some(vec![11.0, 15.0])
        );
        assert_eq!(
            matrix.apply(1, &x, &y, Some(&z), None),
            Some(vec![27.0, 35.0])
        );
        assert_eq!(
            matrix.apply(2, &x, &y, Some(&z), None),
            Some(vec![43.0, 55.0])
        );
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

        assert_eq!(matrix.apply(0, &x, &y, None, None), Some(vec![-1.0, -3.0]));
        assert_eq!(matrix.apply(1, &x, &y, None, None), Some(vec![3.0, 5.0]));
    }

    #[test]
    fn a_shift_moves_each_position_by_its_own_amount() {
        let matrix = AffineMatrix {
            coefficients: [
                0.0, -1.0, 0.0, 0.0, // x' = -y + shift
                1.0, 0.0, 0.0, 0.0, // y' = x
                0.0, 0.0, 1.0, 0.0, // z' = z + shift
            ],
        };
        let (x, y, z) = ([1.0, 3.0], [2.0, 4.0], [5.0, 6.0]);
        let shift = [10.0, 20.0];

        assert_eq!(
            matrix.apply(0, &x, &y, Some(&z), Some(&shift)),
            Some(vec![8.0, 16.0])
        );
        assert_eq!(
            matrix.apply(2, &x, &y, Some(&z), Some(&shift)),
            Some(vec![15.0, 26.0])
        );
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

        assert_eq!(
            matrix.apply(0, &x, &y, Some(&z), None),
            Some(vec![6.0, 7.0])
        );
        assert_eq!(matrix.apply(1, &x, &y, Some(&z), None), None);
        assert_eq!(
            matrix.apply(2, &x, &y, Some(&z), None),
            Some(vec![7.0, 7.0])
        );
    }
}
