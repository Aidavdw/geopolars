//! [geo-traits](geo_traits) views over our Arrow storage.
//!
//! These exist because the `wkb` and `wkt` writers only take geometries through geo-traits.
//! Every view borrows one chunk of the storage and copies nothing:
//! a geometry is a range into the list level below it,
//! down to a range of coordinates.

use geo_traits::{
    CoordTrait, Dimensions, GeometryTrait, LineStringTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait, PointTrait, PolygonTrait, UnimplementedGeometryCollection,
    UnimplementedLine, UnimplementedRect, UnimplementedTriangle,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};

use super::storage::{downcast, CoordsView};
use super::{GeoDimension, Kind};

impl From<GeoDimension> for Dimensions {
    fn from(dim: GeoDimension) -> Self {
        match dim {
            GeoDimension::XY => Dimensions::Xy,
            GeoDimension::XYZ => Dimensions::Xyz,
            GeoDimension::XYM => Dimensions::Xym,
            GeoDimension::XYZM => Dimensions::Xyzm,
        }
    }
}

/// The range of the level below that entry `i` of `list` spans.
fn range(list: &ListArray<i64>, i: usize) -> (usize, usize) {
    list.offsets().start_end(i)
}

/// Coordinate `i`.
#[derive(Clone, Copy)]
pub struct Coord<'a> {
    coords: &'a CoordsView<'a>,
    i: usize,
}

impl CoordTrait for Coord<'_> {
    type T = f64;

    fn dim(&self) -> Dimensions {
        self.coords.dim().into()
    }

    fn x(&self) -> f64 {
        self.coords.nth(self.i, 0)
    }

    fn y(&self) -> f64 {
        self.coords.nth(self.i, 1)
    }

    fn nth_or_panic(&self, n: usize) -> f64 {
        self.coords.nth(self.i, n)
    }
}

/// The point at coordinate `i`.
#[derive(Clone, Copy)]
pub struct Point<'a> {
    pub coords: &'a CoordsView<'a>,
    pub i: usize,
    /// Whether NaN `x` and `y` make this the empty point, as GeoArrow spells it.
    /// Only for a point of its own:
    /// the `wkt` writer cannot write an empty point inside a multipoint.
    pub nan_is_empty: bool,
}

/// The coordinates `start..end`.
#[derive(Clone, Copy)]
pub struct LineString<'a> {
    pub coords: &'a CoordsView<'a>,
    pub start: usize,
    pub end: usize,
}

/// The rings `start..end`, the first of which is the exterior.
#[derive(Clone, Copy)]
pub struct Polygon<'a> {
    pub rings: &'a ListArray<i64>,
    pub coords: &'a CoordsView<'a>,
    pub start: usize,
    pub end: usize,
}

/// The points at coordinates `start..end`.
#[derive(Clone, Copy)]
pub struct MultiPoint<'a> {
    pub coords: &'a CoordsView<'a>,
    pub start: usize,
    pub end: usize,
}

/// The linestrings `start..end`.
#[derive(Clone, Copy)]
pub struct MultiLineString<'a> {
    pub lines: &'a ListArray<i64>,
    pub coords: &'a CoordsView<'a>,
    pub start: usize,
    pub end: usize,
}

/// The polygons `start..end`.
#[derive(Clone, Copy)]
pub struct MultiPolygon<'a> {
    pub polygons: &'a ListArray<i64>,
    pub rings: &'a ListArray<i64>,
    pub coords: &'a CoordsView<'a>,
    pub start: usize,
    pub end: usize,
}

impl<'a> PointTrait for Point<'a> {
    type CoordType<'b>
        = Coord<'a>
    where
        Self: 'b;

    fn coord(&self) -> Option<Coord<'a>> {
        let empty = self.nan_is_empty
            && self.coords.nth(self.i, 0).is_nan()
            && self.coords.nth(self.i, 1).is_nan();
        (!empty).then_some(Coord {
            coords: self.coords,
            i: self.i,
        })
    }
}

impl<'a> LineStringTrait for LineString<'a> {
    type CoordType<'b>
        = Coord<'a>
    where
        Self: 'b;

    fn num_coords(&self) -> usize {
        self.end - self.start
    }

    unsafe fn coord_unchecked(&self, i: usize) -> Coord<'a> {
        Coord {
            coords: self.coords,
            i: self.start + i,
        }
    }
}

impl<'a> Polygon<'a> {
    fn ring(&self, i: usize) -> LineString<'a> {
        let (start, end) = range(self.rings, self.start + i);
        LineString {
            coords: self.coords,
            start,
            end,
        }
    }
}

impl<'a> PolygonTrait for Polygon<'a> {
    type RingType<'b>
        = LineString<'a>
    where
        Self: 'b;

    fn exterior(&self) -> Option<LineString<'a>> {
        (self.start < self.end).then(|| self.ring(0))
    }

    fn num_interiors(&self) -> usize {
        (self.end - self.start).saturating_sub(1)
    }

    unsafe fn interior_unchecked(&self, i: usize) -> LineString<'a> {
        self.ring(i + 1)
    }
}

impl<'a> MultiPointTrait for MultiPoint<'a> {
    type InnerPointType<'b>
        = Point<'a>
    where
        Self: 'b;

    fn num_points(&self) -> usize {
        self.end - self.start
    }

    unsafe fn point_unchecked(&self, i: usize) -> Point<'a> {
        Point {
            coords: self.coords,
            i: self.start + i,
            nan_is_empty: false,
        }
    }
}

impl<'a> MultiLineStringTrait for MultiLineString<'a> {
    type InnerLineStringType<'b>
        = LineString<'a>
    where
        Self: 'b;

    fn num_line_strings(&self) -> usize {
        self.end - self.start
    }

    unsafe fn line_string_unchecked(&self, i: usize) -> LineString<'a> {
        let (start, end) = range(self.lines, self.start + i);
        LineString {
            coords: self.coords,
            start,
            end,
        }
    }
}

impl<'a> MultiPolygonTrait for MultiPolygon<'a> {
    type InnerPolygonType<'b>
        = Polygon<'a>
    where
        Self: 'b;

    fn num_polygons(&self) -> usize {
        self.end - self.start
    }

    unsafe fn polygon_unchecked(&self, i: usize) -> Polygon<'a> {
        let (start, end) = range(self.polygons, self.start + i);
        Polygon {
            rings: self.rings,
            coords: self.coords,
            start,
            end,
        }
    }
}

/// Every view is a geometry of the one variant it is.
macro_rules! impl_geometry {
    ($($view:ident),*) => {$(
        impl<'a> GeometryTrait for $view<'a> {
            type T = f64;
            type PointType<'b> = Point<'a> where Self: 'b;
            type LineStringType<'b> = LineString<'a> where Self: 'b;
            type PolygonType<'b> = Polygon<'a> where Self: 'b;
            type MultiPointType<'b> = MultiPoint<'a> where Self: 'b;
            type MultiLineStringType<'b> = MultiLineString<'a> where Self: 'b;
            type MultiPolygonType<'b> = MultiPolygon<'a> where Self: 'b;
            type GeometryCollectionType<'b> = UnimplementedGeometryCollection<f64> where Self: 'b;
            type RectType<'b> = UnimplementedRect<f64> where Self: 'b;
            type TriangleType<'b> = UnimplementedTriangle<f64> where Self: 'b;
            type LineType<'b> = UnimplementedLine<f64> where Self: 'b;

            fn dim(&self) -> Dimensions {
                self.coords.dim().into()
            }

            fn as_type(
                &self,
            ) -> geo_traits::GeometryType<
                '_,
                Point<'a>,
                LineString<'a>,
                Polygon<'a>,
                MultiPoint<'a>,
                MultiLineString<'a>,
                MultiPolygon<'a>,
                UnimplementedGeometryCollection<f64>,
                UnimplementedRect<f64>,
                UnimplementedTriangle<f64>,
                UnimplementedLine<f64>,
            > {
                geo_traits::GeometryType::$view(self)
            }
        }
    )*};
}

impl_geometry!(
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon
);

/// Any one of the views.
#[derive(Clone, Copy)]
pub enum Geometry<'a> {
    Point(Point<'a>),
    LineString(LineString<'a>),
    Polygon(Polygon<'a>),
    MultiPoint(MultiPoint<'a>),
    MultiLineString(MultiLineString<'a>),
    MultiPolygon(MultiPolygon<'a>),
}

impl<'a> GeometryTrait for Geometry<'a> {
    type T = f64;
    type PointType<'b>
        = Point<'a>
    where
        Self: 'b;
    type LineStringType<'b>
        = LineString<'a>
    where
        Self: 'b;
    type PolygonType<'b>
        = Polygon<'a>
    where
        Self: 'b;
    type MultiPointType<'b>
        = MultiPoint<'a>
    where
        Self: 'b;
    type MultiLineStringType<'b>
        = MultiLineString<'a>
    where
        Self: 'b;
    type MultiPolygonType<'b>
        = MultiPolygon<'a>
    where
        Self: 'b;
    type GeometryCollectionType<'b>
        = UnimplementedGeometryCollection<f64>
    where
        Self: 'b;
    type RectType<'b>
        = UnimplementedRect<f64>
    where
        Self: 'b;
    type TriangleType<'b>
        = UnimplementedTriangle<f64>
    where
        Self: 'b;
    type LineType<'b>
        = UnimplementedLine<f64>
    where
        Self: 'b;

    fn dim(&self) -> Dimensions {
        match self {
            Geometry::Point(g) => g.dim(),
            Geometry::LineString(g) => g.dim(),
            Geometry::Polygon(g) => g.dim(),
            Geometry::MultiPoint(g) => g.dim(),
            Geometry::MultiLineString(g) => g.dim(),
            Geometry::MultiPolygon(g) => g.dim(),
        }
    }

    fn as_type(
        &self,
    ) -> geo_traits::GeometryType<
        '_,
        Point<'a>,
        LineString<'a>,
        Polygon<'a>,
        MultiPoint<'a>,
        MultiLineString<'a>,
        MultiPolygon<'a>,
        UnimplementedGeometryCollection<f64>,
        UnimplementedRect<f64>,
        UnimplementedTriangle<f64>,
        UnimplementedLine<f64>,
    > {
        match self {
            Geometry::Point(g) => geo_traits::GeometryType::Point(g),
            Geometry::LineString(g) => geo_traits::GeometryType::LineString(g),
            Geometry::Polygon(g) => geo_traits::GeometryType::Polygon(g),
            Geometry::MultiPoint(g) => geo_traits::GeometryType::MultiPoint(g),
            Geometry::MultiLineString(g) => geo_traits::GeometryType::MultiLineString(g),
            Geometry::MultiPolygon(g) => geo_traits::GeometryType::MultiPolygon(g),
        }
    }
}

/// Calls `f` with `row(i)` for every row `i` of `chunk`, or `None` where it is null.
fn rows<'a>(
    chunk: &dyn Array,
    f: &mut impl FnMut(Option<Geometry<'a>>) -> PolarsResult<()>,
    row: impl Fn(usize) -> Geometry<'a>,
) -> PolarsResult<()> {
    (0..chunk.len()).try_for_each(|i| f((!chunk.is_null(i)).then(|| row(i))))
}

/// Calls `f` with every row of `chunk`, a chunk of a `kind` column's storage:
/// the geometry, or `None` for a null.
pub fn for_each_geometry(
    chunk: &dyn Array,
    kind: Kind,
    mut f: impl FnMut(Option<Geometry<'_>>) -> PolarsResult<()>,
) -> PolarsResult<()> {
    match kind {
        Kind::Point => {
            let coords = CoordsView::new(chunk)?;
            rows(chunk, &mut f, |i| {
                Geometry::Point(Point {
                    coords: &coords,
                    i,
                    nan_is_empty: true,
                })
            })
        }
        Kind::LineString | Kind::MultiPoint => {
            let list = downcast::<ListArray<i64>>(chunk, "a list of coordinates")?;
            let coords = CoordsView::new(list.values().as_ref())?;
            rows(chunk, &mut f, |i| {
                let (start, end) = range(list, i);
                let coords = &coords;
                match kind {
                    Kind::LineString => Geometry::LineString(LineString { coords, start, end }),
                    _ => Geometry::MultiPoint(MultiPoint { coords, start, end }),
                }
            })
        }
        Kind::Polygon | Kind::MultiLineString => {
            let list = downcast::<ListArray<i64>>(chunk, "a list of rings or linestrings")?;
            let lines =
                downcast::<ListArray<i64>>(list.values().as_ref(), "a list of coordinates")?;
            let coords = CoordsView::new(lines.values().as_ref())?;
            rows(chunk, &mut f, |i| {
                let (start, end) = range(list, i);
                let coords = &coords;
                match kind {
                    Kind::Polygon => Geometry::Polygon(Polygon {
                        rings: lines,
                        coords,
                        start,
                        end,
                    }),
                    _ => Geometry::MultiLineString(MultiLineString {
                        lines,
                        coords,
                        start,
                        end,
                    }),
                }
            })
        }
        Kind::MultiPolygon => {
            let list = downcast::<ListArray<i64>>(chunk, "a list of polygons")?;
            let polygons = downcast::<ListArray<i64>>(list.values().as_ref(), "a list of rings")?;
            let rings =
                downcast::<ListArray<i64>>(polygons.values().as_ref(), "a list of coordinates")?;
            let coords = CoordsView::new(rings.values().as_ref())?;
            rows(chunk, &mut f, |i| {
                let (start, end) = range(list, i);
                Geometry::MultiPolygon(MultiPolygon {
                    polygons,
                    rings,
                    coords: &coords,
                    start,
                    end,
                })
            })
        }
    }
}
