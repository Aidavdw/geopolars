//! The GeoParquet metadata: the JSON under a Parquet file's `geo` key,
//!
//! GeoParquet keeps per column what GeoArrow keeps in a dtype's metadata,
//! but in its own format (see `docs/geoparquet-1.1.0p1.md`).
//! This module converts between our metadata representation and GeoParquet's.
//! Only GeoParquet 1.x is supported, as 2.0rc breaks with upstream

use std::sync::Arc;

use geoarrow_schema::Edges;
use indexmap::IndexMap;
use polars::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::geoarrow::crs::ExtensionMetadata;
use crate::geoarrow::encoded::{describe_encoded, Encoded, Encoding};
use crate::geoarrow::{coord, describe, Geo, Kind};

/// The GeoParquet version this crate writes.
const VERSION: &str = "1.1.0";

/// The CRS of a column without a `crs` key.
/// Follows geoparquet v1.1.0.
const DEFAULT_CRS: &str = "OGC:CRS84";

/// The value of the `geo` key in a Parquet file's metadata.
///
/// Fields we don't use (such as `bbox` or `covering`) are ignored when reading,
/// as the spec asks of readers.
#[derive(Debug, Serialize, Deserialize)]
pub struct GeoParquetMetadata {
    pub version: String,
    pub primary_column: String,
    /// Keyed by column name, in schema order.
    pub columns: IndexMap<String, GeoParquetColumn>,
}

/// How one geometry column is stored.
#[derive(Debug, Serialize, Deserialize)]
pub struct GeoParquetColumn {
    pub encoding: ColumnEncoding,
    /// `Point`, `Polygon Z`, ...; empty if not known.
    pub geometry_types: Vec<String>,
    /// A missing key (`None`) means OGC:CRS84.
    /// An explicit `null` (`Some(None)`) is an unknown CRS.
    /// Different from GeoArrow, where a missing CRS is unknown.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub crs: Option<Option<Map<String, Value>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<ParquetEdges>,
    /// Only read so that it can be refused: we don't support dynamic CRSs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
}

/// The layout of a geometry column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnEncoding {
    Wkb,
    /// GeoArrow's separated coordinates.
    Native(Kind),
}

/// The edges GeoParquet 1.1 knows.
/// GeoArrow knows a couple more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParquetEdges {
    Planar,
    Spherical,
}

impl GeoParquetMetadata {
    /// The metadata describing every geometry column of `schema`.
    /// `None` if it has none.
    pub fn from_schema(schema: &Schema) -> PolarsResult<Option<Self>> {
        let mut columns = IndexMap::new();
        for (name, dtype) in schema.iter() {
            if is_geometry(dtype) {
                columns.insert(name.to_string(), GeoParquetColumn::from_dtype(dtype)?);
            }
        }
        let Some(primary_column) = columns.keys().next().cloned() else {
            return Ok(None);
        };
        Ok(Some(Self {
            version: VERSION.to_owned(),
            primary_column,
            columns,
        }))
    }

    /// Parses and validates the value of a file's `geo` key.
    pub fn parse(geo: &str) -> PolarsResult<Self> {
        let metadata: Self = serde_json::from_str(geo)
            .map_err(|e| polars_err!(ComputeError: "invalid GeoParquet metadata: {e}"))?;
        polars_ensure!(
            metadata.version.starts_with("1."),
            ComputeError: "GeoParquet {} is not supported yet, only 1.x is", metadata.version
        );
        polars_ensure!(
            metadata.columns.contains_key(&metadata.primary_column),
            ComputeError: "the GeoParquet primary column `{}` is not one of its geometry columns",
            metadata.primary_column
        );
        for (name, column) in &metadata.columns {
            polars_ensure!(
                column.epoch.is_none(),
                ComputeError: "the GeoParquet column `{name}` has a coordinate epoch, \
                but dynamic CRSs are not supported"
            );
        }
        Ok(metadata)
    }

    /// The value for a file's `geo` key.
    pub fn to_json(&self) -> String {
        // Every key is a string and every number finite, so this cannot fail.
        serde_json::to_string(self).unwrap()
    }
}

impl GeoParquetColumn {
    /// How a column of `dtype` is written.
    pub fn from_dtype(dtype: &DataType) -> PolarsResult<Self> {
        let (encoding, geometry_types, metadata) = match dtype {
            DataType::Extension(typ, _) if typ.name() == Encoding::Wkb.name() => (
                ColumnEncoding::Wkb,
                // Knowing them would mean reading every value.
                vec![],
                describe_encoded(dtype, Encoding::Wkb)?,
            ),
            DataType::Extension(typ, _) if typ.name() == Encoding::Wkt.name() => polars_bail!(
                SchemaMismatch: "GeoParquet cannot store WKT; \
                convert it with `from_wkt` or `to_wkb` first"
            ),
            _ => {
                let column = describe(dtype)?;
                polars_ensure!(
                    !column.dim.has_m(),
                    SchemaMismatch: "GeoParquet 1.1 cannot store M values, got: {dtype}"
                );
                let z = if column.dim.has_z() { " Z" } else { "" };
                (
                    ColumnEncoding::Native(column.kind),
                    vec![format!("{}{z}", column.kind.type_name())],
                    column.metadata.clone(),
                )
            }
        };

        let crs = match metadata.projjson()? {
            // Leaving the key out would claim the data is in OGC:CRS84.
            None => Some(None),
            Some(projjson) if is_crs84(&projjson) => None,
            Some(projjson) => Some(Some(projjson)),
        };
        let edges = match metadata.edges()? {
            None => None,
            Some(Edges::Spherical) => Some(ParquetEdges::Spherical),
            Some(edges) => polars_bail!(
                ComputeError: "GeoParquet 1.1 only has planar and spherical edges, got: {}",
                serde_json::to_value(edges).unwrap()
            ),
        };
        Ok(Self {
            encoding,
            geometry_types,
            crs,
            edges,
            epoch: None,
        })
    }

    /// The dtype of this column, given the `storage` Polars reads it from a file as.
    pub fn dtype(&self, storage: &DataType) -> PolarsResult<DataType> {
        let mut metadata = ExtensionMetadata::default();
        match &self.crs {
            None => metadata = metadata.with_crs(DEFAULT_CRS),
            Some(None) => {}
            Some(Some(projjson)) => {
                metadata = metadata.with_crs(&serde_json::to_string(projjson).unwrap())
            }
        }
        if self.edges == Some(ParquetEdges::Spherical) {
            metadata = metadata.with_edges(Edges::Spherical);
        }
        let metadata = Arc::new(metadata);

        match self.encoding {
            ColumnEncoding::Native(kind) => {
                // Not from `geometry_types`, which is allowed to be empty.
                let Some(dim) = coord::dimension_of_storage(storage, kind.nesting()) else {
                    polars_bail!(
                        SchemaMismatch: "a GeoParquet `{}` column must be stored as separated \
                        x/y[/z] coordinates, got: {storage}", kind.display()
                    );
                };
                Ok(Geo::new(kind, dim, metadata).dtype())
            }
            ColumnEncoding::Wkb => {
                polars_ensure!(
                    *storage == DataType::Binary,
                    SchemaMismatch: "a GeoParquet WKB column must be stored as binary, got: {storage}"
                );
                Ok(Encoded::new(Encoding::Wkb, metadata).dtype())
            }
        }
    }
}

impl ColumnEncoding {
    /// How the encoding is spelled in the metadata.
    fn name(self) -> &'static str {
        match self {
            ColumnEncoding::Wkb => "WKB",
            // This happens to coincide with our display names.
            // If we ever change the display names, we'll have to change this.
            ColumnEncoding::Native(kind) => kind.display(),
        }
    }
}

impl Serialize for ColumnEncoding {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ColumnEncoding {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        if name == ColumnEncoding::Wkb.name() {
            return Ok(ColumnEncoding::Wkb);
        }
        Kind::from_display(&name)
            .map(ColumnEncoding::Native)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown encoding {name:?}")))
    }
}

/// Tells an explicit `null` apart from a missing key
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

/// is does GeoParquet have to describe this?
fn is_geometry(dtype: &DataType) -> bool {
    let DataType::Extension(typ, _) = dtype else {
        return false;
    };
    let name = typ.name();
    Kind::from_name(&name).is_some() || Encoding::ALL.iter().any(|e| e.name() == name)
}

/// Whether a PROJJSON CRS is OGC:CRS84, by its `id` as the spec suggests.
fn is_crs84(projjson: &Map<String, Value>) -> bool {
    projjson
        .get("id")
        .is_some_and(|id| id["authority"] == "OGC" && id["code"] == "CRS84")
}

#[cfg(test)]
mod tests {
    //! Not reachable from Python until the reader and writer exist.

    use serde_json::json;

    use super::*;
    use crate::geoarrow::GeoDimension;

    fn metadata(json: Option<&str>) -> Arc<ExtensionMetadata> {
        Arc::new(ExtensionMetadata::parse(json).unwrap())
    }

    fn geo(kind: Kind, dim: GeoDimension, json: Option<&str>) -> DataType {
        Geo::new(kind, dim, metadata(json)).dtype()
    }

    fn wkb(json: Option<&str>) -> DataType {
        Encoded::new(Encoding::Wkb, metadata(json)).dtype()
    }

    fn written(dtype: &DataType) -> Value {
        serde_json::to_value(GeoParquetColumn::from_dtype(dtype).unwrap()).unwrap()
    }

    fn write_error(dtype: &DataType) -> String {
        GeoParquetColumn::from_dtype(dtype).unwrap_err().to_string()
    }

    /// A file's metadata with `column` as its only column, `g`.
    fn file(column: &str) -> String {
        format!(r#"{{"version":"1.1.0","primary_column":"g","columns":{{"g":{column}}}}}"#)
    }

    fn read(column: &str) -> GeoParquetColumn {
        let mut metadata = GeoParquetMetadata::parse(&file(column)).unwrap();
        metadata.columns.shift_remove("g").unwrap()
    }

    fn storage(dtype: &DataType) -> &DataType {
        let DataType::Extension(_, storage) = dtype else {
            panic!("not an extension type: {dtype}");
        };
        storage
    }

    #[test]
    fn writes_a_native_column() {
        let v = written(&geo(
            Kind::Point,
            GeoDimension::XY,
            Some(r#"{"crs":"EPSG:4326"}"#),
        ));
        assert_eq!(v["encoding"], "point");
        assert_eq!(v["geometry_types"], json!(["Point"]));
        assert_eq!(v["crs"]["id"], json!({"authority": "EPSG", "code": 4326}));
        assert!(v.get("edges").is_none(), "{v}");
        assert!(v.get("epoch").is_none(), "{v}");
    }

    #[test]
    fn a_z_coordinate_is_in_the_geometry_type() {
        let v = written(&geo(Kind::MultiLineString, GeoDimension::XYZ, None));
        assert_eq!(v["encoding"], "multilinestring");
        assert_eq!(v["geometry_types"], json!(["MultiLineString Z"]));
    }

    #[test]
    fn refuses_m_values() {
        for dim in [GeoDimension::XYM, GeoDimension::XYZM] {
            let err = write_error(&geo(Kind::Polygon, dim, None));
            assert!(err.contains("M values"), "{err}");
        }
    }

    #[test]
    fn no_crs_is_written_as_null() {
        for json in [None, Some(r#"{"crs":null}"#)] {
            let v = written(&geo(Kind::Point, GeoDimension::XY, json));
            assert_eq!(v.get("crs"), Some(&Value::Null), "{v}");
        }
    }

    #[test]
    fn crs84_is_left_out() {
        let v = written(&geo(
            Kind::Point,
            GeoDimension::XY,
            Some(r#"{"crs":"OGC:CRS84"}"#),
        ));
        assert!(v.get("crs").is_none(), "{v}");
    }

    #[test]
    fn refuses_what_geoparquet_cannot_hold() {
        let srid = geo(
            Kind::Point,
            GeoDimension::XY,
            Some(r#"{"crs":"4326","crs_type":"srid"}"#),
        );
        assert!(write_error(&srid).contains("SRID"));
        let karney = geo(Kind::Point, GeoDimension::XY, Some(r#"{"edges":"karney"}"#));
        assert!(write_error(&karney).contains("karney"));
        let wkt = Encoded::new(Encoding::Wkt, metadata(None)).dtype();
        assert!(write_error(&wkt).contains("WKT"));
    }

    #[test]
    fn writes_spherical_edges() {
        let v = written(&geo(
            Kind::LineString,
            GeoDimension::XY,
            Some(r#"{"edges":"spherical"}"#),
        ));
        assert_eq!(v["edges"], "spherical");
    }

    #[test]
    fn writes_wkb_without_geometry_types() {
        let v = written(&wkb(None));
        assert_eq!(v["encoding"], "WKB");
        assert_eq!(v["geometry_types"], json!([]));
    }

    #[test]
    fn describes_every_geometry_column_of_a_schema() {
        let schema = Schema::from_iter([
            Field::new("id".into(), DataType::Int64),
            Field::new("a".into(), geo(Kind::Point, GeoDimension::XY, None)),
            Field::new("name".into(), DataType::String),
            Field::new("b".into(), wkb(None)),
        ]);
        let metadata = GeoParquetMetadata::from_schema(&schema).unwrap().unwrap();
        assert_eq!(metadata.version, "1.1.0");
        assert_eq!(metadata.primary_column, "a");
        assert_eq!(metadata.columns.keys().collect::<Vec<_>>(), ["a", "b"]);

        let schema = Schema::from_iter([Field::new("id".into(), DataType::Int64)]);
        assert!(GeoParquetMetadata::from_schema(&schema).unwrap().is_none());
    }

    #[test]
    fn a_missing_crs_is_crs84() {
        let column = read(r#"{"encoding":"point","geometry_types":["Point"]}"#);
        let crs84 = geo(
            Kind::Point,
            GeoDimension::XY,
            Some(r#"{"crs":"OGC:CRS84"}"#),
        );
        assert_eq!(column.dtype(storage(&crs84)).unwrap(), crs84);
    }

    #[test]
    fn reads_a_null_or_projjson_crs() {
        let column = read(r#"{"encoding":"point","geometry_types":[],"crs":null}"#);
        let none = geo(Kind::Point, GeoDimension::XY, None);
        assert_eq!(column.dtype(storage(&none)).unwrap(), none);

        let column =
            read(r#"{"encoding":"point","geometry_types":[],"crs":{"type":"GeographicCRS"}}"#);
        let projjson = geo(
            Kind::Point,
            GeoDimension::XY,
            Some(r#"{"crs":{"type":"GeographicCRS"}}"#),
        );
        assert_eq!(column.dtype(storage(&projjson)).unwrap(), projjson);
    }

    #[test]
    fn the_dimension_comes_from_the_storage() {
        let column = read(r#"{"encoding":"polygon","geometry_types":[],"crs":null}"#);
        let xyz = geo(Kind::Polygon, GeoDimension::XYZ, None);
        assert_eq!(column.dtype(storage(&xyz)).unwrap(), xyz);
    }

    #[test]
    fn ignores_what_it_does_not_use() {
        let column = r#"{
            "encoding": "WKB", "geometry_types": ["Polygon"], "orientation": "counterclockwise",
            "bbox": [0, 0, 1, 1], "vendor": {"a": 1},
            "covering": {"bbox": {"xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
                                  "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"]}}
        }"#;
        let geo = format!(
            r#"{{"version":"1.1.0","primary_column":"g","columns":{{"g":{column}}},"creator":"x"}}"#
        );
        let metadata = GeoParquetMetadata::parse(&geo).unwrap();
        assert_eq!(metadata.columns["g"].encoding, ColumnEncoding::Wkb);
    }

    #[test]
    fn refuses_metadata_it_cannot_honour() {
        let point = r#"{"encoding":"point","geometry_types":[]}"#;
        for (geo, reason) in [
            (
                file(r#"{"encoding":"point","geometry_types":[],"epoch":2021.47}"#),
                "epoch",
            ),
            (
                file(r#"{"encoding":"geometry","geometry_types":[]}"#),
                "encoding",
            ),
            (file(point).replace("1.1.0", "2.0.0"), "2.0.0"),
            (
                file(point).replace(r#""primary_column":"g""#, r#""primary_column":"h""#),
                "`h`",
            ),
        ] {
            let err = GeoParquetMetadata::parse(&geo).unwrap_err().to_string();
            assert!(err.contains(reason), "{err}");
        }
    }

    #[test]
    fn refuses_storage_that_does_not_match_the_encoding() {
        let point = read(r#"{"encoding":"point","geometry_types":[]}"#);
        assert!(point.dtype(&GeoDimension::XY.storage(1)).is_err());
        let wkb = read(r#"{"encoding":"WKB","geometry_types":[]}"#);
        assert!(wkb.dtype(&DataType::String).is_err());
    }

    #[test]
    fn round_trips_through_the_metadata() {
        let crs84 = Some(r#"{"crs":"OGC:CRS84"}"#);
        for dtype in [
            geo(Kind::Point, GeoDimension::XY, crs84),
            geo(Kind::Point, GeoDimension::XYZ, None),
            geo(
                Kind::LineString,
                GeoDimension::XY,
                Some(r#"{"crs":"OGC:CRS84","edges":"spherical"}"#),
            ),
            geo(
                Kind::MultiPolygon,
                GeoDimension::XY,
                Some(r#"{"crs":{"type":"GeographicCRS"}}"#),
            ),
            wkb(crs84),
            wkb(None),
        ] {
            let schema = Schema::from_iter([Field::new("g".into(), dtype.clone())]);
            let json = GeoParquetMetadata::from_schema(&schema)
                .unwrap()
                .unwrap()
                .to_json();
            let back = GeoParquetMetadata::parse(&json).unwrap().columns["g"]
                .dtype(storage(&dtype))
                .unwrap();
            assert_eq!(back, dtype, "{json}");
        }
    }
}
