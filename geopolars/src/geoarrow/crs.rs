//! The coordinate reference system a geometry column declares.
//!
//! GeoArrow keeps the CRS in `ARROW:extension:metadata`, a JSON object along
//! the lines of `{"crs": ..., "crs_type": ..., "edges": ...}`. [`ExtensionMetadata`]
//! parses and gives back that raw string, so nothing has to downcast to
//! [`Geo`](super::Geo) to get at it.

use geoarrow_schema::{Crs, CrsType};
use polars::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value};

/// A column's `ARROW:extension:metadata`.
///
/// Lossless. Even data that cannot be parsed is kept,
/// so it can be put back on the output type.
/// This is a slight deviation from the GeoArrow schema,
/// but makes chaining with proprietary data producers more versatile.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtensionMetadata(Map<String, Value>);

impl ExtensionMetadata {
    /// Parses the raw metadata string.
    /// If no metadata is found, it produces an empty object.
    pub fn parse(metadata: Option<&str>) -> PolarsResult<Self> {
        // Some producers write an empty string rather than leaving it out.
        let metadata = match metadata.map(str::trim) {
            None | Some("") => return Ok(Self::default()),
            Some(metadata) => metadata,
        };
        match serde_json::from_str(metadata) {
            Ok(Value::Object(map)) => Ok(Self(map)),
            Ok(_) => polars_bail!(
                ComputeError: "GeoArrow extension metadata must be a JSON object, got: {metadata}"
            ),
            Err(e) => polars_bail!(
                ComputeError: "invalid GeoArrow extension metadata {metadata}: {e}"
            ),
        }
    }

    /// The metadata in a JSON-formatted string.
    /// If empty, this returns `None`.
    pub fn serialize(&self) -> Option<String> {
        (!self.0.is_empty()).then(|| serde_json::to_string(&self.0).unwrap())
    }

    /// Gets the CRS in a form PROJ accepts
    ///
    /// Errors if there is no CRS.
    pub fn crs(&self) -> PolarsResult<String> {
        let crs = Crs::deserialize(&self.0).map_err(|e| {
            let metadata = self.serialize().unwrap_or_default();
            polars_err!(ComputeError: "invalid CRS in GeoArrow metadata {metadata}: {e}")
        })?;

        match (crs.crs_value(), crs.crs_type()) {
            (None, _) => polars_bail!(
                ComputeError: "the column does not declare a CRS, so it cannot be reprojected"
            ),
            // Only the producer knows what an SRID refers to.
            (Some(srid), Some(CrsType::Srid)) => polars_bail!(
                ComputeError: "the column's CRS is the opaque SRID {srid}, which cannot be \
                reprojected from"
            ),
            // PROJJSON, WKT, `AUTH:CODE`, or a string of unstated kind: PROJ gets to try.
            (Some(Value::String(crs)), _) => Ok(crs.clone()),
            (Some(projjson @ Value::Object(_)), _) => Ok(projjson.to_string()),
            (Some(other), _) => polars_bail!(
                ComputeError: "a CRS must be a PROJJSON object or a string, got: {other}"
            ),
        }
    }

    /// Consuming setter for CRS.
    pub fn with_crs(mut self, crs: &str) -> Self {
        self.0.insert("crs".to_owned(), crs_value(crs));
        self.0.shift_remove("crs_type");
        self
    }
}

fn crs_value(crs: &str) -> Value {
    match serde_json::from_str(crs) {
        Ok(object @ Value::Object(_)) => object,
        _ => Value::String(crs.to_owned()),
    }
}

/// The metadata fields a constructor can be asked to set, as keyword arguments.
///
/// A fixed set: every one is a field GeoArrow defines, never an arbitrary key.
/// A new field is one more `Option` here, and one more keyword on the Python
/// side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetadataKwargs {
    /// The CRS the coordinates are in, in any form PROJ accepts.
    pub crs: Option<String>,
}

impl MetadataKwargs {
    /// The metadata of a geometry built from parts carrying `metadata`.
    ///
    /// Without fields to set, `metadata` comes back untouched, byte for byte.
    ///
    /// This labels and never reprojects. Parts that already declare a different
    /// CRS are an error rather than being relabelled into the wrong place.
    pub fn apply(&self, metadata: Option<String>) -> PolarsResult<Option<String>> {
        let Some(crs) = &self.crs else {
            return Ok(metadata);
        };

        let parsed = ExtensionMetadata::parse(metadata.as_deref())?;
        match parsed.0.get("crs") {
            None | Some(Value::Null) => Ok(parsed.with_crs(crs).serialize()),
            Some(declared) if *declared == crs_value(crs) => Ok(metadata),
            Some(declared) => polars_bail!(
                SchemaMismatch: "the parts already declare the CRS {declared}, not {}; \
                use `to_crs` to reproject them instead",
                crs_value(crs)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach: metadata other producers write.

    use super::*;

    fn crs_of(metadata: Option<&str>) -> PolarsResult<String> {
        ExtensionMetadata::parse(metadata)?.crs()
    }

    fn with_crs(metadata: Option<&str>, crs: &str) -> Option<String> {
        ExtensionMetadata::parse(metadata)
            .unwrap()
            .with_crs(crs)
            .serialize()
    }

    #[test]
    fn reads_every_crs_representation() {
        let projjson = r#"{"crs":{"type":"GeographicCRS","name":"WGS 84"},"crs_type":"projjson"}"#;
        assert_eq!(
            crs_of(Some(projjson)).unwrap(),
            r#"{"type":"GeographicCRS","name":"WGS 84"}"#
        );
        let code = r#"{"crs":"EPSG:4326","crs_type":"authority_code"}"#;
        assert_eq!(crs_of(Some(code)).unwrap(), "EPSG:4326");
        assert_eq!(crs_of(Some(r#"{"crs":"OGC:CRS84"}"#)).unwrap(), "OGC:CRS84");
    }

    #[test]
    fn refuses_what_proj_cannot_use() {
        for metadata in [
            None,
            Some(""),
            Some("{}"),
            Some(r#"{"edges":"spherical"}"#),
            Some(r#"{"crs":null}"#),
            Some(r#"{"crs":"4326","crs_type":"srid"}"#),
            Some(r#"{"crs":4326}"#),
            Some(r#"["crs"]"#),
            Some("not json"),
        ] {
            assert!(crs_of(metadata).is_err(), "{metadata:?}");
        }
    }

    #[test]
    fn keeps_every_other_key_in_place() {
        let metadata = r#"{"vendor":{"a":[1,2]},"crs":"EPSG:4326","crs_type":"authority_code","edges":"karney","zz":true}"#;
        assert_eq!(
            with_crs(Some(metadata), "EPSG:3857").unwrap(),
            r#"{"vendor":{"a":[1,2]},"crs":"EPSG:3857","edges":"karney","zz":true}"#
        );
        // Even keys and values GeoArrow does not define.
        let metadata = r#"{"edges":"geodesic-ish","x-extra":1}"#;
        assert_eq!(
            with_crs(Some(metadata), "EPSG:3857").unwrap(),
            r#"{"edges":"geodesic-ish","x-extra":1,"crs":"EPSG:3857"}"#
        );
    }

    #[test]
    fn writes_a_json_crs_unescaped() {
        let out = with_crs(None, r#"{"type": "GeographicCRS"}"#).unwrap();
        assert_eq!(out, r#"{"crs":{"type":"GeographicCRS"}}"#);
        assert_eq!(crs_of(Some(&out)).unwrap(), r#"{"type":"GeographicCRS"}"#);
    }

    fn apply(metadata: Option<&str>, crs: Option<&str>) -> PolarsResult<Option<String>> {
        let kwargs = MetadataKwargs {
            crs: crs.map(str::to_owned),
        };
        kwargs.apply(metadata.map(str::to_owned))
    }

    #[test]
    fn nothing_to_set_keeps_metadata_verbatim() {
        let metadata = r#"{ "crs" : "EPSG:4326", "x-extra": 1 }"#;
        assert_eq!(apply(Some(metadata), None).unwrap().unwrap(), metadata);
        assert_eq!(apply(None, None).unwrap(), None);
    }

    #[test]
    fn sets_a_crs_next_to_the_other_keys() {
        assert_eq!(
            apply(None, Some("EPSG:4326")).unwrap().unwrap(),
            r#"{"crs":"EPSG:4326"}"#
        );
        assert_eq!(
            apply(Some(r#"{"edges":"spherical"}"#), Some("EPSG:4326"))
                .unwrap()
                .unwrap(),
            r#"{"edges":"spherical","crs":"EPSG:4326"}"#
        );
    }

    #[test]
    fn the_same_crs_again_is_not_a_conflict() {
        // Untouched: `crs_type` survives, where `with_crs` would have dropped it.
        let code = r#"{"crs":"EPSG:4326","crs_type":"authority_code"}"#;
        assert_eq!(apply(Some(code), Some("EPSG:4326")).unwrap().unwrap(), code);
        let projjson = r#"{"crs":{"type":"GeographicCRS"}}"#;
        assert_eq!(
            apply(Some(projjson), Some(r#"{ "type": "GeographicCRS" }"#))
                .unwrap()
                .unwrap(),
            projjson
        );
    }

    #[test]
    fn refuses_to_relabel_a_declared_crs() {
        let err = apply(Some(r#"{"crs":"EPSG:4326"}"#), Some("EPSG:28992")).unwrap_err();
        assert!(err.to_string().contains("to_crs"), "{err}");
    }

    #[test]
    fn no_keys_is_no_metadata() {
        assert_eq!(
            ExtensionMetadata::parse(Some("{}")).unwrap().serialize(),
            None
        );
    }
}
