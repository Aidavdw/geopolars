# GeoParquet 1.1 test data

Copied from [opengeospatial/geoparquet](https://github.com/opengeospatial/geoparquet)
at tag `v1.1.0+p1` (`540f6bf547587284e632c47530bc08d9e43bb045`),
licensed under Apache-2.0.

- `example.parquet`: from `examples/`.
  WKB, a PROJJSON CRS, and a `bbox` covering column.
- `data-<geometry>-encoding_{native,wkb}.parquet`: from `test_data/`.
  The same geometries in both encodings.
  `data-<geometry>-wkt.csv` holds them as WKT, which makes it the reference for both files.
