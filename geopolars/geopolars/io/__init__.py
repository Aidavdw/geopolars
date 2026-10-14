"""Reading and writing geometries from and to files."""

from geopolars.io.parquet import from_geoparquet, geoparquet_metadata, with_covering

__all__ = ["from_geoparquet", "geoparquet_metadata", "with_covering"]
