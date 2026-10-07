"""Reading and writing geometries from and to files."""

from geopolars.io.parquet import (
    read_parquet,
    scan_parquet,
    sink_parquet,
    write_parquet,
)

__all__ = ["read_parquet", "scan_parquet", "sink_parquet", "write_parquet"]
