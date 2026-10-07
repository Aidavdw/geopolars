"""Reading and writing GeoParquet files.

A GeoParquet file is a Parquet file with a `geo` key in its metadata.
It says which columns hold geometries, how, and in which CRS.
Reading and writing the file is done with polars.
This module only gives the geometry their dtypes on reading,
and writes the `geo` key on writing.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, Literal

import polars as pl

# Private, but the same expansion `pl.scan_parquet` does, and there is no public one.
from polars.io._expand_paths import _expand_paths

from geopolars import geopolars as _rust

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polars._typing import (
        ParquetMetadata,
        ParquetMetadataContext,
        StorageOptionsDict,
    )
    from polars.io.cloud import CredentialProviderFunction


def scan_parquet(
    source: str | Path | Sequence[str | Path],
    *,
    check_every_file: bool = True,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    **kwargs: Any,
) -> pl.LazyFrame:
    """Lazily read GeoParquet, like `pl.scan_parquet`.

    The columns that have been read get geopolars dtypes, with the parquet's metadata.
    Native encodings become e.g. `PolygonXY`,
    and WKB becomes `Wkb` (which you can decode to native yourself with `geo.from_wkb`).
    A file without GeoParquet metadata is read as plain Parquet.

    The metadata of every file in `source` is read and has to agree,
    which costs reading each file's footer.
    With `check_every_file=False`, only the first file's is read and the rest are trusted.
    If another file has different metadata, the failure might be silent.

    Other keyword arguments are passed on to `pl.scan_parquet`.

    ```python
    gpl.scan_parquet("countries/*.parquet").select(
        geo.from_wkb("geometry", MultiPolygonXY)
    )
    ```
    """
    lf = pl.scan_parquet(
        source,
        storage_options=storage_options,
        credential_provider=credential_provider,
        **kwargs,
    )
    paths = (
        _expand_paths(
            source,
            glob=kwargs.get("glob", True),
            hidden_file_prefix=kwargs.get("hidden_file_prefix"),
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        .collect()
        .to_series()
        .to_list()
    )
    if not check_every_file:
        paths = paths[:1]
    geos = [
        pl.read_parquet_metadata(
            path,
            storage_options=storage_options,
            credential_provider=credential_provider,
        ).get("geo")
        for path in paths
    ]
    if all(geo is None for geo in geos):
        return lf

    # Polars reads with the first file's schema too.
    schema = lf.collect_schema()
    first: dict[str, pl.DataType] | None = None
    for path, geo in zip(paths, geos, strict=True):
        if geo is None:
            msg = f"{path!r} has no GeoParquet metadata, but {paths[0]!r} does"
            raise pl.exceptions.SchemaError(msg)
        dtypes = _geometry_dtypes(geo, schema)
        if first is None:
            first = dtypes
        elif dtypes != first:
            msg = (
                f"{path!r} describes its geometry columns as {dtypes}, "
                f"but {paths[0]!r} as {first}; "
                "pass `check_every_file=False` to read all of them as the first"
            )
            raise pl.exceptions.SchemaError(msg)

    assert first is not None
    return lf.with_columns(
        pl.col(name).ext.storage().ext.to(dtype)
        for name, dtype in first.items()
        # A file Polars wrote can already have it.
        if schema[name] != dtype
    )


def read_parquet(
    source: str | Path | Sequence[str | Path], **kwargs: Any
) -> pl.DataFrame:
    """Read GeoParquet into memory: `scan_parquet(source, **kwargs).collect()`."""
    return scan_parquet(source, **kwargs).collect()


def _geometry_dtypes(geo: str, schema: pl.Schema) -> dict[str, pl.DataType]:
    """The dtype of every geometry column the `geo` metadata describes,
    for a file Polars reads with `schema`.
    Converted in Rust, like all metadata.
    """
    frame = _rust.geoparquet_dtypes(geo, pl.DataFrame(schema=schema))
    return dict(frame.schema)


def write_parquet(
    df: pl.DataFrame,
    file: str | Path | IO[bytes],
    *,
    metadata: ParquetMetadata | None = None,
    **kwargs: Any,
) -> None:
    """Write GeoParquet, like `DataFrame.write_parquet`.

    Every geometry column is described in the file's `geo` metadata,
    next to any `metadata` of your own.
    The geometries are written in their own encoding: native, or WKB for a `Wkb` column.
    The WKB we write is modern, so some older readers may not accept it.
    A frame without geometry columns is written as plain Parquet.

    GeoParquet 1.1 does not support M values and no WKT.
    Columns with such data are refused.
    CRS that cannot be written with PROJJSON are also refused.

    You can pass parquet metadata directly as an argument.
    Other keyword arguments are passed on to `DataFrame.write_parquet`.
    """
    df.write_parquet(file, metadata=_with_geo_metadata(df.schema, metadata), **kwargs)


def sink_parquet(
    lf: pl.LazyFrame,
    path: str | Path | IO[bytes],
    *,
    metadata: ParquetMetadata | None = None,
    **kwargs: Any,
) -> pl.LazyFrame | None:
    """Stream GeoParquet to `path`, like `LazyFrame.sink_parquet`.

    See `write_parquet`.
    The metadata follows from the schema alone,
    so a column GeoParquet cannot hold is refused before anything is written.

    Other keyword arguments (such as `lazy`) are passed on to `LazyFrame.sink_parquet`.
    """
    return lf.sink_parquet(
        path, metadata=_with_geo_metadata(lf.collect_schema(), metadata), **kwargs
    )


def _with_geo_metadata(
    schema: pl.Schema, metadata: ParquetMetadata | None
) -> ParquetMetadata | None:
    geo = _rust.geoparquet_metadata(pl.DataFrame(schema=schema))
    if geo is None:
        return metadata
    if metadata is None:
        return {"geo": geo}
    if callable(metadata):
        user_fn = metadata

        def with_geo(ctx: ParquetMetadataContext) -> dict[str, str]:
            return _add_geo(user_fn(ctx), geo)

        return with_geo
    return _add_geo(metadata, geo)


def _add_geo(metadata: dict[str, str], geo: str) -> dict[str, str]:
    if "geo" in metadata:
        msg = "the `geo` metadata key is written by geopolars, and cannot be set"
        raise ValueError(msg)
    return {**metadata, "geo": geo}
