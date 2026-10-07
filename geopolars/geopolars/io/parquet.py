"""Reading GeoParquet files.

A GeoParquet file is a Parquet file with a `geo` key in its metadata.
It says which columns hold geometries, how, and in which CRS.
Reading the file is done with polars.
This module only gives the geometry their dtypes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import polars as pl

# Private, but the same expansion `pl.scan_parquet` does, and there is no public one.
from polars.io._expand_paths import _expand_paths

from geopolars import geopolars as _rust

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polars._typing import StorageOptionsDict
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
