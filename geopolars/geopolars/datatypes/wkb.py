"""The `geoarrow.wkb` extension type.
Mirrors `Wkb` in `src/geoarrow/wkb.rs`.

Geometries encoded as [Well-Known Binary](https://libgeos.org/specifications/wkb/),
one value per row, over `Binary` storage.
This is for exchange only (e.g. GeoParquet): no operation works on it,
other than converting it to and from the native geometries.
That is why it is not a `GeoArrowType`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from geopolars import geopolars as _rust

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class Wkb(pl.datatypes.BaseExtension):
    """Geometries encoded as WKB: `geoarrow.wkb` over `Binary`.
    WKB is supported for exchange, but all operations require it to be in arrow format.
    You'll likely want to convert this using `from_wkb`.
    """

    _extension_name: ClassVar[str] = "geoarrow.wkb"

    def __init__(self, *, crs: str | None = None) -> None:
        """`crs` labels the coordinates, see `GeoArrowType.__init__`."""
        super().__init__(
            name=self._extension_name,
            storage=pl.Binary,
            metadata=_rust.extension_metadata(crs=crs),
        )

    def _declares_crs(self) -> bool:
        """Whether the metadata names a CRS. Read in Rust, like all metadata."""
        return _rust.declares_crs(self.ext_metadata())

    def __repr__(self) -> str:
        return type(self).__name__

    def _string_repr(self) -> str:
        return "wkb"

    @classmethod
    def ext_from_params(
        cls, name: str, storage: PolarsDataType, metadata: str | None
    ) -> Any:
        """Rebuild the type from what was crossed over the boundary."""
        if storage != pl.Binary:
            msg = f"unsupported {name!r} storage: {storage!r}; expected binary"
            raise ValueError(msg)

        # Bypasses __init__ on purpose, so `metadata` is carried through
        # verbatim rather than reset to None.
        slf = cls.__new__(cls)
        slf._name = name
        slf._storage = storage
        slf._metadata = metadata
        return slf
