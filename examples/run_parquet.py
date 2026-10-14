import json
import tempfile
from pathlib import Path

import polars as pl

import geopolars as gpl
from geopolars import MultiPolygonXY, geo

df = pl.DataFrame(
    {
        "city": ["Amsterdam", "Delft", "Utrecht"],
        "lon": [4.9041, 4.3571, 5.1214],
        "lat": [52.3676, 52.0116, 52.0907],
    }
).select("city", geometry=geo.point("lon", "lat", crs="OGC:CRS84"))
print(f"We start with points: {df}")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    print("\n1. Polars' own writer and reader need nothing extra")
    path = tmp / "plain.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)
    print("  dtype after the roundtrip:", repr(back.schema["geometry"]))
    print("  identical:", back.equals(df))
    print(
        "  The The dtype is kept in the Arrow schema, but there is no `geo` key,\n\
  so other GeoParquet readers (geopandas, GDAL, ...) won't see geometries."
    )
    print("  footer keys:", list(pl.read_parquet_metadata(path)))

    print(
        "\n2. Writing GeoParquet:\n\
  to enable interop with other tools, we have to add the `geo` metadata ourselves"
    )
    path = tmp / "geo.parquet"
    # Optional: a bbox covering per geometry column, that readers can filter on.
    covered = df.pipe(gpl.with_covering)
    covered.write_parquet(
        path,
        metadata={
            **gpl.geoparquet_metadata(covered, covering=True),
            "owner": "the example",
        },
    )
    geo_key = json.loads(pl.read_parquet_metadata(path)["geo"])
    print("  `geo` key of the column:", geo_key["columns"]["geometry"])
    # Streaming works the same way, with `sink_parquet`.
    lf = df.lazy().pipe(gpl.with_covering)
    # The magic line!
    lf.sink_parquet(
        tmp / "sunk.parquet", metadata=gpl.geoparquet_metadata(lf, covering=True)
    )
    # Polars still reads it back as-is, the covering as a box.
    print("  read back:", pl.read_parquet(path).schema)

    print("\n3. Reading GeoParquet from another writer: convert after reading")
    # The example file of the GeoParquet spec has WKB geometries and a covering,
    # and only the `geo` key to say so.
    path = Path(__file__).parents[1] / "geopolars/tests/data/geoparquet/example.parquet"
    plain = pl.scan_parquet(path)
    print("  plain Polars:   ", plain.collect_schema())
    # `from_geoparquet` reads the `geo` key from the file, so it needs the path too.
    # The magic line!
    lf = plain.pipe(gpl.from_geoparquet, path)
    print("  from_geoparquet:", lf.collect_schema())
    # It stays lazy: decode the WKB and filter like any other column.
    countries = lf.select(
        "name", "continent", geo.from_wkb("geometry", MultiPolygonXY)
    ).filter(pl.col("continent") == "North America")
    print(countries.collect())
