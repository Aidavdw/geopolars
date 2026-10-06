"""Reprojecting geometries from one coordinate reference system to another.

The CRS lives in the column's extension metadata, not in the data. `to_crs`
reads the source CRS from there, sends every x/y through PROJ, and writes the
target CRS back onto the output dtype.
"""

import polars as pl

import geopolars as gpl
from geopolars import geo

# Longitude/latitude on the WGS 84 ellipsoid: what GPS gives you.
WGS84 = "EPSG:4326"


def crs_of(df: pl.DataFrame, name: str) -> str | None:
    return df.schema[name].ext_metadata()


df = pl.DataFrame(
    {
        "city": ["Amsterdam", "Delft", "Utrecht"],
        "lon": [4.9041, 4.3571, 5.1214],
        "lat": [52.3676, 52.0116, 52.0907],
        "elevation": [-2.0, 1.0, 5.0],
        "distance": [0.0, 58_000.0, 94_000.0],
    }
)

print("=== Points ===\n")

# Every constructor takes `crs=`. It only labels the column: the coordinates
# are taken to be in that CRS already, and nothing is moved.
points = df.select(
    "city",
    xy=geo.point("lon", "lat", crs=WGS84),
    xyzm=geo.point("lon", "lat", z="elevation", m="distance", crs=WGS84),
)
print(f"Points in longitude/latitude: {points}")
print("declared CRS:", crs_of(points, "xy"), "\n")

# EPSG:28992 is Amersfoort / RD New, the Dutch national grid, in metres.
# `to_crs` is also in the `.geo` namespace, like every other operation.
rd = points.select(
    "city",
    gpl.col("xy").geo.to_crs("EPSG:28992"),
    gpl.col("xyzm").geo.to_crs("EPSG:28992"),
)
print(f"Reprojected to the Dutch national grid (EPSG:28992), in metres: {rd}")
print("declared CRS:", crs_of(rd, "xy"))
# The CRS is part of the dtype: still a PointXY, but no longer the same dtype.
print("still a PointXY:", isinstance(rd.schema["xy"], gpl.PointXY))
print("the same dtype as before:", rd.schema["xy"] == points.schema["xy"])
print("the dtype it is:", rd.schema["xy"] == gpl.PointXY(crs="EPSG:28992"), "\n")

print("Only x and y are reprojected. z and m are carried through untouched.")
for field in ("z", "m"):
    before = points["xyzm"].ext.storage().struct.field(field).to_list()
    after = rd["xyzm"].ext.storage().struct.field(field).to_list()
    print(f"  {field}: {before} -> {after}")
print()

print("Any CRS PROJ understands will do. Web Mercator (EPSG:3857), for example:")
print(points.select("city", geo.to_crs("xy", "EPSG:3857")), "\n")

print("Going back to EPSG:4326 lands where we started, up to rounding.")
back = rd.select("city", geo.to_crs("xy", "EPSG:4326"))
for name, start, end in zip(
    df["city"], points["xy"].ext.storage().to_list(), back["xy"].ext.storage().to_list()
):
    print(f"  {name:>9}: dx={end['x'] - start['x']:.1e} dy={end['y'] - start['y']:.1e}")
print()

print("Reprojecting needs to know where you start from.")
print("(PROJ also reports the unknown code on stderr itself.)")
try:
    df.select(geo.to_crs(geo.point("lon", "lat"), "EPSG:28992"))
except Exception as e:
    detail = next(l for l in str(e).splitlines() if "CRS" in l)
    print(f"  no CRS -> {type(e).__name__}: {detail.strip()}")
try:
    points.select(geo.to_crs("xy", "EPSG:not-a-code"))
except Exception as e:
    detail = next(l for l in str(e).splitlines() if "cannot reproject" in l)
    print(f"  unknown CRS -> {type(e).__name__}: {detail.strip()}")

print("\n=== Lines ===\n")

vertices = pl.DataFrame(
    {
        # Two cycling routes, sampled at a handful of points each.
        "route": ["north", "north", "north", "coast", "coast"],
        "lon": [4.9041, 4.8000, 4.7000, 3.6000, 3.5000],
        "lat": [52.3676, 52.4000, 52.5000, 51.5000, 51.4000],
    }
)

# The CRS can go on the vertices: building a linestring from them carries
# the metadata over, so the line knows its CRS without being told twice.
lines = vertices.group_by("route", maintain_order=True).agg(
    line=geo.linestring_from_vertices(geo.point("lon", "lat", crs=WGS84).implode())
)
print(f"Routes in longitude/latitude: {lines}")
print("declared CRS:", crs_of(lines, "line"), "\n")

print("Or on the line itself, when it is built from coordinate columns.")
direct = (
    vertices.group_by("route", maintain_order=True)
    .agg("lon", "lat")
    .select("route", line=geo.linestring_from_columns("lon", "lat", crs=WGS84))
)
print("  the same lines:", direct.equals(lines), "\n")

print("Declaring a different CRS than the vertices already have is refused:")
print("that would put every vertex somewhere else without moving it.")
try:
    lines_lf = (
        vertices.lazy()
        .group_by("route")
        .agg(
            geo.linestring_from_vertices(
                geo.point("lon", "lat", crs=WGS84).implode(), crs="EPSG:28992"
            )
        )
    )
    lines_lf.collect_schema()
except Exception as e:
    detail = next(l for l in str(e).splitlines() if "already declare" in l)
    print(f"  {type(e).__name__}: {detail.strip()}")
print()

rd_lines = lines.select("route", gpl.col("line").geo.to_crs("EPSG:28992"))
print(f"Reprojected to the Dutch national grid: {rd_lines}")
print("declared CRS:", crs_of(rd_lines, "line"))
print(
    "still a LineStringXY:", isinstance(rd_lines.schema["line"], gpl.LineStringXY), "\n"
)

print("Every vertex moves on its own; the routes keep their vertices.")
print(
    pl.DataFrame(
        {
            "route": lines["route"],
            "before": lines["line"].ext.storage().list.len(),
            "after": rd_lines["line"].ext.storage().list.len(),
        }
    ),
    "\n",
)

print("A line reprojects to the same coordinates as its vertices do one by one.")
one_by_one = (
    vertices.select(geo.to_crs(geo.point("lon", "lat", crs=WGS84), "EPSG:28992"))
    .to_series()
    .ext.storage()
)
print(
    "  identical:",
    rd_lines["line"].ext.storage().explode().equals(one_by_one, check_names=False),
    "\n",
)

print("Nulls stay null, and an empty line stays empty.")
# A dtype takes `crs=` too, for labelling storage you already have.
edge_cases = pl.DataFrame(
    {"line": [[{"x": 4.9, "y": 52.4}], [], None]},
    schema={"line": gpl.LineStringXY().ext_storage()},
).select(pl.col("line").ext.to(gpl.LineStringXY(crs=WGS84)))
print(edge_cases.select(geo.to_crs("line", "EPSG:28992")))
