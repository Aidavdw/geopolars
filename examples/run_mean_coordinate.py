"""Example showing the calculation of the mean coordinate of a geometry.
Shows the following steps:
- changing columns into points
- calculating the mean coordinate of the points.
Ideally this should just be an abstraction on the plugin layer.
It uses only polars under the hood, no custom nodes and no magic!
"""

import polars as pl

import geopolars as gpl
from geopolars import PointXYM, geo

# Four soil samples: where they were taken, and what was measured there.
# `value` is the measure a point carries as `m`.
lf = pl.LazyFrame(
    {
        "x": [0.0, 4.0, 4.0, 0.0],
        "y": [0.0, 0.0, 3.0, 3.0],
        "value": [10.0, 20.0, 30.0, 40.0],
    }
)

print("Three plain float columns, one row per sample:")
print(lf.collect())

# Passing `m` is what makes these XYM points: x and y say where,
# `value` says  what was measured there.
points = lf.select(point=geo.point("x", "y", m="value"))

print("\nOne point per row, with the measurement carried along:")
print(points.collect())
print("dtype:", points.collect_schema()["point"])
assert points.collect_schema()["point"] == PointXYM()

collection = points.select(pl.col("point").implode()).select(
    samples=geo.multipoint("point")
)

print("\nGathered into one geometry, the four samples are one row:")
print(collection.collect())

center = collection.select(center=geo.mean_coordinate("samples"))

print(
    "The plan behind it: Ideally, all abstractions fall away, \
        and it is as cheap as manually building everything \
        after Polars is allowed to optimise it."
)
print(center.explain())
print("\nThe physical plan, node by node:")
center.show_graph(plan_stage="physical", engine="streaming", optimized=True)

print("\nThe mean coordinate of the four samples:")
print(center.collect())

print("\nThe same numbers, averaged column by column:")
print(
    lf.select(pl.col("x").mean(), pl.col("y").mean(), pl.col("value").mean()).collect()
)

print("\nThe whole thing is one lazy query, so it streams:")
print(center.collect(engine="streaming"))

print("\nThe namespace spelling is the same expression:")
print(collection.select(center=gpl.col("samples").geo.mean_coordinate()).collect())
