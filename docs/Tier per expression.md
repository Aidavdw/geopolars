# Tier per operation

Which tier each operation uses, per geometry type.
`A` is a native kernel, `B` forwarded to an external library, `C` Polars expressions,
and `-` means the operation refuses that geometry.
If a constant value is returned for an operation, the value is also provided
(e.g. when passing a flag to get 0.0 for area of points, instead of erroring).
All dimensions of a geometry share a column.
Geometries usually carry a CRS, so the tier given is for a geometry with a CRS;
the comment says when one without a CRS takes a different tier.

| Operation | Point | | LineString | | Polygon | | MultiPoint | | MultiLineString | | MultiPolygon | |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| `area` | - | ANP | - | ANP | A/B | fwd `*_planar` `*_geodesic` | - | ANP | - | ANP | A/B | sum of parts |
| `area_planar` | - | ANP | - | ANP | A | shoelace; CRS ignored | - | ANP | - | ANP | A | sum of parts |
| `area_geodesic` | - | ANP | - | ANP | B | PROJ + GGL; needs a CRS | - | ANP | - | ANP | B | sum of parts |
| `length` | - | ANL | A/B | Reuses bits of `distance` [^2] | - | ANL | - | ANL | A/B | sum of parts | - | ANL |
| `centroid` | C | returns the point | A | length-weighted; keeps z/m | A | shoelace; z/m area-weighted | C | fwd `mean_coordinate` | A | length-weighted parts | A | area-weighted parts |
| `exterior` | - | | - | | A | ring 0 | - | | - | | A | LPP |
| `interior` | - | | - | | A | rings after the first | - | | - | | A | LPP |
| `boundary` | - | | C | | C | every ring | - | | C | LPP | C | LPP |
| `convex_hull` | - | | B | straight line marked missing | B | GEOS; z kept, m dropped | B | single polygon around all | B | single polygon around all | B | single polygon around all |
| `simplify` | C | returns the point | B | GEOS; z kept, m dropped | B | split by `buffer(0)` is a per-row error | C | returns the multipoint | B | | B | |
| `is_ring` | - | | A | ≥ 4 vertices, ends ≈  x/y/z | - | | - | | A | LPP | - | |
| `self_intersects` | - | | B | rsgeo sweep; x/y only | B | per ring; holes fwd `rings_overlap` | - | | B | LPP | B | LPP |
| `rings_overlap` | - | | B | hacked as polygon's RGR | - | | - | | - | | - | |
| `is_standard_orientation` | - | | A | CCW | A | | - | | A | LPP | A | LPP |
| `is_valid` | A | bounds from PROJ (plan-time) | A | | A/B | rings via rsgeo relate | A | EP | A | EP | A/B | EP and RGR |
| `null_out_invalid` | A | nulls what `is_valid` rejects | A | | A | | A | | A | | A | |
| `distance` | A/B | [^2] | - | | - | | - | | - | | - | |
| `mean_coordinate` | C | returns the point | C | | C | SCC | C | | C | | C | SCC |
| `translate` | A | impl differs for const or col | A | | A | | A | | A | | A | |
| `rotate` | A | | A | | A | | A | | A | | A | |
| `rotate_about_centroid` | - | | C | CAA | C | | - | | - | | - | |
| `skew` | A | `origin` may be a point column | A | | A | | A | | A | | A | |
| `skew_about_centroid` | - | | C | CAA | C | | - | | - | | - | |
| `scale` | A | impl differs for const or col | A | | A | | A | | A | | A | |
| `scale_about_centroid` | - | | C | CAA | C | | - | | - | | - | |
| `affine_transform` | A | `origin` may be a point column | A | | A | | A | | A | | A | |
| `bounds` | C | repeats its coordinate | C | | C | | C | LPP of `box` | C | LPP of `box` | C | LPP of `box` |
| `set_crs` | C | relabels only | C | | C | | C | | C | | C | |
| `to_crs` | B | PROJ | B | PROJ | B | PROJ | B | PROJ | B | PROJ | B | PROJ |
| `is_geographic` | C | scalar, calls PROJ at plan | C | | C | | C | | C | | C | |
| `x` | C | `f64`, drops the CRS | C | `list[f64]` | C | `list²[f64]` | C | `list[f64]` | C | `list²[f64]` | C | `list³[f64]` |
| `y` | C | `f64`, drops the CRS | C | `list[f64]` | C | `list²[f64]` | C | `list[f64]` | C | `list²[f64]` | C | `list³[f64]` |
| `z` | C | like `x`; only with a `z` | C | | C | | C | | C | | C | |
| `m` | C | like `x`; only with an `m` | C | | C | | C | | C | | C | |
| `count_coordinates` | C | `0` if empty | C | | C | SCC | C | | C | | C | SCC |
| `is_empty` | C | NaN `x` and `y` | C | no verts or only empty | C | only empty rings | C | EP | C | EP | C | EP |
| `wrap_longitude` | C | needs geographic CRS | C | | C | | C | | C | | C | |
| `to_wkb` | A | `wkb` crate | A | | A | | A | | A | | A | |
| `from_wkb` | A | `wkb` crate | A | | A | | A | promotes a point | A | promotes a linestring | A | promotes a poly |
| `to_wkt` | A | `wkt` crate | A | | A | | A | | A | | A | |
| `from_wkt` | A | `wkt` crate | A | | A | | A | promotes a point | A | promotes a linestring | A | promotes a poly |

In this table:

- GGL = 'geographiclib'
- CRS = Coordinate Reference System, stored in metadata
- EP = true when every part passes the single variant's check, one `bool` per row
  (e.g. a multipolygon is only valid if each of its polygons is)
- ANP = refused, unless `allow_non_polygons` is set, then C returning `0.0` (null stays null)
- ANL = refused, unless `allow_non_lines` is set, then C returning `0.0` (null stays null)
- LPP = returns a list, one result per part (computed as for the single variant)
- CAA = composed of `centroid` + `affine_about`
- SCC = skips the closing coordinate of each ring
- RGR = uses `rsgeo`'s `relate` to get an intersection matrix

[^2]: user chooses method, all of which differ in implementation method.
      Planar: A; geodesic: PROJ + GGL, needs a CRS.
