"""Building a geometry out of the parts it is made of."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import GeoBox, GeoPoint

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _coord(value: IntoExprColumn, name: str) -> pl.Expr:
    """One coordinate column of a geometry's storage struct."""
    return _named(value, name).cast(pl.Float64)


def _named(value: IntoExprColumn, name: str) -> pl.Expr:
    """One argument, under the name of the coordinate it carries.

    The plugin reads the dimension back off these names, so they are the spec's
    (`x`, `y`, `z`, `m`) rather than whatever the column was called.
    """
    if isinstance(value, str):
        expr = pl.col(value)
    elif isinstance(value, pl.Series):
        expr = pl.lit(value)
    else:
        expr = value
    return expr.alias(name)


def _given(
    x: IntoExprColumn,
    y: IntoExprColumn | None,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
) -> dict[str, IntoExprColumn]:
    """The coordinates that were passed, in the field order the spec fixes."""
    given = {"x": x, "y": y, "z": z, "m": m}
    return {name: value for name, value in given.items() if value is not None}


def _metadata(crs: str | None) -> dict[str, str | None]:
    """Mirrors `MetadataKwargs` in `src/geoarrow/crs.rs`."""
    return {"crs": crs}


def _decode_kwargs(
    geometry: type[GeoArrowType], crs: str | None
) -> dict[str, str | None]:
    """Mirrors `DecodeKwargs` in `src/geoarrow/decode.rs`:
    the concrete geometry `from_wkb` / `from_wkt` decode into.
    """
    dimension = getattr(geometry, "_dimension", ())
    if not dimension:
        msg = (
            f"expected a concrete geometry type such as `PolygonXY`, got: {geometry!r}"
        )
        raise TypeError(msg)
    return {
        "kind": geometry._display,
        "dimension": "".join(dimension),
        **_metadata(crs),
    }


def _from_columns(
    function_name: str,
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
    crs: str | None,
) -> pl.Expr:
    """Zip one nested coordinate column per axis into one geometry per row."""
    # A plugin call rather than expressions:
    # Polars can transpose a struct of lists into a list of structs only by exploding and regrouping,
    # where the plugin reuses the offsets the columns already have.
    return register_plugin_function(
        plugin_path=LIB,
        args=[_named(value, name) for name, value in _given(x, y, z, m).items()],
        function_name=function_name,
        is_elementwise=True,
        kwargs=_metadata(crs),
    )


def _gather(function_name: str, parts: IntoExprColumn, crs: str | None) -> pl.Expr:
    """Gather a list column of a geometry's parts into one geometry per row."""
    # Unlike `point`, this is a plugin call: the output dimension is only
    # knowable from the input's dtype, which `output_type_func` gets to see and
    # Python does not.
    return register_plugin_function(
        plugin_path=LIB,
        args=[parts],
        function_name=function_name,
        is_elementwise=True,
        kwargs=_metadata(crs),
    )


def validate(geometry: IntoExprColumn) -> pl.Expr:
    """Null out every geometry that is missing a part, or a coordinate of one.
    GeoArrow allows nulls only at the outermost level:
    A geometry is whole or it is null.
    <https://geoarrow.org/format.html#missing-values-null>

    Every constructor in this module already holds to that,
    so a geometry built by one never needs this.
    This function is for the routes into a geometry column that go
    around them and check nothing:

    ```python
    pl.scan_parquet("routes.parquet").select(geo.validate("route"))
    ```

    `ext.to` is a relabelling and reads no data, so it cannot check either:

    ```python
    df.select(pl.col("route").ext.to(gpl.LineStringXY())).select(
        geo.validate("route")
    )
    ```

    Operations over coordinates take the guarantee as given rather than paying
    for it on every call, so a column that arrived one of those ways is worth
    putting through this once. On a column that is already whole it finds
    nothing and copies nothing.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="validate",
        is_elementwise=True,
    )


def point(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.point` column from its coordinate columns.
    Whether you pass `z` and/or `m` decides the dimension, and so the dtype:

    | passed      | dtype       |
    |-------------|-------------|
    | -           | `PointXY`   |
    | `z`         | `PointXYZ`  |
    | `m`         | `PointXYM`  |
    | `z` and `m` | `PointXYZM` |

    `z` is an elevation.
    `m` is an arbitrary measure carried along with the vertex,
    such as a timestamp or a distance along a route.

    ```python
    df.select(geo.point("lon", "lat", crs="EPSG:4326"))
    ```
    """
    given = _given(x, y, z, m)
    dimension = tuple(given)
    coords = [_coord(value, name) for name, value in given.items()]

    # No plugin call is needed:
    # the coordinates are never copied or crossed over FFI.
    #
    # Missing coordinate must invalidate the entire point.
    # https://geoarrow.org/format.html#missing-values-null
    # TODO: The spec can be interpreted as saying that 'm' may also not be null.
    # That really limits usability, no?
    complete = pl.all_horizontal([coord.is_not_null() for coord in coords])
    dtype = GeoPoint.of_dimension(dimension)
    return pl.when(complete).then(pl.struct(coords)).ext.to(dtype(crs=crs))


def box(
    xmin: IntoExprColumn,
    ymin: IntoExprColumn,
    xmax: IntoExprColumn,
    ymax: IntoExprColumn,
    *,
    zmin: IntoExprColumn | None = None,
    zmax: IntoExprColumn | None = None,
    mmin: IntoExprColumn | None = None,
    mmax: IntoExprColumn | None = None,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.box` column from its bounds.
    Whether you pass the `z` and/or `m` bounds decides the dimension, and so the dtype:

    | passed                         | dtype     |
    |--------------------------------|-----------|
    | -                              | `BoxXY`   |
    | `zmin`, `zmax`                 | `BoxXYZ`  |
    | `mmin`, `mmax`                 | `BoxXYM`  |
    | `zmin`, `zmax`, `mmin`, `mmax` | `BoxXYZM` |

    The bounds are taken as given: `xmin > xmax` is a box that crosses the antimeridian,
    and a range from `inf` to `-inf` is empty.

    ```python
    df.select(geo.box("west", "south", "east", "north", crs="EPSG:4326"))
    ```
    """
    given = _given(xmin, ymin, zmin, mmin)
    for axis, low, high in (("z", zmin, zmax), ("m", mmin, mmax)):
        if (low is None) != (high is None):
            msg = f"pass both {axis}min and {axis}max, or neither"
            raise ValueError(msg)
    dimension = tuple(given)
    maxima = _given(xmax, ymax, zmax, mmax)
    bounds = [_coord(value, f"{axis}min") for axis, value in given.items()] + [
        _coord(value, f"{axis}max") for axis, value in maxima.items()
    ]

    # A missing bound invalidates the entire box.
    complete = pl.all_horizontal([bound.is_not_null() for bound in bounds])
    dtype = GeoBox.of_dimension(dimension)
    return pl.when(complete).then(pl.struct(bounds)).ext.to(dtype(crs=crs))


def linestring_from_vertices(
    vertices: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.linestring` column out of lists of vertices.

    `vertices` is a list column holding one list per linestring,
    of either `geoarrow.point`s or the bare coordinate structs a point wraps.

    Vertices normally arrive grouped:

    ```python
    df.group_by("route").agg(
        geo.linestring_from_vertices(
            geo.point("lon", "lat").implode()
        ).alias("route")
    )
    ```
    """
    return _gather("linestring", vertices, crs)


def linestring_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column from one coordinate column per axis.
    Each is a `List[Float64]` holding one list of coordinates per linestring

    ```python
    df.select(geo.linestring_from_columns("lon", "lat", z="elevation"))
    ```
    """
    return _from_columns("linestring_coords", x, y, z, m, crs)


def linestring(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column, from vertices or from coordinates.
    Dispatches to either:
    - [`linestring_from_vertices`][geopolars.geo.linestring_from_vertices]
    - [`linestring_from_columns`][geopolars.geo.linestring_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "linestring() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call linestring_from_vertices()"
            )
            raise TypeError(msg)
        return linestring_from_vertices(x, crs=crs)

    return linestring_from_columns(x, y, z, m, crs=crs)


def multipoint_from_points(
    points: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column out of lists of points.

    ```python
    df.group_by("survey").agg(
        geo.multipoint_from_points(
            geo.point("lon", "lat").implode()
        ).alias("sightings")
    )
    ```
    """
    return _gather("multipoint", points, crs)


def multipoint_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column from one coordinate column per axis.
    Each is a `List[Float64]` holding one list of coordinates per multipoint.

    ```python
    df.select(geo.multipoint_from_columns("lon", "lat", m="seen_at"))
    ```
    """
    return _from_columns("multipoint_coords", x, y, z, m, crs)


def multipoint(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column, from points or from coordinates.
    Dispatches to either:
    - [`multipoint_from_points`][geopolars.geo.multipoint_from_points]
    - [`multipoint_from_columns`][geopolars.geo.multipoint_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multipoint() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call multipoint_from_points()"
            )
            raise TypeError(msg)
        return multipoint_from_points(x, crs=crs)

    return multipoint_from_columns(x, y, z, m, crs=crs)


def multilinestring_from_linestrings(
    linestrings: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column out of lists of linestrings.
    Takes either a list of linestrings, or bare vertex lists.

    ```python
    (
        df.group_by("river", "branch", maintain_order=True)
        .agg(
            geo.linestring_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("branch")
        )
        .group_by("river", maintain_order=True)
        .agg(
            geo.multilinestring_from_linestrings(
                pl.col("branch").implode()
            ).alias("river")
        )
    )
    ```
    """
    return _gather("multilinestring", linestrings, crs)


def multilinestring_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column from one column per axis.
    Each is a `List[List[Float64]]`.

    ```python
    df.select(geo.multilinestring_from_columns("lon", "lat"))
    ```
    """
    return _from_columns("multilinestring_coords", x, y, z, m, crs)


def multilinestring(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column, from linestrings or coordinates.
    Dispatches to either:
    - [`multilinestring_from_linestrings`][geopolars.geo.multilinestring_from_linestrings]
    - [`multilinestring_from_columns`][geopolars.geo.multilinestring_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multilinestring() got a z or m coordinate without a y "
                "coordinate; pass x and y as columns, or call "
                "multilinestring_from_linestrings()"
            )
            raise TypeError(msg)
        return multilinestring_from_linestrings(x, crs=crs)

    return multilinestring_from_columns(x, y, z, m, crs=crs)


def polygon_from_rings(rings: IntoExprColumn, *, crs: str | None = None) -> pl.Expr:
    """Build a `geoarrow.polygon` column out of lists of rings.

    `rings` is a list column holding one list per polygon, of either
    `geoarrow.linestring`s or the bare vertex lists a linestring wraps. The
    first ring of a polygon is its exterior ring; the rest are its holes.

    A ring is a closed linestring: its last vertex has to repeat its first.
    That is the caller's to get right this does not check it,
    and does not close a ring for you.

    Rings normally arrive grouped, one linestring at a time:

    ```python
    (
        df.group_by("plot", "ring", maintain_order=True)
        .agg(
            geo.linestring_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("plot", maintain_order=True)
        .agg(geo.polygon_from_rings(pl.col("ring").implode()).alias("plot"))
    )
    ```
    """
    return _gather("polygon", rings, crs)


def polygon_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column from one coordinate column per axis.
    Each is a `List[List[Float64]]`.

    ```python
    df.select(geo.polygon_from_columns("lon", "lat"))
    ```

    The first ring of a polygon is its exterior ring,
    and the rest are its holes.
    """
    return _from_columns("polygon_coords", x, y, z, m, crs)


def polygon(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column, from rings or from coordinates.
    Dispatches to either:
    - [`polygon_from_rings`][geopolars.geo.polygon_from_rings]
    - [`polygon_from_columns`][geopolars.geo.polygon_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "polygon() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call polygon_from_rings()"
            )
            raise TypeError(msg)
        return polygon_from_rings(x, crs=crs)

    return polygon_from_columns(x, y, z, m, crs=crs)


def multipolygon_from_polygons(
    polygons: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column out of lists of polygons.
    Takes either a list of polygons, or bare lists of rings.

    ```python
    (
        df.group_by("country", "island", "ring", maintain_order=True)
        .agg(
            geo.linestring_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("country", "island", maintain_order=True)
        .agg(geo.polygon_from_rings(pl.col("ring").implode()).alias("island"))
        .group_by("country", maintain_order=True)
        .agg(
            geo.multipolygon_from_polygons(
                pl.col("island").implode()
            ).alias("country")
        )
    )
    ```

    As for `polygon_from_rings`, every ring has to be closed already.
    """
    return _gather("multipolygon", polygons, crs)


def multipolygon_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column from one column per axis.
    Each is a `List[List[List[Float64]]]`: polygons, then rings, then coordinates.

    ```python
    df.select(geo.multipolygon_from_columns("lon", "lat"))
    ```

    The first ring of every polygon is its exterior ring,
    and the rest are its holes.
    """
    return _from_columns("multipolygon_coords", x, y, z, m, crs)


def multipolygon(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column, from polygons or from coordinates.
    Dispatches to either:
    - [`multipolygon_from_polygons`][geopolars.geo.multipolygon_from_polygons]
    - [`multipolygon_from_columns`][geopolars.geo.multipolygon_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multipolygon() got a z or m coordinate without a y "
                "coordinate; pass x and y as columns, or call "
                "multipolygon_from_polygons()"
            )
            raise TypeError(msg)
        return multipolygon_from_polygons(x, crs=crs)

    return multipolygon_from_columns(x, y, z, m, crs=crs)
