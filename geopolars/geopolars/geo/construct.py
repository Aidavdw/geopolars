"""Building a geometry out of the parts it is made of."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import BoxType, PointType

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


def _polygon_kwargs(crs: str | None, close: bool) -> dict[str, str | bool | None]:
    """Mirrors `PolygonKwargs` in `src/expr/construct.rs`."""
    return {**_metadata(crs), "close": close}


def _decode_kwargs(dtype: type[GeoArrowType], crs: str | None) -> dict[str, str | None]:
    """Mirrors `DecodeKwargs` in `src/geoarrow/decode.rs`:
    the concrete geometry `from_wkb` / `from_wkt` decode into.
    """
    dimension = getattr(dtype, "_dimension", ())
    if not dimension:
        msg = f"expected a concrete geometry type such as `PolygonXY`, got: {dtype!r}"
        raise TypeError(msg)
    return {
        "kind": dtype._display,
        "dimension": "".join(dimension),
        **_metadata(crs),
    }


def _from_columns(
    function_name: str,
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
    kwargs: dict[str, Any],
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
        kwargs=kwargs,
    )


def _gather(
    function_name: str, parts: IntoExprColumn, kwargs: dict[str, Any]
) -> pl.Expr:
    """Gather a list column of a geometry's parts into one geometry per row."""
    # Unlike `point`, this is a plugin call: the output dimension is only
    # knowable from the input's dtype, which `output_type_func` gets to see and
    # Python does not.
    return register_plugin_function(
        plugin_path=LIB,
        args=[parts],
        function_name=function_name,
        is_elementwise=True,
        kwargs=kwargs,
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
    dtype = PointType.of_dimension(dimension)
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
    dtype = BoxType.of_dimension(dimension)
    return pl.when(complete).then(pl.struct(bounds)).ext.to(dtype(crs=crs))


def line_string_from_vertices(
    vertices: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.linestring` column out of lists of vertices.

    `vertices` is a list column holding one list per linestring,
    of either `geoarrow.point`s or the bare coordinate structs a point wraps.

    Vertices normally arrive grouped:

    ```python
    df.group_by("route").agg(
        geo.line_string_from_vertices(
            geo.point("lon", "lat").implode()
        ).alias("route")
    )
    ```
    """
    return _gather("linestring", vertices, _metadata(crs))


def line_string_from_columns(
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
    df.select(geo.line_string_from_columns("lon", "lat", z="elevation"))
    ```
    """
    return _from_columns("linestring_coords", x, y, z, m, _metadata(crs))


def line_string(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column, from vertices or from coordinates.
    Dispatches to either:
    - [`line_string_from_vertices`][geopolars.geo.line_string_from_vertices]
    - [`line_string_from_columns`][geopolars.geo.line_string_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "line_string() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call line_string_from_vertices()"
            )
            raise TypeError(msg)
        return line_string_from_vertices(x, crs=crs)

    return line_string_from_columns(x, y, z, m, crs=crs)


def multi_point_from_points(
    points: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column out of lists of points.

    ```python
    df.group_by("survey").agg(
        geo.multi_point_from_points(
            geo.point("lon", "lat").implode()
        ).alias("sightings")
    )
    ```
    """
    return _gather("multipoint", points, _metadata(crs))


def multi_point_from_columns(
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
    df.select(geo.multi_point_from_columns("lon", "lat", m="seen_at"))
    ```
    """
    return _from_columns("multipoint_coords", x, y, z, m, _metadata(crs))


def multi_point(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column, from points or from coordinates.
    Dispatches to either:
    - [`multi_point_from_points`][geopolars.geo.multi_point_from_points]
    - [`multi_point_from_columns`][geopolars.geo.multi_point_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multi_point() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call multi_point_from_points()"
            )
            raise TypeError(msg)
        return multi_point_from_points(x, crs=crs)

    return multi_point_from_columns(x, y, z, m, crs=crs)


def multi_line_string_from_line_strings(
    line_strings: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column out of lists of linestrings.
    Takes either a list of linestrings, or bare vertex lists.

    ```python
    (
        df.group_by("river", "branch", maintain_order=True)
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("branch")
        )
        .group_by("river", maintain_order=True)
        .agg(
            geo.multi_line_string_from_line_strings(
                pl.col("branch").implode()
            ).alias("river")
        )
    )
    ```
    """
    return _gather("multilinestring", line_strings, _metadata(crs))


def multi_line_string_from_columns(
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
    df.select(geo.multi_line_string_from_columns("lon", "lat"))
    ```
    """
    return _from_columns("multilinestring_coords", x, y, z, m, _metadata(crs))


def multi_line_string(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column, from linestrings or coordinates.
    Dispatches to either:
    - [`multi_line_string_from_line_strings`][geopolars.geo.multi_line_string_from_line_strings]
    - [`multi_line_string_from_columns`][geopolars.geo.multi_line_string_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multi_line_string() got a z or m coordinate without a y "
                "coordinate; pass x and y as columns, or call "
                "multi_line_string_from_line_strings()"
            )
            raise TypeError(msg)
        return multi_line_string_from_line_strings(x, crs=crs)

    return multi_line_string_from_columns(x, y, z, m, crs=crs)


def polygon_from_rings(
    rings: IntoExprColumn, *, crs: str | None = None, close: bool = False
) -> pl.Expr:
    """Build a `geoarrow.polygon` column out of lists of rings.

    `rings` is a list column holding one list per polygon, of either
    `geoarrow.linestring`s or the bare vertex lists a linestring wraps. The
    first ring of a polygon is its exterior ring; the rest are its holes.

    Every ring has to be a ring (see [`is_ring`][geopolars.geo.is_ring]):
    at least 4 vertices, the last one repeating the first.
    A polygon with a ring that is not one is missing in the result.
    With `close=True`, an open ring is first closed by repeating its first vertex
    at its end. A ring that is still too short after that stays refused.

    Rings normally arrive grouped, one linestring at a time:

    ```python
    (
        df.group_by("plot", "ring", maintain_order=True)
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("plot", maintain_order=True)
        .agg(geo.polygon_from_rings(pl.col("ring").implode()).alias("plot"))
    )
    ```
    """
    return _gather("polygon", rings, _polygon_kwargs(crs, close))


def polygon_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: bool = False,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column from one coordinate column per axis.
    Each is a `List[List[Float64]]`.

    ```python
    df.select(geo.polygon_from_columns("lon", "lat"))
    ```

    The first ring of a polygon is its exterior ring,
    and the rest are its holes.
    Rings are checked (and closed with `close=True`)
    as in [`polygon_from_rings`][geopolars.geo.polygon_from_rings].
    """
    return _from_columns("polygon_coords", x, y, z, m, _polygon_kwargs(crs, close))


def polygon(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: bool = False,
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
        return polygon_from_rings(x, crs=crs, close=close)

    return polygon_from_columns(x, y, z, m, crs=crs, close=close)


def multi_polygon_from_polygons(
    polygons: IntoExprColumn, *, crs: str | None = None, close: bool = False
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column out of lists of polygons.
    Takes either a list of polygons, or bare lists of rings.

    ```python
    (
        df.group_by("country", "island", "ring", maintain_order=True)
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("country", "island", maintain_order=True)
        .agg(geo.polygon_from_rings(pl.col("ring").implode()).alias("island"))
        .group_by("country", maintain_order=True)
        .agg(
            geo.multi_polygon_from_polygons(
                pl.col("island").implode()
            ).alias("country")
        )
    )
    ```

    As for `polygon_from_rings`, every ring has to be a ring, or be closed with `close=True`.
    A multipolygon with a polygon that fails this is missing in the result.
    """
    return _gather("multipolygon", polygons, _polygon_kwargs(crs, close))


def multi_polygon_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: bool = False,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column from one column per axis.
    Each is a `List[List[List[Float64]]]`: polygons, then rings, then coordinates.

    ```python
    df.select(geo.multi_polygon_from_columns("lon", "lat"))
    ```

    The first ring of every polygon is its exterior ring,
    and the rest are its holes.
    Rings are checked (and closed with `close=True`)
    as in [`multi_polygon_from_polygons`][geopolars.geo.multi_polygon_from_polygons].
    """
    return _from_columns("multipolygon_coords", x, y, z, m, _polygon_kwargs(crs, close))


def multi_polygon(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: bool = False,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column, from polygons or from coordinates.
    Dispatches to either:
    - [`multi_polygon_from_polygons`][geopolars.geo.multi_polygon_from_polygons]
    - [`multi_polygon_from_columns`][geopolars.geo.multi_polygon_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multi_polygon() got a z or m coordinate without a y "
                "coordinate; pass x and y as columns, or call "
                "multi_polygon_from_polygons()"
            )
            raise TypeError(msg)
        return multi_polygon_from_polygons(x, crs=crs, close=close)

    return multi_polygon_from_columns(x, y, z, m, crs=crs, close=close)
