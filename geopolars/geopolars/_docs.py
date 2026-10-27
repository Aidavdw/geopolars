"""Docstring sections that several functions share.

A docstring names a section as `{section}`, and `doc` fills it in.
Some sections have placeholders of their own, such as `{output}`.
The keyword arguments of `doc` fill those.
"""

from __future__ import annotations

import inspect
from string import Formatter
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

_shared_docs = {
    "computed_by_geos_dropping_z_and_m": """\
Computed by GEOS.
This operation drops z and m.""",
    "computed_by_geos_keeping_z": """\
Computed by GEOS.
This operation keeps z, but drops m.""",
    "coincidence_axes": """\
Coordinates are compared in x and y.
With `unique_z=True`, the default, they also have to match in z.
With `unique_m=True`, they also have to match in m.
m is a measure rather than a position, so it is not compared by default.
Either flag is ignored for a geometry without that axis.""",
    "coincidence_tolerance": """\
With `abs_tol=None` and `rel_tol=None`, the default, the coordinates have to be exactly equal.
Otherwise they only have to be close on every compared axis, as in `pl.Expr.is_close` (see `geo.is_close`).
A tolerance that is not given then counts as `0.0`.""",
    "empty_and_missing": """\
An empty geometry gives an empty {output}, and a missing one a missing {output}.""",
    "geos_errors": """\
A geometry GEOS fails on raises GEOS's error, failing the query.
With `ignore_per_row_errors=True`, it gives a missing {output} instead.""",
    "geos_non_finite": """\
GEOS supports neither empty points inside a geometry (a vertex with NaN x and y)
nor any other NaN or infinite x or y.
By default they go to GEOS as they are.""",
    "early_finite_check": """\
With `early_finite_check=True`, a geometry with such a vertex is caught before it reaches GEOS.
It then raises, or gives a missing {output} with `ignore_per_row_errors=True`.
This only adds a short path.
It pays off when you expect many non-finite coordinates in your data.""",
    "ccw_outer": """\
The exterior runs as GEOS gives it, clockwise.
With `ccw_outer=True`, it runs counter-clockwise instead at a minimal performance cost.""",
    "origin_per_column_or_row": """\
- one point for the whole column, as `(x, y)` or `(x, y, z)`, or
- a `geoarrow.point` column (or a single point) with a point for every geometry,
  such as `geo.mean_coordinate("shape")` to {verb} each geometry about its own middle.
  A geometry whose origin is missing comes out missing.
  An origin column that declares a CRS has to declare the geometry's.""",
    "about_centroid_accepts": """\
Only linestrings and polygons are accepted:
a multi-geometry has a centroid per part rather than one,
and a point would stay where it is.
A geometry without a centroid (missing, empty, a line without length,
or a polygon that encloses nothing) comes out missing.""",
    "decode_into_dtype": """\
{encoding} can mix geometries row by row, but a Polars column has one dtype,
so `dtype` names the concrete type to decode into, e.g. `PolygonXY`.
A single geometry is promoted into its multi geometry
(e.g. a polygon into `MultiPolygonXY`).
Anything else that does not fit, including a different dimension, is an error.

The metadata of a `{encoding_dtype}` column is carried over, and `crs` labels the result
the same way it does for the constructors.""",
    "box_of_coordinates": """\
The bounds are the minimum and maximum of the stored coordinates.
z and m are bounded like x and y.""",
    "box_around_the_globe": """\
With a geographic CRS, `x` is then brought back into `[-180, 180]`
(or the same in the CRS's unit).
`xmin` lands in `[-180, 180]`, and `xmax` moves by the same turns:
coordinates running from 170 to 190 get a box from 170 to -170,
which is how a box crosses the antimeridian (`xmin > xmax`),
and ones from 540 to 545 get a box from -180 to -175.
Ones spanning a full turn or more get `-180` to `180`.
Coordinates that cross the antimeridian without leaving that range,
such as a line from 170 straight to -170, cannot be told apart
from ones going the long way round, and get a box from -170 to 170.
Without a geographic CRS the bounds are left as they are.""",
    "empty_box": """\
Without any coordinates (or with only empty points), the box is empty:
`inf` to `-inf` in every range.""",
    "crs_forms": """\
`crs` is anything PROJ accepts:
e.g. PROJJSON, WKT or an `AUTH:CODE` format such as `"EPSG:4326"`.""",
    "projected_plane_xy": """\
This looks at a projected plane.
Only x and y are looked at. z and m are ignored.""",
    "missing_ring_input": """\
A missing {input} gives a missing result,
and so does one with a missing vertex, or one that is not finite in x or y.""",
    "point_measure": """\
`m` stands for 'measure', an arbitrary per-vertex value.
Typical uses:
- a timestamp
- a distance along a route
- an airspeed measurement""",
    "linestring_measure": """\
`m` stands for 'measure', an arbitrary value carried per *vertex* rather
than per linestring. Typical uses:
- a timestamp, making the linestring a trajectory
- a distance along a route
- an airspeed measurement""",
    "polygon_measure": """\
`m` stands for 'measure', an arbitrary value carried per *vertex* rather
than per ring or per polygon. Typical uses:
- a timestamp
- a distance along the boundary
- a sensor reading taken at that vertex""",
}


def doc(**kwargs: str) -> Callable[[T], T]:
    """Fill the `{section}`s of a function's or class's docstring.

    The docstring is dedented first,
    so a section's lines line up with the lines around it.
    """

    def decorator(obj: T) -> T:
        if obj.__doc__ is None:  # Stripped with `python -OO`.
            return obj
        template = inspect.cleandoc(obj.__doc__)
        names = {name for _, name, _, _ in Formatter().parse(template) if name}
        shared = {
            name: _shared_docs[name].format(**kwargs)
            for name in names & _shared_docs.keys()
        }
        obj.__doc__ = template.format(**{**shared, **kwargs})
        return obj

    return decorator
