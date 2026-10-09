"""Whether two points lie at almost the same place.

Works on the points' storage, so it can be used inside `list.eval`
as well as wrapped into an expression between two point columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


def _is_close(
    first: pl.Expr,
    second: pl.Expr,
    axes: Iterable[str],
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 1e-9,
) -> pl.Expr:
    """Whether two points (as storage) are close on every axis in `axes`.

    Per axis, as `pl.Expr.is_close`:
    `|a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)`.
    """
    return pl.all_horizontal(
        first.struct.field(axis).is_close(
            second.struct.field(axis), abs_tol=abs_tol, rel_tol=rel_tol
        )
        for axis in axes
    )
