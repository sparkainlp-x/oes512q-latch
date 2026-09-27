"""Vectorised OES block score, bit-identical to ``oes32_hilbert_hook.block_score``.

``block_score`` sums with the built-in ``sum()``. On CPython < 3.12 that is a plain
left-to-right float sum; from 3.12 on, ``sum()`` of floats uses Neumaier compensated
summation. ``_pysum_rows`` reproduces whichever algorithm the running interpreter uses,
column by column across all blocks at once, so every score (not just every latch decision)
is the same IEEE-754 double. The remaining operations (abs, max, /n, sqrt, weighted sum in
the same order) are correctly rounded in both implementations. Equivalence is tested in
``tests/test_benchmark.py`` on every CI Python version.

For widths other than 32 (sensitivity analysis only) the same formula is applied to the
wider/narrower block; that is **not** the OES-32 contract.
"""

from __future__ import annotations

import sys

import numpy as np

from oes32_hilbert_hook import W_MEAN, W_PEAK, W_RMS

NEUMAIER_SUM = sys.version_info >= (3, 12)


def _pysum_rows(m: np.ndarray) -> np.ndarray:
    """Row sums of ``m`` computed exactly like CPython's ``sum()`` over each row."""
    rows, cols = m.shape
    f = np.zeros(rows)
    if not NEUMAIER_SUM:
        for j in range(cols):
            f = f + m[:, j]
        return f
    c = np.zeros(rows)
    for j in range(cols):
        x = m[:, j]
        t = f + x
        c = c + np.where(np.abs(f) >= np.abs(x), (f - t) + x, (x - t) + f)
        f = t
    return np.where((c != 0) & np.isfinite(c), f + c, f)


def block_scores(r: np.ndarray) -> np.ndarray:
    """Weighted score S = 0.45*peak + 0.35*RMS + 0.20*mean|r| for each row of ``r``.

    Fails closed on non-2-D input, empty blocks or non-finite values.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[1] == 0:
        raise ValueError(f"expected a 2-D array of blocks, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("blocks must be finite")
    n = r.shape[1]
    # Overflow to inf mirrors the pure-Python path; silence numpy's warnings only.
    with np.errstate(over="ignore", invalid="ignore"):
        a = np.abs(r)
        peak = a.max(axis=1)
        mean_abs = _pysum_rows(a) / n
        rms = np.sqrt(_pysum_rows(r * r) / n)
        return W_PEAK * peak + W_RMS * rms + W_MEAN * mean_abs
