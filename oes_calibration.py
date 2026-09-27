#!/usr/bin/env python3
"""OES latch v2: weighted score + self-calibrated tau (standard library only).

The OES-32 weighted score ``S = 0.45*peak + 0.35*RMS + 0.20*mean|r|`` and the rule
``latch = S >= tau`` are unchanged; the fixed default ``tau = 0.50`` is retained and remains the
default everywhere. This module adds an *optional*, per-series calibration of ``tau`` and of
the residual scale from a warm-up segment that is assumed to be mostly normal:

1. residual per warm-up block: ``x - reference`` (reference: scalar, per-block scalar, or
   per-block length-32 vector; e.g. a causal trailing median);
2. robust scale over all warm-up residual values: ``1.4826 * MAD``; if MAD is 0 (mostly
   constant or count data), ``sqrt(pi/2) * mean absolute deviation from the median``; if that
   is 0 too, calibration fails closed;
3. scaled residual ``r = (x - reference) / scale``; score each warm-up block with
   ``oes32_hilbert_hook.block_score``;
4. ``tau`` = the ``quantile`` (default 0.99) of the warm-up block scores, Hyndman-Fan type 7
   linear interpolation, computed with the same floating-point operations as numpy's default
   ``np.quantile(..., method="linear")``.

Labels are never an input. Everything is deterministic (no randomness, order-independent).
This is an extension in this repository, not the normative OES-32 residual (ADR-001).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Real

from oes32_hilbert_hook import N, block_score, validate_vector
from oes512_hilbert_hook import BLOCK, N_BLOCKS, N_CHANNELS

QUANTILE_DEFAULT = 0.99
MIN_WARMUP_BLOCKS = 4
MAD_TO_SIGMA = 1.4826
MEANAD_TO_SIGMA = math.sqrt(math.pi / 2)  # 1.2533...
METHOD = "warmup-quantile-type7"


class CalibrationError(ValueError):
    """The warm-up data cannot support a calibration (too few blocks, zero spread, ...)."""


# ---------------------------------------------------------------- numeric building blocks


def _check_real(v: object, label: str) -> float:
    if isinstance(v, bool) or not isinstance(v, Real):
        raise TypeError(f"{label} must be a real number, got {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{label} must be finite, got {v!r}")
    return f


def validate_quantile(q: float) -> float:
    """Require a real quantile strictly inside (0, 1)."""
    if isinstance(q, bool) or not isinstance(q, Real):
        raise TypeError(f"quantile must be a real number, got {type(q).__name__}")
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"quantile must be in the open interval (0, 1), got {q!r}")
    return q


def median(values: Sequence[float]) -> float:
    """Median (mean of the two middle values for even length), as numpy computes it."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise CalibrationError("median of an empty sequence")
    h = n // 2
    return s[h] if n % 2 else (s[h - 1] + s[h]) / 2


def quantile_linear(values: Sequence[float], q: float) -> float:
    """Hyndman-Fan type 7 quantile, bit-identical to numpy's default ``np.quantile``.

    Reproduces numpy's virtual index ``(n - 1) * q`` and its two-sided lerp
    (``a + (b-a)*g`` for g < 0.5, ``b - (b-a)*(1-g)`` otherwise).
    """
    q = validate_quantile(q)
    s = sorted(_check_real(v, f"values[{i}]") for i, v in enumerate(values))
    n = len(s)
    if n == 0:
        raise CalibrationError("quantile of an empty sequence")
    virtual = (n - 1) * q
    if virtual >= n - 1:
        return s[-1]
    lo = math.floor(virtual)
    g = virtual - lo
    a, b = s[lo], s[lo + 1]
    diff = b - a
    return b - diff * (1 - g) if g >= 0.5 else a + diff * g


def robust_scale(values: Sequence[float]) -> tuple[float, str]:
    """Robust sigma: ``1.4826*MAD``; if 0, ``sqrt(pi/2)*mean|v - median|`` (``math.fsum``).

    Returns ``(scale, method)`` with method ``"mad"`` or ``"mean_abs_dev"``. Fails closed with
    ``CalibrationError`` if both are 0 (constant values) and ``ValueError`` on non-finite input.
    """
    vals = [_check_real(v, f"values[{i}]") for i, v in enumerate(values)]
    if not vals:
        raise CalibrationError("robust scale of an empty sequence")
    m = median(vals)
    dev = [abs(v - m) for v in vals]
    mad = MAD_TO_SIGMA * median(dev)
    if math.isfinite(mad) and mad > 0:
        return mad, "mad"
    meanad = MEANAD_TO_SIGMA * (math.fsum(dev) / len(dev))
    if math.isfinite(meanad) and meanad > 0:
        return meanad, "mean_abs_dev"
    raise CalibrationError("zero spread in warm-up residuals (constant data); cannot calibrate")


# ---------------------------------------------------------------- calibration object


@dataclass(frozen=True)
class TauCalibration:
    """Result of :func:`calibrate_tau`. Immutable; use with the ``calibrated_*`` functions."""

    tau: float
    scale: float
    quantile: float
    n_blocks: int
    method: str
    scale_method: str

    def __post_init__(self) -> None:
        tau = _check_real(self.tau, "tau")
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau!r}")
        scale = _check_real(self.scale, "scale")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale!r}")
        validate_quantile(self.quantile)
        if isinstance(self.n_blocks, bool) or not isinstance(self.n_blocks, int):
            raise TypeError("n_blocks must be an int")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")

    def residual(self, block: Sequence[float], reference=0.0) -> list[float]:
        """Scaled residual ``(block - reference) / scale`` of one 32-value block."""
        return _scaled_residual(block, reference, self.scale, "block")

    def score(self, block: Sequence[float], reference=0.0) -> float:
        return block_score(self.residual(block, reference))

    def latch(self, block: Sequence[float], reference=0.0) -> int:
        return int(self.score(block, reference) >= self.tau)

    def as_dict(self) -> dict:
        return asdict(self)


def _scaled_residual(block: Sequence[float], reference, scale: float, label: str) -> list[float]:
    validate_vector(block, N, label)
    if isinstance(reference, Real) and not isinstance(reference, bool):
        ref = _check_real(reference, f"{label} reference")
        out = [(float(v) - ref) / scale for v in block]
    else:
        validate_vector(reference, N, f"{label} reference")
        out = [(float(v) - float(rv)) / scale for v, rv in zip(block, reference, strict=True)]
    for v in out:
        if not math.isfinite(v):
            raise ValueError(f"{label} scaled residual is not finite")
    return out


def calibrate_tau(
    warmup_blocks: Sequence[Sequence[float]],
    reference=None,
    quantile: float = QUANTILE_DEFAULT,
    scale="robust",
    min_blocks: int = MIN_WARMUP_BLOCKS,
) -> TauCalibration:
    """Self-calibrate the latch threshold from warm-up blocks (no labels, deterministic).

    Parameters
    ----------
    warmup_blocks:
        Sequence of 32-value blocks from a warm-up segment assumed to be mostly normal.
    reference:
        ``None`` (blocks are already residuals), one real for all blocks, or one entry per
        block that is either a real or a 32-value vector.
    quantile:
        Quantile of warm-up block scores used as tau, in the open interval (0, 1).
    scale:
        ``"robust"`` (1.4826*MAD with mean-absolute-deviation fallback over all warm-up residual
        values) or a positive finite real to use as-is (``scale_method = "fixed"``).
    min_blocks:
        Minimum number of warm-up blocks (default 4); fewer raises ``CalibrationError``.
    """
    q = validate_quantile(quantile)
    if isinstance(min_blocks, bool) or not isinstance(min_blocks, int) or min_blocks < 1:
        raise ValueError(f"min_blocks must be an int >= 1, got {min_blocks!r}")
    if isinstance(warmup_blocks, (str, bytes)):
        raise TypeError("warmup_blocks must be a sequence of blocks")
    blocks = list(warmup_blocks)
    if len(blocks) < min_blocks:
        raise CalibrationError(f"need at least {min_blocks} warm-up blocks, got {len(blocks)}")
    for i, b in enumerate(blocks):
        validate_vector(b, N, f"warmup_blocks[{i}]")

    if reference is None:
        refs: list = [0.0] * len(blocks)
    elif isinstance(reference, Real) and not isinstance(reference, bool):
        refs = [_check_real(reference, "reference")] * len(blocks)
    else:
        refs = list(reference)
        if len(refs) != len(blocks):
            raise ValueError(f"reference must have {len(blocks)} entries, got {len(refs)}")

    raw = [
        _scaled_residual(b, ref, 1.0, f"warmup_blocks[{i}]")
        for i, (b, ref) in enumerate(zip(blocks, refs, strict=True))
    ]

    if isinstance(scale, str):
        if scale != "robust":
            raise ValueError(f"scale must be 'robust' or a positive real, got {scale!r}")
        sigma, scale_method = robust_scale([v for row in raw for v in row])
    else:
        sigma = _check_real(scale, "scale")
        if sigma <= 0:
            raise ValueError(f"scale must be positive, got {scale!r}")
        scale_method = "fixed"

    scaled = [
        _scaled_residual(b, ref, sigma, f"warmup_blocks[{i}]")
        for i, (b, ref) in enumerate(zip(blocks, refs, strict=True))
    ]
    tau = quantile_linear([block_score(r) for r in scaled], q)
    if not math.isfinite(tau) or tau < 0:
        raise CalibrationError(f"calibrated tau is not a finite non-negative number: {tau!r}")
    return TauCalibration(
        tau=tau,
        scale=sigma,
        quantile=q,
        n_blocks=len(blocks),
        method=METHOD,
        scale_method=scale_method,
    )


# ---------------------------------------------------------------- latch functions


def _require_calibration(calibration: object) -> TauCalibration:
    if not isinstance(calibration, TauCalibration):
        raise TypeError(f"calibration must be a TauCalibration, got {type(calibration).__name__}")
    return calibration


def calibrated_block_score(block: Sequence[float], calibration: TauCalibration, reference=0.0):
    """Weighted score of ``(block - reference) / calibration.scale``."""
    return _require_calibration(calibration).score(block, reference)


def calibrated_latch_bit(block: Sequence[float], calibration: TauCalibration, reference=0.0) -> int:
    """1 iff the scaled-residual score is >= ``calibration.tau`` (equality latches)."""
    return _require_calibration(calibration).latch(block, reference)


def calibrated_coarse_syndrome(
    frame: Sequence[float], calibration: TauCalibration, reference=0.0
) -> list[int]:
    """16-bit OES-512 syndrome with a calibrated tau.

    ``reference`` is one real, 16 per-block reals, or a 512-value vector.
    """
    cal = _require_calibration(calibration)
    validate_vector(frame, N_CHANNELS, "OES-512 frame")
    if isinstance(reference, Real) and not isinstance(reference, bool):
        refs: list = [reference] * N_BLOCKS
    elif len(reference) == N_BLOCKS:
        refs = list(reference)
    elif len(reference) == N_CHANNELS:
        refs = [list(reference[i * BLOCK : (i + 1) * BLOCK]) for i in range(N_BLOCKS)]
    else:
        raise ValueError(f"reference must be a real, {N_BLOCKS} reals or {N_CHANNELS} values")
    return [cal.latch(list(frame[i * BLOCK : (i + 1) * BLOCK]), refs[i]) for i in range(N_BLOCKS)]


def main() -> None:
    """SYNTHETIC demo: calibrate on a quiet warm-up, then latch a quiet and a burst block."""
    import random

    rng = random.Random(42)
    warmup = [[rng.gauss(0.0, 1.0) for _ in range(N)] for _ in range(64)]
    cal = calibrate_tau(warmup)
    quiet = [rng.gauss(0.0, 1.0) for _ in range(N)]
    burst = list(quiet)
    burst[5] += 12.0
    print(
        f"tau={cal.tau:.4f} scale={cal.scale:.4f} ({cal.scale_method}) q={cal.quantile} "
        f"n={cal.n_blocks} quiet_latch={calibrated_latch_bit(quiet, cal)} "
        f"burst_latch={calibrated_latch_bit(burst, cal)}"
    )


if __name__ == "__main__":
    main()
