#!/usr/bin/env python3
"""OES-32 classical syndrome register + optional 5-qubit amplitude encode.
Not a stabilizer code. Not a 32-qubit Hilbert space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

N = 32
N_QUBITS_FEATURE = 5
TAU_DEFAULT = 0.50
EPS = 1e-15
W_PEAK, W_RMS, W_MEAN = 0.45, 0.35, 0.20


def validate_vector(x: Sequence[float], n: int, label: str) -> None:
    """Fail closed: require exactly ``n`` finite real (non-bool) values.

    Raises ``TypeError`` for a non-real element and ``ValueError`` for a wrong
    length or a non-finite value (NaN/inf would otherwise make ``S >= tau``
    evaluate to False and silently admit the block).
    """
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of real numbers, got {type(x).__name__}")
    if len(x) != n:
        raise ValueError(f"{label} must have length {n}, got {len(x)}")
    for i, v in enumerate(x):
        if isinstance(v, bool) or not isinstance(v, Real):
            raise TypeError(f"{label}[{i}] must be a real number, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ValueError(f"{label}[{i}] must be finite, got {v!r}")


def validate_tau(tau: float) -> None:
    """Require a finite, non-negative real threshold."""
    if isinstance(tau, bool) or not isinstance(tau, Real):
        raise TypeError(f"tau must be a real number, got {type(tau).__name__}")
    if not math.isfinite(tau) or tau < 0:
        raise ValueError(f"tau must be finite and non-negative, got {tau!r}")


def block_score(r: Sequence[float]) -> float:
    validate_vector(r, N, "OES-32 block")
    abs_r = [abs(x) for x in r]
    peak = max(abs_r)
    mean_abs = sum(abs_r) / N
    rms = math.sqrt(sum(x * x for x in r) / N)
    return W_PEAK * peak + W_RMS * rms + W_MEAN * mean_abs


def latch_bit(r: Sequence[float], tau: float = TAU_DEFAULT) -> int:
    validate_tau(tau)
    return int(block_score(r) >= tau)


def _normalize_real(r: Sequence[float]) -> list[float]:
    nrm = math.sqrt(sum(x * x for x in r))
    if nrm < EPS:
        out = [0.0] * N
        out[0] = 1.0
        return out
    return [x / nrm for x in r]


def amplitude_encode_oes32(r: Sequence[float]) -> list[complex]:
    validate_vector(r, N, "OES-32 block")
    return [complex(a, 0.0) for a in _normalize_real(r)]


def state_norm2(psi: Sequence[complex]) -> float:
    return sum((z.real * z.real + z.imag * z.imag) for z in psi)


def oes32_register(r: Sequence[float], tau: float = TAU_DEFAULT) -> dict:
    validate_tau(tau)
    score = block_score(r)
    s = int(score >= tau)
    psi = amplitude_encode_oes32(r)
    return {
        "space": "R^32 classical residual + optional C^32 feature state",
        "n_channels": N,
        "feature_qubits_if_encoded": N_QUBITS_FEATURE,
        "not": "32-qubit Hilbert space / stabilizer QEC",
        "score": score,
        "tau": tau,
        "syndrome_bit": s,
        "latch": bool(s),
        "statevector": psi,
        "norm2": state_norm2(psi),
    }


def main() -> None:
    quiet = [0.01] * N
    burst = [0.01] * N
    burst[0] = 2.0
    for name, vec in (("quiet", quiet), ("burst", burst)):
        out = oes32_register(vec)
        print(
            f"{name}: latch={out['latch']} score={out['score']:.4f} "
            f"norm2={out['norm2']:.12f} qubits_feature={out['feature_qubits_if_encoded']}"
        )


if __name__ == "__main__":
    main()
