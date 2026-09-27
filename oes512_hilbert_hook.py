#!/usr/bin/env python3
"""OES-512: 16-bit coarse syndrome first, optional 9-qubit feature encode."""

from __future__ import annotations

from collections.abc import Sequence

from oes32_hilbert_hook import (
    TAU_DEFAULT,
    N,
    amplitude_encode_oes32,
    block_score,
    latch_bit,
    state_norm2,
    validate_tau,
    validate_vector,
)

N_CHANNELS = 512
N_BLOCKS = 16
BLOCK = N
N_QUBITS_FEATURE = 9
EPS = 1e-15


def split_blocks(x: Sequence[float]) -> list[list[float]]:
    validate_vector(x, N_CHANNELS, "OES-512 frame")
    return [list(x[i * BLOCK : (i + 1) * BLOCK]) for i in range(N_BLOCKS)]


def coarse_syndrome(x: Sequence[float], tau: float = TAU_DEFAULT) -> list[int]:
    return [latch_bit(block, tau) for block in split_blocks(x)]


def admitted(syndrome: Sequence[int]) -> bool:
    if len(syndrome) != N_BLOCKS:
        raise ValueError(f"syndrome must have length {N_BLOCKS}, got {len(syndrome)}")
    for i, bit in enumerate(syndrome):
        if not isinstance(bit, int) or bit not in (0, 1):
            raise ValueError(f"syndrome[{i}] must be 0 or 1, got {bit!r}")
    return sum(syndrome) == 0


def _normalize_real_512(x: Sequence[float]) -> list[float]:
    nrm = sum(v * v for v in x) ** 0.5
    if nrm < EPS:
        out = [0.0] * N_CHANNELS
        out[0] = 1.0
        return out
    return [v / nrm for v in x]


def amplitude_encode_oes512(x: Sequence[float]) -> list[complex]:
    validate_vector(x, N_CHANNELS, "OES-512 frame")
    return [complex(a, 0.0) for a in _normalize_real_512(x)]


def tile_encode_oes32(x: Sequence[float]) -> list[list[complex]]:
    return [amplitude_encode_oes32(block) for block in split_blocks(x)]


def oes512_register(x: Sequence[float], tau: float = TAU_DEFAULT) -> dict:
    validate_tau(tau)
    blocks = split_blocks(x)
    block_scores = [block_score(b) for b in blocks]
    s = [int(sc >= tau) for sc in block_scores]
    psi = amplitude_encode_oes512(x)
    return {
        "space": "R^512 classical residual + optional C^512 feature state",
        "n_channels": N_CHANNELS,
        "n_blocks": N_BLOCKS,
        "block_width": BLOCK,
        "feature_qubits_if_encoded": N_QUBITS_FEATURE,
        "tile_feature_qubits": 5,
        "not": "512-qubit Hilbert space / stabilizer QEC / IQM Cloud",
        "tau": tau,
        "scores": block_scores,
        "coarse_syndrome": s,
        "syndrome_weight": int(sum(s)),
        "admitted": admitted(s),
        "latched_blocks": [i for i, bit in enumerate(s) if bit],
        "statevector": psi,
        "norm2": state_norm2(psi),
    }


def synthetic_frame(kind: str) -> list[float]:
    quiet = [0.01] * N_CHANNELS
    if kind == "stable":
        return quiet
    if kind == "burst":
        frame = list(quiet)
        frame[3 * BLOCK] = 2.0
        return frame
    if kind == "shock":
        return [1.5] * N_CHANNELS
    raise ValueError(f"unknown kind: {kind}")


def main() -> None:
    for kind in ("stable", "burst", "shock"):
        out = oes512_register(synthetic_frame(kind))
        print(
            f"{kind:6s} admit={out['admitted']} wt={out['syndrome_weight']:2d} "
            f"blocks={out['latched_blocks']} norm2={out['norm2']:.12f} "
            f"q={out['feature_qubits_if_encoded']}"
        )


if __name__ == "__main__":
    main()
