#!/usr/bin/env python3
"""S_i^{blk} = I - 2 Π_i on the 9-qubit feature. Not 16 independent Paulis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from oes512_hilbert_hook import (
    BLOCK,
    N_BLOCKS,
    N_CHANNELS,
    amplitude_encode_oes512,
    coarse_syndrome,
    synthetic_frame,
)

N_QUBITS = 9
PAYLOAD_QUBITS = (0, 1, 2, 3, 4)
ADDRESS_QUBITS = (5, 6, 7, 8)


def _check_index(value: int, upper: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    if not 0 <= value < upper:
        raise ValueError(f"{label} must be 0..{upper - 1}, got {value}")


def _check_state(psi: Sequence[complex], n: int = N_CHANNELS) -> None:
    if len(psi) != n:
        raise ValueError(f"state must have length {n}, got {len(psi)}")


def block_index(k: int) -> int:
    """Block id of channel k. Since BLOCK = 32 = 2**5, k // 32 == k >> 5 (address bits 5..8)."""
    _check_index(k, N_CHANNELS, "channel index")
    return k // BLOCK


def block_projector_diag(i: int) -> list[float]:
    if isinstance(i, bool) or not isinstance(i, int):
        raise TypeError(f"block id must be an int, got {type(i).__name__}")
    if not 0 <= i < N_BLOCKS:
        raise ValueError("block id must be 0..15")
    diag = [0.0] * N_CHANNELS
    start = i * BLOCK
    for j in range(BLOCK):
        diag[start + j] = 1.0
    return diag


def block_observable_diag(i: int) -> list[float]:
    return [1.0 - 2.0 * p for p in block_projector_diag(i)]


def expectation_diag(psi: Sequence[complex], diag: Sequence[float]) -> float:
    if len(psi) != len(diag):
        raise ValueError(f"state length {len(psi)} != observable length {len(diag)}")
    return sum((z.real * z.real + z.imag * z.imag) * d for z, d in zip(psi, diag, strict=True))


def block_masses(psi: Sequence[complex]) -> list[float]:
    _check_state(psi)
    masses = [0.0] * N_BLOCKS
    for k, z in enumerate(psi):
        masses[block_index(k)] += z.real * z.real + z.imag * z.imag
    return masses


def pauli_z_diag(qubit: int) -> list[float]:
    if isinstance(qubit, bool) or not isinstance(qubit, int):
        raise TypeError(f"qubit must be an int, got {type(qubit).__name__}")
    if not 0 <= qubit < N_QUBITS:
        raise ValueError("qubit must be 0..8")
    bit = 1 << qubit
    return [1.0 if (k & bit) == 0 else -1.0 for k in range(N_CHANNELS)]


def stabilizer_generators():
    payload = [(f"B_{j}", j, pauli_z_diag(j)) for j in PAYLOAD_QUBITS]
    address = [(f"A_{t}", 5 + t, pauli_z_diag(5 + t)) for t in range(4)]
    return {"payload": payload, "address": address}


def z_expectations(psi: Sequence[complex]) -> dict[str, float]:
    _check_state(psi)
    gens = stabilizer_generators()
    return {
        name: expectation_diag(psi, diag) for name, _, diag in gens["payload"] + gens["address"]
    }


def fire_from_mass(masses: Sequence[float], mass_tau: float = 0.50) -> list[int]:
    if isinstance(mass_tau, bool) or not isinstance(mass_tau, Real) or not math.isfinite(mass_tau):
        raise ValueError(f"mass_tau must be a finite real number, got {mass_tau!r}")
    return [int(p >= mass_tau) for p in masses]


def compare_classical_vs_encoded(x: Sequence[float], tau: float = 0.50) -> dict:
    s_classical = coarse_syndrome(x, tau)
    psi = amplitude_encode_oes512(x)
    masses = block_masses(psi)
    s_mass = fire_from_mass(masses, mass_tau=0.50)
    return {
        "not": "stabilizer QEC / 16 independent Pauli generators on 9 qubits",
        "n_qubits_feature": N_QUBITS,
        "max_independent_stabilizers": N_QUBITS,
        "classical_syndrome": s_classical,
        "classical_weight": int(sum(s_classical)),
        "block_masses": masses,
        "mass_syndrome": s_mass,
        "same_pattern_as_classical": s_classical == s_mass,
    }


def main() -> None:
    print("S_i^{blk} = I - 2 Pi_i ; Paulis Z_0..Z_4 and Z_5..Z_8")
    for kind in ("stable", "burst", "shock"):
        report = compare_classical_vs_encoded(synthetic_frame(kind))
        print(
            f"{kind:6s} class_wt={report['classical_weight']} "
            f"mass_wt={sum(report['mass_syndrome'])} "
            f"same={report['same_pattern_as_classical']}"
        )


if __name__ == "__main__":
    main()
