"""Tests for oes512_stabilizers (block projectors and Z observables on the 9-qubit feature).

The block observables here are diagonal operators on a 512-dim feature vector;
this is NOT a stabilizer QEC code. SYNTHETIC inputs only.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

import oes512_hilbert_hook as o512
import oes512_stabilizers as st


def _random_state(seed: int) -> list[complex]:
    rng = random.Random(seed)
    return o512.amplitude_encode_oes512([rng.gauss(0.0, 1.0) for _ in range(512)])


@pytest.mark.parametrize("i", range(16))
def test_block_projector_properties(i):
    p = st.block_projector_diag(i)
    assert len(p) == 512
    assert all(v in (0.0, 1.0) for v in p)  # idempotent: P^2 = P for 0/1 diagonal
    assert sum(p) == 32  # rank 32
    assert [k for k, v in enumerate(p) if v] == list(range(32 * i, 32 * (i + 1)))
    s = st.block_observable_diag(i)
    assert all(v * v == 1.0 for v in s)  # S^2 = I (involution)
    assert sum(s) == 512 - 64  # trace = 512 - 2*32


def test_block_projectors_are_orthogonal_and_complete():
    projs = [st.block_projector_diag(i) for i in range(16)]
    for a, b in itertools.combinations(range(16), 2):
        assert all(x * y == 0 for x, y in zip(projs[a], projs[b], strict=True))
    assert [sum(col) for col in zip(*projs, strict=True)] == [1.0] * 512


@pytest.mark.parametrize("i", [-1, 16])
def test_block_projector_range(i):
    with pytest.raises(ValueError):
        st.block_projector_diag(i)


def test_block_projector_type():
    with pytest.raises(TypeError):
        st.block_projector_diag(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_block_masses_sum_to_one(seed):
    m = st.block_masses(_random_state(seed))
    assert len(m) == 16
    assert all(v >= 0 for v in m)
    assert sum(m) == pytest.approx(1.0, abs=1e-12)


def test_block_observable_expectation_matches_mass():
    psi = _random_state(7)
    m = st.block_masses(psi)
    for i in range(16):
        assert st.expectation_diag(psi, st.block_observable_diag(i)) == pytest.approx(
            1.0 - 2.0 * m[i], abs=1e-12
        )


def test_block_index_is_address_bits_5_to_8():
    for k in range(512):
        assert st.block_index(k) == k >> 5 == (k >> 5) & 0b1111
        assert k % 32 == k & 0b11111  # payload = bits 0..4
    assert st.ADDRESS_QUBITS == (5, 6, 7, 8)
    assert st.PAYLOAD_QUBITS == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError):
        st.block_index(512)


def test_pauli_z_diag_consistent_with_2_pow_9():
    assert 2**st.N_QUBITS == st.N_CHANNELS == 512
    for q in range(9):
        z = st.pauli_z_diag(q)
        assert len(z) == 512
        assert sum(z) == 0  # traceless
        assert all(z[k] == (1.0 if not (k >> q) & 1 else -1.0) for k in range(512))


@pytest.mark.parametrize("q", [-1, 9, 10])
def test_pauli_z_diag_range_check(q):
    with pytest.raises(ValueError, match="0..8"):
        st.pauli_z_diag(q)


def test_pauli_z_diag_type_check():
    with pytest.raises(TypeError):
        st.pauli_z_diag(True)


@pytest.mark.parametrize("seed", range(5))
def test_z_expectations_in_range(seed):
    z = st.z_expectations(_random_state(seed))
    assert set(z) == {"B_0", "B_1", "B_2", "B_3", "B_4", "A_0", "A_1", "A_2", "A_3"}
    assert all(-1.0 - 1e-12 <= v <= 1.0 + 1e-12 for v in z.values())


def test_address_z_expectations_are_block_mass_marginals():
    # <Z_{5+t}> = sum_b M_b * (-1)^{bit_t(b)}: the address expectations are a
    # linear function of the 16 block masses.
    for seed in range(5):
        psi = _random_state(seed)
        m = st.block_masses(psi)
        z = st.z_expectations(psi)
        for t in range(4):
            expected = sum(m[b] * (-1) ** ((b >> t) & 1) for b in range(16))
            assert z[f"A_{t}"] == pytest.approx(expected, abs=1e-12)


def _mass_state(masses: list[float]) -> list[complex]:
    """Put mass M_b uniformly on block b."""
    return [complex(math.sqrt(masses[k // 32] / 32), 0.0) for k in range(512)]


def test_four_address_expectations_do_not_determine_block_masses():
    # 4 single-qubit expectations cannot fix 15 free mass parameters.
    uniform = [1 / 16] * 16
    split = [0.0] * 16
    split[0] = split[15] = 0.5
    za, zb = st.z_expectations(_mass_state(uniform)), st.z_expectations(_mass_state(split))
    for t in range(4):
        assert za[f"A_{t}"] == pytest.approx(0.0, abs=1e-12)
        assert zb[f"A_{t}"] == pytest.approx(0.0, abs=1e-12)
    assert st.block_masses(_mass_state(uniform)) != pytest.approx(split)


def test_address_z_string_correlators_determine_block_masses():
    # All 16 products prod_{t in T} Z_{5+t} (T subset of {0..3}) determine the
    # masses exactly via the inverse Walsh-Hadamard transform.
    psi = _random_state(11)
    zs = [st.pauli_z_diag(q) for q in st.ADDRESS_QUBITS]
    corr = {}
    for T in range(16):
        diag = [1.0] * 512
        for t in range(4):
            if (T >> t) & 1:
                diag = [a * b for a, b in zip(diag, zs[t], strict=True)]
        corr[T] = st.expectation_diag(psi, diag)
    recovered = [
        sum(corr[T] * (-1) ** bin(T & b).count("1") for T in range(16)) / 16 for b in range(16)
    ]
    assert recovered == pytest.approx(st.block_masses(psi), abs=1e-12)


def test_expectation_diag_length_mismatch():
    with pytest.raises(ValueError):
        st.expectation_diag([1 + 0j] * 511, [1.0] * 512)
    with pytest.raises(ValueError):
        st.block_masses([1 + 0j] * 32)


def test_stable_and_burst_mass_fire_match_classical():
    for kind, weight in (("stable", 0), ("burst", 1)):
        r = st.compare_classical_vs_encoded(o512.synthetic_frame(kind))
        assert r["classical_weight"] == weight
        assert r["same_pattern_as_classical"] is True


def test_shock_mass_fire_mismatch_is_expected_behaviour():
    """Documented, by design: the normalized state spreads a uniform shock evenly,
    so each block holds mass 1/16 = 0.0625 < 0.50 and the mass-fire syndrome is
    empty, while the classical latch (on raw amplitudes) fires all 16 blocks.
    The encoded feature is NOT a substitute for the classical latch."""
    r = st.compare_classical_vs_encoded(o512.synthetic_frame("shock"))
    assert r["classical_weight"] == 16
    assert sum(r["mass_syndrome"]) == 0
    assert r["block_masses"] == pytest.approx([1 / 16] * 16, abs=1e-12)
    assert r["same_pattern_as_classical"] is False
    assert r["max_independent_stabilizers"] == 9


def test_fire_from_mass_rejects_bad_tau():
    with pytest.raises(ValueError):
        st.fire_from_mass([0.0] * 16, mass_tau=math.nan)


def test_stabilizer_generators_indices():
    g = st.stabilizer_generators()
    assert [q for _, q, _ in g["payload"]] == [0, 1, 2, 3, 4]
    assert [q for _, q, _ in g["address"]] == [5, 6, 7, 8]


def test_demo_output(capsys):
    st.main()
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "stable class_wt=0 mass_wt=0 same=True"
    assert out[2] == "burst  class_wt=1 mass_wt=1 same=True"
    assert out[3] == "shock  class_wt=16 mass_wt=0 same=False"


def test_encode_is_scale_invariant_stable_equals_shock():
    # Normalization discards overall amplitude: the quiet "stable" frame (0.01)
    # and the "shock" frame (1.5) map to the same feature state.
    a = o512.amplitude_encode_oes512(o512.synthetic_frame("stable"))
    b = o512.amplitude_encode_oes512(o512.synthetic_frame("shock"))
    assert a == pytest.approx(b, abs=1e-15)
