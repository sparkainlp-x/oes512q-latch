"""Tests for oes512_hilbert_hook: 16 x 32 coarse syndrome + optional 9-qubit encode (SYNTHETIC)."""

from __future__ import annotations

import math
import random

import pytest

import oes32_hilbert_hook as o32
import oes512_hilbert_hook as o512


def test_layout_constants():
    assert o512.N_CHANNELS == o512.N_BLOCKS * o512.BLOCK == 512
    assert 2**o512.N_QUBITS_FEATURE == o512.N_CHANNELS


def test_split_blocks_contiguous():
    x = [float(k) for k in range(512)]
    blocks = o512.split_blocks(x)
    assert len(blocks) == 16
    assert all(len(b) == 32 for b in blocks)
    for i, b in enumerate(blocks):
        assert b == [float(k) for k in range(32 * i, 32 * (i + 1))]


@pytest.mark.parametrize("n", [0, 32, 511, 513])
def test_split_blocks_wrong_length(n):
    with pytest.raises(ValueError, match="length 512"):
        o512.split_blocks([0.0] * n)
    with pytest.raises(ValueError, match="length 512"):
        o512.amplitude_encode_oes512([0.0] * n)


def test_non_finite_frame_fails_closed():
    x = o512.synthetic_frame("stable")
    x[40] = math.nan
    with pytest.raises(ValueError, match="finite"):
        o512.coarse_syndrome(x)
    with pytest.raises(ValueError, match="finite"):
        o512.oes512_register(x)


def test_coarse_syndrome_stable():
    s = o512.coarse_syndrome(o512.synthetic_frame("stable"))
    assert s == [0] * 16
    assert o512.admitted(s)


def test_coarse_syndrome_burst_latches_block_3_only():
    s = o512.coarse_syndrome(o512.synthetic_frame("burst"))
    assert s == [1 if i == 3 else 0 for i in range(16)]
    assert not o512.admitted(s)


def test_coarse_syndrome_shock_all_blocks():
    s = o512.coarse_syndrome(o512.synthetic_frame("shock"))
    assert s == [1] * 16
    assert not o512.admitted(s)


def test_admitted():
    assert o512.admitted([0] * 16) is True
    for i in range(16):
        s = [0] * 16
        s[i] = 1
        assert o512.admitted(s) is False


@pytest.mark.parametrize("bad", [[1, -1] + [0] * 14, [2] + [0] * 15, [0.5] + [0] * 15])
def test_admitted_rejects_non_bits(bad):
    # Without this check [1, -1, 0, ...] would sum to 0 and be admitted.
    with pytest.raises(ValueError):
        o512.admitted(bad)


def test_admitted_rejects_wrong_length():
    with pytest.raises(ValueError):
        o512.admitted([0] * 15)


def test_unknown_synthetic_kind():
    with pytest.raises(ValueError):
        o512.synthetic_frame("noisy")


def test_register_report():
    out = o512.oes512_register(o512.synthetic_frame("burst"))
    assert out["syndrome_weight"] == 1
    assert out["latched_blocks"] == [3]
    assert out["admitted"] is False
    assert out["feature_qubits_if_encoded"] == 9
    assert "512-qubit" in out["not"]
    assert out["norm2"] == pytest.approx(1.0, abs=1e-12)


def test_encode_zero_fallback_and_norm():
    psi = o512.amplitude_encode_oes512([0.0] * 512)
    assert psi[0] == 1 and o32.state_norm2(psi) == 1.0
    for kind in ("stable", "burst", "shock"):
        psi = o512.amplitude_encode_oes512(o512.synthetic_frame(kind))
        assert o32.state_norm2(psi) == pytest.approx(1.0, abs=1e-12)


def test_tile_encode():
    tiles = o512.tile_encode_oes32(o512.synthetic_frame("burst"))
    assert len(tiles) == 16
    for t in tiles:
        assert len(t) == 32
        assert o32.state_norm2(t) == pytest.approx(1.0, abs=1e-12)


def _reference_syndrome(x: list[float], tau: float) -> list[int]:
    """Independent, numpy-free recompute of the 16-bit coarse syndrome."""
    bits = []
    for i in range(16):
        blk = x[32 * i : 32 * i + 32]
        a = [abs(v) for v in blk]
        peak = max(a)
        rms = math.sqrt(sum(v * v for v in a) / 32)
        mean = sum(a) / 32
        bits.append(1 if 0.45 * peak + 0.35 * rms + 0.20 * mean >= tau else 0)
    return bits


@pytest.mark.parametrize("seed", range(25))
def test_property_syndrome_matches_reference(seed):
    rng = random.Random(seed)
    scale = rng.choice([0.05, 0.3, 0.6, 1.5])
    x = [rng.gauss(0.0, scale) for _ in range(512)]
    for _ in range(rng.randrange(4)):  # optional localized bursts
        x[rng.randrange(512)] = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
    tau = rng.choice([0.25, 0.50, 0.75])
    s = o512.coarse_syndrome(x, tau)
    assert s == _reference_syndrome(x, tau)
    assert o512.admitted(s) == (sum(s) == 0)


def test_demo_output(capsys):
    o512.main()
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("stable admit=True wt= 0 blocks=[]")
    assert out[1].startswith("burst  admit=False wt= 1 blocks=[3]")
    assert out[2].startswith("shock  admit=False wt=16")
