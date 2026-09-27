"""Tests for oes32_hilbert_hook: classical OES-32 latch + optional 5-qubit encode (SYNTHETIC)."""

from __future__ import annotations

import math

import pytest

import oes32_hilbert_hook as o32


def quiet() -> list[float]:
    return [0.01] * o32.N


def burst() -> list[float]:
    v = quiet()
    v[0] = 2.0
    return v


def test_block_score_quiet_known_value():
    assert o32.block_score(quiet()) == pytest.approx(0.01, abs=1e-12)
    assert f"{o32.block_score(quiet()):.4f}" == "0.0100"


def test_block_score_burst_known_value():
    peak = 2.0
    mean_abs = (2.0 + 31 * 0.01) / 32
    rms = math.sqrt((4.0 + 31 * 0.0001) / 32)
    expected = 0.45 * peak + 0.35 * rms + 0.20 * mean_abs
    score = o32.block_score(burst())
    assert score == pytest.approx(expected, rel=1e-12)
    assert f"{score:.4f}" == "1.0382"


def test_block_score_is_sign_invariant():
    v = burst()
    assert o32.block_score([-x for x in v]) == pytest.approx(o32.block_score(v))


def test_weights_sum_to_one():
    assert o32.W_PEAK + o32.W_RMS + o32.W_MEAN == pytest.approx(1.0)


def test_latch_at_exact_tau_boundary_fires():
    v = burst()
    s = o32.block_score(v)
    assert o32.latch_bit(v, tau=s) == 1  # S >= tau: equality latches
    assert o32.latch_bit(v, tau=math.nextafter(s, math.inf)) == 0


def test_latch_at_default_tau_boundary():
    # A constant block of c has peak = RMS = mean|x| = c, so S == c exactly.
    v = [0.5] * o32.N
    assert o32.block_score(v) == 0.5
    assert o32.latch_bit(v) == 1
    below = [math.nextafter(0.5, 0.0)] * o32.N
    assert o32.latch_bit(below) == 0


def test_quiet_admitted_burst_latches():
    assert o32.latch_bit(quiet()) == 0
    assert o32.latch_bit(burst()) == 1


@pytest.mark.parametrize("n", [0, 1, 31, 33, 512])
def test_wrong_length_raises(n):
    with pytest.raises(ValueError, match="length 32"):
        o32.block_score([0.0] * n)
    with pytest.raises(ValueError, match="length 32"):
        o32.amplitude_encode_oes32([0.0] * n)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_fails_closed(bad):
    v = [3.0] * o32.N
    v[0] = bad
    with pytest.raises(ValueError, match="finite"):
        o32.latch_bit(v)
    with pytest.raises(ValueError, match="finite"):
        o32.amplitude_encode_oes32(v)


@pytest.mark.parametrize("bad", ["1.0", None, 1j, True])
def test_non_real_element_raises_type_error(bad):
    v = [0.0] * o32.N
    v[5] = bad
    with pytest.raises(TypeError):
        o32.block_score(v)


def test_string_input_rejected():
    with pytest.raises(TypeError):
        o32.block_score("x" * 32)


@pytest.mark.parametrize("tau", [-0.1, math.nan, math.inf])
def test_invalid_tau_raises(tau):
    with pytest.raises(ValueError):
        o32.latch_bit(quiet(), tau=tau)


def test_tau_wrong_type_raises():
    with pytest.raises(TypeError):
        o32.latch_bit(quiet(), tau="0.5")


def test_zero_vector_encode_fallback():
    psi = o32.amplitude_encode_oes32([0.0] * o32.N)
    assert psi[0] == 1 + 0j
    assert all(z == 0 for z in psi[1:])
    assert o32.state_norm2(psi) == 1.0


@pytest.mark.parametrize("vec", [quiet(), burst(), [(-1.0) ** k * k for k in range(32)]])
def test_encode_norm2_is_one(vec):
    psi = o32.amplitude_encode_oes32(vec)
    assert len(psi) == 2**o32.N_QUBITS_FEATURE == o32.N
    assert o32.state_norm2(psi) == pytest.approx(1.0, abs=1e-12)
    assert all(z.imag == 0.0 for z in psi)


def test_register_report():
    out = o32.oes32_register(burst())
    assert out["latch"] is True and out["syndrome_bit"] == 1
    assert out["feature_qubits_if_encoded"] == 5
    assert "stabilizer" in out["not"]
    assert out["norm2"] == pytest.approx(1.0)
    assert o32.oes32_register(quiet())["latch"] is False


def test_demo_output(capsys):
    o32.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("quiet: latch=False score=0.0100")
    assert lines[1].startswith("burst: latch=True score=1.0382")
