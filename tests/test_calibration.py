"""Tests for oes_calibration: OES latch v2 self-calibrated tau (SYNTHETIC, offline)."""

from __future__ import annotations

import dataclasses
import inspect
import math
import random
import statistics

import pytest

import oes32_hilbert_hook as o32
import oes512_hilbert_hook as o512
import oes_calibration as oc


def _gauss_blocks(n: int, seed: int = 42, sigma: float = 1.0, mu: float = 0.0):
    rng = random.Random(seed)
    return [[rng.gauss(mu, sigma) for _ in range(32)] for _ in range(n)]


# ---------------------------------------------------------------- quantile / median / scale


def test_quantile_linear_known_values():
    assert oc.quantile_linear([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert oc.quantile_linear([3.0, 1.0, 2.0], 0.5) == 2.0
    assert oc.quantile_linear([5.0], 0.99) == 5.0
    assert oc.quantile_linear([0.0, 10.0], 0.99) == pytest.approx(9.9)
    assert oc.quantile_linear([0.0, 10.0], 0.25) == pytest.approx(2.5)


def test_quantile_linear_matches_statistics_inclusive():
    rng = random.Random(7)
    for n in (2, 3, 10, 101):
        xs = [rng.random() for _ in range(n)]
        qs = statistics.quantiles(xs, n=100, method="inclusive")
        assert oc.quantile_linear(xs, 0.99) == pytest.approx(qs[98], rel=1e-12)


def test_quantile_linear_bit_identical_to_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    for n in [*range(1, 40), 97, 300]:
        for _ in range(10):
            x = rng.standard_normal(n) * rng.choice([1e-6, 1.0, 1e6])
            if rng.random() < 0.3:
                x = np.round(x, 1)  # ties
            for q in (0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
                assert oc.quantile_linear(x.tolist(), q) == np.quantile(x, q)
            assert oc.median(x.tolist()) == np.median(x)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf])
def test_quantile_outside_open_unit_interval_rejected(q):
    with pytest.raises(ValueError, match="quantile"):
        oc.quantile_linear([1.0, 2.0], q)
    with pytest.raises(ValueError, match="quantile"):
        oc.calibrate_tau(_gauss_blocks(8), quantile=q)


@pytest.mark.parametrize("q", [True, "0.99", None])
def test_quantile_wrong_type_rejected(q):
    with pytest.raises(TypeError):
        oc.calibrate_tau(_gauss_blocks(8), quantile=q)


def test_robust_scale_mad_and_fallback():
    s, how = oc.robust_scale([1.0, 2.0, 3.0, 4.0, 100.0])
    assert how == "mad"
    assert s == pytest.approx(1.4826 * 1.0)
    s, how = oc.robust_scale([0.0] * 90 + [1.0] * 10)
    assert how == "mean_abs_dev"
    assert s == pytest.approx(math.sqrt(math.pi / 2) * 0.1)
    with pytest.raises(oc.CalibrationError, match="zero spread"):
        oc.robust_scale([3.0] * 50)
    with pytest.raises(ValueError, match="finite"):
        oc.robust_scale([1.0, math.nan])


# ---------------------------------------------------------------- calibrate_tau


def test_calibrate_tau_basic_object():
    cal = oc.calibrate_tau(_gauss_blocks(64))
    assert isinstance(cal, oc.TauCalibration)
    assert cal.quantile == 0.99
    assert cal.n_blocks == 64
    assert cal.method == "warmup-quantile-type7"
    assert cal.scale_method == "mad"
    assert 0.8 < cal.scale < 1.2  # unit-variance Gaussian warm-up
    assert cal.tau > o32.TAU_DEFAULT
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.tau = 0.1  # type: ignore[misc]
    assert cal.as_dict()["tau"] == cal.tau


def test_calibrate_tau_rule_by_hand():
    blocks = _gauss_blocks(20, seed=3, sigma=5.0, mu=7.0)
    cal = oc.calibrate_tau(blocks, reference=7.0, quantile=0.9)
    flat = [v - 7.0 for b in blocks for v in b]
    scale, _ = oc.robust_scale(flat)
    scores = sorted(o32.block_score([(v - 7.0) / scale for v in b]) for b in blocks)
    h = (len(scores) - 1) * 0.9
    lo = math.floor(h)
    expected = scores[lo] + (scores[lo + 1] - scores[lo]) * (h - lo)
    assert cal.scale == scale
    assert cal.tau == pytest.approx(expected, rel=1e-15)


def test_calibrate_tau_is_deterministic_and_order_independent():
    blocks = _gauss_blocks(40, seed=11)
    a = oc.calibrate_tau(blocks)
    b = oc.calibrate_tau([list(x) for x in blocks])
    c = oc.calibrate_tau(list(reversed(blocks)))
    assert a == b == c


def test_calibrate_tau_reference_forms_agree():
    blocks = _gauss_blocks(12, seed=5, mu=3.0)
    refs_scalar = [3.0] * 12
    refs_vec = [[3.0] * 32 for _ in range(12)]
    shifted = [[v - 3.0 for v in b] for b in blocks]
    a = oc.calibrate_tau(blocks, reference=3.0)
    b = oc.calibrate_tau(blocks, reference=refs_scalar)
    c = oc.calibrate_tau(blocks, reference=refs_vec)
    d = oc.calibrate_tau(shifted)
    assert a == b == c
    assert d.tau == pytest.approx(a.tau, rel=1e-12)


def test_calibrate_tau_fixed_scale():
    blocks = _gauss_blocks(10)
    cal = oc.calibrate_tau(blocks, scale=2.0)
    assert cal.scale == 2.0
    assert cal.scale_method == "fixed"
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="scale"):
            oc.calibrate_tau(blocks, scale=bad)
    with pytest.raises(ValueError, match="scale"):
        oc.calibrate_tau(blocks, scale="mad")


def test_calibrate_tau_zero_spread_fallback_and_fail_closed():
    blocks = [[0.0] * 32 for _ in range(9)] + [[1.0] * 32]
    cal = oc.calibrate_tau(blocks)
    assert cal.scale_method == "mean_abs_dev"
    with pytest.raises(oc.CalibrationError, match="zero spread"):
        oc.calibrate_tau([[2.0] * 32 for _ in range(10)])


@pytest.mark.parametrize("n", [0, 1, 3])
def test_calibrate_tau_too_few_blocks(n):
    with pytest.raises(oc.CalibrationError, match="at least 4"):
        oc.calibrate_tau(_gauss_blocks(n) if n else [])


def test_calibrate_tau_min_blocks_validation():
    assert oc.calibrate_tau(_gauss_blocks(2), min_blocks=2).n_blocks == 2
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="min_blocks"):
            oc.calibrate_tau(_gauss_blocks(8), min_blocks=bad)


def test_calibrate_tau_rejects_bad_blocks_and_references():
    good = _gauss_blocks(6)
    bad_len = [*good[:5], [0.0] * 31]
    with pytest.raises(ValueError, match="length 32"):
        oc.calibrate_tau(bad_len)
    nan_block = [*good[:5], [0.0] * 31 + [math.nan]]
    with pytest.raises(ValueError, match="finite"):
        oc.calibrate_tau(nan_block)
    inf_block = [*good[:5], [math.inf] + [0.0] * 31]
    with pytest.raises(ValueError, match="finite"):
        oc.calibrate_tau(inf_block)
    with pytest.raises(TypeError):
        oc.calibrate_tau([[*good[0][:31], "x"], *good[1:]])
    with pytest.raises(TypeError):
        oc.calibrate_tau("not blocks")
    with pytest.raises(ValueError, match="entries"):
        oc.calibrate_tau(good, reference=[0.0] * 5)
    with pytest.raises(ValueError, match="finite"):
        oc.calibrate_tau(good, reference=math.nan)
    with pytest.raises(ValueError, match="length 32"):
        oc.calibrate_tau(good, reference=[[0.0] * 31] * 6)


def test_overflowing_residual_fails_closed():
    blocks = _gauss_blocks(6)
    blocks[0][0] = 1e308
    with pytest.raises(ValueError, match="finite"):
        oc.calibrate_tau(blocks, scale=1e-10)


def test_tau_calibration_object_validation():
    ok = {"tau": 1.0, "scale": 1.0, "quantile": 0.99, "n_blocks": 4, "method": "m"}
    oc.TauCalibration(**ok, scale_method="mad")
    for field, bad in (
        ("tau", -0.1),
        ("tau", math.nan),
        ("tau", math.inf),
        ("scale", 0.0),
        ("quantile", 1.0),
        ("n_blocks", 0),
    ):
        with pytest.raises(ValueError):
            oc.TauCalibration(**{**ok, field: bad}, scale_method="mad")
    with pytest.raises(TypeError):
        oc.TauCalibration(**{**ok, "n_blocks": 4.0}, scale_method="mad")


def test_calibration_never_takes_labels():
    params = set(inspect.signature(oc.calibrate_tau).parameters)
    assert params == {"warmup_blocks", "reference", "quantile", "scale", "min_blocks"}
    assert not any("label" in p for p in params)


# ---------------------------------------------------------------- latch functions


def test_calibrated_latch_equality_and_scaling():
    cal = oc.TauCalibration(1.0, 2.0, 0.99, 4, "manual", "fixed")
    at = [2.0] * 32  # residual 1.0 after scaling -> S = 1.0 == tau -> latch
    below = [1.9] * 32
    assert oc.calibrated_block_score(at, cal) == pytest.approx(1.0)
    assert oc.calibrated_latch_bit(at, cal) == 1
    assert oc.calibrated_latch_bit(below, cal) == 0
    assert oc.calibrated_latch_bit([12.0] * 32, cal, reference=10.0) == 1
    assert oc.calibrated_latch_bit([12.0] * 32, cal, reference=[10.5] * 32) == 0


def test_calibrated_latch_rejects_bad_inputs():
    cal = oc.calibrate_tau(_gauss_blocks(8))
    with pytest.raises(TypeError, match="TauCalibration"):
        oc.calibrated_latch_bit([0.0] * 32, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length 32"):
        oc.calibrated_latch_bit([0.0] * 31, cal)
    with pytest.raises(ValueError, match="finite"):
        oc.calibrated_latch_bit([math.nan] * 32, cal)
    with pytest.raises(ValueError, match="finite"):
        oc.calibrated_latch_bit([0.0] * 32, cal, reference=math.inf)


def test_calibrated_coarse_syndrome_matches_per_block_latch():
    cal = oc.calibrate_tau(_gauss_blocks(32))
    rng = random.Random(1)
    frame = [rng.gauss(0.0, 1.0) for _ in range(512)]
    frame[3 * 32 + 4] += 25.0
    syn = oc.calibrated_coarse_syndrome(frame, cal)
    assert syn == [cal.latch(frame[i * 32 : (i + 1) * 32]) for i in range(16)]
    assert syn[3] == 1
    assert oc.calibrated_coarse_syndrome(frame, cal, [0.0] * 16) == syn
    assert oc.calibrated_coarse_syndrome(frame, cal, [0.0] * 512) == syn
    with pytest.raises(ValueError, match="reference"):
        oc.calibrated_coarse_syndrome(frame, cal, [0.0] * 7)
    with pytest.raises(ValueError, match="length 512"):
        oc.calibrated_coarse_syndrome(frame[:511], cal)


def test_fixed_default_contract_unchanged():
    """v2 is additive: tau = 0.50 remains the default of the existing latch functions."""
    assert o32.TAU_DEFAULT == 0.50
    assert inspect.signature(o32.latch_bit).parameters["tau"].default == 0.50
    assert inspect.signature(o512.coarse_syndrome).parameters["tau"].default == 0.50
    cal = oc.TauCalibration(0.50, 1.0, 0.99, 4, "manual", "fixed")
    rng = random.Random(9)
    frame = [rng.gauss(0.0, 0.4) for _ in range(512)]
    assert oc.calibrated_coarse_syndrome(frame, cal) == o512.coarse_syndrome(frame)


def test_demo_runs(capsys):
    oc.main()
    out = capsys.readouterr().out
    assert "quiet_latch=0" in out
    assert "burst_latch=1" in out


# ---------------------------------------------------------------- benchmark equivalence


def test_benchmark_threshold_equals_package_calibration():
    """The benchmark's per-series self-calibrated tau is exactly calibrate_tau (SYNTHETIC)."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from benchmarks import run_benchmark as rb

    rng = np.random.default_rng(0)
    n = 160 * 32
    t = np.arange(n)
    x = 10 + np.sin(2 * np.pi * t / 288) + 0.1 * rng.standard_normal(n)
    y = np.zeros(n, dtype=int)
    for b in (60, 61, 100, 140):
        x[b * 32 + 5 : b * 32 + 20] += 8.0
        y[b * 32 : (b + 1) * 32] = 1
    res = rb.evaluate(x, y, 32, seeds=(42,), time_repeats=1)

    # Independent pure-Python recomputation of the warm-up blocks and trailing medians.
    base = rb.baseline_len(n)
    n_total = n // 32
    first = -(-base // 32)
    n_cal = int(rb.CAL_FRACTION * n_total)
    xs = x.tolist()
    warm = [xs[b * 32 : (b + 1) * 32] for b in range(first, n_cal)]
    refs = [statistics.median(xs[b * 32 - base : b * 32]) for b in range(first, n_cal)]
    cal = oc.calibrate_tau(warm, refs, quantile=rb.CAL_QUANTILE)
    assert cal.tau == res["modes"]["calibrated"]["oes_latch"]["threshold"]
    assert cal.scale == res["residual_scale"]
    assert res["latch_calibration"]["calibrated"] == cal.as_dict()
    assert res["checks"]["calibrate_tau_matches"] is True

    # Labels never enter the calibration: scrambling them leaves every threshold unchanged.
    y2 = np.roll(y, 1000)
    y2[:200] = 1
    res2 = rb.evaluate(x, y2, 32, seeds=(42,), time_repeats=1)
    for mode in ("calibrated", "trimmed"):
        for det in rb.DETECTORS:
            assert res2["modes"][mode][det]["threshold"] == res["modes"][mode][det]["threshold"]
    assert res2["latch_calibration"] == res["latch_calibration"]


def test_readme_v2_example_output_is_real(capsys):
    """The README usage example prints exactly the output shown under it (SYNTHETIC)."""
    import pathlib
    import re

    readme = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text("utf-8")
    section = readme.split("## OES latch v2", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    shown = re.search(r"```text\n(.*?)```", section, re.S).group(1)
    exec(compile(code, "README-example", "exec"), {})
    assert capsys.readouterr().out == shown
