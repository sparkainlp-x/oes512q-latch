"""Offline tests for the benchmark harness (tiny SYNTHETIC fixtures; no network, no NAB data)."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

import oes32_hilbert_hook as o32  # noqa: E402
from benchmarks import download_nab as dl  # noqa: E402
from benchmarks import fast_score as fs  # noqa: E402
from benchmarks import nab_score as ns  # noqa: E402
from benchmarks import run_benchmark as rb  # noqa: E402

N_BLOCKS = 160
ANOMALY_BLOCKS = (60, 61, 100, 140)


def _fixture(seed: int = 0, n_blocks: int = N_BLOCKS, block: int = 32):
    """Seeded sine + noise series with large bursts in a few 32-sample blocks (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    n = n_blocks * block
    t = np.arange(n)
    x = 10 + np.sin(2 * np.pi * t / 288) + 0.1 * rng.standard_normal(n)
    y = np.zeros(n, dtype=int)
    for b in ANOMALY_BLOCKS:
        s = b * 32
        x[s + 5 : s + 20] += 8.0
        y[s : s + 32] = 1
    return x, y


# ---------------------------------------------------------------- fast score equivalence


def _python_scores(r):
    return np.array([o32.block_score([float(v) for v in row]) for row in r])


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1.0, 7.0, 1e4, 1e12, 1e160])
def test_fast_score_bit_identical_to_block_score(scale):
    rng = np.random.default_rng(int(math.log10(scale) + 20))
    r = rng.standard_normal((400, 32)) * scale
    r[::5] *= rng.choice([1e-12, 1.0, 1e12], size=(1, 32))  # mixed magnitudes / cancellation
    r[1::7, ::3] = 0.0
    fast = fs.block_scores(r)
    py = _python_scores(r)
    assert np.array_equal(fast, py)  # same doubles (incl. inf on overflow), not just close
    for tau in (0.0, o32.TAU_DEFAULT, 1.0, 3.0):
        assert np.array_equal(fast >= tau, py >= tau)


def test_fast_score_bit_identical_near_threshold():
    """Blocks whose score lands on or next to tau latch identically in both paths."""
    rnd = random.Random(42)
    rows = []
    for _ in range(300):
        c = o32.TAU_DEFAULT + rnd.choice([-1, 0, 1]) * rnd.random() * 1e-15
        row = [c * rnd.choice([-1.0, 1.0]) for _ in range(32)]
        rows.append(row)
    r = np.array(rows)
    assert np.array_equal(fs.block_scores(r), _python_scores(r))


def test_fast_score_known_values_and_fail_closed():
    assert fs.block_scores(np.full((1, 32), 0.5))[0] == pytest.approx(0.5)
    burst = np.full((1, 32), 0.01)
    burst[0, 0] = 2.0
    assert fs.block_scores(burst)[0] == o32.block_score(burst[0].tolist())
    with pytest.raises(ValueError, match="finite"):
        fs.block_scores(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError, match="2-D"):
        fs.block_scores(np.zeros(32))


# ---------------------------------------------------------------- protocol pieces


def test_make_blocks_shapes_and_causal_reference():
    x, y = _fixture()
    blocks = rb.make_blocks(x, y, 32, 288)
    first = -(-288 // 32)
    assert blocks["ids"][0] == first
    assert blocks["raw"].shape == (N_BLOCKS - first, 32)
    s = first * 32
    assert blocks["ref"][0] == pytest.approx(np.median(x[s - 288 : s]))
    assert set(np.flatnonzero(blocks["labels"]) + first) == set(ANOMALY_BLOCKS)


def test_baseline_len_rule():
    assert rb.baseline_len(22695) == 288
    assert rb.baseline_len(1624) == 81
    assert rb.baseline_len(10) == 1


def test_split_is_fixed_fraction_not_label_driven():
    x, y = _fixture()
    blocks = rb.make_blocks(x, y, 32, 288)
    cal, test = rb.split_masks(blocks["ids"], blocks["n_total_blocks"])
    n_cal = int(rb.CAL_FRACTION * N_BLOCKS)
    assert blocks["ids"][cal].max() == n_cal - 1
    assert blocks["ids"][test].min() == n_cal
    assert not (cal & test).any()


def test_residual_scale_fallback_and_degenerate():
    c = np.array([0.0] * 90 + [1.0] * 10)  # MAD = 0
    scale, how = rb.residual_scale(c)
    assert how == "mean_abs_dev"
    assert scale == pytest.approx(rb.MEANAD_TO_SIGMA * 0.1)
    assert rb.residual_scale(np.array([0.0, 1.0, 2.0]))[1] == "mad"
    with pytest.raises(rb.SkipSeries):
        rb.residual_scale(np.zeros(10))


def test_trim_mask_is_label_free_and_drops_outliers():
    rng = np.random.default_rng(3)
    r = rng.standard_normal((50, 32))
    r[7, 4] = 60.0
    keep = rb.trim_mask(r)
    assert not keep[7]
    assert keep.sum() >= 45


def test_cusum_is_causal_and_nonnegative():
    r = np.zeros((4, 32))
    r[2, :] = 3.0
    s = rb.score_cusum(r)
    assert s[0] == s[1] == 0.0
    assert s[2] == pytest.approx(32 * (3.0 - rb.CUSUM_K))
    assert s[3] > 0  # no reset: the statistic decays, it does not jump back to zero


def test_prf_and_best_f1_known_values():
    y = np.array([1, 1, 0, 0, 1])
    m = rb.prf(y, np.array([1, 0, 1, 0, 1]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["f1"] == pytest.approx(2 / 3)
    assert rb.prf(np.array([0, 1]), np.array([0, 0]))["f1"] == 0.0
    best = rb.best_f1(y, np.array([0.9, 0.8, 0.1, 0.2, 0.7]))
    assert best["f1"] == pytest.approx(1.0)
    assert best["threshold"] == pytest.approx(0.7)
    tie = rb.best_f1(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.1]))
    assert tie["f1"] == pytest.approx(max(2 * 1 / (2 + 1 + 0), 2 * 2 / (2 * 2 + 1)))


# ---------------------------------------------------------------- end to end


def test_evaluate_end_to_end_is_deterministic_and_consistent():
    x, y = _fixture()
    a = rb.evaluate(x, y, 32, seeds=(42, 43), time_repeats=1)
    b = rb.evaluate(x, y, 32, seeds=(42, 43), time_repeats=1)
    assert a["modes"] == b["modes"]
    assert a["iforest_seeds"] == b["iforest_seeds"]
    oes = a["modes"]
    assert oes["default_tau"]["oes_latch"]["recall"] == 1.0
    assert oes["oracle"]["oes_latch"]["f1"] == pytest.approx(1.0)
    assert a["counts"]["positive_blocks_test"] == len(ANOMALY_BLOCKS)
    assert a["checks"]["fast_equals_block_score_blocks"] == a["counts"]["blocks_used"]
    assert set(a["iforest_seeds"]) == {"42", "43"}
    assert a["reference"]["flag_all"]["recall"] == 1.0
    assert a["reference"]["prevalence"] == pytest.approx(4 / a["counts"]["blocks_test"])
    for mode in ("calibrated", "trimmed"):
        assert set(oes[mode]) == set(rb.DETECTORS)


def test_evaluate_sensitivity_block_sizes_and_skip():
    x, y = _fixture()
    for block in (16, 64):
        res = rb.evaluate(x, y, block, seeds=(42,), time_repeats=1)
        assert res["checks"] == {}  # contract checks only at 32
    with pytest.raises(rb.SkipSeries, match="warm-up blocks"):
        rb.evaluate(x[:1200], y[:1200], 64, seeds=(42,), time_repeats=1)


def _fake_rows(n=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        modes = {}
        for mode in ("calibrated", "trimmed", "oracle"):
            modes[mode] = {}
            for i, d in enumerate(rb.DETECTORS):
                f1 = float(np.clip(rng.random() * 0.5 + 0.1 * i, 0, 1))
                modes[mode][d] = {
                    "f1": f1,
                    "ap": f1,
                    "precision": f1,
                    "recall": f1,
                    "flag_rate": 0.1,
                }
        modes["default_tau"] = {"oes_latch": dict(modes["calibrated"]["oes_latch"])}
        seeds = {"42": {m: {"f1": 0.2, "ap": 0.3} for m in ("calibrated", "trimmed")}}
        rows.append({"modes": modes, "iforest_seeds": seeds, "tau_sweep_f1": [0.1] * 40})
        rows[-1]["reference"] = {"flag_all": {"f1": 0.2}, "prevalence": 0.1}
    return rows


def test_aggregate_wins_and_paired_stats():
    rows = _fake_rows()
    agg = rb.aggregate(rows)
    for mode in ("calibrated", "trimmed", "oracle"):
        assert sum(agg[mode]["wins_f1"].values()) == pytest.approx(len(rows))
    st = rb.paired_stats(rows)
    c = st["calibrated"]["f1"]["cusum"]
    lo, hi = c["bootstrap_95ci_mean_diff"]
    assert lo <= c["mean_diff"] <= hi
    assert 0.0 <= c["wilcoxon_p"] <= c["wilcoxon_p_holm"] <= 1.0
    assert rb.iforest_seed_summary(rows, (42,))["calibrated"]["mean_f1_range"] == pytest.approx(
        [0.2, 0.2]
    )


def test_holm_and_ties():
    adj = rb.holm({"a": 0.01, "b": 0.04, "c": None})
    assert adj == {"a": 0.02, "b": 0.04, "c": None}
    wins = rb.win_counts([{"modes": {"m": {d: {"f1": 0.5} for d in rb.DETECTORS}}}], "m", "f1")
    assert all(v == pytest.approx(0.25) for v in wins.values())
    assert rb.wilcoxon_p(np.zeros(5)) is None


def test_export_nab_one_detection_per_flagged_block(tmp_path):
    t0 = datetime(2020, 1, 1)
    ts = [t0 + timedelta(minutes=5 * i) for i in range(96)]
    flags = np.array([True, False])
    e = {"ts": ts, "x": np.arange(96.0), "block": 32, "test_starts": np.array([32, 64])}
    e["flags"] = dict.fromkeys(rb.DETECTORS, flags)
    rb.export_nab({"realX/s.csv": e}, tmp_path)
    lines = (tmp_path / "oes_latch" / "realX" / "oes_latch_s.csv").read_text().splitlines()
    assert lines[0] == "timestamp,value,anomaly_score"
    hits = [i - 1 for i, ln in enumerate(lines) if ln.endswith(",1.0")]
    assert hits == [63]


def test_nab_filter_windows():
    w = {"a/x.csv": [], "b/y.csv": [["s", "e"]]}
    assert ns.filter_windows(w, ["b/y.csv"]) == {"b/y.csv": [["s", "e"]]}
    with pytest.raises(KeyError):
        ns.filter_windows(w, ["c/z.csv"])


def test_make_chart_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.make_chart import make_chart

    x, y = _fixture()
    res = rb.evaluate(x, y, 32, seeds=(42,), time_repeats=1)
    report = {
        "dataset": {"nab_commit": "0" * 40},
        "code": {"commit": "1" * 40},
        "series": {"realX/a.csv": {"blocks": {"32": res}}, "realY/b.csv": {"blocks": {"32": res}}},
    }
    out = tmp_path / "c.png"
    make_chart(report, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------- data loading / manifest


def test_manifest_pins_47_real_series():
    m = dl.MANIFEST
    assert len(m["nab_commit"]) == 40
    assert len(m["series"]) == 47
    assert {k.split("/")[0] for k in m["series"]} == set(m["folders"])
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in m["series"].values())
    assert dl.FILES[dl.LABELS_NAME][0].endswith("labels/combined_windows.json")


def test_load_series_and_labels(tmp_path):
    t0 = datetime(2020, 1, 1)
    rows = [f"{(t0 + timedelta(minutes=5 * i)).isoformat(' ')},{float(i)}" for i in range(6)]
    csv = tmp_path / "s.csv"
    csv.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    lab = tmp_path / "l.json"
    lab.write_text(
        json.dumps({"k": [["2020-01-01 00:05:00.000000", "2020-01-01 00:10:00.000000"]]}),
        encoding="utf-8",
    )
    ts, x = rb.load_series(csv)
    y = rb.sample_labels(ts, rb.load_windows(lab, "k"))
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert y.tolist() == [0, 1, 1, 0, 0, 0]


def test_load_series_rejects_non_finite(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("timestamp,value\n2020-01-01 00:00:00,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        rb.load_series(csv)


def test_verify_fails_closed_on_missing_or_bad_hash(tmp_path, monkeypatch):
    good = tmp_path / "a.csv"
    good.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    monkeypatch.setattr(dl, "FILES", {"a.csv": ("https://example.invalid/a.csv", digest)})
    dl.verify(tmp_path)
    good.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dl.verify(tmp_path)
    good.unlink()
    with pytest.raises(FileNotFoundError):
        dl.verify(tmp_path)


def test_readme_tables_match_results_json():
    """Claim hygiene: README benchmark tables are exactly what results.json renders to."""
    from benchmarks import report_tables as rt

    root = rt.ROOT
    report = json.loads((root / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert rt.splice(readme, rt.render(report)) == readme
    assert report["code"]["dirty"] is False
    for key in ("run_benchmark.py", "fast_score.py", "download_nab.py", "nab_manifest.json"):
        path = root / "benchmarks" / key
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert report["code"]["files_sha256"][f"benchmarks/{key}"] == digest, key
