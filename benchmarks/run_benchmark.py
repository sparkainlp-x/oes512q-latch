#!/usr/bin/env python3
"""Block-level anomaly benchmark: OES-32/512 weighted latch vs standard detectors on NAB.

Protocol (identical for every detector; see README "Benchmark (REPORTED)"):

1. Each series is cut into contiguous, non-overlapping blocks of ``block`` samples. 32 is the
   OES-32 contract (16 blocks = one OES-512 frame); 16 and 64 are a sensitivity analysis.
2. Reference: median of the trailing ``baseline`` samples ending just before each block
   (strictly causal, no labels); ``baseline = min(288, floor(0.05 * n))``. Blocks without a
   full trailing baseline are dropped for all detectors.
3. Residual ``r = (x - reference) / scale``; ``scale`` = 1.4826 * MAD of the raw residuals on
   the warm-up blocks (robust-sigma units). OES latch, CUSUM and IsolationForest consume r.
4. Split: the first 15% of blocks (NAB probationary-period convention, fixed a priori) is the
   warm-up/calibration segment; the rest is the test segment.
5. Thresholds, applied identically to every detector, from warm-up scores only:
   - ``calibrated``: 0.99 quantile of the detector's warm-up block scores (for the latch this
     is the per-series self-calibrated tau);
   - ``trimmed``: same, after a shared, label-free trimming of warm-up blocks whose max |r|
     exceeds the Tukey far-out fence Q3 + 3*IQR of warm-up max |r| (IsolationForest is refit
     on the retained blocks). This targets anomalies that leak into the warm-up.
   The latch is also run at its fixed contract tau = 0.50 (``default_tau``, untuned; weights
   0.45/0.35/0.20 are never tuned). ``oracle`` = best-F1 threshold chosen on the test labels
   (TUNED; optimistic upper bound, not deployable).
6. Labels are used only for scoring: a block is positive if any sample lies in a NAB window.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.fast_score import NEUMAIER_SUM, block_scores  # noqa: E402
from oes32_hilbert_hook import TAU_DEFAULT, block_score  # noqa: E402
from oes32_hilbert_hook import N as CONTRACT_BLOCK  # noqa: E402
from oes512_hilbert_hook import N_BLOCKS as FRAME_BLOCKS  # noqa: E402
from oes512_hilbert_hook import coarse_syndrome  # noqa: E402
from oes_calibration import (  # noqa: E402
    MAD_TO_SIGMA,  # noqa: F401  (re-exported; the scale constants live in the package)
    MEANAD_TO_SIGMA,  # noqa: F401
    CalibrationError,
    calibrate_tau,
    quantile_linear,
    robust_scale,
)

SEED = 42
IF_SEEDS = (42, 43, 44, 45, 46)
BLOCK_SIZES = (16, 32, 64)
BASELINE_MAX = 288  # one day at NAB's most common 5-minute sampling
BASELINE_FRAC = 0.05
CAL_FRACTION = 0.15
CAL_QUANTILE = 0.99
MIN_CAL_BLOCKS = 4
TRIM_IQR_K = 3.0
CUSUM_K = 0.5
# The robust scale, the quantile rule and the latch calibration come from oes_calibration
# (OES latch v2 in the package), so benchmark and package share one implementation.
IFOREST_TREES = 100
N_BOOT = 10_000
TAU_SWEEP = tuple(round(0.25 * i, 2) for i in range(1, 41))  # 0.25 .. 10.00
DETECTORS = ("oes_latch", "rolling_zscore", "cusum", "isolation_forest")
BASELINES = ("rolling_zscore", "cusum", "isolation_forest")
THRESHOLD_MODES = ("calibrated", "trimmed")

EVIDENCE = {
    "calibrated": "REPORTED (warm-up q0.99 threshold per detector and series; labels unused)",
    "trimmed": "REPORTED (warm-up q0.99 after label-free Tukey trimming; labels unused)",
    "default_tau": "REPORTED (fixed contract tau=0.50; untuned)",
    "oracle": "REPORTED, TUNED on test labels (best-F1 per series; optimistic upper bound)",
    "tau_sweep": "REPORTED, TUNED on test labels (single tau for all series)",
    "block_16_64": "REPORTED sensitivity analysis; not the OES-32 contract (block = 32)",
    "reference": "REPORTED trivial references: flag-all F1; random-ranking AP = prevalence",
}


CODE_FILES = (
    "oes32_hilbert_hook.py",
    "oes_calibration.py",
    "oes512_hilbert_hook.py",
    "benchmarks/run_benchmark.py",
    "benchmarks/fast_score.py",
    "benchmarks/download_nab.py",
    "benchmarks/nab_manifest.json",
    "benchmarks/nab_score.py",
    "benchmarks/make_chart.py",
)


class SkipSeries(Exception):
    """Series/config cannot be evaluated under the protocol (reason in the message)."""


# ---------------------------------------------------------------- data loading


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_series(csv_path: Path) -> tuple[list[datetime], np.ndarray]:
    """Read a NAB ``timestamp,value`` CSV. Fails closed on non-finite values."""
    ts: list[datetime] = []
    vals: list[float] = []
    with csv_path.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        if header != ["timestamp", "value"]:
            raise ValueError(f"unexpected header {header!r}")
        for line in fh:
            if not line.strip():
                continue
            t, v = line.rsplit(",", 1)
            ts.append(_parse_ts(t))
            vals.append(float(v))
    x = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite values")
    return ts, x


def load_windows(json_path: Path, key: str) -> list[tuple[datetime, datetime]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [(_parse_ts(a), _parse_ts(b)) for a, b in data[key]]


def sample_labels(ts: Sequence[datetime], windows: Sequence[tuple[datetime, datetime]]):
    """1 for samples inside any [start, end] window (inclusive), else 0."""
    y = np.zeros(len(ts), dtype=int)
    for i, t in enumerate(ts):
        if any(a <= t <= b for a, b in windows):
            y[i] = 1
    return y


# ---------------------------------------------------------------- blocking


def baseline_len(n: int) -> int:
    return max(1, min(BASELINE_MAX, int(math.floor(BASELINE_FRAC * n))))


def make_blocks(x: np.ndarray, y: np.ndarray, block: int = CONTRACT_BLOCK, baseline: int = 288):
    """Cut into contiguous blocks and compute causal trailing-baseline statistics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    n_total = len(x) // block
    first = math.ceil(baseline / block)
    if n_total <= first:
        raise SkipSeries("series too short for the requested baseline")
    ids = np.arange(first, n_total)
    starts = ids * block
    raw = np.stack([x[s : s + block] for s in starts])
    hist = np.stack([x[s - baseline : s] for s in starts])
    return {
        "n_total_blocks": int(n_total),
        "ids": ids,
        "starts": starts,
        "raw": raw,
        "labels": np.array([int(y[s : s + block].any()) for s in starts]),
        "ref": np.median(hist, axis=1),
        "hmean": hist.mean(axis=1),
        "hstd": hist.std(axis=1),
    }


def split_masks(ids: np.ndarray, n_total_blocks: int, cal_fraction: float = CAL_FRACTION):
    n_cal = int(math.floor(cal_fraction * n_total_blocks))
    cal = ids < n_cal
    return cal, ~cal


def residual_scale(c: np.ndarray) -> tuple[float, str]:
    """Robust sigma of warm-up residuals via ``oes_calibration.robust_scale`` (label-free):
    1.4826*MAD; if MAD = 0, 1.2533 * mean absolute deviation from the median."""
    try:
        return robust_scale(np.asarray(c, dtype=float).ravel().tolist())
    except CalibrationError as exc:
        raise SkipSeries(f"degenerate warm-up scale: {exc}") from exc


def warmup_threshold(scores: np.ndarray) -> float:
    """Warm-up q0.99 threshold for any detector (``oes_calibration.quantile_linear``)."""
    return quantile_linear(np.asarray(scores, dtype=float).tolist(), CAL_QUANTILE)


def residuals(blocks: dict, cal: np.ndarray) -> tuple[np.ndarray, float, str]:
    raw_res = blocks["raw"] - blocks["ref"][:, None]
    scale, how = residual_scale(raw_res[cal].ravel())
    return raw_res / scale, scale, how


def trim_mask(r_cal: np.ndarray, k: float = TRIM_IQR_K) -> np.ndarray:
    """Label-free: keep warm-up blocks whose max |r| is within Q3 + k*IQR (Tukey far-out)."""
    peak = np.abs(r_cal).max(axis=1)
    q1, q3 = (quantile_linear(peak.tolist(), q) for q in (0.25, 0.75))
    return peak <= q3 + k * (q3 - q1)


# ---------------------------------------------------------------- detectors


def score_oes_python(r: np.ndarray) -> np.ndarray:
    """Repo's OES-32 ``block_score`` on each residual block (pure-Python contract code)."""
    return np.array([block_score([float(v) for v in row]) for row in r])


def score_oes(r: np.ndarray) -> np.ndarray:
    """Vectorised score; bit-identical to ``block_score`` at width 32 (see fast_score)."""
    return block_scores(r)


def oes512_frame_bits(r: np.ndarray, tau: float) -> np.ndarray:
    """Latch bits via the repo's OES-512 ``coarse_syndrome`` on complete 16-block frames."""
    n_frames = len(r) // FRAME_BLOCKS
    bits: list[int] = []
    for f in range(n_frames):
        frame = r[f * FRAME_BLOCKS : (f + 1) * FRAME_BLOCKS].ravel()
        bits.extend(coarse_syndrome([float(v) for v in frame], tau))
    return np.array(bits, dtype=int)


def score_zscore(blocks: dict) -> np.ndarray:
    """Rolling z-score: max_j |x_j - trailing mean| / trailing std within the block."""
    std = np.where(blocks["hstd"] > 0, blocks["hstd"], np.finfo(float).tiny)
    z = (blocks["raw"] - blocks["hmean"][:, None]) / std[:, None]
    return np.minimum(np.abs(z).max(axis=1), np.finfo(float).max)


def score_cusum(r: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    """Two-sided tabular CUSUM run continuously over the residual stream (no reset)."""
    flat = r.ravel().tolist()
    sp = sn = 0.0
    out = [0.0] * len(flat)
    for i, v in enumerate(flat):
        sp = max(0.0, sp + v - k)
        sn = max(0.0, sn - v - k)
        out[i] = sp if sp > sn else sn
    return np.asarray(out).reshape(r.shape).max(axis=1)


def fit_iforest(r_fit: np.ndarray, seed: int = SEED):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(n_estimators=IFOREST_TREES, random_state=seed).fit(r_fit)


def score_iforest(model, r: np.ndarray) -> np.ndarray:
    return -model.score_samples(r)


# ---------------------------------------------------------------- metrics


def prf(y: np.ndarray, flag: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    flag = np.asarray(flag, dtype=int)
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": p,
        "recall": rc,
        "f1": f1,
        "flag_rate": float(flag.mean()) if len(flag) else 0.0,
    }


def best_f1(y: np.ndarray, scores: np.ndarray) -> dict:
    """Oracle: threshold (among observed test scores) maximizing F1 on the test labels."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, y_sorted = scores[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]  # flag all ties at a threshold
    pos = int(y.sum())
    f1 = np.where(tp[last] > 0, 2 * tp[last] / (tp[last] + fp[last] + pos), 0.0)
    i = int(np.argmax(f1))
    thr = float(s_sorted[last][i])
    return {**prf(y, scores >= thr), "threshold": thr}


def average_precision(y: np.ndarray, scores: np.ndarray) -> float | None:
    if int(np.sum(y)) == 0:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, scores))


def _timed(fn: Callable[[], np.ndarray], n_blocks: int, repeats: int):
    best = math.inf
    out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best, best / n_blocks * 1e6


def _round(obj, nd: int = 4):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj)
        if obj != 0 and abs(obj) < 10**-nd:
            return float(f"{obj:.3g}")  # keep small p-values informative
        return round(obj, nd)
    if isinstance(obj, dict):
        return {str(k): _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.generic):
        return _round(obj.item(), nd)
    return obj


# ---------------------------------------------------------------- per-series evaluation


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    block: int = CONTRACT_BLOCK,
    seeds: Sequence[int] = IF_SEEDS,
    time_repeats: int = 3,
    baseline: int | None = None,
    keep_scores: bool = False,
) -> dict:
    """Run the full protocol on one series ``x`` with per-sample labels ``y``."""
    baseline = baseline_len(len(x)) if baseline is None else baseline
    blocks = make_blocks(x, y, block, baseline)
    cal, test = split_masks(blocks["ids"], blocks["n_total_blocks"])
    if int(cal.sum()) < MIN_CAL_BLOCKS:
        raise SkipSeries(f"only {int(cal.sum())} warm-up blocks (< {MIN_CAL_BLOCKS})")
    if not test.any():
        raise SkipSeries("empty test segment")
    r, scale, scale_how = residuals(blocks, cal)
    keep = trim_mask(r[cal])
    cal_idx = np.flatnonzero(cal)
    cal_trim = np.zeros_like(cal)
    cal_trim[cal_idx[keep]] = True
    yt = blocks["labels"][test]
    n_all = len(r)

    scores: dict[str, np.ndarray] = {}
    runtime: dict[str, float] = {}
    scores["oes_latch"], _, runtime["oes_latch"] = _timed(lambda: score_oes(r), n_all, time_repeats)
    scores["rolling_zscore"], _, runtime["rolling_zscore"] = _timed(
        lambda: score_zscore(blocks), n_all, time_repeats
    )
    scores["cusum"], _, runtime["cusum"] = _timed(lambda: score_cusum(r), n_all, time_repeats)

    checks = {}
    if block == CONTRACT_BLOCK:
        py, py_s, runtime["oes_latch_python"] = _timed(
            lambda: score_oes_python(r), n_all, time_repeats
        )
        _, fast_s, _ = _timed(lambda: score_oes(r), n_all, time_repeats)
        if not np.array_equal(py, scores["oes_latch"]):
            raise AssertionError("vectorised OES score differs from block_score")
        frame_bits = oes512_frame_bits(r, TAU_DEFAULT)
        if not np.array_equal(frame_bits, (py[: len(frame_bits)] >= TAU_DEFAULT).astype(int)):
            raise AssertionError("OES-512 coarse_syndrome disagrees with OES-32 latch bits")
        checks = {
            "fast_equals_block_score_blocks": n_all,
            "oes512_frames_checked": len(frame_bits) // FRAME_BLOCKS,
            "python_seconds": py_s,
            "fast_seconds": fast_s,
        }

    # IsolationForest: fit on warm-up (all / trimmed), several seeds; primary = seeds[0].
    if_scores: dict[int, dict[str, np.ndarray]] = {}
    for sd in seeds:
        m_all = fit_iforest(r[cal], sd)
        m_trim = fit_iforest(r[cal_trim], sd)
        if sd == seeds[0]:
            s_all, _, runtime["isolation_forest"] = _timed(
                lambda m=m_all: score_iforest(m, r), n_all, time_repeats
            )
        else:
            s_all = score_iforest(m_all, r)
        if_scores[sd] = {"calibrated": s_all, "trimmed": score_iforest(m_trim, r)}

    def det_scores(det: str, mode: str) -> np.ndarray:
        return if_scores[seeds[0]][mode] if det == "isolation_forest" else scores[det]

    def thresholded(s: np.ndarray, mask: np.ndarray) -> dict:
        thr = warmup_threshold(s[mask])
        return {**prf(yt, s[test] >= thr), "threshold": thr, "ap": average_precision(yt, s[test])}

    modes: dict[str, dict] = {}
    for mode, mask in (("calibrated", cal), ("trimmed", cal_trim)):
        modes[mode] = {det: thresholded(det_scores(det, mode), mask) for det in DETECTORS}
    modes["oracle"] = {
        det: {**best_f1(yt, det_scores(det, "calibrated")[test])}
        if yt.any()
        else {**prf(yt, np.zeros_like(yt)), "threshold": None}
        for det in DETECTORS
    }
    s_oes = scores["oes_latch"][test]
    modes["default_tau"] = {
        "oes_latch": {
            **prf(yt, s_oes >= TAU_DEFAULT),
            "threshold": TAU_DEFAULT,
            "ap": average_precision(yt, s_oes),
        }
    }
    tau_sweep = [prf(yt, s_oes >= t)["f1"] for t in TAU_SWEEP]

    latch_calibration = {}
    if block == CONTRACT_BLOCK:
        # The package API must reproduce the benchmark's self-calibrated tau exactly.
        full = calibrate_tau(blocks["raw"][cal].tolist(), blocks["ref"][cal].tolist(), CAL_QUANTILE)
        trim = calibrate_tau(
            blocks["raw"][cal_trim].tolist(),
            blocks["ref"][cal_trim].tolist(),
            CAL_QUANTILE,
            scale=scale,
            min_blocks=1,
        )
        if (full.tau, full.scale) != (modes["calibrated"]["oes_latch"]["threshold"], scale):
            raise AssertionError("calibrate_tau disagrees with the benchmark threshold")
        if trim.tau != modes["trimmed"]["oes_latch"]["threshold"]:
            raise AssertionError("calibrate_tau (trimmed) disagrees with the benchmark")
        latch_calibration = {"calibrated": full.as_dict(), "trimmed": trim.as_dict()}
        checks["calibrate_tau_matches"] = True

    seed_results = {}
    for sd, d in if_scores.items():
        seed_results[str(sd)] = {
            mode: {
                "f1": prf(yt, d[mode][test] >= warmup_threshold(d[mode][mask]))["f1"],
                "ap": average_precision(yt, d[mode][test]),
            }
            for mode, mask in (("calibrated", cal), ("trimmed", cal_trim))
        }

    out = {
        "counts": {
            "samples": len(x),
            "baseline_samples": baseline,
            "blocks_total": blocks["n_total_blocks"],
            "blocks_used": n_all,
            "blocks_warmup": int(cal.sum()),
            "blocks_warmup_after_trim": int(cal_trim.sum()),
            "blocks_test": int(test.sum()),
            "positive_blocks_warmup": int(blocks["labels"][cal].sum()),
            "positive_blocks_warmup_after_trim": int(blocks["labels"][cal_trim].sum()),
            "positive_blocks_test": int(yt.sum()),
        },
        "residual_scale": scale,
        "residual_scale_estimator": scale_how,
        # Trivial references: flag every test block; random ranking has expected AP = prevalence.
        "reference": {
            "flag_all": prf(yt, np.ones_like(yt)),
            "prevalence": float(yt.mean()),
        },
        "modes": modes,
        "iforest_seeds": seed_results,
        "tau_sweep_f1": tau_sweep,
        "runtime_us_per_block": runtime,
        "checks": checks,
        "latch_calibration": latch_calibration,
    }
    if keep_scores:
        out["_scores"] = {
            "test_starts": blocks["starts"][test],
            "flags": {
                det: det_scores(det, "calibrated")[test] >= modes["calibrated"][det]["threshold"]
                for det in DETECTORS
            },
        }
    return out


# ---------------------------------------------------------------- aggregation & statistics


def _vals(rows: list[dict], mode: str, det: str, key: str) -> np.ndarray:
    return np.array([row["modes"][mode][det][key] for row in rows], dtype=float)


def win_counts(rows: list[dict], mode: str, key: str) -> dict[str, float]:
    """Per series, the detector(s) with the highest ``key``; ties share the win equally."""
    wins = dict.fromkeys(DETECTORS, 0.0)
    for row in rows:
        v = {d: row["modes"][mode][d][key] for d in DETECTORS}
        best = max(v.values())
        top = [d for d in DETECTORS if v[d] == best]
        for d in top:
            wins[d] += 1.0 / len(top)
    return wins


def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilcoxon_p(diffs: np.ndarray) -> float | None:
    if not np.any(diffs != 0):
        return None
    from scipy.stats import wilcoxon

    return float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m = len(items)
    out: dict[str, float | None] = dict.fromkeys(pvals)
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def aggregate(rows: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for mode in ("calibrated", "trimmed", "oracle", "default_tau"):
        dets = ("oes_latch",) if mode == "default_tau" else DETECTORS
        agg[mode] = {}
        for det in dets:
            f1 = _vals(rows, mode, det, "f1")
            e = {
                "mean_f1": float(f1.mean()),
                "median_f1": float(np.median(f1)),
                "mean_precision": float(_vals(rows, mode, det, "precision").mean()),
                "mean_recall": float(_vals(rows, mode, det, "recall").mean()),
                "mean_flag_rate": float(_vals(rows, mode, det, "flag_rate").mean()),
            }
            if mode != "oracle":
                ap = _vals(rows, mode, det, "ap")
                e["mean_ap"] = float(ap.mean())
                e["median_ap"] = float(np.median(ap))
            agg[mode][det] = e
        if mode != "default_tau":
            agg[mode]["wins_f1"] = win_counts(rows, mode, "f1")
            if mode != "oracle":
                agg[mode]["wins_ap"] = win_counts(rows, mode, "ap")
    return agg


def paired_stats(rows: list[dict]) -> dict:
    """Latch vs each baseline across series: mean diff, bootstrap 95% CI, Wilcoxon, Holm."""
    out: dict[str, dict] = {}
    for mode in THRESHOLD_MODES:
        out[mode] = {}
        for key in ("f1", "ap"):
            comp = {}
            for base in BASELINES:
                d = _vals(rows, mode, "oes_latch", key) - _vals(rows, mode, base, key)
                comp[base] = {
                    "n_series": len(d),
                    "mean_diff": float(d.mean()),
                    "median_diff": float(np.median(d)),
                    "latch_better": int((d > 0).sum()),
                    "latch_worse": int((d < 0).sum()),
                    "ties": int((d == 0).sum()),
                    "bootstrap_95ci_mean_diff": bootstrap_ci(d),
                    "wilcoxon_p": wilcoxon_p(d),
                }
            adj = holm({b: comp[b]["wilcoxon_p"] for b in BASELINES})
            for b in BASELINES:
                comp[b]["wilcoxon_p_holm"] = adj[b]
            out[mode][key] = comp
    # Fixed contract tau=0.50 vs CUSUM (calibrated), F1 only.
    d = _vals(rows, "default_tau", "oes_latch", "f1") - _vals(rows, "calibrated", "cusum", "f1")
    out["default_tau_vs_cusum_calibrated_f1"] = {
        "n_series": len(d),
        "mean_diff": float(d.mean()),
        "bootstrap_95ci_mean_diff": bootstrap_ci(d),
        "wilcoxon_p": wilcoxon_p(d),
    }
    return out


def iforest_seed_summary(rows: list[dict], seeds: Sequence[int]) -> dict:
    out = {}
    for mode in THRESHOLD_MODES:
        per_seed = {}
        for sd in seeds:
            f1 = np.array([r["iforest_seeds"][str(sd)][mode]["f1"] for r in rows])
            ap = np.array([r["iforest_seeds"][str(sd)][mode]["ap"] for r in rows])
            per_seed[str(sd)] = {"mean_f1": float(f1.mean()), "mean_ap": float(ap.mean())}
        mf1 = [v["mean_f1"] for v in per_seed.values()]
        map_ = [v["mean_ap"] for v in per_seed.values()]
        out[mode] = {
            "per_seed": per_seed,
            "mean_f1_range": [min(mf1), max(mf1)],
            "mean_ap_range": [min(map_), max(map_)],
        }
    return out


def tau_sweep_summary(rows: list[dict]) -> dict:
    m = np.array([r["tau_sweep_f1"] for r in rows]).mean(axis=0)
    i = int(np.argmax(m))
    return {
        "tau": list(TAU_SWEEP),
        "mean_f1": [float(v) for v in m],
        "best_tau": TAU_SWEEP[i],
        "best_mean_f1": float(m[i]),
    }


def speed_summary(rows_by_key: dict[str, dict]) -> dict:
    py = sum(r["checks"]["python_seconds"] for r in rows_by_key.values())
    fast = sum(r["checks"]["fast_seconds"] for r in rows_by_key.values())
    n = sum(r["checks"]["fast_equals_block_score_blocks"] for r in rows_by_key.values())
    return {
        "blocks": n,
        "python_block_score_us_per_block": py / n * 1e6,
        "numpy_block_scores_us_per_block": fast / n * 1e6,
        "speedup": py / fast,
        "bit_identical_scores": True,
        "sum_algorithm": "Neumaier (CPython >= 3.12)" if NEUMAIER_SUM else "sequential",
        "note": "best of 3 per series, summed over evaluated block-32 series; machine-dependent",
    }


# ---------------------------------------------------------------- driver


def _git_commit() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    import hashlib

    files = {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        for f in CODE_FILES
        if (ROOT / f).is_file()
    }
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", *CODE_FILES)),
        "files_sha256": files,
        "note": "commit is the commit the run was made at; files_sha256 pins the exact code",
    }


def run_all(data_dir: Path, keys: Sequence[str], blocks: Sequence[int], seeds, repeats: int):
    from benchmarks import download_nab as dl

    dl.verify(data_dir, [*keys, dl.LABELS_NAME])
    per_series: dict[str, dict] = {}
    skipped: dict[str, dict] = {}
    exports: dict[str, dict] = {}
    for key in keys:
        ts, x = load_series(data_dir / key)
        y = sample_labels(ts, load_windows(data_dir / dl.LABELS_NAME, key))
        per_series[key] = {"samples": len(x), "positive_samples": int(y.sum()), "blocks": {}}
        for b in blocks:
            try:
                res = evaluate(x, y, b, seeds, repeats, keep_scores=(b == CONTRACT_BLOCK))
            except SkipSeries as exc:
                skipped.setdefault(str(b), {})[key] = str(exc)
                continue
            if b == CONTRACT_BLOCK:
                sc = res.pop("_scores")
                exports[key] = {"ts": ts, "x": x, "block": b, **sc}
            per_series[key]["blocks"][str(b)] = res
        print(f"done {key}", file=sys.stderr)
    return per_series, skipped, exports


def summarize(per_series: dict, blocks: Sequence[int], seeds) -> dict:
    summary: dict[str, dict] = {}
    for b in blocks:
        evaluated = {k: v["blocks"][str(b)] for k, v in per_series.items() if str(b) in v["blocks"]}
        scored = {k: v for k, v in evaluated.items() if v["counts"]["positive_blocks_test"] > 0}
        rows = list(scored.values())
        s = {
            "series_evaluated": len(evaluated),
            "series_scored": len(rows),
            "excluded_no_test_positives": sorted(set(evaluated) - set(scored)),
            "aggregate": aggregate(rows),
            "by_folder": {},
            "iforest_seeds": iforest_seed_summary(rows, seeds),
            "tau_sweep_tuned": tau_sweep_summary(rows),
            "reference": {
                "flag_all_mean_f1": float(
                    np.mean([r["reference"]["flag_all"]["f1"] for r in rows])
                ),
                "flag_all_median_f1": float(
                    np.median([r["reference"]["flag_all"]["f1"] for r in rows])
                ),
                "mean_prevalence_random_ap": float(
                    np.mean([r["reference"]["prevalence"] for r in rows])
                ),
            },
        }
        folders = sorted({k.split("/")[0] for k in scored})
        for fo in folders:
            fr = [v for k, v in scored.items() if k.startswith(fo + "/")]
            s["by_folder"][fo] = {
                "n": len(fr),
                "calibrated_mean_f1": {
                    d: float(_vals(fr, "calibrated", d, "f1").mean()) for d in DETECTORS
                },
                "calibrated_mean_ap": {
                    d: float(_vals(fr, "calibrated", d, "ap").mean()) for d in DETECTORS
                },
            }
        if b == CONTRACT_BLOCK:
            s["paired_stats"] = paired_stats(rows)
            s["speed"] = speed_summary(evaluated)
        summary[str(b)] = s
    return summary


def main(argv: list[str] | None = None) -> int:
    from benchmarks import download_nab as dl

    p = argparse.ArgumentParser(description="OES latch vs standard detectors on NAB real data.")
    p.add_argument("--data-dir", type=Path, default=dl.DEFAULT_DIR)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results.json")
    p.add_argument("--series", nargs="*", default=list(dl.SERIES_KEYS))
    p.add_argument("--blocks", nargs="*", type=int, default=list(BLOCK_SIZES))
    p.add_argument("--seeds", nargs="*", type=int, default=list(IF_SEEDS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--chart", type=Path, default=Path(__file__).resolve().parent / "nab_f1_ap.png")
    p.add_argument("--nab-export", type=Path, default=None, help="write NAB-format result CSVs")
    args = p.parse_args(argv)

    per_series, skipped, exports = run_all(
        args.data_dir, args.series, args.blocks, args.seeds, args.repeats
    )
    summary = summarize(per_series, args.blocks, args.seeds)

    import scipy
    import sklearn

    report = {
        "dataset": {
            "name": "Numenta Anomaly Benchmark (NAB), labelled real-data folders",
            "folders": dl.MANIFEST["folders"],
            "nab_commit": dl.NAB_COMMIT,
            "base_url": dl.NAB_RAW,
            "manifest": "benchmarks/nab_manifest.json (per-file SHA-256)",
            "license": dl.MANIFEST["license"],
            "citation": (
                "Ahmad, S., Lavin, A., Purdy, S., Agha, Z. (2017). Unsupervised real-time "
                "anomaly detection for streaming data. Neurocomputing 262, 134-147."
            ),
        },
        "code": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "block_sizes": list(args.blocks),
            "contract_block": CONTRACT_BLOCK,
            "baseline": f"min({BASELINE_MAX}, floor({BASELINE_FRAC} * n)) trailing samples",
            "reference": "median of trailing baseline samples before each block (causal)",
            "residual_scale": (
                "1.4826*MAD of raw residuals on warm-up blocks; if MAD = 0, "
                "1.2533*mean absolute deviation from the median"
            ),
            "warmup_fraction": CAL_FRACTION,
            "threshold_quantile": CAL_QUANTILE,
            "min_warmup_blocks": MIN_CAL_BLOCKS,
            "trim_rule": f"drop warm-up blocks with max|r| > Q3 + {TRIM_IQR_K}*IQR",
            "cusum_k": CUSUM_K,
            "iforest_trees": IFOREST_TREES,
            "iforest_seeds": list(args.seeds),
            "primary_seed": args.seeds[0],
            "tau_default": TAU_DEFAULT,
            "oes_weights": [0.45, 0.35, 0.20],
            "oes_weights_tuned": False,
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": SEED,
            "wilcoxon": "scipy.stats.wilcoxon two-sided, zero_method='wilcox'; Holm over 3",
            "aggregates_exclude": "series with zero positive test blocks",
        },
        "evidence": EVIDENCE,
        "skipped": skipped,
        "summary": summary,
        "series": per_series,
    }
    report = _round(report, 4)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if args.nab_export is not None:
        export_nab(exports, args.nab_export)
    if args.chart is not None:
        from benchmarks.make_chart import make_chart

        make_chart(report, args.chart)
        print(f"wrote {args.chart}")
    print_summary(report)
    return 0


def export_nab(exports: dict, out_dir: Path) -> None:
    """NAB result CSVs: anomaly_score = 1 at the last sample of each flagged test block.

    A block decision is only available once the block is complete, so the detection is
    placed on its final timestamp; all other samples get 0 (calibrated thresholds, block 32).
    """
    for det in DETECTORS:
        for key, e in exports.items():
            score = np.zeros(len(e["x"]))
            ends = e["test_starts"][e["flags"][det]] + e["block"] - 1
            score[ends] = 1.0
            folder, fname = key.split("/")
            path = out_dir / det / folder / f"{det}_{fname}"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                fh.write("timestamp,value,anomaly_score\n")
                for t, v, s in zip(e["ts"], e["x"], score, strict=True):
                    fh.write(f"{t.strftime('%Y-%m-%d %H:%M:%S')},{v!r},{s:.1f}\n")


def print_summary(report: dict) -> None:
    for b, s in report["summary"].items():
        print(f"\n== block {b}: {s['series_scored']} series scored")
        for mode, dets in s["aggregate"].items():
            for det, e in dets.items():
                if det.startswith("wins"):
                    print(f"  {mode:12s} {det}: {e}")
                    continue
                ap = f"{e['mean_ap']:.3f}/{e['median_ap']:.3f}" if "mean_ap" in e else "-"
                print(
                    f"  {mode:12s} {det:17s} F1 mean/med {e['mean_f1']:.3f}/{e['median_f1']:.3f}"
                    f"  AP {ap}"
                )


if __name__ == "__main__":
    sys.exit(main())
