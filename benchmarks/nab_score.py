#!/usr/bin/env python3
"""Optional: NAB's own scorer on the benchmark's block-32 detections (standard profile etc.).

Uses the official scoring code of a local NAB checkout at the pinned commit (``--nab``) on a
sub-corpus made of the series the benchmark evaluated at block 32. Our detectors' result files
come from ``run_benchmark.py --nab-export`` (binary anomaly_score = 1 at the end of each
flagged test block, warm-up-calibrated thresholds). They are scored at threshold 0.5 with **no**
NAB threshold optimisation, i.e. without looking at labels. For context, published NAB
detectors are re-scored on the same sub-corpus with their official thresholds from
``config/thresholds.json`` (which NAB optimised on the full 58-file corpus, labels included).
Normalisation uses the null detector re-scored on the same sub-corpus.

Requires pandas (not needed for the main benchmark). Results are merged into results.json
under ``nab_score``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

OURS = ("oes_latch", "rolling_zscore", "cusum", "isolation_forest")
# NAB's normaliser splits detector names on "_", so ours get underscore-free aliases.
NAB_NAME = {
    "oes_latch": "oesLatch",
    "rolling_zscore": "rollingZscore",
    "cusum": "cusum",
    "isolation_forest": "isolationForest",
}
PUBLISHED = (
    "numenta",
    "windowedGaussian",
    "relativeEntropy",
    "knncad",
    "bayesChangePt",
    "skyline",
    "random",
)
PROFILES = ("standard", "reward_low_FP_rate", "reward_low_FN_rate")


def filter_windows(windows: dict, keys: list[str]) -> dict:
    missing = [k for k in keys if k not in windows]
    if missing:
        raise KeyError(f"no NAB windows for {missing}")
    return {k: windows[k] for k in keys}


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nab", type=Path, required=True, help="NAB checkout at the pinned commit")
    p.add_argument("--export", type=Path, required=True, help="dir from --nab-export")
    p.add_argument("--work", type=Path, required=True, help="scratch dir (will be replaced)")
    p.add_argument("--data-dir", type=Path, default=here / "data")
    p.add_argument("--results", type=Path, default=here / "results.json")
    args = p.parse_args(argv)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.nab, capture_output=True, text=True, check=True
    ).stdout.strip()
    report = json.loads(args.results.read_text(encoding="utf-8"))
    if commit != report["dataset"]["nab_commit"]:
        raise SystemExit(f"NAB checkout at {commit}, expected {report['dataset']['nab_commit']}")
    keys = sorted(k for k, v in report["series"].items() if "32" in v["blocks"])

    if args.work.exists():
        shutil.rmtree(args.work)
    data, res = args.work / "data", args.work / "results"
    for k in keys:
        (data / k).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.data_dir / k, data / k)
    windows = json.loads((args.data_dir / "combined_windows.json").read_text(encoding="utf-8"))
    (args.work / "windows.json").write_text(json.dumps(filter_windows(windows, keys), indent=1))

    thresholds = json.loads((args.nab / "config" / "thresholds.json").read_text())
    for det in OURS:
        thresholds[NAB_NAME[det]] = {pr: {"threshold": 0.5, "score": None} for pr in PROFILES}
    (args.work / "thresholds.json").write_text(json.dumps(thresholds, indent=1))

    for det in (*OURS, "null", *PUBLISHED):
        src_root = args.export / det if det in OURS else args.nab / "results" / det
        name = NAB_NAME.get(det, det)
        for k in keys:
            folder, fname = k.split("/")
            src = src_root / folder / f"{det}_{fname}"
            dst = res / name / folder / f"{name}_{fname}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    dets = ",".join((*(NAB_NAME[d] for d in OURS), "null", *PUBLISHED))
    cmd = [
        sys.executable,
        "run.py",
        "--score",
        "--normalize",
        "--skipConfirmation",
        "-d",
        dets,
        "--dataDir",
        str(data),
        "--resultsDir",
        str(res),
        "--windowsFile",
        str(args.work / "windows.json"),
        "--thresholdsFile",
        str(args.work / "thresholds.json"),
        "-n",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=args.nab, capture_output=True, text=True, check=False)
    (args.work / "nab_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        raise SystemExit(f"NAB scorer failed with exit code {proc.returncode}")
    final = json.loads((res / "final_results.json").read_text())
    report["nab_score"] = {
        "evidence": "REPORTED (NAB official scorer on a sub-corpus; see note)",
        "nab_commit": commit,
        "series": len(keys),
        "note": (
            "Sub-corpus = the series evaluated at block 32. Our detectors: binary detections "
            "at warm-up-calibrated thresholds, scored at threshold 0.5 without NAB "
            "optimisation. Published detectors: official NAB result files and thresholds "
            "(optimised by NAB on the full corpus with labels). Normalised against the null "
            "detector on the same sub-corpus. Not comparable to the official NAB scoreboard."
        ),
        "normalized_scores": {
            det: {pr: round(float(v), 2) for pr, v in final[NAB_NAME.get(det, det)].items()}
            for det in (*OURS, *PUBLISHED)
        },
    }
    args.results.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for det, v in report["nab_score"]["normalized_scores"].items():
        print(det, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
