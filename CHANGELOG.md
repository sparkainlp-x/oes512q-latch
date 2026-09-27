# Changelog

All notable changes to this project. Evidence tags follow the README (SYNTHETIC / REPORTED / TARGET / UNRUN).

## Unreleased

### Changed
- `.zenodo.json` adds the `spark-ai-nlp` Zenodo community, so future releases are archived there.

## 0.2.0 — 2026-09-27

### Added
- **OES latch v2: weighted score + self-calibrated τ** (`oes_calibration.py`, standard library only). It is an optional extension here, not part of the normative ADR-001 residual.
  - `calibrate_tau(warmup_blocks, reference=None, quantile=0.99, scale="robust", min_blocks=4)` returns a frozen `TauCalibration` (`tau`, `scale`, `quantile`, `n_blocks`, `method`, `scale_method`).
  - `calibrated_block_score`, `calibrated_latch_bit`, `calibrated_coarse_syndrome`.
  - Helpers `robust_scale` (1.4826·MAD, with a mean-absolute-deviation fallback) and `quantile_linear` (Hyndman–Fan type 7, bit-identical to numpy's default `np.quantile`).
  - Fails closed on too few warm-up blocks, non-finite values, zero spread, a non-finite or negative τ, and a quantile outside (0, 1). Labels are never an input.
- `tests/test_calibration.py`: edge cases, determinism, numpy equivalence, and exact equivalence with the benchmark's per-series τ; a check that labels cannot influence calibration.
- `CHANGELOG.md`.

### Changed
- The benchmark (`benchmarks/run_benchmark.py`) now takes its robust scale, warm-up quantile thresholds and the latch calibration from `oes_calibration`, so the package and the benchmark share one implementation. Every block-32 series asserts that `calibrate_tau` reproduces the benchmark threshold exactly. Every stored benchmark number is unchanged. The only floating-point difference is one mean-absolute-deviation fallback scale, 1 ulp apart (`math.fsum` vs numpy's pairwise mean; `ec2_cpu_utilization_24ae8d`, block 64, sensitivity only), with no effect on any reported metric.
- Version 0.2.0 (`pyproject.toml`, `CITATION.cff`).

### Unchanged
- The fixed default τ = 0.50, the weights 0.45/0.35/0.20, the S ≥ τ rule (equality latches), the 16 × 32 layout and every existing public function and test.

## 0.1.0

- OES-32 / OES-512 classical latch, optional amplitude encode, block observables, tests, CI.
- Benchmarks: single NAB series, then all 47 NAB real-data series.

Versions up to 0.2.0 were developed in a private archive. This public repository (`oes512q-latch`) starts from a single clean snapshot of 0.2.0.
