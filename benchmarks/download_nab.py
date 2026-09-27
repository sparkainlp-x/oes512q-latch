#!/usr/bin/env python3
"""Download the pinned NAB real-data series + labels used by the benchmark; verify SHA-256.

Dataset: Numenta Anomaly Benchmark (NAB), all series in the labelled real-data folders
realAWSCloudwatch, realAdExchange, realKnownCause, realTraffic and realTweets (47 files), plus
labels/combined_windows.json, pinned to the NAB commit in ``nab_manifest.json`` (MIT license).
Files are written to ``benchmarks/data/<folder>/<file>.csv`` (git-ignored). Standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "nab_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
NAB_COMMIT = MANIFEST["nab_commit"]
NAB_RAW = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}"
LABELS_NAME = "combined_windows.json"
SERIES_KEYS = tuple(sorted(MANIFEST["series"]))


def _build_files(manifest: dict) -> dict[str, tuple[str, str]]:
    """Map local relative path -> (pinned URL, expected SHA-256)."""
    files = {key: (f"{NAB_RAW}/data/{key}", sha) for key, sha in sorted(manifest["series"].items())}
    files[LABELS_NAME] = (f"{NAB_RAW}/{manifest['labels']['path']}", manifest["labels"]["sha256"])
    return files


FILES = _build_files(MANIFEST)
DEFAULT_DIR = HERE / "data"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data_dir: Path, names: list[str] | None = None) -> None:
    """Fail closed: raise if any file is missing or its SHA-256 does not match."""
    for name in names if names is not None else list(FILES):
        _, expected = FILES[name]
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run: python benchmarks/download_nab.py")
        got = sha256_file(path)
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: got {got}, expected {expected}")


def download(data_dir: Path = DEFAULT_DIR) -> None:
    for name, (url, expected) in FILES.items():
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and sha256_file(path) == expected:
            print(f"ok (cached) {name}")
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (pinned https URL)
            tmp.write_bytes(resp.read())
        got = sha256_file(tmp)
        if got != expected:
            tmp.unlink()
            raise ValueError(f"SHA-256 mismatch for {url}: got {got}, expected {expected}")
        tmp.replace(path)
        print(f"ok {name} sha256={got}")
    verify(data_dir)
    print(f"verified {len(FILES)} files at NAB commit {NAB_COMMIT}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args(argv)
    download(args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
