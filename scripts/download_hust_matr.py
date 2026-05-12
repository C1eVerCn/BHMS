#!/usr/bin/env python3
"""Resume-friendly downloader for the original HUST and MATR releases."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART_SIZE = 64 * 1024 * 1024
WORKERS = 16


@dataclass(frozen=True)
class FileSpec:
    label: str
    url: str
    destination: Path
    total_size: int
    magic_prefix: bytes


FILES: tuple[FileSpec, ...] = (
    FileSpec(
        label="HUST",
        url="https://data.mendeley.com/public-files/datasets/nsc7hnsg4s/files/5ca0ac3e-d598-4d07-8dcb-879aa047e98b/file_downloaded",
        destination=PROJECT_ROOT / "data/raw/hust/hust_data.zip",
        total_size=1188136932,
        magic_prefix=b"PK\x03\x04",
    ),
    FileSpec(
        label="MATR 20170512",
        url="https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
        destination=PROJECT_ROOT / "data/raw/matr/MATR_batch_20170512.mat",
        total_size=3025320241,
        magic_prefix=b"MATLAB",
    ),
    FileSpec(
        label="MATR 20170630",
        url="https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
        destination=PROJECT_ROOT / "data/raw/matr/MATR_batch_20170630.mat",
        total_size=2007331155,
        magic_prefix=b"MATLAB",
    ),
    FileSpec(
        label="MATR 20180412",
        url="https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
        destination=PROJECT_ROOT / "data/raw/matr/MATR_batch_20180412.mat",
        total_size=3236690412,
        magic_prefix=b"MATLAB",
    ),
    FileSpec(
        label="MATR 20190124",
        url="https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download",
        destination=PROJECT_ROOT / "data/raw/matr/MATR_batch_20190124.mat",
        total_size=2601295745,
        magic_prefix=b"MATLAB",
    ),
)


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def iter_ranges(total_size: int, part_size: int = PART_SIZE) -> list[tuple[int, int]]:
    parts = []
    for start in range(0, total_size, part_size):
        end = min(start + part_size, total_size) - 1
        parts.append((start, end))
    return parts


def download_range(url: str, destination: Path, start: int, end: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    curl_cmd = [
        "curl",
        "-k",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "-r",
        f"{start}-{end}",
        "-o",
        str(destination),
        url,
    ]
    subprocess.run(curl_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def expected_part_size(start: int, end: int) -> int:
    return end - start + 1


def merge_parts(spec: FileSpec, part_paths: list[Path]) -> None:
    tmp_path = spec.destination.with_suffix(spec.destination.suffix + ".assembling")
    with tmp_path.open("wb") as handle:
        for part_path in part_paths:
            with part_path.open("rb") as part_handle:
                shutil.copyfileobj(part_handle, handle, length=8 * 1024 * 1024)
    tmp_path.replace(spec.destination)


def validate_download(spec: FileSpec) -> None:
    if not spec.destination.exists():
        raise FileNotFoundError(spec.destination)
    actual_size = spec.destination.stat().st_size
    if actual_size != spec.total_size:
        raise ValueError(f"{spec.destination} size mismatch: {actual_size} != {spec.total_size}")
    with spec.destination.open("rb") as handle:
        prefix = handle.read(len(spec.magic_prefix))
    if prefix != spec.magic_prefix:
        raise ValueError(f"{spec.destination} has unexpected file signature: {prefix!r}")


def download_file(spec: FileSpec, workers: int = WORKERS) -> None:
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    part_dir = spec.destination.parent / f".{spec.destination.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    ranges = iter_ranges(spec.total_size)
    part_paths = [part_dir / f"part-{index:04d}" for index in range(len(ranges))]

    completed_bytes = 0
    pending: list[tuple[int, int, Path]] = []
    for (start, end), part_path in zip(ranges, part_paths, strict=True):
        size = expected_part_size(start, end)
        if part_path.exists() and part_path.stat().st_size == size:
            completed_bytes += size
            continue
        pending.append((start, end, part_path))

    print(
        f"[{spec.label}] target={spec.destination} total={human_bytes(spec.total_size)} "
        f"parts={len(ranges)} resume={human_bytes(completed_bytes)}"
    )

    started_at = time.time()
    last_report_at = started_at

    def report_progress(force: bool = False) -> None:
        nonlocal last_report_at
        now = time.time()
        if not force and now - last_report_at < 2:
            return
        elapsed = max(now - started_at, 0.001)
        rate = completed_bytes / elapsed
        percent = completed_bytes / spec.total_size * 100
        remaining = spec.total_size - completed_bytes
        eta = remaining / rate if rate > 0 else math.inf
        eta_text = "n/a" if math.isinf(eta) else time.strftime("%H:%M:%S", time.gmtime(eta))
        print(
            f"[{spec.label}] {percent:5.1f}% "
            f"{human_bytes(completed_bytes)}/{human_bytes(spec.total_size)} "
            f"rate={human_bytes(rate)}/s eta={eta_text}"
        )
        last_report_at = now

    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {
                pool.submit(download_range, spec.url, part_path, start, end): (start, end, part_path)
                for start, end, part_path in pending
            }
            for future in as_completed(futures):
                start, end, part_path = futures[future]
                future.result()
                size = expected_part_size(start, end)
                actual_size = part_path.stat().st_size
                if actual_size != size:
                    raise ValueError(f"{part_path} size mismatch: {actual_size} != {size}")
                completed_bytes += size
                report_progress()

    report_progress(force=True)
    print(f"[{spec.label}] assembling {len(part_paths)} parts")
    merge_parts(spec, part_paths)
    validate_download(spec)
    shutil.rmtree(part_dir)
    elapsed = max(time.time() - started_at, 0.001)
    print(f"[{spec.label}] done in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download original HUST and MATR raw assets")
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for spec in FILES:
            download_file(spec, workers=max(1, args.workers))
    except KeyboardInterrupt:
        print("Interrupted; part files were kept for resume.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
