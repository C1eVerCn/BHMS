#!/usr/bin/env python3
"""Attempt to fetch external lifecycle datasets and always leave audit-ready status notes."""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings


SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


@dataclass
class DownloadSpec:
    url: str
    filename: str
    required: bool = True


@dataclass
class DatasetSpec:
    source: str
    license_name: str
    landing_url: str
    description: str
    expected_files: list[str]
    validate_commands: list[str]
    downloads: list[DownloadSpec] = field(default_factory=list)
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "hust": DatasetSpec(
        source="hust",
        license_name="Mendeley Data original HUST release",
        landing_url="https://data.mendeley.com/datasets/nsc7hnsg4s/2",
        description="HUST 77-cell lifecycle benchmark used for main lifecycle training.",
        downloads=[
            DownloadSpec(
                "https://data.mendeley.com/public-files/datasets/nsc7hnsg4s/files/5ca0ac3e-d598-4d07-8dcb-879aa047e98b/file_downloaded",
                "hust_data.zip",
            ),
        ],
        expected_files=["hust_data.zip", "extracted HUST raw files"],
        validate_commands=[
            "ls -lh data/raw/hust",
            "unzip -l data/raw/hust/hust_data.zip | head",
        ],
        note="Original Mendeley file is used because the Zenodo processed mirror blocked automated access in this environment.",
    ),
    "matr": DatasetSpec(
        source="matr",
        license_name="MATR original release via data.matr.io",
        landing_url="https://data.matr.io/",
        description="MATR fast-charging cycle life benchmark used for main lifecycle training.",
        downloads=[
            DownloadSpec(
                "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
                "MATR_batch_20170512.mat",
            ),
            DownloadSpec(
                "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
                "MATR_batch_20170630.mat",
            ),
            DownloadSpec(
                "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
                "MATR_batch_20180412.mat",
            ),
            DownloadSpec(
                "https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download",
                "MATR_batch_20190124.mat",
            ),
        ],
        expected_files=[
            "MATR_batch_20170512.mat",
            "MATR_batch_20170630.mat",
            "MATR_batch_20180412.mat",
            "MATR_batch_20190124.mat",
        ],
        validate_commands=[
            "ls -lh data/raw/matr",
            "file data/raw/matr/MATR_batch_20170512.mat",
        ],
        note="Original MATR batch MAT files are used because the Zenodo processed mirror blocked automated access in this environment.",
    ),
    "oxford": DatasetSpec(
        source="oxford",
        license_name="Oxford ORA terms of use (ODbL-linked page on ORA)",
        landing_url="https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac",
        description="Oxford Battery Degradation Dataset 1 used for trajectory auxiliary study.",
        downloads=[
            DownloadSpec(
                "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m5ac36a1e2073852e4f1f7dee647909a7",
                "Oxford_Battery_Degradation_Dataset_1.mat",
            ),
            DownloadSpec(
                "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m43cc05e7c5f1245f4895d9dbd495e52f",
                "Readme.txt",
                required=False,
            ),
        ],
        expected_files=["Oxford_Battery_Degradation_Dataset_1.mat", "Readme.txt"],
        validate_commands=[
            "ls -lh data/raw/oxford",
            "file data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat",
        ],
        note="The current BHMS Oxford adapter is CSV-first; if MAT conversion is still needed, keep the raw MAT here and add a conversion step before import.",
    ),
    "pulsebat": DatasetSpec(
        source="pulsebat",
        license_name="GitHub repository terms / see upstream repository",
        landing_url="https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation",
        description="Pulse-driven diagnostic repository used for anomaly / GraphRAG enhancement assets.",
        downloads=[
            DownloadSpec(
                "https://codeload.github.com/terencetaothucb/Pulse-Voltage-Response-Generation/zip/refs/heads/main",
                "Pulse-Voltage-Response-Generation-main.zip",
            ),
        ],
        expected_files=["Pulse-Voltage-Response-Generation-main.zip or extracted repository contents"],
        validate_commands=[
            "ls -lh data/raw/pulsebat",
            "unzip -l data/raw/pulsebat/Pulse-Voltage-Response-Generation-main.zip | head",
        ],
        note="Repository archive is fetched as the raw source drop. Extract it here if you want to inspect or adapt the pulse-response assets.",
    ),
}


def download_file(url: str, destination: Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    request = Request(url, headers=headers)
    with urlopen(request, context=SSL_CONTEXT, timeout=60) as response:
        response_status = getattr(response, "status", None)
        mode = "ab" if existing_size > 0 and response_status == 206 else "wb"
        if mode == "wb" and existing_size > 0:
            existing_size = 0
        with destination.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "resumed_from": existing_size,
    }


def probe_remote_size(url: str) -> int | None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
    try:
        with urlopen(request, context=SSL_CONTEXT, timeout=30) as response:
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                try:
                    return int(content_range.rsplit("/", 1)[-1])
                except ValueError:
                    pass
            length = response.headers.get("Content-Length")
    except Exception:
        return None
    if not length:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def write_status_note(target_dir: Path, spec: DatasetSpec, payload: dict[str, Any]) -> None:
    lines = [
        f"# {spec.source.upper()} acquisition status",
        "",
        f"- source: {spec.source}",
        f"- description: {spec.description}",
        f"- license: {spec.license_name}",
        f"- landing_url: {spec.landing_url}",
        f"- status: {payload['status']}",
        f"- note: {spec.note or '--'}",
        "",
        "## expected_files",
    ]
    lines.extend(f"- {item}" for item in spec.expected_files)
    lines.extend(["", "## attempted_downloads"])
    for item in payload.get("attempted_downloads", []):
        lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
    lines.extend(["", "## validate_commands"])
    lines.extend(f"- {command}" for command in spec.validate_commands)
    (target_dir / "DATASET_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    (target_dir / "dataset_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def acquire_source(spec: DatasetSpec, raw_root: Path) -> dict[str, Any]:
    target_dir = raw_root / spec.source
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source": spec.source,
        "status": "ready" if any(target_dir.iterdir()) else "pending",
        "landing_url": spec.landing_url,
        "license": spec.license_name,
        "expected_files": spec.expected_files,
        "attempted_downloads": [],
    }
    succeeded = False
    for download in spec.downloads:
        destination = target_dir / download.filename
        remote_size = probe_remote_size(download.url)
        if destination.exists() and destination.stat().st_size > 0 and remote_size is not None and destination.stat().st_size >= remote_size:
            payload["attempted_downloads"].append(
                {
                    "url": download.url,
                    "path": str(destination),
                    "status": "skipped_existing_complete",
                    "size_bytes": destination.stat().st_size,
                    "remote_size_bytes": remote_size,
                }
            )
            succeeded = True
            continue
        try:
            result = download_file(download.url, destination)
            payload["attempted_downloads"].append(
                {
                    "url": download.url,
                    "path": result["path"],
                    "status": "downloaded",
                    "size_bytes": result["size_bytes"],
                    "resumed_from": result["resumed_from"],
                    "remote_size_bytes": remote_size,
                }
            )
            succeeded = True
        except HTTPError as exc:
            if exc.code == 416 and destination.exists() and destination.stat().st_size > 0:
                payload["attempted_downloads"].append(
                    {
                        "url": download.url,
                        "path": str(destination),
                        "status": "skipped_existing_complete",
                        "size_bytes": destination.stat().st_size,
                    }
                )
                succeeded = True
                continue
            payload["attempted_downloads"].append(
                {
                    "url": download.url,
                    "path": str(destination),
                    "status": "blocked_http",
                    "http_status": exc.code,
                    "reason": str(exc),
                }
            )
        except URLError as exc:
            payload["attempted_downloads"].append(
                {
                    "url": download.url,
                    "path": str(destination),
                    "status": "network_error",
                    "reason": str(exc.reason),
                }
            )
        except Exception as exc:  # noqa: BLE001 - status note should keep the exact failure
            payload["attempted_downloads"].append(
                {
                    "url": download.url,
                    "path": str(destination),
                    "status": "failed",
                    "reason": str(exc),
                }
            )
    if succeeded:
        payload["status"] = "downloaded_or_present"
    elif spec.downloads:
        payload["status"] = "external_blocked"
    write_status_note(target_dir, spec, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or stage external BHMS lifecycle datasets")
    parser.add_argument("--source", choices=[*DATASETS.keys(), "all"], default="all")
    args = parser.parse_args()

    settings = get_settings()
    raw_root = settings.data_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    selected = DATASETS.values() if args.source == "all" else [DATASETS[args.source]]
    results = [acquire_source(spec, raw_root) for spec in selected]
    print(json.dumps({"generated_at": "now", "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
