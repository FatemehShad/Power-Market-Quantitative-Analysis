"""Download the public Kaggle archive, with TLS verification and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

DATASET = "lakshmi2305/european-wholesale-electricity-prices"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
MAX_BYTES = 512 * 1024 * 1024


def download(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        size = 0
        with requests.get(URL, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("Archive exceeds the 512 MiB safety limit.")
                archive.write(chunk)
                digest.update(chunk)
        archive.seek(0)
        if not zipfile.is_zipfile(archive):
            raise ValueError("Kaggle returned no ZIP archive. Download the CSV manually from the dataset page.")
        archive.seek(0)
        with zipfile.ZipFile(archive) as bundle:
            # The archive also contains a >500 MiB JSON copy; CSVs suffice.
            files = [item for item in bundle.infolist() if not item.is_dir() and item.filename.lower().endswith(".csv")]
            if not files:
                raise ValueError("Archive contains no CSV files.")
            if sum(item.file_size for item in files) > MAX_BYTES:
                raise ValueError("Unpacked archive exceeds the 512 MiB safety limit.")
            targets = []
            for item in files:
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path.")
                if target.exists():
                    raise FileExistsError(f"Will not overwrite {target}; choose a fresh output directory.")
                targets.append(target)
            manifest = destination / "download_manifest.json"
            if manifest.exists() or manifest.resolve() in targets:
                raise FileExistsError("Manifest already exists; choose a fresh output directory.")
            for item, target in zip(files, targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as reader, target.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
            manifest.write_text(json.dumps({
                "dataset": DATASET, "url": URL,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive_sha256": digest.hexdigest(),
                "files": [{"path": str(path.relative_to(destination.resolve())),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in targets],
                "license": "Check the Kaggle dataset page; data is not covered by the code license.",
            }, indent=2) + "\n")
    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    try:
        for path in download(args.output):
            print(path)
    except (requests.RequestException, ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Download failed: {exc}\nYou can instead download from https://www.kaggle.com/datasets/{DATASET} and import the CSV.\n")
