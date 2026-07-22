#!/usr/bin/env python3
"""Download and validate the official WordNet 3.0 noun taxonomy."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase602_three_track_semantics"
SOURCE_DIR = OUT_DIR / "source"
ARCHIVE_PATH = SOURCE_DIR / "WordNet-3.0.tar.gz"
EXTRACTED_DIR = SOURCE_DIR / "WordNet-3.0"
DATA_NOUN_PATH = EXTRACTED_DIR / "dict/data.noun"
INDEX_NOUN_PATH = EXTRACTED_DIR / "dict/index.noun"
LICENSE_PATH = EXTRACTED_DIR / "LICENSE"
SOURCE_URL = "https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz"
EXPECTED_SIZE = 11_537_239
EXPECTED_SHA256 = "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52"
MEMBERS = (
    "WordNet-3.0/dict/data.noun",
    "WordNet-3.0/dict/index.noun",
    "WordNet-3.0/LICENSE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path = ARCHIVE_PATH) -> dict[str, object]:
    if path.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"Unexpected WordNet archive size: {path.stat().st_size}")
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Unexpected WordNet archive SHA-256: {digest}")
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        missing = set(MEMBERS) - names
        if missing:
            raise RuntimeError(f"Missing WordNet members: {sorted(missing)}")
    return {
        "source_url": SOURCE_URL,
        "archive_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def extract_required(path: Path = ARCHIVE_PATH) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        for name in MEMBERS:
            member = archive.getmember(name)
            destination = SOURCE_DIR / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read WordNet member: {name}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            with temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            os.replace(temporary, destination)


def download() -> dict[str, object]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists():
        temporary = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".part")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "OpenOne-Phase602/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(temporary, ARCHIVE_PATH)
    result = validate_archive()
    extract_required()
    result["extracted_files"] = {
        "data.noun": {"size_bytes": DATA_NOUN_PATH.stat().st_size, "sha256": sha256_file(DATA_NOUN_PATH)},
        "index.noun": {"size_bytes": INDEX_NOUN_PATH.stat().st_size, "sha256": sha256_file(INDEX_NOUN_PATH)},
        "LICENSE": {"size_bytes": LICENSE_PATH.stat().st_size, "sha256": sha256_file(LICENSE_PATH)},
    }
    return result


if __name__ == "__main__":
    print(json.dumps(download(), indent=2, sort_keys=True))
