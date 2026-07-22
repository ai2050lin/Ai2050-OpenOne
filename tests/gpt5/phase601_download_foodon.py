#!/usr/bin/env python3
"""Download and verify the frozen FoodOn source used by Phase601."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase601_foodon_public_ontology/source"
SOURCE_PATH = OUT_DIR / "foodon-v2025-02-01.owl"
SOURCE_URL = (
    "https://raw.githubusercontent.com/FoodOntology/foodon/"
    "v2025-02-01/foodon.owl"
)
EXPECTED_SIZE = 40_429_965
EXPECTED_SHA256 = "1e11fc50283c6498697a7aca9606c9d914f1cda71cc5510e006d949c32df7db0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"FoodOn size mismatch: {size} != {EXPECTED_SIZE}")
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"FoodOn SHA-256 mismatch: {digest}")
    for _event, element in ET.iterparse(path, events=("end",)):
        element.clear()
    return {
        "path": str(path.relative_to(ROOT)),
        "source_url": SOURCE_URL,
        "version": "v2025-02-01",
        "size_bytes": size,
        "sha256": digest,
        "xml_well_formed": True,
    }


def download(force: bool = False) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_PATH.exists() and not force:
        return validate(SOURCE_PATH)
    fd, temporary_name = tempfile.mkstemp(prefix="foodon-", suffix=".owl", dir=OUT_DIR)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "Ai2050-OpenOne-Phase601/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        result = validate(temporary_path)
        temporary_path.replace(SOURCE_PATH)
        result["path"] = str(SOURCE_PATH.relative_to(ROOT))
        return result
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download(args.force))


if __name__ == "__main__":
    main()
