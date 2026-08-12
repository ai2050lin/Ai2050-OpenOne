#!/usr/bin/env python3
"""Resume and verify the frozen Qwen3-14B weight snapshot sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "hf" / "Qwen3-14B"
CACHE_ROOT = MODEL_ROOT / ".cache" / "huggingface" / "download"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1118_qwen3_14b_fp16_offload_smoke"
COMMIT = "40c069824f4251a91eefaf281ebe4c544efd3e18"
BASE_URL = "https://www.modelscope.cn/models/Qwen/Qwen3-14B/resolve/master"
SHARDS = (
    ("model-00001-of-00008.safetensors", 3_841_788_544, "e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e"),
    ("model-00002-of-00008.safetensors", 3_963_750_816, "f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9"),
    ("model-00003-of-00008.safetensors", 3_963_750_880, "dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9"),
    ("model-00004-of-00008.safetensors", 3_963_750_880, "eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d"),
    ("model-00005-of-00008.safetensors", 3_963_750_880, "97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6"),
    ("model-00006-of-00008.safetensors", 3_963_750_880, "9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e"),
    ("model-00007-of-00008.safetensors", 3_963_750_880, "0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750"),
    ("model-00008-of-00008.safetensors", 1_912_371_880, "0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0"),
)
TAINTED_TAIL_HASHES = {
    "e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e",
    "f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9",
    "dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9",
    "eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d",
}


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def largest_partial(sha256: str) -> Path:
    candidates = list(CACHE_ROOT.glob(f"*.{sha256}.*.incomplete"))
    if not candidates:
        return CACHE_ROOT / f"manual.{sha256}.resume.incomplete"
    return max(candidates, key=lambda path: path.stat().st_size)


def download_shard(entry: tuple[str, int, str]) -> dict[str, object]:
    name, expected_size, expected_sha256 = entry
    destination = MODEL_ROOT / name
    if destination.exists():
        initial_size = destination.stat().st_size
        actual_sha256 = file_sha256(destination) if initial_size == expected_size else None
        if initial_size != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError(f"existing destination failed verification: {name}")
        return {
            "name": name,
            "initial_size": initial_size,
            "expected_size": expected_size,
            "actual_size": initial_size,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "passed": True,
        }

    partial = largest_partial(expected_sha256)
    initial_size = partial.stat().st_size if partial.exists() else 0
    if initial_size > expected_size:
        raise RuntimeError(f"partial exceeds expected size: {name}")
    tail = CACHE_ROOT / f"manual.{expected_sha256}.tail"
    tail_size = tail.stat().st_size if tail.exists() else 0
    if initial_size + tail_size > expected_size:
        raise RuntimeError(f"partial plus tail exceeds expected size: {name}")
    print(
        f"resume {name}: {(initial_size + tail_size) / (1024 ** 3):.3f}/"
        f"{expected_size / (1024 ** 3):.3f} GiB",
        flush=True,
    )
    attempts = 0
    while initial_size + tail_size < expected_size:
        attempts += 1
        if attempts > 20:
            raise RuntimeError(f"retry budget exhausted for {name}")
        range_start = initial_size + tail_size
        before = tail_size
        command = [
            "curl.exe",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "60",
            "--max-time",
            "1800",
            "--speed-time",
            "300",
            "--speed-limit",
            "1024",
            "--range",
            f"{range_start}-{expected_size - 1}",
            "--output",
            "-",
            f"{BASE_URL}/{name}",
        ]
        with tail.open("ab") as handle:
            completed = subprocess.run(command, stdout=handle, check=False)
        tail_size = tail.stat().st_size
        if initial_size + tail_size > expected_size:
            raise RuntimeError(f"server returned too many bytes for {name}")
        if completed.returncode != 0:
            print(
                f"retry {name}: curl={completed.returncode}, "
                f"downloaded={(tail_size - before) / (1024 ** 2):.1f} MiB",
                flush=True,
            )
        if tail_size == before:
            time.sleep(min(30, attempts * 3))

    assembled = CACHE_ROOT / f"manual.{expected_sha256}.assembling"
    with assembled.open("wb") as output:
        if initial_size:
            with partial.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        if tail_size:
            with tail.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    actual_size = assembled.stat().st_size
    actual_sha256 = file_sha256(assembled) if actual_size == expected_size else None
    passed = actual_size == expected_size and actual_sha256 == expected_sha256
    if not passed:
        raise RuntimeError(
            f"verification failed for {name}: size={actual_size}, sha256={actual_sha256}"
        )
    assembled.replace(destination)
    print(f"verified {name}", flush=True)
    return {
        "name": name,
        "initial_size": initial_size,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-tainted-first-four",
        action="store_true",
        help="Discard only the four tails exposed to a duplicate-writer incident.",
    )
    args = parser.parse_args()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.reset_tainted_first_four:
        for sha256 in TAINTED_TAIL_HASHES:
            for suffix in ("tail", "assembling"):
                target = CACHE_ROOT / f"manual.{sha256}.{suffix}"
                target.unlink(missing_ok=True)
        print("reset four tainted tail/assembly files", flush=True)
    started = time.time()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_shard, entry): entry[0] for entry in SHARDS}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row["name"]))

    core = {
        "schema_version": "phase1118_qwen3_14b_prefetch.v1",
        "phase": 1118,
        "repo_commit": COMMIT,
        "download_transport": BASE_URL,
        "identity_authority": "official Hugging Face LFS SHA-256 and byte size at the frozen commit",
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "all_checks_passed": all(bool(row["passed"]) for row in rows),
    }
    core["audit_digest"] = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = OUT_ROOT / "download" / "prefetch_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(core, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(core, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
