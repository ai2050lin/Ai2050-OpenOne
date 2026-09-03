#!/usr/bin/env python3
"""Independent audit for C235."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C235"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    mask = np.load(OUT / "raw/token_mask.bool.npy", mmap_mode="r")
    rows = core.rows(OUT / "raw/hidden_index.jsonl")
    checks = {
        "final": final["all_checks_passed"],
        "fields": fields.shape == (640, 37, 128, 2560),
        "mask": mask.shape == (640, 128),
        "rows": len(rows) == 640,
        "both_orders": {row["order"] for row in rows} == {1, -1},
        "real_tokens": bool(np.all(mask.sum(axis=1) == np.asarray([row["length"] for row in rows]))),
        "bf16": core.load(OUT / "raw/run_metadata.json")["quantization"]["has_bf16_parameters"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1769_c235_qwen_all_layer_full_token_capture.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1769, "campaign": "C235", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
