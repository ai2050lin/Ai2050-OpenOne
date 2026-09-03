#!/usr/bin/env python3
"""C332: capture every Qwen graph token, checkpoint and coordinate."""
from __future__ import annotations

from transformers import AutoTokenizer
import torch

import phase1844_c310_c335_dual_axis_common as common
from model_utils import MODEL_CONFIGS


def main() -> None:
    parent = common.core.load(common.OUTS["C331"] / "analysis/final.json")
    rows = common.core.rows(common.OUTS["C331"] / "material/cases.jsonl")
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 384, "cuda": torch.cuda.is_available(), "all_coordinate_policy": True}
    protocol = {
        "status": "qwen_graph_full_field_frozen",
        "model": "Qwen3-4B BF16 CUDA unquantized",
        "interface": "strict_chat frozen before behavior",
        "archive": "all 384 prompts x embedding + 36 block outputs + final norm x all 144 token slots x all 2560 coordinates; six semantic-role means are also archived",
        "behavior_gate": common.core.load(common.OUTS["C331"] / "protocol/preregistration.json")["behavior_gate"],
        "claim_boundary": "The archive is a complete activation observation under this prompt compiler; it is not a parameter trace or an attention/MLP circuit.",
    }
    out = common.prepare("C332", protocol, checks)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    compiled = common.compile_general(tokenizer, rows, "strict_chat")
    common.core.write_rows(out / "compiled/qwen3.jsonl", compiled)
    capture = common.batch_capture_qwen(rows, compiled, out, full_selector=lambda _row: True, batch_size=8, field_width=144)
    behavior = common.core.rows(out / "raw/behavior.jsonl")
    source = {r["case_id"]: r for r in rows}
    accuracy = sum(r["correct"] for r in behavior) / len(behavior)
    by_depth = {str(d): sum(r["correct"] for r in behavior if source[r["case_id"]]["depth"] == d) / sum(1 for r in behavior if source[r["case_id"]]["depth"] == d) for d in range(1, 5)}
    by_surface = {s: sum(r["correct"] for r in behavior if source[r["case_id"]]["surface"] == s) / sum(1 for r in behavior if source[r["case_id"]]["surface"] == s) for s in ("registry", "briefing")}
    gate = protocol["behavior_gate"]
    eligible = accuracy >= gate["global_min"] and min(by_depth.values()) >= gate["depth_min"] and min(by_surface.values()) >= gate["surface_min"]
    headline = {"status": "qwen_graph_full_field_closed", "accuracy": accuracy, "by_depth_accuracy": by_depth, "by_surface_accuracy": by_surface, "behavior_eligible": eligible, **capture, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C332", headline, {"behavior_rows": len(behavior) == 384, "role_shape": capture["role_shape"] == [384, 38, 6, 2560], "full_shape": capture["full_shape"] == [384, 38, 144, 2560], "finite": common.finite_dict(headline)}, "C333_graph_depth_operator_atlas")


if __name__ == "__main__":
    main()
