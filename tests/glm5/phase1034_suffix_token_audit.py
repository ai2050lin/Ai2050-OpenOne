#!/usr/bin/env python3
"""Decode Phase1034 anchors so chat-template tokens are not semanticized."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_registry import get_model_spec
import phase1034_post_query_component_protocol as protocol


def main() -> None:
    rows = []
    for model in protocol.MODELS:
        tokenizer = AutoTokenizer.from_pretrained(
            str(get_model_spec(model).local_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        cases = protocol.read_jsonl(
            protocol.SOURCE_ROOT
            / "protocol"
            / f"cases.{model}.jsonl"
        )
        for template_index in (0, 1):
            case = next(
                row
                for row in cases
                if int(row["template_index"]) == template_index
            )
            positions = protocol.anchor_positions(case)
            anchor_rows = []
            for name, position in zip(protocol.ANCHORS, positions):
                token_id = int(case["input_ids"][position])
                anchor_rows.append(
                    {
                        "anchor": name,
                        "position": position,
                        "token_id": token_id,
                        "decoded": tokenizer.decode(
                            [token_id], skip_special_tokens=False
                        ),
                    }
                )
            rows.append(
                {
                    "model": model,
                    "template_index": template_index,
                    "suffix_length": (
                        int(case["role_spans"]["pre_output"][1])
                        - int(case["role_spans"]["query_nonce"][1])
                    ),
                    "unique_anchor_position_count": len(set(positions)),
                    "anchors": anchor_rows,
                }
            )
        del tokenizer
    output = {
        "schema_version": "phase1034_suffix_token_audit.v1",
        "phase": protocol.PHASE,
        "purpose": (
            "Identify punctuation, chat-control, and reasoning-control tokens "
            "at normalized suffix anchors before interpreting response bands."
        ),
        "rows": rows,
    }
    protocol.write_json(
        protocol.OUT_ROOT / "protocol" / "suffix_token_audit.json",
        output,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
