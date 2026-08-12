#!/usr/bin/env python3
"""Run the Phase1072 bidirectional pattern-specificity atlas in FP16."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1070_process_answer_scan as engine
import phase1072_bidirectional_pattern_protocol as protocol


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[(
            row["relation"],
            row["path_name"],
            int(row["answer_branch"]),
            int(row["lexical_branch"]),
            row["split"],
        )].append(row)
    selected = []
    for condition in protocol.RELATION_NAMES:
        for path_name in protocol.PATH_NAMES.values():
            for answer in protocol.ANSWER_BRANCHES:
                for lexical in protocol.LEXICAL_BRANCHES:
                    for split in protocol.SPLITS:
                        values = grouped[(
                            condition,
                            path_name,
                            answer,
                            lexical,
                            split,
                        )]
                        ranked = sorted(
                            values,
                            key=lambda row: hashlib.sha256(
                                (
                                    "phase1072-natural|"
                                    + str(row["record_id"])
                                ).encode("utf-8")
                            ).hexdigest(),
                        )
                        if not ranked:
                            raise RuntimeError(
                                "missing natural audit cell"
                            )
                        selected.append(ranked[0])
    expected = (
        len(protocol.RELATION_NAMES)
        * len(protocol.PATH_NAMES)
        * protocol.NATURAL_AUDIT_PER_PATH
    )
    if len(selected) != expected:
        raise RuntimeError(
            f"natural selection drift: "
            f"{len(selected)} != {expected}"
        )
    return selected


engine.protocol = protocol
engine.natural_selection = natural_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    engine.run(args.model)


if __name__ == "__main__":
    main()
