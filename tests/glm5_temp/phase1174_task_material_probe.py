from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.glm5 import phase1172_cross_quotient_event_time_prediction as p1172


P = 61


def candidate_functions():
    eligible = {
        "power2_outer1": lambda a, b: (a + pow(b, 2, P)) % P,
        "power3_outer7": lambda a, b: pow((a + pow(b, 3, P)) % P, 7, P),
        "power5_outer11": lambda a, b: pow((a + pow(b, 5, P)) % P, 11, P),
        "power10_outer13": lambda a, b: pow((a + pow(b, 10, P)) % P, 13, P),
        "power15_outer17": lambda a, b: pow((a + pow(b, 15, P)) % P, 17, P),
        "power30_outer19": lambda a, b: pow((a + pow(b, 30, P)) % P, 19, P),
        "power4_outer23": lambda a, b: pow((a + pow(b, 4, P)) % P, 23, P),
        "power12_outer29": lambda a, b: pow((a + pow(b, 12, P)) % P, 29, P),
        "power60_outer31": lambda a, b: pow((a + pow(b, 60, P)) % P, 31, P),
    }
    abstain = {
        "interaction_one": lambda a, b: (a * pow(b, 3, P) + 7 * a + 11 * b + 13) % P,
        "interaction_two": lambda a, b: (((a + 2) * pow((b + 3) % P, 2, P)) + 5 * a + 17) % P,
        "interaction_three": lambda a, b: (((a * a + 3) * ((b + 5) % P)) + 7 * a + 19) % P,
    }
    return eligible, abstain


def table(function):
    return np.asarray([[function(a, b) for b in range(P)] for a in range(P)], dtype=np.int64)


def signature(values):
    payload = p1172.quotient_invariant_payload(values)
    return p1172.base.digest(payload)


def relation_consistency(values, delta):
    mapping = {}
    conflicts = 0
    edges = 0
    for a in range(P):
        for b in range(P):
            source = int(values[a, b])
            target = int(values[(a + delta) % P, b])
            edges += 1
            if source in mapping and mapping[source] != target:
                conflicts += 1
            else:
                mapping[source] = target
    return {
        "edges": edges,
        "source_coverage": len(mapping) / P,
        "consistency": 1.0 - conflicts / edges,
        "injectivity": len(set(mapping.values())) / max(1, len(mapping)),
    }


def main():
    old = {task.name: p1172.quotient_signature(task.name)["digest"] for task in p1172.TASK_SPECS}
    old_by_digest = {digest: name for name, digest in old.items()}
    eligible, abstain = candidate_functions()
    rows = []
    for expected, functions in (("eligible", eligible), ("abstain", abstain)):
        for name, function in functions.items():
            values = table(function)
            digest = signature(values)
            rows.append(
                {
                    "name": name,
                    "expected": expected,
                    "quotient_digest": digest,
                    "old_collision": old_by_digest.get(digest),
                    "relations": {str(delta): relation_consistency(values, delta) for delta in (1, 2, 3)},
                }
            )
    payload = {
        "all_candidate_signatures_unique": len({row["quotient_digest"] for row in rows}) == len(rows),
        "old_collision_count": sum(row["old_collision"] is not None for row in rows),
        "rows": rows,
    }
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
