from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase601_foodon_public_ontology"
CASES = OUT / "phase601_registered_cases.jsonl.gz"
PROTOCOL = OUT / "phase601_frozen_protocol.json"
AUDIT = OUT / "phase601_static_audit.json"
SOURCE = OUT / "source/foodon-v2025-02-01.owl"
EXPECTED_SOURCE_SHA256 = "1e11fc50283c6498697a7aca9606c9d914f1cda71cc5510e006d949c32df7db0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cases() -> list[dict]:
    with gzip.open(CASES, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Phase601FoodOnProtocolTest(unittest.TestCase):
    def test_public_source_is_frozen(self) -> None:
        self.assertEqual(SOURCE.stat().st_size, 40_429_965)
        self.assertEqual(sha256_file(SOURCE), EXPECTED_SOURCE_SHA256)

    def test_denominator_and_hash_are_frozen(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        rows = cases()
        self.assertEqual(len(rows), 1_920)
        self.assertEqual(protocol["registered_concept_count"], 480)
        self.assertEqual(protocol["registered_case_count"], 1_920)
        self.assertEqual(protocol["cases_sha256"], sha256_file(CASES))

    def test_concepts_are_split_isolated_and_balanced(self) -> None:
        rows = cases()
        concept_splits = defaultdict(set)
        concept_surfaces = defaultdict(set)
        concepts = {}
        for row in rows:
            concept_splits[row["concept_id"]].add(row["split"])
            concept_surfaces[row["concept_id"]].add(row["surface_id"])
            concepts[row["concept_id"]] = (row["split"], row["family"], row["cluster_key"])
        self.assertEqual(len(concepts), 480)
        self.assertTrue(all(len(values) == 1 for values in concept_splits.values()))
        self.assertTrue(all(len(values) == 4 for values in concept_surfaces.values()))
        counts = Counter((split, family) for split, family, _cluster in concepts.values())
        for family in ("fruit", "nut", "root_vegetable", "seed_vegetable", "animal_food"):
            self.assertEqual(counts[("discovery", family)], 48)
            self.assertEqual(counts[("independent_confirmation", family)], 24)
            self.assertEqual(counts[("heldout", family)], 24)
        self.assertEqual(len({cluster for _split, _family, cluster in concepts.values()}), 480)

    def test_hard_negatives_and_answer_order_are_balanced(self) -> None:
        rows = cases()
        self.assertTrue(all(row["family_root_id"] != row["false_root_id"] for row in rows))
        self.assertTrue(all(row["exclusive_family_membership"] for row in rows))
        self.assertEqual(Counter(row["target_letter"] for row in rows), {"A": 960, "B": 960})
        self.assertEqual(
            Counter(row["surface_id"] for row in rows),
            {f"surface_{index}": 480 for index in range(4)},
        )
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertTrue(audit["false_root_is_not_an_ancestor"])

    def test_audit_has_nonlexical_and_depth_coverage(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertTrue(audit["source_valid"])
        self.assertEqual(audit["selected_concept_count"], 480)
        self.assertGreater(audit["nonlexical_concept_count"], 300)
        self.assertEqual(audit["depth_bucket_count"], {"deep": 60, "direct": 30, "near": 390})
        for model, ledger in audit["answer_token_ledger_by_model"].items():
            self.assertEqual(len(ledger["A"]), 1, model)
            self.assertEqual(len(ledger["B"]), 1, model)
            self.assertNotEqual(ledger["A"], ledger["B"], model)


if __name__ == "__main__":
    unittest.main()
