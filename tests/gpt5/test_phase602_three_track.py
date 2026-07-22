from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase602_three_track_semantics"
CASES = OUT / "phase602_registered_cases.jsonl.gz"
PROTOCOL = OUT / "phase602_frozen_protocol.json"
AUDIT = OUT / "phase602_static_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rows() -> list[dict]:
    with gzip.open(CASES, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Phase602ThreeTrackProtocolTest(unittest.TestCase):
    def test_sources_and_denominator_are_frozen(self) -> None:
        frozen = json.loads(PROTOCOL.read_text())
        self.assertEqual(frozen["wordnet_archive_sha256"], "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52")
        self.assertEqual(frozen["registered_concept_count"], 120)
        self.assertEqual(frozen["registered_case_count"], 1_440)
        self.assertEqual(frozen["cases_sha256"], sha256_file(CASES))

    def test_same_objects_are_matched_across_tracks(self) -> None:
        cases = rows()
        by_concept = defaultdict(list)
        for row in cases:
            by_concept[row["concept_id"]].append(row)
        self.assertEqual(len(by_concept), 120)
        for values in by_concept.values():
            self.assertEqual(len(values), 12)
            self.assertEqual({row["track"] for row in values}, {"technical", "daily", "explicit_evidence"})
            self.assertEqual(len({row["split"] for row in values}), 1)
            self.assertTrue(all(row["foodon_wordnet_binary_agreement"] for row in values))

    def test_balance_novelty_and_roles(self) -> None:
        cases = rows()
        concepts = {row["concept_id"]: row for row in cases}
        self.assertEqual(Counter(row["fruit_member"] for row in concepts.values()), {True: 60, False: 60})
        self.assertEqual(Counter(row["entity_role"] for row in concepts.values()), {
            "raw_fruit": 60, "seed_vegetable": 10, "meat": 25, "dairy": 21, "seafood": 4,
        })
        self.assertEqual(Counter(row["target_letter"] for row in cases), {"A": 720, "B": 720})
        audit = json.loads(AUDIT.read_text())
        self.assertTrue(audit["all_selected_clusters_novel_to_phase601"])
        self.assertTrue(audit["selected_cluster_unique"])
        self.assertFalse(audit["full_five_family_matched_panel_feasible"])


if __name__ == "__main__":
    unittest.main()
