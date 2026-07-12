from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase364_projection_sufficiency_audit/offline_projection_audit"
P361 = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace/phase361_frozen_predictive_candidates.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase364ProjectionSufficiencyAuditTests(unittest.TestCase):
    def test_projection_denominator_and_candidate_freeze(self) -> None:
        summary = json.loads((OUT / "phase364_projection_audit_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["projection_matrix_count"], 9)
        self.assertEqual(summary["denominator"]["candidate_feature_count"], 7)
        self.assertEqual(summary["frozen_inputs"]["phase361_candidate_sha256"], hashlib.sha256(P361.read_bytes()).hexdigest())
        self.assertEqual(summary["frozen_inputs"]["new_model_execution_count"], 0)
        self.assertFalse(summary["frozen_inputs"]["physical_confirmation_read"])

    def test_p0_is_lossy_without_overclaiming_total_impossibility(self) -> None:
        summary = json.loads((OUT / "phase364_projection_audit_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["results"]["p0_structurally_noninjective"])
        self.assertGreater(summary["results"]["p0_time_zero_all_case_duplicate_pair_record_count"], 0)
        self.assertFalse(summary["claim_boundary"]["p0_is_proven_sufficient_state"])
        self.assertFalse(summary["claim_boundary"]["p0_is_proven_insufficient_for_every_possible_nonlinear_mapping"])

    def test_anchor_capabilities_are_exactly_bounded(self) -> None:
        rows = read_jsonl(OUT / "phase364_anchor_capability_rows.jsonl")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["attention_source_edge_offline_replayable"] for row in rows))
        self.assertTrue(all(row["mlp_shard_write_offline_replayable"] for row in rows))
        self.assertFalse(any(row["mlp_single_neuron_write_self_contained"] for row in rows))
        self.assertFalse(any(row["dynamic_flow_bundle_schema_present"] for row in rows))

    def test_next_execution_remains_blocked_and_frontend_is_summary_only(self) -> None:
        protocol = json.loads((OUT / "phase365_instrumentation_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["phase365_fixed_engineering_denominator"]["total_case_count"], 96)
        self.assertFalse(protocol["new_model_execution_authorized"])
        public = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
        manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["phase364"]["status"], "p0_lossy_skeleton_p2_p3_instrumentation_incomplete")
        self.assertFalse(manifest["phase364"]["raw_tensors_frontend_exported"])


if __name__ == "__main__":
    unittest.main()
