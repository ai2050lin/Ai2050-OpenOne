"""Regression tests for the client research accumulation database and API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.research_workspace_service import (
    ClosureApplicationInput,
    ClosureGatePatch,
    ConstructionInput,
    FieldRecordInput,
    GearCandidateInput,
    InterventionInput,
    LanguageEdgeInput,
    LanguageNodeInput,
    LanguageObjectInput,
    LanguageOperationInput,
    PairAlignmentInput,
    ProbeResponseInput,
    ResearchCaseInput,
    ResearchWorkspaceStore,
    TheoryClaimInput,
    get_research_workspace_store,
    router,
)


class ResearchWorkspaceStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResearchWorkspaceStore(Path(self.temp.name) / "research_workspace.sqlite3")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_seed_snapshot_is_explicitly_untested(self) -> None:
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["schema_version"], "research_workspace.v2")
        self.assertEqual(snapshot["overview"]["language_object_count"], 10)
        self.assertEqual(snapshot["overview"]["operation_count"], 6)
        self.assertEqual(snapshot["overview"]["field_record_count"], 0)
        self.assertEqual(snapshot["overview"]["passed_gate_count"], 0)
        self.assertTrue(all(row["evidence_level"] == "E0" for row in snapshot["language_objects"]))
        self.assertTrue(all(row["behavior_status"] == "untested" for row in snapshot["operations"]))

    def test_typed_graph_operation_case_and_pair_roundtrip(self) -> None:
        form_node = self.store.create_language_node(
            LanguageNodeInput(node_type="form", label="苹果", normalized_form="苹果", language="zh")
        )
        concept_node = self.store.create_language_node(
            LanguageNodeInput(node_type="concept", label="水果", normalized_form="fruit", language="und")
        )
        edge = self.store.create_language_edge(
            LanguageEdgeInput(
                source_node_id=form_node["id"],
                target_node_id=concept_node["id"],
                relation="denotes",
                condition="中文名词语境",
            )
        )
        operation = self.store.create_operation(
            LanguageOperationInput(
                family_type="custom_taxonomy",
                label="实例到类别",
                invariants=["对象身份"],
                changed_factors=["抽象层级"],
                next_evidence_gap="建立未见实例锁箱测试",
            )
        )
        construction = self.store.create_construction(
            ConstructionInput(
                family="classification",
                label="X 是一种 Y",
                typed_slots=[{"name": "X", "type": "concept"}, {"name": "Y", "type": "category"}],
                operation_ids=[operation["id"]],
                surfaces=["苹果是一种水果"],
            )
        )
        baseline = self.store.create_case(
            ResearchCaseInput(
                operation_id=operation["id"],
                construction_id=construction["id"],
                label="苹果基线",
                input_text="苹果是一种水果。",
                invariants=["对象身份"],
                split="test",
            )
        )
        variant = self.store.create_case(
            ResearchCaseInput(
                operation_id=operation["id"],
                construction_id=construction["id"],
                label="梨变体",
                input_text="梨是一种水果。",
                changed_factors=["对象词面"],
                split="lockbox",
            )
        )
        pair = self.store.create_pair_alignment(
            PairAlignmentInput(
                operation_id=operation["id"],
                baseline_case_id=baseline["id"],
                variant_case_id=variant["id"],
                token_alignment={"苹果": "梨"},
                role_alignment={"subject": "subject"},
                status="aligned",
            )
        )
        snapshot = self.store.snapshot()
        self.assertEqual(edge["condition"], "中文名词语境")
        self.assertEqual(pair["token_alignment"], {"苹果": "梨"})
        self.assertEqual(snapshot["overview"]["language_node_count"], 2)
        self.assertEqual(snapshot["overview"]["case_count"], 2)
        self.assertEqual(snapshot["overview"]["pair_count"], 1)

    def test_full_field_record_preserves_dimensions_and_artifact_links(self) -> None:
        language_object = self.store.create_language_object(
            LanguageObjectInput(
                object_type="token",
                family="fruit",
                label="苹果",
                normalized_form="apple",
                status="collecting",
                sample_count=128,
            )
        )
        field = self.store.create_field_record(
            FieldRecordInput(
                language_object_id=language_object["id"],
                model_id="qwen3",
                case_id="fruit_context_001",
                run_id="run_full_field_001",
                token_count=32,
                layer_count=36,
                hidden_size=2560,
                embedding_parameter_count=81920,
                hiddenstate_parameter_count=2949120,
                embedding_artifact="tests/glm5/result/example/embedding.safetensors",
                hiddenstate_artifact="tests/glm5/result/example/hiddenstates.safetensors",
                coverage_scope="full",
                status="validated",
                evidence_level="E2",
            )
        )
        snapshot = self.store.snapshot()
        self.assertEqual(field["hiddenstate_parameter_count"], 2_949_120)
        self.assertEqual(snapshot["overview"]["full_field_count"], 1)
        self.assertEqual(snapshot["field_records"][0]["language_object_label"], "苹果")

    def test_theory_and_closure_updates_are_persisted(self) -> None:
        claim = self.store.create_claim(
            TheoryClaimInput(
                title="成对条件响应",
                statement="语境改变应产生可复查的逐层响应差分。",
                status="hypothesis",
                next_test="运行留出构式对照。",
            )
        )
        self.assertTrue(claim["id"].startswith("claim_"))
        gate = self.store.update_closure_gate(
            "gate_language_coverage",
            ClosureGatePatch(status="in_progress", evidence_count=12),
        )
        self.assertEqual(gate["status"], "in_progress")
        self.assertEqual(gate["evidence_count"], 12)

    def test_probe_gear_intervention_and_closure_request_are_auditable(self) -> None:
        operation = self.store.create_operation(
            LanguageOperationInput(family_type="causal_test", label="否定调用测试")
        )
        probe = self.store.create_probe_response(
            ProbeResponseInput(
                operation_id=operation["id"],
                run_id="run_probe_001",
                source_checkpoint="layer_8",
                target_checkpoint="layer_16",
                source_coordinate=101,
                target_coordinate=202,
                dose=0.25,
                response_sign=-1,
                response_amplitude=0.42,
                output_effect=-0.18,
                artifact_path="tests/glm5/result/probe_001/responses.safetensors",
            )
        )
        gear = self.store.create_gear_candidate(
            GearCandidateInput(
                operation_id=operation["id"],
                label="否定范围齿轮候选",
                condition_domain="单层否定句",
                source_nodes=[{"layer": 8, "coordinate": 101}],
                target_nodes=[{"layer": 16, "coordinate": 202}],
                sign_structure="negative",
            )
        )
        intervention = self.store.create_intervention(
            InterventionInput(
                operation_id=operation["id"],
                gear_candidate_id=gear["id"],
                run_id="run_delete_001",
                intervention_type="delete",
                target={"layer": 8, "coordinate": 101},
                dose=1.0,
                decision="inconclusive",
            )
        )
        request = self.store.create_closure_application(
            ClosureApplicationInput(
                gate_id="gate_causal",
                rationale="请求审核 Probe 与删除检验是否满足因果门。",
                evidence_ids=[probe["id"], intervention["id"]],
            )
        )
        snapshot = self.store.snapshot()
        causal_gate = next(item for item in snapshot["closure_gates"] if item["id"] == "gate_causal")
        self.assertEqual(request["review_status"], "pending")
        self.assertEqual(request["evidence_ids"], [probe["id"], intervention["id"]])
        self.assertEqual(causal_gate["status"], "open")
        self.assertEqual(snapshot["overview"]["pending_closure_application_count"], 1)

    def test_loop_result_is_transactionally_written_back(self) -> None:
        result = self.store.record_loop_result(
            run_id="airnd_test_001",
            objective="验证否定构式的逐层场变化",
            loop_number=1,
            mode="manual",
            status="completed",
            decision="inconclusive",
            summary="完整产物尚缺一个留出对照。",
            master_model="main-agent",
            analyst_models=["reviewer-a", "reviewer-b"],
            artifact_audit={"decision": "inconclusive", "missing_required_artifacts": ["cases.jsonl"]},
        )
        snapshot = self.store.snapshot()
        self.assertEqual(result["analyst_models"], ["reviewer-a", "reviewer-b"])
        self.assertEqual(snapshot["overview"]["loop_run_count"], 1)
        self.assertEqual(snapshot["loop_runs"][0]["run_id"], "airnd_test_001")

    def test_http_api_roundtrip(self) -> None:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_research_workspace_store] = lambda: self.store
        client = TestClient(app)

        created = client.post(
            "/api/research-workspace/language-objects",
            json={"object_type": "construction", "family": "voice", "label": "被动构式"},
        )
        self.assertEqual(created.status_code, 201)
        snapshot = client.get("/api/research-workspace/snapshot")
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.json()["overview"]["language_object_count"], 11)

        operation = client.post(
            "/api/research-workspace/operations",
            json={"family_type": "http_test", "label": "HTTP 操作", "next_evidence_gap": "补齐 Case"},
        )
        self.assertEqual(operation.status_code, 201)
        self.assertEqual(client.get("/api/research-workspace/snapshot").json()["overview"]["operation_count"], 7)


if __name__ == "__main__":
    unittest.main()
