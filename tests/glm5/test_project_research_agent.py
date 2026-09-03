import unittest

from server.ai_rnd_service import (
    _advance_project_agent,
    _default_project_agent_state,
    build_project_agent_plan,
)


class FakeSession:
    def __init__(self, tasks, *, max_loops=3, inconclusive_limit=3):
        self.project_agent = {
            **_default_project_agent_state(),
            "enabled": True,
            "status": "running",
            "config": {
                "max_loops": max_loops,
                "execution_mode": "auto",
                "stop_on_accepted": False,
                "stop_on_rejected": False,
                "max_consecutive_inconclusive": inconclusive_limit,
            },
            "plan": {"tasks": tasks},
        }
        self.research_state = {}
        self.persist_count = 0
        self.events = []

    def persist_state(self):
        self.persist_count += 1

    def push_event(self, event_type, **payload):
        self.events.append((event_type, payload))


def task(task_id, objective):
    return {
        "id": task_id,
        "title": task_id,
        "objective": objective,
        "completion_rule": "artifact audit",
        "status": "pending",
        "decision": None,
        "run_id": None,
    }


class ProjectResearchAgentTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "overview": {
                "language_object_count": 8,
                "full_field_count": 3,
                "open_claim_count": 2,
                "passed_gate_count": 1,
                "gate_count": 5,
            },
            "claims": [
                {
                    "id": "claim_challenged",
                    "title": "受挑战主张",
                    "status": "challenged",
                    "next_test": "先寻找受挑战主张的反例。",
                },
                {
                    "id": "claim_supported",
                    "title": "初步支持主张",
                    "status": "supported",
                    "next_test": "扩大样本复核初步支持主张。",
                },
            ],
            "closure_gates": [
                {
                    "id": "gate_blocked",
                    "title": "跨模型闭合",
                    "status": "blocked",
                    "blocking_reason": "缺少独立模型复现",
                }
            ],
        }
        self.evidence = {
            "available": True,
            "open_gaps": [
                {
                    "id": "gap_01",
                    "title": "完整场缺口",
                    "next_test": "采集完整 Layer × Token × HiddenSize 场。",
                }
            ],
        }

    def test_plan_prioritizes_user_goal_then_evidence_gap(self):
        plan = build_project_agent_plan(
            "验证当前项目核心假设",
            4,
            workspace_snapshot=self.snapshot,
            evidence_context=self.evidence,
        )

        self.assertEqual(plan["schema_version"], "project_research_plan.v1")
        self.assertEqual(len(plan["tasks"]), 4)
        self.assertEqual(plan["tasks"][0]["source_type"], "project_goal")
        self.assertEqual(plan["tasks"][1]["source_type"], "evidence_gap")
        self.assertEqual(plan["tasks"][2]["source_id"], "claim_challenged")
        self.assertIn("不代表理论成立", plan["completion_policy"]["plan_exhausted"])

    def test_plan_is_bounded_and_adds_full_field_coverage(self):
        plan = build_project_agent_plan(
            max_tasks=12,
            workspace_snapshot={
                "overview": self.snapshot["overview"],
                "claims": [],
                "closure_gates": [],
            },
            evidence_context={"available": False, "open_gaps": []},
        )

        self.assertEqual(len(plan["tasks"]), 1)
        self.assertEqual(plan["tasks"][0]["source_id"], "hiddenstate_full_field")
        self.assertIn("全部 Layer", plan["tasks"][0]["completion_rule"])

    def test_plan_reads_language_operation_and_pending_closure_request(self):
        plan = build_project_agent_plan(
            max_tasks=2,
            workspace_snapshot={
                "overview": {
                    **self.snapshot["overview"],
                    "full_field_count": 8,
                    "operation_count": 1,
                    "pending_closure_application_count": 1,
                },
                "operations": [
                    {
                        "id": "operation_negation",
                        "label": "否定与作用域",
                        "behavior_status": "untested",
                        "next_evidence_gap": "冻结否定范围 Case 与错误角色反事实控制。",
                    }
                ],
                "claims": [],
                "closure_applications": [
                    {
                        "id": "closure_request_01",
                        "gate_id": "gate_causal",
                        "review_status": "pending",
                        "rationale": "核验删除和救援干预。",
                    }
                ],
                "closure_gates": [],
            },
            evidence_context={"available": False, "open_gaps": []},
        )

        self.assertEqual(plan["tasks"][0]["source_type"], "language_operation")
        self.assertEqual(plan["tasks"][1]["source_type"], "closure_application")
        self.assertIn("不得直接批准", plan["tasks"][1]["completion_rule"])

    def test_agent_advances_to_next_bounded_task(self):
        session = FakeSession([task("one", "目标一"), task("two", "目标二")], max_loops=2)

        should_stop = _advance_project_agent(session, decision="inconclusive", run_id="run_01")

        self.assertFalse(should_stop)
        self.assertEqual(session.project_agent["current_task_index"], 1)
        self.assertEqual(session.project_agent["plan"]["tasks"][0]["run_id"], "run_01")
        self.assertEqual(session.research_state["research_objective"], "目标二")
        self.assertEqual(session.events[-1][0], "project_agent_progress")

    def test_agent_stops_at_plan_boundary_without_claiming_theory_complete(self):
        session = FakeSession([task("one", "目标一")], max_loops=1)

        should_stop = _advance_project_agent(session, decision="inconclusive", run_id="run_01")

        self.assertTrue(should_stop)
        self.assertFalse(session.project_agent["enabled"])
        self.assertEqual(session.project_agent["status"], "plan_completed")
        self.assertIn("人工确认", session.project_agent["stop_reason"])

    def test_agent_stops_after_repeated_inconclusive_results(self):
        session = FakeSession(
            [task("one", "目标一"), task("two", "目标二"), task("three", "目标三")],
            max_loops=3,
            inconclusive_limit=1,
        )

        should_stop = _advance_project_agent(session, decision="inconclusive", run_id="run_01")

        self.assertTrue(should_stop)
        self.assertEqual(session.project_agent["status"], "review_required")
        self.assertIn("安全停止", session.project_agent["stop_reason"])


if __name__ == "__main__":
    unittest.main()
