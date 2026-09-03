import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ai2050_research_os" / "scripts" / "researchctl.py"
SPEC = importlib.util.spec_from_file_location("researchctl", SCRIPT)
researchctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(researchctl)


class ResearchSnapshotTests(unittest.TestCase):
    def test_workspace_discovery_uses_repository_root(self):
        self.assertEqual(researchctl.WORKSPACE, ROOT)

    def test_legacy_manifest_path_maps_without_rewriting_manifest(self):
        resolved = researchctl.resolve_manifest_artifact(
            "research/ai2050_research_os/schemas/experiment_contract.schema.json"
        )
        self.assertEqual(resolved, researchctl.SCHEMAS / "experiment_contract.schema.json")
        self.assertTrue(resolved.is_file())

    def test_snapshot_is_deterministic_and_tracks_registry(self):
        data = researchctl.load_all()
        first = researchctl.build_snapshot(data)
        self.assertEqual(first, researchctl.build_snapshot(data))
        self.assertEqual(first["current"]["phase"], data["project"]["latest_recorded_phase"])
        self.assertEqual(first["current"]["campaign_id"], data["project"]["active_campaign_id"])
        self.assertEqual(first["information_architecture"]["technical_stage_mapping"]["M8"], "R8")
        self.assertEqual(first["framework"]["current_stage_id"], "R1")
        self.assertEqual(len(first["summaries"]["puzzles"]), len(data["puzzles"]))
        self.assertEqual(researchctl.validate_snapshot(first), [])

    def test_exported_snapshot_matches_canonical_snapshot(self):
        canonical = json.loads((researchctl.SNAPSHOTS / "current" / "snapshot.json").read_text(encoding="utf-8"))
        exported = json.loads(researchctl.CLIENT_SNAPSHOT.read_text(encoding="utf-8"))
        self.assertEqual(exported, canonical)

    def test_client_has_no_legacy_current_state_source(self):
        self.assertEqual(researchctl.client_drift_findings(), [])

    def test_app_uses_single_research_center_entry(self):
        app_source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
        center_source = (ROOT / "frontend" / "src" / "researchCenter" / "ResearchCenter.jsx").read_text(encoding="utf-8")
        self.assertIn("./researchCenter/ResearchCenter", app_source)
        self.assertNotIn("./HLAIBlueprint", app_source)
        self.assertIn("lazy(() => import('../HLAIBlueprint')", center_source)

    def test_heatmap_route_uses_real_embedding_and_hidden_state_trace(self):
        source = (ROOT / "frontend" / "src" / "components" / "app" / "ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
        self.assertIn("event.event_type === STATE_HEATMAP_ROUTE.embeddingEvent", source)
        self.assertIn("hiddenByLayer", source)
        self.assertIn("top_units", source)
        self.assertNotIn("HEATMAP_ROUTE_PREVIEW", source)

    def test_primary_research_surfaces_are_not_phase_led(self):
        paths = [
            ROOT / "frontend" / "src" / "researchCenter" / "ResearchCenter.jsx",
            ROOT / "frontend" / "src" / "components" / "app" / "ResearchSpaceOverlay.jsx",
            ROOT / "frontend" / "src" / "components" / "app" / "ResearchEvidenceCockpit.jsx",
            ROOT / "frontend" / "src" / "components" / "app" / "ResearchHeatmapRoute.jsx",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("PHASE {", combined)
        self.assertNotIn("Phase {", combined)


if __name__ == "__main__":
    unittest.main()
