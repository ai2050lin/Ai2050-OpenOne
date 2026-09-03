import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.research_asset_service import resolve_research_asset, router


class ResearchAssetPathTests(unittest.TestCase):
    def test_accepts_relative_and_legacy_prefix(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            expected = root / "research_kernel" / "manifest.json"
            self.assertEqual(resolve_research_asset("research_kernel/manifest.json", root=root), expected)
            self.assertEqual(resolve_research_asset("/vis_data/research_kernel/manifest.json", root=root), expected)

    def test_blocks_traversal(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            for path in ("../secret.json", "research_kernel/../../secret.json"):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(ValueError, "escapes"):
                        resolve_research_asset(path, root=root)

    def test_http_api_serves_assets_from_backend_store(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            payload = {"schema_version": "test.v1"}
            (root / "source_registry.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"AI2050_RESEARCH_ASSET_ROOT": str(root)}):
                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)
                health = client.get("/api/research-assets/health")
                response = client.get("/api/research-assets/file/source_registry.json")
                self.assertTrue(health.json()["available"])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), payload)


if __name__ == "__main__":
    unittest.main()
