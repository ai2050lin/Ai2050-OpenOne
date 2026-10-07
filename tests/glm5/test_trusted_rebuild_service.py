"""Read-only API contract checks on the saved evidence; no CUDA or model load."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rdc_trusted_rebuild_service import router


class EvidenceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app=FastAPI();app.include_router(router);cls.client=TestClient(app)

    def test_claim_retains_withdrawal_and_replacement(self):
        res=self.client.get('/trusted-rebuild/claims/RDC-M01')
        self.assertEqual(res.status_code,200)
        self.assertEqual(res.json()['status'],'invalid_legacy_formula')
        self.assertIn('4B',res.json()['replacement_results'])

    def test_unknown_inputs_are_not_filesystem_paths(self):
        self.assertEqual(self.client.get('/trusted-rebuild/models/not-a-model').status_code,422)
        self.assertEqual(self.client.get('/trusted-rebuild/claims/nonexistent').status_code,404)
        self.assertEqual(self.client.get('/trusted-rebuild/models/4B/field?kind=weights').status_code,422)

    def test_complete_unsorted_coordinate_row(self):
        res=self.client.get('/trusted-rebuild/models/4B/field?layer=36')
        self.assertEqual(res.status_code,200)
        self.assertEqual(len(res.json()['values']),2560)
        self.assertEqual(res.json()['shape'],[37,2560])
        self.assertEqual(self.client.get('/trusted-rebuild/models/4B/field?layer=-1').status_code,422)


if __name__=='__main__':unittest.main()
