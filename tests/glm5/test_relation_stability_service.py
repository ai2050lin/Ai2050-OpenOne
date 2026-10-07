"""Protect evidence scope, native dimensions, identity and lookup bounds."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rdc_relation_stability_service import router

class RelationEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app=FastAPI();app.include_router(router);cls.client=TestClient(app)
    def test_model_and_field_scope(self):
        r=self.client.get('/relation-stability/field/4B?factor=truth&layer=36');self.assertEqual(r.status_code,200)
        self.assertEqual(len(r.json()['values']),2560);self.assertEqual(r.json()['shape'][1:], [8,38,2560])
        for suffix in ('/field/unknown','/field/4B?row=-1','/field/4B?factor=secret','/field/4B?layer=38'):
            self.assertEqual(self.client.get('/relation-stability'+suffix).status_code,422)
    def test_factorial_world_identity(self):
        r=self.client.get('/relation-stability/worlds/role_entity_00');self.assertEqual(r.status_code,200)
        rows=r.json()['rows'];self.assertEqual(len(rows),16);self.assertEqual(sum(x['truth_sign']>0 for x in rows),8)
        self.assertEqual(self.client.get('/relation-stability/worlds/missing').status_code,404)
    def test_units_are_complete_and_causal_claim_limited(self):
        r=self.client.get('/relation-stability/units/33');self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['unit_count'],9728);self.assertEqual(len(r.json()['values']),9728)
        self.assertIn('observational',r.json()['scope'])
        self.assertEqual(self.client.get('/relation-stability/units/0').status_code,422)
        self.assertEqual(self.client.get('/relation-stability/units/33?row=12').status_code,422)

if __name__=='__main__':unittest.main()
