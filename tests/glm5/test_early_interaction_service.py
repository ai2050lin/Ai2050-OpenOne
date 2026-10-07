"""Evidence contracts: unit of inference, captured dimensions, failure visibility."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rdc_early_interaction_service import router

class EarlyEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app=FastAPI();app.include_router(router);cls.client=TestClient(app)
    def test_split_and_failure_preserved(self):
        r=self.client.get('/early-interaction/summary');self.assertEqual(r.status_code,200)
        d=r.json();self.assertEqual(d['worlds'],384);self.assertEqual(d['prompts'],3072)
        self.assertGreater(d['splits']['fresh_surface']['methods']['target8']['interaction'],1)
        self.assertLessEqual(self.client.get('/early-interaction/readout').json()['splits']['fresh_surface']['target8']['accuracy'],.55)
    def test_all_coordinates_and_bounds(self):
        r=self.client.get('/early-interaction/field');self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['shape'],[16,2560]);self.assertEqual(len(r.json()['values']),2560)
        for suffix in ('?kind=secret','?row=-1','?row=16'):
            self.assertEqual(self.client.get('/early-interaction/field'+suffix).status_code,422)
    def test_world_clusters_and_graph(self):
        r=self.client.get('/early-interaction/worlds/category_fresh_entity_00')
        self.assertEqual(r.status_code,200);self.assertEqual(len(r.json()['rows']),8)
        self.assertEqual(len(r.json()['annotations']),2)
        self.assertEqual(self.client.get('/early-interaction/worlds/missing').status_code,404)

if __name__=='__main__':unittest.main()
