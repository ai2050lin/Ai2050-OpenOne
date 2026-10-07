"""Evidence API contract and scope checks; no GPU or network calls."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rdc_context_interaction_service import router


class ContextEvidenceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app=FastAPI()
        app.include_router(router)
        cls.client=TestClient(app)

    def test_measurement_is_not_claimed_universal(self):
        res=self.client.get('/context-interaction/claims')
        self.assertEqual(res.status_code,200)
        claims={r['id']:r for r in res.json()['claims']}
        self.assertEqual(claims['REF01']['status'],'incorrect_generalization')
        self.assertEqual(claims['REF05']['status'],'wrong_measured_object')

    def test_full_native_field_and_model_validation(self):
        res=self.client.get('/context-interaction/models/4B/field?layer=36')
        self.assertEqual(res.status_code,200)
        self.assertEqual(len(res.json()['values']),2560)
        self.assertEqual(res.json()['shape'],[38,2560])
        for url in ('/models/unknown','/models/14B?control=assertion','/models/4B/field?layer=-1','/models/4B/field?kind=secret'):
            self.assertEqual(self.client.get('/context-interaction'+url).status_code,422)

    def test_world_identity_and_repeated_measure_grouping(self):
        res=self.client.get('/context-interaction/worlds/category_train_00')
        self.assertEqual(res.status_code,200)
        self.assertEqual(len(res.json()['rows']),12)
        self.assertEqual({r['split'] for r in res.json()['rows']},{'train','wording'})
        self.assertEqual(self.client.get('/context-interaction/worlds/nonexistent').status_code,404)


if __name__=='__main__':
    unittest.main()
