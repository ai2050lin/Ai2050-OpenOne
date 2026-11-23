"""Regression tests for serving a previously cached, incomplete center skeleton."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]
from deploy import platform_service


SNAPSHOT = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
MODEL_KEY = 'qwen3-14b|cfg-e73c3664ca09|bf16'


class VerifiedModelLayoutTests(unittest.TestCase):
    def setUp(self):
        self.skeleton = {'model': MODEL_KEY, 'model_snapshot': SNAPSHOT, 'n_layers': 40,
                         'd_model': 5120, 'n_heads': None, 'd_ff': None}
        snapshot = {'data': {'skeletons': {MODEL_KEY: self.skeleton}}}
        replacement = patch.object(platform_service, '_lang_atlas', return_value=snapshot)
        replacement.start()
        self.addCleanup(replacement.stop)
        app = FastAPI()
        app.include_router(platform_service.lang_router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_list_single_and_architecture_endpoints_fill_old_snapshot(self):
        listed = self.client.get('/api/model/skeleton').json()['models'][0]
        single = self.client.get('/api/model/skeleton', params={'model_key': MODEL_KEY})
        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json(), {**listed, 'pending': False})
        self.assertEqual(listed['model_snapshot'], SNAPSHOT)
        self.assertEqual([listed[k] for k in ('n_heads', 'n_kv_heads', 'head_dim', 'd_ff')],
                         [40, 8, 128, 17408])
        self.assertEqual(listed['structure_source']['kind'], 'verified_snapshot_config')
        self.assertIsNone(self.skeleton['n_heads'])
        self.assertIsNone(self.skeleton['d_ff'])
        with patch.object(platform_service, '_hf_config', side_effect=AssertionError('read model weights')):
            arch = self.client.get('/api/model/arch', params={'model_key': MODEL_KEY}).json()
        self.assertTrue(arch['available'])
        self.assertEqual(arch['attn']['mats'][0]['shape'], [5120, 5120])
        self.assertEqual(arch['mlp']['mats'][0]['shape'], [5120, 17408])

    def test_model_alias_alone_cannot_supply_dimensions(self):
        self.skeleton['model_snapshot'] = 'a' * 64
        result = self.client.get('/api/model/skeleton').json()['models'][0]
        self.assertIsNone(result['n_heads'])
        self.assertIsNone(result['d_ff'])
        self.assertNotIn('structure_source', result)

    def test_conflicting_original_dimensions_are_preserved(self):
        self.skeleton['n_heads'] = 32
        result = self.client.get('/api/model/skeleton').json()['models'][0]
        self.assertEqual(result['n_heads'], 32)
        self.assertIsNone(result['d_ff'])
        self.assertNotIn('structure_source', result)


if __name__ == '__main__':
    unittest.main()
