import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deploy.datapack import canonical, circuit_source
from server.neuron_census_service import router


class ArchivedCircuitTest(unittest.TestCase):
    def test_reads_only_the_requested_record_and_preserves_legacy_identity(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            file = circuit_source(root, 'census-fixture')
            file.parent.mkdir()
            query = {'model_key': 'census-fixture', 'run': 'r1', 'block': 3, 'unit': 2}
            key = canonical('/api/neurons/circuit?model_key=census-fixture&run=r1&block=3&unit=2&k=5')
            with gzip.open(file, 'wt', encoding='utf-8') as stream:
                json.dump({'schema': 'neuron_circuit_pack.v1', 'model_key': 'census-fixture', 'run': 'r1', 'k': 5,
                           'entries': {key: {'available': True, 'block': 3, 'unit': 2, 'upstream': [{'block': 2, 'unit': 9}],
                                             'downstream': [{'block': 4, 'unit': 5}]}}}, stream)
            app = FastAPI()
            app.include_router(router)
            with patch('deploy.research_library.services', return_value=SimpleNamespace(_DATA_DIR=root)), TestClient(app) as client:
                response = client.get('/api/neurons/circuit-archive', params=query)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIsNone(data['model_snapshot'])
                self.assertEqual((data['upstream'][0]['unit'], data['downstream'][0]['unit']), (9, 5))
                self.assertFalse(client.get('/api/neurons/circuit-archive', params={**query, 'run': 'r2'}).json()['available'])
                self.assertFalse(client.get('/api/neurons/circuit-archive', params={**query, 'unit': 10}).json()['available'])
