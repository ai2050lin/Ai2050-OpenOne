"""State publication, custody, evidence and native 3D index consistency. No model inference."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from deploy.research_library import Library
from shared.dist_protocol.mainline_state import MainlineStore, RELEASE, digest, validate


class MainlineStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lib = Library(Path(self.temp.name) / 'research_library')
        self.store = MainlineStore(self.lib)
        self.bundle = json.loads(RELEASE.read_text(encoding='utf-8'))

    def next_bundle(self):
        b = copy.deepcopy(self.bundle)
        b['state']['sequence'] += 1
        b['state']['parent_revision'] = digest(self.bundle['state'])
        return b

    def test_bootstrap_all_proofs_and_native_objects(self):
        result = self.store.ensure()
        self.assertEqual(result['revision'], digest(self.bundle['state']))
        state = json.loads(result['record'])
        self.assertEqual(state['active_branch']['status'], 'not_supported')
        self.assertEqual(state['view']['LOCKBOXES'][0]['state'], 'opened')
        for sha in self.bundle['artifacts']:
            self.assertEqual(digest(self.lib.artifact(sha)), sha)
        with patch('server.research_space_service.library', return_value=self.lib):
            from server.research_space_service import research_space
            space = research_space()
        self.assertEqual(space['mainline_revision'], result['revision'])
        self.assertEqual({s['run_id'] for s in space['studies']}, {'MA-H3-v4', 'MA-H3b-v1'})
        self.assertGreater(len(space['objects']), 600)
        self.assertTrue(all(o['type'] == 'feature' for o in space['objects']))
        for obj in space['objects']:
            self.assertEqual(obj['body']['evidence']['level'], 'observation')
            self.assertFalse(obj['body']['scope']['individual_causality_established'])
            if obj['body']['index']['space'] == 'attention':
                self.assertLess(obj['body']['index']['head'], 32)
            else:
                self.assertEqual(obj['body']['index']['field'], 'residual_write')
                self.assertNotIn('unit', obj['body']['index'])
        self.assertEqual(space['models'][0]['n_layers'], 36)

    def test_duplicate_does_not_increase_counts_and_reopen_persists(self):
        a = self.store.ensure()
        with self.lib.db() as db:
            before = db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        self.assertEqual(self.store.publish(self.bundle), a)
        with self.lib.db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], before)
        self.assertEqual(MainlineStore(Library(self.lib.root)).ensure(), a)

    def test_opened_confirm_cannot_be_resealed_or_dropped(self):
        self.store.ensure()
        for remove in (False, True):
            b = self.next_bundle()
            if remove:
                b['state']['custody'] = b['state']['custody'][1:]
            else:
                b['state']['custody'][0]['state'] = 'sealed'
                next(x for x in b['state']['view']['LOCKBOXES'] if x['id'] == b['state']['custody'][0]['id'])['state'] = 'sealed'
            with self.assertRaisesRegex(ValueError, 'resealed|custody'):
                self.store.publish(b)

    def test_missing_corrupt_proof_or_wrong_model_fail_before_pointer_change(self):
        old = self.store.ensure()
        for mode in ('missing', 'corrupt', 'model'):
            b = self.next_bundle()
            ref = b['state']['records'][0]['report_sha256']
            if mode == 'missing':
                del b['artifacts'][ref]
            elif mode == 'corrupt':
                b['artifacts'][ref]['report']['n_items'] = 0
            else:
                b['state']['records'][0]['model_snapshot'] = 'a' * 64
            with self.assertRaises(ValueError):
                self.store.publish(b)
            self.assertEqual(self.store.ensure(), old)

    def test_compare_and_swap_and_sequential_update(self):
        self.store.ensure()
        b = self.next_bundle()
        b['state']['parent_revision'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'parent'):
            self.store.publish(b)
        b = self.next_bundle()
        result = self.store.publish(b)
        self.assertEqual(result['sequence'], self.bundle['state']['sequence'] + 1)
        self.assertFalse(result['evidence_promoted'])
        with self.assertRaisesRegex(ValueError, 'increase'):
            self.store.publish(self.bundle)

    def test_http_public_read_authenticated_write(self):
        from deploy.mainline_state_service import router
        app = FastAPI()
        app.include_router(router)
        with patch('deploy.mainline_state_service.store', return_value=self.store):
            client = TestClient(app)
            read = client.get('/api/research/mainline')
            self.assertEqual(read.status_code, 200)
            self.assertEqual(read.headers['cache-control'], 'no-store')
            self.assertEqual(read.headers['etag'], '"' + read.json()['revision'] + '"')
            with patch('deploy.research_library._registry_admin', side_effect=HTTPException(403, 'admin required')):
                self.assertEqual(client.post('/api/research/mainline', json=self.bundle).status_code, 403)
            with patch('deploy.research_library._registry_admin'):
                self.assertEqual(client.post('/api/research/mainline', json=self.bundle).status_code, 200)
                self.assertEqual(client.post('/api/research/mainline', json=self.next_bundle()).status_code, 200)

    def test_real_source_values_not_independent_prediction(self):
        state = validate(self.bundle)
        v = state['view']
        self.assertTrue(all(s['value'] is None for s in v['SCORES']))
        self.assertIsNone(v['TESTS'][0]['rows'][0][1])
        h3, h3b = v['METHOD_RESULTS']
        self.assertAlmostEqual(h3['value'], .9884567856788635)
        self.assertLess(h3b['value'], .06)
        self.assertTrue(all(not r['replicated'] for r in v['RUNS'] if r['id'].startswith('MA-H3')))

    def test_interrupted_publication_cannot_expose_partial_graph(self):
        old = self.store.ensure()
        b = self.next_bundle()
        old_ref = b['state']['graph_sha256'][0]
        graph = copy.deepcopy(b['artifacts'][old_ref])
        graph['feature_cards'][0]['label'] = 'unpublished changed candidate'
        new_ref = digest(graph)
        b['artifacts'][new_ref] = graph
        b['state']['graph_sha256'][0] = new_ref
        original = self.lib.ingest

        def interrupted(doc, *args, **kwargs):
            result = original(doc, *args, **kwargs)
            if digest(doc) == new_ref:
                raise OSError('injected interruption after graph ingestion')
            return result

        with patch.object(self.lib, 'ingest', side_effect=interrupted):
            with self.assertRaises(OSError):
                self.store.publish(b)
        self.assertEqual(self.store.ensure(), old)
        self.assertNotIn(new_ref, {f['sha'] for f in self.lib.fragments(limit=1000)['features']})
        self.assertNotIn(new_ref, {o['sha'] for o in self.lib.graph(limit=1000)['objects']})
        with patch('server.research_space_service.library', return_value=self.lib):
            from server.research_space_service import research_space
            self.assertNotIn(new_ref, {o['sha'] for o in research_space()['objects']})
        self.assertEqual(self.store.publish(b)['sequence'], b['state']['sequence'])

    def test_custody_requires_actual_opening_event_even_on_new_center(self):
        b = copy.deepcopy(self.bundle)
        b['state']['custody'][0]['state'] = 'sealed'
        next(x for x in b['state']['view']['LOCKBOXES'] if x['id'] == b['state']['custody'][0]['id'])['state'] = 'sealed'
        with self.assertRaisesRegex(ValueError, 'opening event'):
            validate(b)


if __name__ == '__main__':
    unittest.main()
