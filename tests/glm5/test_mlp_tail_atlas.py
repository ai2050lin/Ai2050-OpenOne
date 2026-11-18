"""Real atlas serving and rejected transport/misidentification; no model inference."""
import copy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server import mlp_tail_service as S
from deploy import mlp_tail_pack as P

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'tests/glm5/result/mlp_tail_goodfire_v1'
EXTENSION=DATA.with_name('mlp_tail_b28_b31_v1')


class TailAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=patch.dict(os.environ,{'AI2050_MLP_TAIL_DIR':str(DATA)});cls.env.start()
        app=FastAPI();app.include_router(S.router);cls.client=TestClient(app)
        cls.report=json.loads((DATA/'report.json').read_text(encoding='utf-8'))
    @classmethod
    def tearDownClass(cls):cls.client.close();cls.env.stop();S._json.cache_clear();S._tokenizer.cache_clear()

    def get(self,p):
        r=self.client.get('/api/research/mlp-tail'+p);self.assertEqual(r.status_code,200,r.text[:200]);return r.json()

    def test_all_four_native_arrays_not_semantic_promotion(self):
        for b in range(32,36):
            d=self.get(f'/layers/{b}')
            self.assertEqual(d['width'],9728);self.assertEqual(len(d['candidate_labels']),9728)
            for values in d['arrays'].values():self.assertEqual(len(values),9728);self.assertTrue(np.isfinite(values).all())
            self.assertEqual(sum(d['semantic_confirmed']),0)
            self.assertTrue(np.allclose(sum(np.array(d['arrays'][f'sign_{i}']) for i in range(4)),1))

    def test_full_parameters_and_all_unit_boundaries(self):
        for b in range(32,36):
            for u in [0,9727]:
                d=self.get(f'/units/{b}/{u}')
                self.assertEqual(len(d['examples']),8)
                self.assertEqual(len(d['cue_correlations']),5)
                for key,values in d['weights'].items():
                    self.assertEqual(len(values),2560)
                    raw=np.load(DATA/f'B{b}-W-{key}.npy',mmap_mode='r')
                    expected=S.unbits(raw[:,u] if key=='down' else raw[u])
                    self.assertTrue(np.array_equal(values,expected))

    def test_pole_values_and_causal_prefix(self):
        d=self.get('/units/35/1244?field=A&split=confirm&pole=negative')
        tok=S._tokenizer(str(DATA))
        docs=S.read('documents.json')['docs']
        for e in d['examples']:
            self.assertGreaterEqual(e['document_index'],1024);self.assertLess(e['document_index'],1536)
            ids=docs[e['document_index']]['tokens'][:e['token_position']+1]
            self.assertEqual(e['prefix'],tok.decode(ids,skip_special_tokens=False))
            self.assertEqual(e['A'],float(S.unbits(np.load(DATA/'B35-A.npy',mmap_mode='r')[e['document_index']*16+S.read('documents.json')['positions'].index(e['token_position']),1244])))

    def test_complete_rows_columns_all_matrix_types(self):
        for b in [32,33,34]:
            for kind in ['rho-discovery','rho-confirm','rho-holdout','weight-gate','weight-up']:
                out=self.get(f'/relations/{b}/9727?kind={kind}')
                inc=self.get(f'/relations/{b+1}/0?kind={kind}&direction=incoming')
                self.assertEqual(len(out['values']),9728);self.assertEqual(len(inc['values']),9728)
                self.assertEqual(out['values'][0],inc['values'][9727])
                self.assertNotEqual(out['kind'],'causal')

    def test_causal_identical_valid_document_subset(self):
        rows=S.read('path-cases.json')['rows']
        d=self.get('/relations/34/3?kind=causal&factor=2')
        selected=[(i,r) for i,r in enumerate(rows) if r['source']==34 and r['unit']==3 and r['factor']==2]
        native={r['doc_index']:i for i,r in selected if r['condition']=='native_unit' and r['norm_control_valid']}
        ctrl={r['doc_index']:i for i,r in selected if r['condition']=='matched_noise' and r['norm_control_valid']}
        ids=sorted(native.keys()&ctrl.keys());arr=np.load(DATA/'path-target-deltas.npy',mmap_mode='r')
        self.assertEqual(d['n_valid'],len(ids));self.assertGreater(len(ids),0)
        self.assertTrue(np.allclose(d['values'],np.asarray(arr[[native[k] for k in ids]],np.float32).mean(0),atol=1e-5))
        self.assertTrue(np.allclose(d['control_values'],np.asarray(arr[[ctrl[k] for k in ids]],np.float32).mean(0),atol=1e-5))

    def test_missing_intervention_not_invented(self):
        d=self.get('/relations/32/9727?kind=causal');self.assertEqual(d['kind'],'not_tested');self.assertIsNone(d['values'])

    def test_native_parameter_lens_not_context_effect(self):
        d=self.get('/units/35/9727');r=d['parameter_readout']['positive'][0]
        e=S.unbits(np.load(DATA/'unembedding.npy',mmap_mode='r')[r['token_id']]);w=np.array(d['weights']['down'])
        self.assertAlmostEqual(r['cosine'],float(e@w/np.linalg.norm(e)/np.linalg.norm(w)),delta=1e-5)

    def test_bounds_and_unavailable_boundaries(self):
        for path in ['/layers/31','/units/35/9728','/units/32/-1','/relations/35/0','/relations/32/0?direction=incoming','/relations/33/0?kind=invalid']:
            self.assertEqual(self.client.get('/api/research/mlp-tail'+path).status_code,422)

    def test_no_data_and_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(S,'folder',return_value=Path(tmp)):
            self.assertEqual(self.client.get('/api/research/mlp-tail').status_code,404)
            man=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'));(Path(tmp)/'manifest.json').write_text(json.dumps(man),encoding='utf-8')
            (Path(tmp)/'report.json').write_text('{}',encoding='utf-8')
            self.assertEqual(self.client.get('/api/research/mlp-tail').status_code,409)

    def test_negative_rollout_and_preservation_scope_retained(self):
        d=self.get('');self.assertEqual(d['coverage']['semantic_explanations_confirmed'],0)
        self.assertEqual(d['path']['rows'],9216);self.assertIn('not established',d['path']['capability'])
        self.assertTrue(any(r['fvu']['direct']>1 for r in d['prediction']['rollout']))


class TailTransportTests(unittest.TestCase):
    def setUp(self):self.man=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    def test_all_required_files_and_source_hashes(self):
        required=P.validate_manifest(self.man);self.assertGreater(len(required),40)
        self.assertNotIn('B35-A.npy',{r['path'] for r in required})
        for r in required:self.assertEqual((DATA/r['path']).stat().st_size,r['bytes'])
    def test_reject_missing_files_and_wrong_snapshot(self):
        bad=copy.deepcopy(self.man);bad['files']=[r for r in bad['files'] if r['path']!='B35-profile.json']
        with self.assertRaises(ValueError):P.validate_manifest(bad)
        bad=copy.deepcopy(self.man);bad['model_snapshot']='0'*64
        with self.assertRaises(ValueError):P.validate_manifest(bad)
    def test_reject_traversal_duplicates_and_symlinks(self):
        for name in ['../outside','C:/outside','..\\outside','/outside','a/b']:
            bad=copy.deepcopy(self.man);bad['files'][0]['path']=name
            with self.assertRaises(ValueError):P.validate_manifest(bad)
        bad=copy.deepcopy(self.man);bad['files'].append(bad['files'][0])
        with self.assertRaises(ValueError):P.validate_manifest(bad)
        with tempfile.TemporaryDirectory() as t:
            archive=Path(t)/'bad.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                raw=json.dumps(self.man).encode();member=tarfile.TarInfo('manifest.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
                member=tarfile.TarInfo('report.json');member.type=tarfile.SYMTYPE;member.linkname='../../outside';tar.addfile(member)
            with self.assertRaises(ValueError):P.install(archive,Path(t)/'data')
            self.assertEqual(list((Path(t)/'data/mlp_tail').iterdir()),[])


@unittest.skipUnless((EXTENSION/'manifest.json').is_file(),'real B28–B31 census has not been completed')
class TailExtensionTests(unittest.TestCase):
    def setUp(self):
        env=patch.dict(os.environ,{'AI2050_MLP_TAIL_DIR':str(DATA),'AI2050_MLP_TAIL_EXTENSION_DIR':str(EXTENSION)})
        env.start()
        self.addCleanup(env.stop)
        app=FastAPI()
        app.include_router(S.router)
        self.client=TestClient(app)
        self.addCleanup(self.client.close)

    def get(self,path):
        response=self.client.get('/api/research/mlp-tail'+path)
        self.assertEqual(response.status_code,200,response.text[:250])
        return response.json()

    def test_extended_coverage_and_native_endpoints(self):
        report=self.get('')
        self.assertEqual(report['blocks'],list(range(28,36)))
        self.assertEqual(report['n_units'],77824)
        self.assertEqual(report['prediction_blocks'],list(range(32,36)))
        self.assertEqual(report['coverage']['all_adjacent_pair_matrices'],35)
        self.assertEqual([row['block'] for row in report['replay']['checks']],list(range(28,36)))
        for b in range(28,32):
            data=self.get(f'/layers/{b}')
            self.assertEqual(data['run_id'],S.EXTENSION_RUN)
            self.assertEqual(len(data['candidate_labels']),9728)
            self.assertTrue(all(len(vals)==9728 and np.isfinite(vals).all() for vals in data['arrays'].values()))
            self.assertTrue(np.allclose(sum(np.array(data['arrays'][f'sign_{i}']) for i in range(4)),1))
            self.assertFalse(any(data['semantic_confirmed']))
            for u in [0,9727]:
                card=self.get(f'/units/{b}/{u}?field=A&pole=negative')
                self.assertEqual(len(card['examples']),8)
                self.assertTrue(all(len(card['parameter_readout'][pole])==8 for pole in ('positive','negative')))
                for field,vector in card['weights'].items():
                    raw=np.load(EXTENSION/f'B{b}-W-{field}.npy',mmap_mode='r')
                    self.assertTrue(np.array_equal(vector,S.unbits(raw[:,u] if field=='down' else raw[u])))

    def test_all_new_boundaries_agree_in_both_directions(self):
        for b in range(28,32):
            for kind in ('rho-discovery','rho-confirm','rho-holdout','weight-gate','weight-up'):
                outgoing=self.get(f'/relations/{b}/9727?kind={kind}')
                incoming=self.get(f'/relations/{b+1}/0?kind={kind}&direction=incoming')
                self.assertEqual(len(outgoing['values']),9728)
                self.assertEqual(len(incoming['values']),9728)
                self.assertEqual(outgoing['values'][0],incoming['values'][9727])
                self.assertEqual(outgoing['kind'],'observation' if kind.startswith('rho') else 'parameter_alignment')
        self.assertEqual(self.get('/relations/31/9727?kind=causal')['kind'],'not_tested')

    def test_extension_rejects_mismatched_base_manifest(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(S,'extension_folder',return_value=Path(tmp)):
            bad=json.loads((EXTENSION/'manifest.json').read_text(encoding='utf-8'))
            bad['base_manifest_sha256']='0'*64
            (Path(tmp)/'manifest.json').write_text(json.dumps(bad),encoding='utf-8')
            self.assertEqual(self.client.get('/api/research/mlp-tail').status_code,409)


if __name__=='__main__':unittest.main()
