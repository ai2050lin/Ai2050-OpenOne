"""Software fixtures only. Never indexed as real research evidence."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from deploy import research_library as R,parameter_reader as PR
from shared.dist_protocol.runner_v3 import model_snapshot as reference_snapshot
import phase2768_knowledge_loop as L


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.lib=R.Library(Path(self.tmp.name))
    def tearDown(self):self.tmp.cleanup()
    def graph(self,level='observation',model='a'*64):
        ref=self.lib.ingest({'result':'software_fixture'},'fixture')['sha256']
        return {'schema':'research_graph.v1','kind':'real','model_snapshot':model,'feature_cards':[
            {'id':'feature','layer':18,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':3},'method':'fixture','run_id':'fixture',
             'counterexamples':[],'evidence':{'level':level,'artifact_sha256':ref}}]}
    def test_negative_revision_is_current_and_previous_observation_survives(self):
        a=self.lib.ingest(self.graph(),'first');b=self.lib.ingest(self.graph('negative'),'second')
        f=self.lib.fragments()['features'][0]
        self.assertEqual(f['sha'],b['sha256']);self.assertEqual(f['body']['evidence']['level'],'negative')
        self.assertEqual([h['level'] for h in f['history']],['negative','observation'])
        self.assertEqual(self.lib.artifact(a['sha256'])['feature_cards'][0]['evidence']['level'],'observation')
    def test_different_models_are_never_merged_and_pagination_is_complete(self):
        self.lib.ingest(self.graph(),'a');self.lib.ingest(self.graph(model='b'*64),'b')
        self.assertEqual(self.lib.fragments(limit=1)['total'],2)
        self.assertNotEqual(self.lib.fragments(limit=1)['features'][0]['model_snapshot'],self.lib.fragments(offset=1,limit=1)['features'][0]['model_snapshot'])
    def test_workpackage_revision_and_restart_are_durable(self):
        a={'schema':'research_workpackage.v1','run_id':'fixture','stages':{'discovery':{'state':'done'}}}
        old=self.lib.ingest(a,'producer');b=copy.deepcopy(a);b['stages']['validation']={'state':'not_supported'}
        new=self.lib.ingest(b,'producer');self.lib=R.Library(Path(self.tmp.name))
        self.assertEqual(self.lib.workpackages()['workpackages'][0]['sha'],new['sha256'])
        self.assertNotIn('validation',self.lib.artifact(old['sha256'])['stages'])
    def test_workpackage_schema_and_synthetic_refused(self):
        with self.assertRaises(ValueError):self.lib.ingest({'schema':'research_workpackage.v1','run_id':'x'},'bad')
        with self.assertRaises(ValueError):self.lib.ingest({'kind':'synthetic','schema':'research_workpackage.v1','run_id':'x','stages':{}},'bad')
    def test_offline_extras_include_fragments_workpackages_and_evidence(self):
        self.lib.ingest(self.graph(),'g')
        from deploy import datapack
        with patch.object(R,'library',return_value=self.lib):entries,_=datapack.research_entries()
        self.assertIn('/api/research-library/workpackages',entries)
        key=datapack.canonical('/api/research-library/fragments?offset=0&limit=50')
        self.assertEqual(entries[key]['total'],1)
        for f in entries[key]['features']:self.assertIn('/api/research-library/artifacts/'+f['body']['evidence']['artifact_sha256'],entries)


class SnapshotTests(unittest.TestCase):
    def test_protocol_compatible_full_hash_and_changed_file_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'config.json').write_text('{}');(p/'model.safetensors').write_bytes(b'fixture original')
            first=PR.model_snapshot(p);self.assertEqual(first['sha256'],reference_snapshot(p)['sha256'])
            (p/'model.safetensors').write_bytes(b'fixture changed!')
            self.assertNotEqual(first['sha256'],PR.model_snapshot(p)['sha256'])
            (p/'tokenizer.json').write_text('{}');self.assertEqual(PR.model_snapshot(p)['sha256'],reference_snapshot(p)['sha256'])


class AnalysisTests(unittest.TestCase):
    def test_planted_unit_is_recovered_and_unrelated_labels_fail(self):
        rng=np.random.default_rng(2768);y=np.repeat([0,1],100);X=rng.normal(size=(200,20));X[:,7]+=4*y
        self.assertEqual(int(np.argmax(np.abs(L.effect(X,y)))),7)
        self.assertGreater(L.auc(X[:,7],y),.99)
        shuffled=rng.permutation(y);self.assertLess(abs(L.effect(X,shuffled)[7]),.5)
        self.assertLess(L.auc(X[:,7],shuffled),.65)
    def test_family_averaging_and_frame_holdout_do_not_leak(self):
        spec=L.spec();corpus=L.GR.generate(spec,include_canary=False);items={i['id']:i for i in corpus['items']};ids=list(items)
        A=np.arange(len(ids))[:,None].astype(float)
        names,X,y=L.entity_rows(ids,items,A,'discovery');self.assertEqual(len(names),32)
        held,_,_=L.entity_rows(ids,items,A,'unseen_frame_and_entities')
        self.assertEqual(len(held),16);self.assertFalse(set(names)&set(held))
        for n,row in zip(names,X):
            expected=[k for k,i in enumerate(corpus['items']) if i['filler_id']==n and i['split']=='discovery']
            self.assertEqual(row[0],np.mean(expected))


if __name__=='__main__':unittest.main()
