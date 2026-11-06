"""Implanted CPU fixtures are software tests, never research results."""
import sys,tempfile,json,sqlite3
from pathlib import Path
import unittest
from contextlib import closing
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shared.dist_protocol.native_mlp_atlas import decision_contrast,fit_sign,evaluate,factorial_coefficients,prediction_fvu,power_for_intersection

class AtlasMath(unittest.TestCase):
    def data(self,n=192):
        rng=np.random.default_rng(37);x=rng.normal(0,.01,(n,3,2,2,2,2,4)).astype(np.float32)
        for m in [0,1]:
            for t in [0,1]:
                for q in [0,1]:
                    d=1 if t==q else -1
                    x[:,:,m,t,q,:,0]+=d
                    x[:,:,m,t,q,:,1]+=d*((-1)**m)
                    x[:,:,m,t,q,:,2]+=(2*t-1)
        return x
    def test_abstract_direction_recovers_without_promoting_lexical_or_fact_only(self):
        x=self.data();sign=fit_sign(decision_contrast(x[:32]));r=evaluate(decision_contrast(x),sign,alpha=.05/700416)
        self.assertTrue(r['all_components_passed'][:,0].all());self.assertFalse(r['all_components_passed'][:,1:].any())
    def test_one_failed_code_mapping_blocks_global_transfer(self):
        x=self.data();sign=fit_sign(decision_contrast(x[:32]));x[:,2,1,:,:,:,0]*=-1;r=evaluate(decision_contrast(x),sign,alpha=.05/700416)
        self.assertFalse(r['all_components_passed'][:,0].any())
    def test_factorial_separates_main_fact_and_triple_interaction(self):
        names,mean,rms=factorial_coefficients(self.data());self.assertGreater(mean[0,names.index('decision'),0,0],.9);self.assertLess(mean[0,names.index('lexical_decision'),0,1],-.9);self.assertGreater(mean[0,names.index('fact'),0,2],.9)

    def test_prediction_uses_only_frozen_discovery_means(self):
        x=self.data();r=prediction_fvu(x[:32],x[32:]);self.assertLess(r['fvu'][:,:3].max(),.002)
        noise=np.random.default_rng(2).normal(size=x.shape).astype(np.float32);self.assertGreater(prediction_fvu(noise[:32],noise[32:])['fvu'].mean(),.95)
    def test_zero_variance_never_yields_infinite_evidence(self):
        x=np.zeros((32,3,2,2,2,2,3),np.float32);r=evaluate(decision_contrast(x),fit_sign(decision_contrast(x)),alpha=.001);self.assertFalse(r['passed'].any());self.assertTrue(prediction_fvu(x,x)['zero_variance'].all())
    def test_joint_power_is_not_marginal_power(self):
        r=power_for_intersection(192,.85,alpha=.05/700416);self.assertLess(r['joint_power_union_lower_bound'],r['marginal_power']);self.assertGreater(power_for_intersection(192,.9,alpha=.05/700416)['joint_power_union_lower_bound'],.8)
    def test_symmetric_product_allocation_has_no_unique_semantic_interpretation(self):
        rng=np.random.default_rng(11);s0,s1,v0,v1=rng.normal(size=(4,100))
        gate=.5*(s1-s0)*(v1+v0);value=.5*(v1-v0)*(s1+s0)
        np.testing.assert_allclose(gate+value,s1*v1-s0*v0,atol=1e-14)
        # The identity survives swapping branch names: its names cannot prove causal roles.
        np.testing.assert_allclose(.5*(v1-v0)*(s1+s0),value)
        s1=s0.copy();np.testing.assert_allclose(.5*(s1-s0)*(v1+v0),0)

class BankReadTests(unittest.TestCase):
    def setUp(self):
        from server import mlp_bank_service as S
        from unittest.mock import patch
        self.S=S;self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests/glm5_temp');self.root=Path(self.tmp.name);self.folder=self.root/'mlp_atlas_code_v9/bank';self.folder.mkdir(parents=True);self.run='MLP-ATLAS-CODE-20990101-v9'
        values=np.arange(20,dtype=np.float32).reshape(2,2,5);bits=(values.view(np.uint32)>>16).astype(np.uint16);np.save(self.folder/'B00-A.npy',bits)
        self.d={'schema':'mlp_native_bank.v1','kind':'real','run_id':self.run,'model_snapshot':'a'*64,'capture_contract_sha256':'b'*64,'n_items':2,'completed':1,'state':'collecting','n_layers':1,'d_ff':5,'d_model':4,'n_heads':2,'n_kv_heads':1,'head_dim':2,'file_shapes':{'B00-A.npy':[2,2,5]}}
        (self.folder/'manifest.json').write_text(json.dumps(self.d),encoding='utf-8')
        with closing(sqlite3.connect(self.folder/'samples.sqlite3')) as db:db.execute('CREATE TABLE samples(idx INTEGER PRIMARY KEY,id TEXT,body TEXT)');db.execute('INSERT INTO samples VALUES(0,?,?)',('fixture',json.dumps({'id':'fixture','bank_index':0,'readout_positions':[0,1,2,3]})));db.commit()
        self.patch=patch.object(S,'RESULTS',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def test_uncaptured_preallocated_rows_are_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as r:self.S.sample(self.run,1)
        self.assertEqual(r.exception.status_code,409)
    def test_actual_stage_and_all_native_units_are_returned(self):
        a=self.S.field(self.run,0,0,0,'mlp_activation');b=self.S.field(self.run,0,0,1,'mlp_activation')
        self.assertEqual(a['values'],[0,1,2,3,4]);self.assertEqual(b['values'],[5,6,7,8,9]);self.assertEqual(b['position'],3)
    def test_missing_attention_is_not_invented(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):self.S.field(self.run,0,0,0,'attention_rows')
    def test_invalid_run_and_native_index_are_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):self.S.bank('../../private')
        with self.assertRaises(HTTPException):self.S.field(self.run,0,1,0,'mlp_activation')
    def test_completed_bank_requires_matching_file_hash(self):
        from fastapi import HTTPException
        from shared.dist_protocol.mechanism_trace import file_sha
        self.d['state']='complete';self.d['sample_index_sha256']=file_sha(self.folder/'samples.sqlite3');self.d['files']=[{'path':'B00-A.npy','bytes':(self.folder/'B00-A.npy').stat().st_size,'sha256':'0'*64}];(self.folder/'manifest.json').write_text(json.dumps(self.d),encoding='utf-8')
        with self.assertRaises(HTTPException) as r:self.S.field(self.run,0,0,0,'mlp_activation')
        self.assertEqual(r.exception.status_code,409)
        self.assertIn('原生数据文件哈希',r.exception.detail)
    def test_complete_bank_sample_identity_is_hash_verified(self):
        from fastapi import HTTPException
        self.d.update(state='complete',sample_index_sha256='0'*64);(self.folder/'manifest.json').write_text(json.dumps(self.d),encoding='utf-8')
        with self.assertRaises(HTTPException) as r:self.S.sample(self.run,0)
        self.assertIn('样本索引哈希',r.exception.detail)

class DictionaryReadTests(unittest.TestCase):
    def setUp(self):
        from server import mlp_dictionary_service as S
        from unittest.mock import patch
        from shared.dist_protocol.mechanism_trace import file_sha
        self.S=S;self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests/glm5_temp');self.root=Path(self.tmp.name);p=self.root/'dictionaries';p.mkdir();self.folder=p
        np.savez(p/'B00-transcoder.npz',fixture=[1]);np.savez(p/'B00-associations.npz',all_discovery_correlations=np.arange(9728*4,dtype=np.float32).reshape(9728,4),discovery_partner=np.arange(9728)%4,discovery_partner_correlation=np.ones(9728),confirm_partner_correlation=np.zeros(9728),lockbox_partner_correlation=-np.ones(9728))
        self.row={'m':4,'model_snapshot':'a'*64,'dictionary_sha256':file_sha(p/'B00-transcoder.npz'),'association_sha256':file_sha(p/'B00-associations.npz')};(p/'B00-result.json').write_text(json.dumps(self.row),encoding='utf-8');self.patch=patch.object(S,'OUT',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def test_all_native_units_and_negative_holdout_association_are_preserved(self):
        d=self.S.layer(0);self.assertEqual(len(d['lockbox_partner_correlation']),9728);u=self.S.unit(0,9727);self.assertEqual(u['lockbox_partner_correlation'],-1);self.assertEqual(u['all_discovery_correlations'],[38908,38909,38910,38911])
    def test_missing_layer_and_nonexistent_dictionary_index_are_not_substituted(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as r:self.S.layer(1)
        self.assertEqual(r.exception.status_code,409)
        with self.assertRaises(HTTPException):self.S.feature(0,4)
        with self.assertRaises(HTTPException):self.S.unit(0,9728)
    def test_corrupted_dictionary_is_rejected(self):
        from fastapi import HTTPException
        (self.folder/'B00-transcoder.npz').write_bytes(b'corrupted')
        with self.assertRaises(HTTPException) as r:self.S.layer(0)
        self.assertEqual(r.exception.status_code,409)

if __name__=='__main__':unittest.main()
