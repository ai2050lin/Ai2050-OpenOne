"""Software/identity tests; fixtures do not count as scientific observations."""
from pathlib import Path
import sys,json,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from phase2794_analyze import feature_matrix,ci

class Materials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=json.loads((P.OUT/'formal-corpus.json').read_text(encoding='utf-8'))
    def test_exact_reference_and_full_integer_labels(self):
        es=P.entities();self.assertEqual(len(es),96)
        for r in self.c['items']:
            self.assertEqual(len(r['token_ids']),r['length'])
            self.assertIn('为'+r['prior']+'。',r['reference_text'])
            self.assertNotEqual(r['prior'][0],r['document'][0])
            self.assertEqual(r['expected'],r['document'] if r['scope']=='文档' else r['prior'])
    def test_holdout_combinations_and_entities_do_not_recur(self):
        train={(r['family_id'],r['document']) for r in self.c['items'] if r['split']=='discovery'}
        held={(r['family_id'],r['document']) for r in self.c['items'] if r['split']=='unseen_combination'}
        self.assertFalse(train&held)
        known={r[0] for r in train};novel={r['family_id'] for r in self.c['items'] if r['split']=='unseen_entity'}
        self.assertFalse(known&novel);self.assertEqual(len(known),64);self.assertEqual(len(novel),32)
    def test_features_ignore_expected_answer_and_observed_predictions(self):
        rows=[r for r in self.c['items'] if r['split']=='discovery']
        a=feature_matrix(rows,self.c['entities'])
        b=feature_matrix([{**r,'expected':'999','generated':'999','actual_hidden':[999]} for r in rows],self.c['entities'])
        np.testing.assert_array_equal(a,b)
    def test_document_values_are_observable_in_text(self):
        for r in self.c['items']:
            self.assertIn(r['source_text'],r['text']);self.assertIn(r['document'],r['source_text'])
            self.assertIn(r['reference_text'],r['text'])
    def test_cluster_ci_retains_family_count(self):
        d=ci([0,0,1,1]);self.assertEqual(d['n_entity_clusters'],4);self.assertEqual(d['mean'],.5)
    def test_neuron_dose_changes_only_declared_current_units(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from phase2794_run import install
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8))
        captured=[]
        h=install(model,[(0,7)],.5)[0]
        # Exercise the actual registered hook without a model forward.
        x=torch.ones(1,3,64);y=list(model.model.layers[0].mlp.down_proj._forward_pre_hooks.values())[0](None,(x,))[0]
        self.assertEqual(float(y[0,-1,7]),.5);self.assertEqual(float(y[0,0,7]),1);self.assertEqual(float(y[0,-1,8]),1);h.remove()
if __name__=='__main__':unittest.main()
