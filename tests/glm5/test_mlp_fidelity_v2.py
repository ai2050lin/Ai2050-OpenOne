"""CPU implanted software fixtures; none of these numbers are research observations."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_engine as E
import mlp_fidelity_v2_tail as T
from mlp_fidelity_v2_semantics_v2 import contrasts
from mlp_fidelity_v2_collect_v4 import first_word

class FidelitySoftwareTests(unittest.TestCase):
    def test_source_stratified_development_split(self):
        docs=[{'source':s,'doc_id':f'{s}:{i}'} for s in ['a','b','c','d'] for i in range(512)];a,b=E.indices(docs,7);self.assertEqual(len(a),1536);self.assertEqual(len(b),512);self.assertFalse(set(a)&set(b));self.assertEqual(sum(docs[i]['source']=='a' for i in b),128)
    def test_planted_property_industry_and_future_query_separate(self):
        x=np.zeros((32,2,2,2,2,3),np.float32)
        for r in [0,1]:
            for t in [0,1]:
                for q in [0,1]:x[:,r,t,q,:,0]=2*t-1;x[:,r,t,q,:,1]=2*(t^r)-1;x[:,r,t,q,1,2]=2*q-1
        c=contrasts(x);self.assertTrue((c['location'][...,0]==2).all());self.assertTrue((c['industry'][...,1]==2).all());self.assertTrue((c['query'][:,:,0,2]==0).all());self.assertTrue((c['query'][:,:,1,2]==2).all())
    def test_literal_generation_parser_handles_case_quotes_and_wrong_category(self):
        self.assertEqual(first_word('\n"Coastal".'),'coastal');self.assertNotEqual(first_word('fishing'),'coastal');self.assertNotEqual(first_word('The settlement is coastal'),'coastal')
    def test_no_change_in_tiny_monitor_is_not_noninferiority(self):
        r=T.capability_interval([1]*8,[1]*8,np.ones(8,bool));self.assertEqual(r['status'],'inconclusive')
    def test_large_monitor_checks_noninferiority_and_real_loss(self):
        e=np.ones(2200,bool);self.assertEqual(T.capability_interval(e,e,e)['status'],'noninferior');edited=e.copy();edited[:150]=False;self.assertEqual(T.capability_interval(e,edited,e)['status'],'inferior')
    def test_reference_only_losses_do_not_enter_formal_gate(self):
        base=np.ones(2340,bool);edited=base.copy();edited[2200:]=False;eligible=np.arange(2340)<2200;r=T.capability_interval(base,edited,eligible);self.assertEqual(r['losses'],0);self.assertEqual(r['n'],2200)
    def test_tail_matches_actual_rmsnorm_order_and_native_bfloat_rounding(self):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        torch.manual_seed(2);norm=Qwen3RMSNorm(8).to(torch.bfloat16);head=torch.nn.Linear(8,24,bias=False).to(torch.bfloat16);model=SimpleNamespace(model=SimpleNamespace(norm=norm),lm_head=head);r=torch.randn(3,8).to(torch.bfloat16);y=torch.randn(3,8).to(torch.bfloat16)
        with torch.inference_mode():expected=head(norm((r+y)[:,None]))[:,0];actual=T.tail_logits(model,r,y)
        self.assertTrue(torch.equal(actual,expected))
    def test_full_layout_tail_keeps_original_unembedding_stride(self):
        import torch
        import mlp_fidelity_v2_tail_v2 as V2
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        torch.manual_seed(7);norm=Qwen3RMSNorm(8).to(torch.bfloat16);head=torch.nn.Linear(8,24,bias=False).to(torch.bfloat16);model=SimpleNamespace(model=SimpleNamespace(norm=norm),lm_head=head);r=torch.randn(3,8).to(torch.bfloat16);y=torch.randn(3,8).to(torch.bfloat16);full=torch.randn(3,7,8).to(torch.bfloat16);full[:,-1]=r+y;strides=[];hook=head.register_forward_pre_hook(lambda _m,args:strides.append(tuple(args[0].stride())))
        with torch.inference_mode():expected=head(norm(full)[:,-1:])[:,0];actual=V2.tail_logits(model,r,y,full,[-1],1)
        hook.remove();self.assertTrue(torch.equal(actual,expected));self.assertEqual(strides[0],strides[1]);self.assertEqual(strides[-1][0],56)
    def test_feature_removal_retains_original_reconstruction_error(self):
        import torch
        from shared.dist_protocol.transcoder import TopKTranscoder
        torch.manual_seed(3);t=TopKTranscoder(8,8,16,4);y=torch.randn(5,8);a=torch.randn(5,64);down=torch.randn(8,64);z=t.encode(y);f=int(z.sum(0).argmax());ops,_=T.operations(t,y,a,down,[f]);without=z.clone();without[:,f]=0
        torch.testing.assert_close(ops['selected_1']-t.decode(without),y-t.decode(z),atol=1e-6,rtol=1e-5)
    def test_sparse_engine_recovers_implanted_representation_but_not_unrelated_input(self):
        import torch
        torch.set_num_threads(1);rng=np.random.default_rng(41);latent=rng.uniform(.1,2,(512,2)).astype(np.float32);y=np.zeros((512,8),np.float32);y[:,:2]=latent;y[:,2]=latent[:,0]-latent[:,1];y=torch.tensor(y);noise=torch.tensor(rng.normal(size=(512,8)).astype(np.float32))
        t,_=E.train(y[:256],y[:256],y[256:],y[256:],m=16,k=4,steps=500,batch=64,lr=.01,lr_end=.001,seed=8,checkpoints=[500]);self.assertLess(E.evaluate(t,y[256:],y[256:]),.15)
        n,_=E.train(noise[:256],y[:256],noise[256:],y[256:],m=16,k=4,steps=500,batch=64,lr=.01,lr_end=.001,seed=8,checkpoints=[500]);self.assertGreater(E.evaluate(n,noise[256:],y[256:]),.75)

if __name__=='__main__':unittest.main()
