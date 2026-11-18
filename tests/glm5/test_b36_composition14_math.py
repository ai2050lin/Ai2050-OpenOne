"""CPU-only derivative and causal-mask engineering checks; no research evidence."""
import unittest
import torch
from b36_composition14_math import rms_jvp,last_attention_jvp,silu_product_jvp


class TestDerivatives(unittest.TestCase):
    def setUp(self):torch.manual_seed(418);torch.set_num_threads(2)

    def test_rms_matches_autograd(self):
        x=torch.randn(2,3,7,dtype=torch.float64);d=torch.randn(2,3,7,4,dtype=torch.float64);w=torch.randn(7,dtype=torch.float64)
        for i in range(4):
            fn=lambda z:w*z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)
            _,expected=torch.autograd.functional.jvp(fn,x,d[...,i]);actual=rms_jvp(x,d,w,1e-6)[...,i]
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)

    def test_attention_matches_finite_change(self):
        q=torch.randn(2,3,4,dtype=torch.float64);k=torch.randn(2,3,6,4,dtype=torch.float64);v=torch.randn_like(k)
        dq=torch.randn(2,3,4,2,dtype=torch.float64);dk=torch.randn_like(dq);dv=torch.randn_like(dq);last=torch.tensor([3,5]);scale=.5
        actual,_=last_attention_jvp(q,k,v,dq,dk,dv,last,scale)
        def calc(q_,k_,v_):
            scores=torch.einsum('bhd,bhtd->bht',q_,k_)*scale
            scores=scores.masked_fill(torch.arange(6)[None,None,:]>last[:,None,None],float('-inf'))
            return torch.einsum('bht,bhtd->bhd',scores.softmax(-1),v_)
        eps=1e-5
        for col in range(2):
            kk=k.clone();vv=v.clone();kminus=k.clone();vminus=v.clone()
            for b,t in enumerate(last):
                kk[b,:,t]+=eps*dk[b,...,col];vv[b,:,t]+=eps*dv[b,...,col]
                kminus[b,:,t]-=eps*dk[b,...,col];vminus[b,:,t]-=eps*dv[b,...,col]
            expected=(calc(q+eps*dq[...,col],kk,vv)-calc(q-eps*dq[...,col],kminus,vminus))/(2*eps)
            torch.testing.assert_close(actual[...,col],expected,rtol=1e-7,atol=1e-8)

    def test_future_keys_have_no_effect(self):
        q=torch.randn(1,2,4,dtype=torch.float64);k=torch.randn(1,2,7,4,dtype=torch.float64);v=torch.randn_like(k)
        d=torch.randn(1,2,4,1,dtype=torch.float64);last=torch.tensor([2])
        a,base=last_attention_jvp(q,k,v,d,d,d,last,.5)
        k[:,:,3:]*=100;v[:,:,3:]*=100
        b,base2=last_attention_jvp(q,k,v,d,d,d,last,.5)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(base,base2,rtol=0,atol=0)

    def test_silu_product_joint_term(self):
        g=torch.randn(4,9,dtype=torch.float64);v=torch.randn_like(g);dg=torch.randn(4,9,3,dtype=torch.float64);dv=torch.randn_like(dg)
        actual=silu_product_jvp(g,v,dg,dv)
        for col in range(3):
            _,expected=torch.autograd.functional.jvp(lambda a,b:torch.nn.functional.silu(a)*b,(g,v),(dg[...,col],dv[...,col]))
            torch.testing.assert_close(actual[...,col],expected,rtol=1e-10,atol=1e-10)

    def test_full_native_block_and_clamped_paths(self):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
        from b36_composition14_math import local_step
        from b36_composition14_tail import capture_hooks
        cfg=Qwen3Config(hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4,attention_dropout=0.)
        cfg._attn_implementation='eager';layer=Qwen3DecoderLayer(cfg,layer_idx=0).eval()
        x=torch.randn(2,7,16);pos=torch.arange(7)[None];pe=Qwen3RotaryEmbedding(cfg)(x,pos)
        mask=torch.full((1,1,7,7),float('-inf')).triu(1);last=torch.tensor([3,5]);batch=torch.arange(2)
        cap={};hooks=capture_hooks(layer,cap)
        with torch.no_grad():base=layer(x.clone(),attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
        for hook in hooks:hook.remove()
        direction=torch.randn(2,16);directions=direction[:,:,None].expand(-1,-1,8).clone()
        with torch.no_grad():actual,_,numerical=local_step(layer,x,cap,directions,last,pe,roots=[1,2])
        self.assertLess(numerical['attention_fp32_reconstruction_relative_L2'],1e-5)
        eps=.002
        for col in range(4):
            hooks=[]
            if col in [1,3]:
                def attn_hook(m,a,y):
                    z=y.clone();z[batch,last]=cap['attention'][batch,last];return z
                hooks.append(layer.self_attn.o_proj.register_forward_hook(attn_hook))
            if col in [2,3]:
                def mlp_hook(m,a,y):
                    z=y.clone();z[batch,last]=cap['mlp'][batch,last];return z
                hooks.append(layer.mlp.down_proj.register_forward_hook(mlp_hook))
            plus=x.clone();minus=x.clone();plus[batch,last]+=eps*direction;minus[batch,last]-=eps*direction
            with torch.no_grad():
                yp=layer(plus,attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
                ym=layer(minus,attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
            for hook in hooks:hook.remove()
            expected=(yp[batch,last]-ym[batch,last])/(2*eps)
            torch.testing.assert_close(actual[...,col],expected,rtol=.003,atol=.0002)


if __name__=='__main__':unittest.main(verbosity=2)
