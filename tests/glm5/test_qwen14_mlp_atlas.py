"""Isolated engineering fixtures. No fixture is registered as research evidence."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import qwen14_mlp_intervene as I


class NativeReplayTests(unittest.TestCase):
    def test_cached_prefix_matches_native_causal_block(self):
        import torch
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
        from transformers.cache_utils import DynamicCache
        torch.manual_seed(3)
        cfg=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        cfg._attn_implementation='sdpa'
        layer=Qwen3DecoderLayer(cfg,1).eval();x=torch.randn(2,19,32)
        positions=torch.arange(19)[None].expand(2,-1);pe=Qwen3RotaryEmbedding(cfg)(x,positions)
        mask=torch.full((2,1,19,19),float('-inf')).triu(1);cache=DynamicCache(config=cfg)
        with torch.inference_mode():
            full=layer(x,attention_mask=mask,position_ids=positions,position_embeddings=pe,past_key_values=cache,use_cache=True)
            states=x[:,-1,None].expand(2,3,32).clone()
            replay=I.suffix(layer,states,cache.layers[1].keys,cache.layers[1].values,cfg,1,positions,pe,mask[:,:,-1:,:])
        torch.testing.assert_close(replay,full[:,-1,None].expand_as(replay),atol=2e-6,rtol=2e-6)
        with torch.inference_mode():
            exact=I.exact_suffix(layer,states,x,cache.layers[1].keys,cache.layers[1].values,1,positions,pe,mask)
            self.assertTrue(torch.equal(exact,full[:,-1,None].expand_as(exact)))
            altered=states.clone();altered[:,1,0]+=.7
            tested=I.exact_suffix(layer,altered,x,cache.layers[1].keys,cache.layers[1].values,1,positions,pe,mask)
            x2=x.clone();x2[:,-1]=altered[:,1]
            direct=layer(x2,attention_mask=mask,position_ids=positions,position_embeddings=pe)
            torch.testing.assert_close(tested[:,1],direct[:,-1],atol=2e-6,rtol=2e-6)

    def test_actual_bf16_control_norm(self):
        import torch
        torch.manual_seed(7)
        base=torch.randn(4,5120).to(torch.bfloat16);desired=torch.randn(4,5120)*.15
        result,error,valid=I.norm_control(base,desired,torch.randn(4,5120))
        self.assertTrue(bool(valid.all()))
        actual=(result.float()-base.float()).norm(dim=-1);expected=desired.norm(dim=-1)
        self.assertTrue(bool(((actual-expected).abs()/expected<=.02).all()))

    def test_snapshot_and_boundary_guards(self):
        from fastapi import HTTPException
        from server.qwen14_mlp_service import check
        check(35,17407)
        for args in [(27,0,None),(35,17408,None),(28,0,'wrong')]:
            with self.assertRaises(HTTPException):check(*args)


if __name__=='__main__':unittest.main()
