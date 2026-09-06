"""Tiny random CPU models test instrumentation; no pretrained research evidence."""
import os,sys,inspect
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import torch
import numpy as np
from transformers import Qwen3Config,Qwen3ForCausalLM,Qwen2Config,Qwen2ForCausalLM,GlmConfig,GlmForCausalLM
from phase2620_native_coordinate_contract import RESULT,save
from phase2679_native_source_capture import NativeSourceCapture
from phase2679_source_coordinate_ledger import attention_ledger,conditional_norm_ledger,input_weight_ledger


def main():
    torch.set_num_threads(2);torch.manual_seed(2679);checks=[]
    for name,config_class,model_class in [('qwen3',Qwen3Config,Qwen3ForCausalLM),('qwen2',Qwen2Config,Qwen2ForCausalLM),('glm',GlmConfig,GlmForCausalLM)]:
        cfg=config_class(vocab_size=128,hidden_size=24 if name=='qwen3' else 32,intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,pad_token_id=0,bos_token_id=1,eos_token_id=2)
        cfg._attn_implementation='eager';model=model_class(cfg).eval()
        ids=torch.arange(12).unsqueeze(0);module=inspect.getmodule(type(model.model.layers[0].self_attn));original=module.eager_attention_forward
        with torch.inference_mode():
            baseline=model.model(input_ids=ids,use_cache=False).last_hidden_state.clone()
            with NativeSourceCapture(model,(0,2)) as cap:
                cap.reset(5,11);cap.enabled=True
                captured=model.model(input_ids=ids,use_cache=False).last_hidden_state.clone();cap.enabled=False;data=cap.pack()
            assert module.eager_attention_forward is original and torch.equal(baseline,captured)
            for layer,row in data.items():
                assert row['actual_probability'].shape==(2,4,12),row['actual_probability'].shape
                assert row['actual_query_post_rope'].shape==(2,4,8),row['actual_query_post_rope'].shape
                assert row['actual_key_post_rope'].shape==row['actual_value'].shape==(12,2,8)
                block=model.model.layers[layer];att=block.self_attn;arr=lambda t:t.detach().double().numpy()
                ledger=attention_ledger(row['actual_probability'],row['actual_value'],arr(att.o_proj.weight),row['attention_output'],row['native_head_concat'],None if att.o_proj.bias is None else arr(att.o_proj.bias))
                norm=conditional_norm_ledger(row['residual_before_attention'],ledger,arr(block.post_attention_layernorm.weight),block.post_attention_layernorm.variance_epsilon,row['pre_mlp_norm'],row['mlp_x'])
                unit=7;projection=block.mlp.gate_proj if hasattr(block.mlp,'gate_proj') else block.mlp.gate_up_proj
                gate=input_weight_ledger(norm,arr(projection.weight[unit]),row['gate'][:,unit])
                error=max(float(np.abs(obj['reconstruction_error']).max()) for obj in (ledger,norm,gate))
                assert error<1e-12,error
                # Causal mask source positions after the first query must be zero.
                assert (row['actual_probability'][0,:,6:]==0).all()
                checks.append({'architecture':name,'layer':layer,'exact_noop':True,'all_coordinate_reconstruction_error':error})
            after=model.model(input_ids=ids,use_cache=False).last_hidden_state
            assert torch.equal(baseline,after)
    result={'all_checks_passed':True,'checks':checks,'scope':'Random tiny CPU Qwen3/Qwen2/GLM implementation and wrapper tests. Qwen3 head-concat width differs from residual width. No local pretrained model was loaded; no CUDA call; not evidence of linguistic mechanisms or a completed Phase.'}
    save(RESULT/'phase2676_native_mlp_delivery/analysis/next_capture_cpu_preflight.json',result)
    print('NATIVE CAPTURE SYNTHETIC CPU CHECKS',checks)


if __name__=='__main__':main()
