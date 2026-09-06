"""CPU-only native source->normalizer->single-input identity checks."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from phase2620_native_coordinate_contract import RESULT,save
from phase2677_padded_native_runtime import padded_inputs
from phase2679_native_source_capture import NativeSourceCapture
from phase2679_native_source_ledger import real_token_data
import phase2680_native_mlp_source_paths as paths


def main():
    torch.set_num_threads(2);torch.manual_seed(2680)
    cfg=Qwen3Config(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).bfloat16().eval()
    paths.LAYERS=(0,1);paths.SITES=((0,7),(1,13));ids=[1,3,5,7]
    row={'prompt_ids':ids,'body_end_token':1,'task_end_token':3,'token_regions':[{'role':r} for r in ('body_entity_a','body_other','query_other','answer_rule')]}
    with torch.inference_mode(),NativeSourceCapture(model,(0,1)) as cap:
        cap.reset(1,3);cap.enabled=True;model.model(**padded_inputs(model,ids,2));cap.enabled=False
        data=real_token_data(cap.pack(),len(ids));weights={};meta={}
        for l,j in paths.SITES:
            b=model.model.layers[l];arr=cap.array
            weights[f'L{l}__Wo']=arr(b.self_attn.o_proj.weight);weights[f'L{l}__Wo_bias']=np.zeros(24)
            weights[f'L{l}__gamma']=arr(b.post_attention_layernorm.weight);meta[str(l)]={'epsilon':b.post_attention_layernorm.variance_epsilon}
            for kind in ('gate','up','down'):
                w=getattr(b.mlp,kind+'_proj').weight;weights[f'L{l}__J{j}_{kind}']=arr(w[:,j] if kind=='down' else w[j,:])
    dense,units=paths.one_case(row,data,weights,meta)
    maximum=max(b['reconstruction_max_abs'] for u in units for b in u['branches'].values())
    assert maximum<1e-12 and len(units)==2
    for l,j in paths.SITES:
        for kind in ('gate','up'):
            a=dense[f'L{l}_J{j}__{kind}__native_input']
            assert a.shape==(2,24) and np.array_equal(a,data[l]['mlp_x']*weights[f'L{l}__J{j}_{kind}'][None,:])
    for u in units:
        for b in u['branches'].values():assert all(r is None or -1e-12<=r<=1+1e-12 for r in b['cancellation_fraction'])
    report={'all_checks_passed':True,'all_incoming_coordinates':24,'max_known_identity_error':maximum,'zero_padding_excluded':True,
        'scope':'Synthetic tiny random BF16 CPU engineering check. No real512source data or Phase2680 completion.'}
    save(RESULT/'phase2677_source_role_contract/analysis/mlp_paths_preflight.json',report);print(report)


if __name__=='__main__':main()
