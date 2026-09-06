"""CPU synthetic native-factor roundtrip and complete-coordinate conservation."""
import os,sys,uuid
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from phase2620_native_coordinate_contract import RESULT,save
from phase2677_padded_native_runtime import padded_inputs,PaddedCapture
from phase2671_native_mlp_field import unbits
from phase2679_native_source_capture import NativeSourceCapture
from phase2679_native_source_ledger import exact_bits,pack_case,unpack_case,real_token_data,summarize_case


def main():
    torch.set_num_threads(2);torch.manual_seed(2679)
    cfg=Qwen3Config(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).bfloat16().eval();ids=[1,3,5,7]
    row={'prompt_ids':ids,'body_end_token':1,'task_end_token':3,'token_regions':[{'role':r} for r in ('body_entity_a','body_other','query_other','answer_rule')]}
    cap=PaddedCapture(model,(0,1));cap.reset(1,False,3);cap.enabled=True
    with torch.inference_mode(),NativeSourceCapture(model,(0,1)) as src:
        src.reset(1,3);src.enabled=True;model.model(**padded_inputs(model,ids,2));src.enabled=False
        original=src.pack();trimmed=real_token_data(original,len(ids));field=cap.pack();weights={}
        for l in (0,1):
            weights[f'L{l}__Wo']=src.array(model.model.layers[l].self_attn.o_proj.weight)
            weights[f'L{l}__Wo_bias']=np.zeros(24)
    cap.enabled=False;cap.close()
    path=RESULT/'phase2677_source_role_contract/synthetic_source_test'/f'{uuid.uuid4().hex}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**pack_case(trimmed));restored=unpack_case(path)
    assert all(np.array_equal(restored[l][k],v) for l,d in trimmed.items() for k,v in d.items() if k!='execution_dtype')
    assert all(np.array_equal(unbits(field['h'][l]),d['residual_before_attention']) and np.array_equal(unbits(field['a'][l]),d['mlp_a']) for l,d in restored.items())
    dense,reports=summarize_case(row,restored,weights);errors=[]
    for l in (0,1):
        for suffix in ('signed','absolute'):
            head=dense[f'L{l}__head_signed'] if suffix=='signed' else dense[f'L{l}__head_absolute_source_terms']
            role=dense[f'L{l}__role_signed'] if suffix=='signed' else dense[f'L{l}__role_absolute_head_source_terms']
            errors.append(float(np.max(np.abs(head.sum(axis=1)-role.sum(axis=1)))))
    assert max(errors)<1e-12 and all(r['future_P_zero'] and r['padding_source_contribution_zero'] for r in reports)
    try:exact_bits(np.asarray([0.123456789],dtype=np.float64))
    except AssertionError:refuses_lossy=True
    else:refuses_lossy=False
    assert refuses_lossy
    result={'all_checks_passed':True,'native_bit_roundtrip':True,'no_actual_token_removed':True,'native_H_a_bridge':True,
        'heads_roles_fullcoordinate_conservation_errors':errors,'refuses_nonBF16_lossy_storage':True,
        'scope':'Tiny random CPU BF16 engineering test only; no512pretrainedsourcecases collected here.'}
    save(RESULT/'phase2677_source_role_contract/analysis/source_preflight.json',result);print(result)


if __name__=='__main__':main()
