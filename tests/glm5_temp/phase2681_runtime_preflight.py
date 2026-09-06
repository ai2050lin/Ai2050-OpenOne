"""Synthetic fresh-start, mid-group crash/resume, padding and publication tests."""
import os,sys,uuid
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from phase2620_native_coordinate_contract import RESULT,read,save
from phase2679_native_source_capture import NativeSourceCapture
import phase2680_native_mlp_source_paths as paths
import phase2681_fresh_source_confirmation as runtime


def main():
    torch.set_num_threads(2);torch.manual_seed(2681)
    cfg=Qwen3Config(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).bfloat16().eval()
    class ConstantHead(torch.nn.Module):
        def forward(self,x):
            result=torch.zeros(x.shape[:-1]+(32,),dtype=x.dtype,device=x.device);result[...,2]=1;return result
    model.lm_head=ConstantHead()
    paths.LAYERS=runtime.LAYERS=(0,1);paths.SITES=runtime.SITES=((0,7),(1,13))
    weights={};meta={'layers':{}}
    for l,j in paths.SITES:
        b=model.model.layers[l];arr=NativeSourceCapture.array
        weights[f'L{l}__Wo']=arr(b.self_attn.o_proj.weight);weights[f'L{l}__Wo_bias']=np.zeros(24)
        weights[f'L{l}__gamma']=arr(b.post_attention_layernorm.weight);meta['layers'][str(l)]={'epsilon':b.post_attention_layernorm.variance_epsilon}
        for kind in ('gate','up','down'):
            w=getattr(b.mlp,kind+'_proj').weight;weights[f'L{l}__J{j}_{kind}']=arr(w[:,j] if kind=='down' else w[j,:])
    runtime.source_weights=lambda model:(weights,meta)
    runtime.OUT=RESULT/'phase2677_source_role_contract/synthetic_confirmation_test'/uuid.uuid4().hex
    cases=[]
    for r in read(RESULT/'phase2681_fresh_source_confirmation/material/cases.json')[:256]:
        cases.append({**r,'prompt_ids':[1,3,5,7+r['target_index']],'body_end_token':1,'task_end_token':3,
            'token_regions':[{'role':role} for role in ('body_entity_a','body_other','query_other','answer_rule')]})
    class Tokenizer:
        eos_token_id=2
        def __init__(self,fail=False):self.fail=fail;self.calls=0
        def decode(self,ids,skip_special_tokens):
            self.calls+=1
            if self.fail and self.calls==6:raise RuntimeError('intentional synthetic mid-group interruption')
            return ''
    try:runtime.run(model,Tokenizer(True),cases)
    except RuntimeError as e:assert str(e)=='intentional synthetic mid-group interruption'
    else:raise AssertionError('Expected synthetic interruption did not fire')
    lines=(runtime.OUT/'analysis/records.jsonl').read_text(encoding='utf-8').splitlines();assert len(lines)==5
    records=runtime.run(model,Tokenizer(),cases);again=runtime.run(model,Tokenizer(),cases)
    assert records==again and len(records)==256 and [r['case_index'] for r in records]==list(range(256))
    assert sum(r['source_selected'] for r in records)==32 and len(read(runtime.OUT/'analysis/raw_manifest.json'))==8
    for fn in ('truth','mapped_truth','name','cloze'):
        m=read(runtime.OUT/f'analysis/moments_chronology_en_{fn}.json');assert m=={'cases':64,'actual_tokens':256,'padding_included':0}
    with np.load(runtime.OUT/'maps/fresh_chronology_en.npz') as z:
        assert z['h__all4_same_nonzero'].shape==(4,3,2,24)
        assert all(z[k].min()>=0 and z[k].max()<=8 for k in z.files if z[k].dtype==np.uint16)
    report={'all_checks_passed':True,'synthetic_conditions':256,'partial_group_interrupted_after':5,'exact_replay_no_duplicates':True,'complete_resume_unchanged':True,
        'all_tokens_exclude_padding':True,'32_source_cases_8_published_packs':True,'full_coordinate_counts_bounded':True,
        'scope':'Tiny random BF16 CPU model with constantEOS testhead. Fake tokenIDs and blank answers test engineering only, not real language data, not4096modelcompletion.'}
    save(RESULT/'phase2681_fresh_source_confirmation/analysis/runtime_preflight.json',report);print(report)


if __name__=='__main__':main()
