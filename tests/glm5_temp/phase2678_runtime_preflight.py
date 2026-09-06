"""Synthetic CPU-only fixed-padding/token-boundary/resume engineering tests."""
import os,sys,uuid
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from phase2620_native_coordinate_contract import RESULT,read,save
from phase2677_padded_native_runtime import PaddedCapture,padded_inputs
from phase2671_native_mlp_field import unbits
import phase2678_padded_source_field as runtime


def main():
    torch.set_num_threads(2);torch.manual_seed(2678)
    config=Qwen3Config(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,eos_token_id=2,pad_token_id=0)
    config._attn_implementation='eager';model=Qwen3ForCausalLM(config).bfloat16().eval()
    cap=PaddedCapture(model,(0,1));cap.reset(1,True,3);cap.enabled=True
    with torch.inference_mode():model.model(**padded_inputs(model,[1,3,5,7],2))
    cap.enabled=False;pack=cap.pack();moments=cap.moment_pack();cap.close()
    assert pack['full__h'].shape==(3,4,24)
    for key in ('h','a'):
        a=unbits(pack['full__'+key]).astype('float64')
        assert np.array_equal(unbits(pack[key]),a[:,[1,3]])
        assert np.array_equal(moments[key+'__sum'],a.sum(axis=1))
        assert np.array_equal(moments[key+'__sumsq'],(a*a).sum(axis=1))
    class Tokenizer:
        eos_token_id=2
        def decode(self,ids,skip_special_tokens):return ' '.join(str(i) for i in ids if i!=2)
    runtime.OUT=RESULT/'phase2677_source_role_contract/synthetic_runtime_test'/uuid.uuid4().hex;runtime.LAYERS=(0,1)
    assert not runtime.OUT.exists(),'Exercise a genuinely fresh output directory.'
    source=read(RESULT/'phase2677_source_role_contract/material/cases.json')
    cases=[]
    for i,old in enumerate(source[:4]):
        cases.append({**old,'prompt_ids':[1,3,5,7+i],'body_end_token':1,'task_end_token':3,'published':i<2,'parameter_published':i==0})
    first=runtime.run(model,Tokenizer(),cases);second=runtime.run(model,Tokenizer(),cases)
    assert first==second and len(first)==4
    with np.load(runtime.OUT/'field/case_0001.npz') as z:assert set(z.files)=={'h','a','full__h'}
    with np.load(runtime.OUT/'field/case_0002.npz') as z:assert set(z.files)=={'h','a'}
    report={'all_checks_passed':True,'checks':{'fresh_output_directory_initialization':True,'real_query_indices':True,'zero_padding_in_moments':True,'fullH_only_publication':True,'native_all_boundary_storage':True,'completed_resume_no_duplicate':True},
        'scope':'Synthetic tiny random CPU model, not pretrained/native language observations. Separate synthetic_runtime_test directory; no formal2678 samples.'}
    save(RESULT/'phase2677_source_role_contract/analysis/runtime_preflight.json',report);print(report)


if __name__=='__main__':main()
