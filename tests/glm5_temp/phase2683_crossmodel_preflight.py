"""Tiny synthetic architecture/storage controls; zero pretrained GPU models."""
import sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,Qwen2Config,Qwen2ForCausalLM,GlmConfig,GlmForCausalLM
import phase2683_crossmodel_function_atlas as p


def main():
    torch.set_num_threads(2);torch.manual_seed(88);checks={}
    for name,config,cls in [('q3',Qwen3Config,Qwen3ForCausalLM),('q2',Qwen2Config,Qwen2ForCausalLM),('glm',GlmConfig,GlmForCausalLM)]:
        c=config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,eos_token_id=2,pad_token_id=2)
        c._attn_implementation='eager';m=cls(c).bfloat16().eval();cap=p.PaddedCapture(m,())
        with torch.inference_mode():
            inp=p.padded_inputs(m,[1,7,8,9],2,32);plain=m.model(**inp).last_hidden_state.detach().clone()
            cap.reset(2,True,3);cap.enabled=True;seen=m.model(**inp).last_hidden_state;cap.enabled=False
            assert torch.equal(plain,seen);pack=cap.pack();mom=cap.moment_pack()
            assert pack['h'].shape==(3,2,16) and pack['a'].shape==(2,2,32) and pack['full__h'].shape==(3,4,16)
            assert set(mom)=={k+'__'+s for k in ('h','x','gate','up','a','down') for s in ('sum','sumsq')}
            d=np.stack([p.unbits(pack['h'])]*4);counts=p.sign_counts(d)
            assert all(a.shape==(3,2,16) for a in counts.values())
        cap.close();checks[name+'_noop_fullcoords_unpadded_rows']=True
    temp=p.OUT/'synthetic'/str(uuid.uuid4());temp.mkdir(parents=True)
    q=temp/'sums.npz';v={'x':np.arange(32,dtype=np.float64).reshape(2,16)}
    assert p.accumulate_chunk(q,'first',v);assert not p.accumulate_chunk(q,'first',v);assert p.accumulate_chunk(q,'second',v)
    with np.load(q) as z:assert np.array_equal(z['x'],2*v['x']) and z['completed_chunks'].tolist()==['first','second']
    checks['atomic_replay_idempotent']=True
    assert p.final_text({'prompt':'<think>'},'still reasoning','ds7')==('',False)
    assert p.final_text({'prompt':'<think>'},'reason</think> Yes','ds7')==('Yes',True)
    assert p.final_text({'prompt':'<think></think>'},'Yes','qwen14')==('Yes',True)
    checks['DS_final_boundary']=True
    p.save(p.OUT/'analysis/runtime_preflight.json',{'synthetic_only':True,'checks':checks,'all_checks_passed':all(checks.values()),'research_cases':0});print(checks)


if __name__=='__main__':main()
