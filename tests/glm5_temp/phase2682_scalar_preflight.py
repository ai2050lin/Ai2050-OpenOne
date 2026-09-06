"""Synthetic CPU checks, not research measurements."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
import phase2682_resolved_scalar_paths as p


def main():
    torch.set_num_threads(2);torch.manual_seed(917)
    p.LAYERS=(0,1);p.SITES=((0,3),(1,4))
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,eos_token_id=2)
    cfg._attn_implementation='eager';m=Qwen3ForCausalLM(cfg).float().eval();cap=p.LocalCapture(m)
    r={'prompt_ids':[1,7,8,9],'observed_ids':[3,6,2],'body_end_token':2}
    with torch.inference_mode():
        b=p.forward(m,r,cap,p.LAYERS,True,True);noop=p.forward(m,r,cap,(),True)
        assert b['logprobs']==noop['logprobs'] and np.array_equal(b['state'],noop['state'])
        assert b['embedding'].shape==(6,16) and len(b['h'])==3
        maximum=0.;count=0
        for l,j in p.SITES:
            for kind in ('gate','up','down'):
                w=getattr(m.model.layers[l].mlp,kind+'_proj').weight;k=5;ij=(k,j) if kind=='down' else (j,k)
                old=w[ij].clone();down=p.arr(m.model.layers[l].mlp.down_proj.weight[:,j]).astype(np.float64)
                for delta in (-.02,.02):
                    try:
                        w[ij]=float(old)+delta;actual=p.forward(m,r,cap,(l,),True)
                        pred=p.local_prediction(b['data'],actual['data'],{'layer':l,'unit':j,'coordinate':k,'kind':kind},float(w[ij])-float(old),down)
                        maximum=max(maximum,pred['summary']['local_max_abs_error']);assert pred['actual'].shape==(6,16)
                        assert pred['summary']['local_max_abs_error']<1e-7;count+=1
                    finally:w[ij].copy_(old)
        restored=p.forward(m,r,cap,(),True);assert restored['logprobs64']==b['logprobs64']
    cap.close()
    p.save(p.OUT/'analysis/runtime_preflight.json',{'synthetic_cpu_only':True,'conditions':count,'maximum_local_error':maximum,'allchecks':True,'research_data':False})
    print('SYNTHETIC scalar preflight passed',count,maximum)


if __name__=='__main__':main()
