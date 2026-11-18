"""Tiny randomly initialized HF architectures: isolated engineering calibration only."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import source_binding_design as D
from source_binding_native import Capture
from source_binding_causal import Editor
from source_binding_analysis import sparse_context,choose_heads

def main():
    import torch
    from transformers import Qwen3Config,GlmConfig
    from transformers.models.qwen3 import modeling_qwen3 as Q
    from transformers.models.glm import modeling_glm as G
    from shared.dist_protocol.mechanism_trace import GpuLease
    torch.set_num_threads(4);torch.manual_seed(170);results=[]
    with GpuLease(),torch.inference_mode():
        for config,mod,prefix in [(Qwen3Config,Q,'Qwen3'),(GlmConfig,G,'Glm')]:
            cfg=config(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
            cfg._attn_implementation='sdpa';layer=getattr(mod,prefix+'DecoderLayer')(cfg,0).cuda().bfloat16().eval()
            rt=SimpleNamespace(t=torch,mod=mod,hd=16,heads=4,kv=2)
            x=torch.randn(2,12,64,device='cuda',dtype=torch.bfloat16);pos=torch.arange(12,device='cuda')[None].expand(2,-1)
            rotary=getattr(mod,prefix+'RotaryEmbedding')(cfg,device='cuda');pe=rotary(x,pos);mask=torch.ones((2,1,12,12),device='cuda',dtype=torch.bool).tril()
            rows=[{'tokens':list(range(12)),'records':{str(j):{'span':[j*2,j*2+1]} for j in range(4)},'headers':{'0':[8],'1':[9]},'query_positions':[10,11],'target_record':0,'source':0,'entity':0} for _ in range(2)]
            cap=Capture(rt,layer);y=layer(x,attention_mask=mask,position_embeddings=pe);data=cap.decode(rows,12,{'position_embeddings':pe})
            err=float(data['qkv_error'].max());cap.close()
            c={'heads':[{'block':0,'head':1,'keep':list(range(8)),'random_keep':list(range(8,16))}],'mlps':[{'block':0,'selected':list(range(8)),'matched':list(range(8,16))}]}
            ed=Editor(rt,c);ed.rows=rows;mod.ALL_ATTENTION_FUNCTIONS.register('source_binding_engineering',ed.attention);cfg._attn_implementation='source_binding_engineering'
            noop=layer(x,attention_mask=mask,position_embeddings=pe);assert torch.equal(y,noop)
            changed={};handle=layer.mlp.down_proj.register_forward_pre_hook(ed.mlp_hook(c['mlps'][0]))
            for arm in ['correct_edge','wrong_edge','sparse_qk','random_qk','mlp_selected','mlp_matched']:
                ed.arm=arm;edited=layer(x,attention_mask=mask,position_embeddings=pe)
                assert torch.equal(y[:,:-1],edited[:,:-1]);assert not torch.equal(y[:,-1],edited[:,-1]);changed[arm]=True
            handle.remove();cfg._attn_implementation='sdpa';assert torch.equal(y,layer(x,attention_mask=mask,position_embeddings=pe))
            results.append({'architecture':prefix,'qkv_relative_error':err,'noop_and_restore_exact':True,'all_arms_local_and_effective':changed})
            del layer;torch.cuda.empty_cache()
    rg=np.random.default_rng(2);q=rg.normal(size=(3,8)).astype('f');k=rg.normal(size=(3,10,8)).astype('f');v=rg.normal(size=(3,10,8)).astype('f')
    context,p=sparse_context(q,k,v,list(range(8)),[10,8,5]);assert np.allclose(p.sum(1),1) and np.all(p[2,:5]==0)
    m=np.ones((12,40,4,8),dtype='f')*.01;target=np.arange(12)%4
    for i in range(12):m[i,20,2,target[i]]=.8
    chosen=choose_heads(m,target);assert chosen[0]['block']==20 and chosen[0]['head']==2
    D.save('engineering-calibration.json',{'status':'passed','isolated_random_architectures':results,'not_scientific_model_evidence':True,
            'sparse_prefix_mask_and_implanted_head_selection':True})

if __name__=='__main__':main()
