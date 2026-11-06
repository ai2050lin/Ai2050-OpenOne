"""Real CUDA diagnosis of strict tail recomposition; no criterion relaxation."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import GpuLease,digest,file_sha
from shared.dist_protocol.natural_register_runner import snapshot
from mlp_fidelity_v2_tail import tail_logits

def main():
    import torch
    from transformers import AutoModelForCausalLM
    plan=P.read(P.OUT/'plan.json');c=P.read(P.OUT/'controlled-v4-corpus.json');lookup={(r['background_index'],r['mapping'],r['fact'],r['question']):r for r in c['items']};items=[]
    for b in [320,321]:
        h=int(digest({'seed':P.SEED,'b':b})[:8],16);items +=[lookup[b,(h>>1)&1,h&1,q] for q in [0,1]]
    ctx={};torch.set_num_threads(4)
    def same(a,b):return {'equal':torch.equal(a,b),'max_abs':float((a.float()-b.float()).abs().max()),'relative_L2':float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))}
    with GpuLease():
        assert snapshot(P.MODEL)==plan['model_snapshot'];m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False);l=m.model.layers[35]
        hs=[l.post_attention_layernorm.register_forward_pre_hook(lambda _m,args:ctx.update(r=args[0][:,-1].detach().clone())),l.mlp.register_forward_hook(lambda _m,args,o:ctx.update(y=o[:,-1].detach().clone())),m.model.norm.register_forward_pre_hook(lambda _m,args:ctx.update(norm_input=args[0].detach().clone())),m.model.norm.register_forward_hook(lambda _m,args,o:ctx.update(norm_output=o.detach().clone())),m.lm_head.register_forward_pre_hook(lambda _m,args:ctx.update(head_input=args[0].detach(),head_stride=list(args[0].stride())))]
        with torch.inference_mode():o=m(input_ids=torch.tensor([r['token_ids'] for r in items],device='cuda'),use_cache=True,logits_to_keep=1)
        for h in hs:h.remove()
        with torch.inference_mode():
            r,y=ctx['r'],ctx['y'];smallh=m.model.norm((r+y)[:,None]);small=tail_logits(m,r,y);original_head_replay=m.lm_head(ctx['head_input']);fullraw=ctx['norm_input'].clone();fullraw[:,-1:]=(r+y)[:,None];fullh=m.model.norm(fullraw);full=m.lm_head(fullh[:,-1:]);
            report={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'source_sha256':file_sha(__file__),'residual_sum':same((r+y),ctx['norm_input'][:,-1]),'small_norm':same(smallh,ctx['norm_output'][:,-1:]),'small_head':same(small,o.logits[:,0]),'native_head_replay':same(original_head_replay,o.logits),'full_layout':same(full,o.logits),'head_native_strides':ctx['head_stride'],'head_small_strides':list(smallh.stride()),'head_full_strides':list(fullh[:,-1:].stride())}
        assert report['full_layout']['equal'];assert snapshot(P.MODEL)==plan['model_snapshot'];P.save(P.OUT/'tail-layout-diagnostic.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
