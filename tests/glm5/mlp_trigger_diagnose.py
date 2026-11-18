"""Exposed-data diagnosis only: inspect native G/V/A and upstream residuals."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
from pathlib import Path
from time import monotonic
import numpy as np
import torch
import mlp_condition_design as D
import mlp_condition_run as R
import mlp_condition_stream_load as L
from shared.dist_protocol.mechanism_trace import GpuLease
from tools.power_analysis import check_phase

OUT=D.ROOT/'tests/glm5/result/mlp_trigger_v1'

def main():
    OUT.mkdir(exist_ok=True)
    old=D.READ(D.ROOT/'tests/glm5/result/mlp_condition_v2/prereg.json')
    p={'schema':'mlp_trigger_diagnostic.v1','provenance':{**old['provenance'],'tool':'exposed-prefix-diagnosis'},
       'source_sha256':D.HASH(Path(__file__)),'scope':'Only exposed previous material; no confirmatory claim or selected-sample inferential p-value',
       'power':{'next_confirm_rare':check_phase(.08,384,.1,.005),'next_confirm_common':check_phase(.15,384,.5,.005)}}
    D.SAVE(OUT/'diagnostic-prereg.json',p)
    D.OUT=OUT;D.checked=lambda:p
    R.save=lambda name,body,plan:D.SAVE(OUT/('diagnostic-'+name),{'provenance':p['provenance'],'diagnostic_plan_sha256':D.HASH(OUT/'diagnostic-prereg.json'),**body})
    fs=D.READ(D.ROOT/'tests/glm5/result/mlp_condition_v2/holdout-materials.json')['families']
    tok=L.AutoConfig # tokenizer loaded separately to avoid changing model implementation
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True)
    rows=[fs[f]['rows'][i] for f,i in [(0,0),(148,14),(389,2),(511,0)]]
    records=[];torch.set_num_threads(4);torch.manual_seed(2026110901)
    with GpuLease(),torch.inference_mode():
        model=L.load(D.MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        for row in rows:
            data={};handles=[]
            for b,u in R.UNITS.items():
                m=model.model.layers[b]
                for field,mod in [('G',m.mlp.gate_proj),('V',m.mlp.up_proj)]:
                    handles.append(mod.register_forward_hook(lambda mod,args,o,b=b,u=u,field=field:data.update({f'{field}{b}':o[0,:,u].float().cpu().numpy()})))
                handles.append(m.mlp.down_proj.register_forward_pre_hook(lambda mod,args,b=b,u=u:data.update({f'A{b}':args[0][0,:,u].float().cpu().numpy()})))
                handles.append(m.post_attention_layernorm.register_forward_pre_hook(lambda mod,args,b=b:data.update({f'R{b}':args[0][0].clone()})))
            model(input_ids=torch.tensor([row['tokens']],device='cuda'),use_cache=False,logits_to_keep=1)
            for h in handles:h.remove()
            peaks=np.argsort(abs(data['A33']))[-8:][::-1]
            detail=[]
            for q in peaks:
                r=data['R33'][q].float(); j=torch.topk(abs(r),5).indices
                detail.append({'position':int(q),'token':tok.decode([row['tokens'][q]]),'prefix':tok.decode(row['tokens'][max(0,q-18):q+1]),
                    'gva33':[float(data[k+'33'][q]) for k in 'GVA'],'gva34':[float(data[k+'34'][q]) for k in 'GVA'],
                    'R33_norm':float(r.norm()),'R33_topcoords':j.tolist(),'R33_topvalues':r[j].tolist()})
            arrays={k:v for k,v in data.items() if not k.startswith('R')}
            for b in R.UNITS:arrays[f'R{b}']=R.bits(data[f'R{b}'][peaks.tolist()])
            np.savez_compressed(OUT/f'diagnostic-{row["id"]}.npz',**arrays)
            records.append({'sample':row,'peaks':detail})
            print(json.dumps({'sample':row['id'],'peaks':detail[:3]},ensure_ascii=False),flush=True)
        D.SAVE(OUT/'diagnostic.json',{'provenance':p['provenance'],'records':records})
if __name__=='__main__':main()
