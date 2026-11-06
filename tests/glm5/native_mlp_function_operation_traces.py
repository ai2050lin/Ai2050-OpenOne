"""Trace every changed capability item and representative applied joint edits."""
import sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_run_v4 import Recorder as BaseRecorder,status
from native_mlp_long_validate import install
from shared.dist_protocol.mechanism_trace import GpuLease,TraceStore,digest,save_json,file_sha
from shared.dist_protocol.natural_register_runner import snapshot

class Recorder(BaseRecorder):
    def __init__(self,*args,annotation=None,**kwargs):super().__init__(*args,**kwargs);self.annotation=annotation or {}
    def finish(self,*args,**kwargs):
        d,p=super().finish(*args,**kwargs);d.update(self.annotation)
        d['causal_visibility']='Readouts refer to the actual padded orunpadded tensor positions; attentionmask is saved when padding was used. No generated answer is part of input.'
        d.pop('trace_id');d['trace_id']=digest(d);save_json(p,d);return d,p
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    prov=P.read(P.OUT/'observations.json')['provenance'];groups=P.read(P.OUT/'dose-groups.json')['groups'];store=TraceStore();torch.set_num_threads(4);out=[]
    with GpuLease():
        for _ in range(30):
            if torch.cuda.mem_get_info()[0]>12*2**30:break
            time.sleep(2)
        assert torch.cuda.mem_get_info()[0]>12*2**30 and snapshot(P.MODEL)==prov['model_revision']
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);tok.padding_side='left';m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        qs=P.read(P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json');qidx={q['id']:i for i,q in enumerate(qs)};base={r['id']:r for r in P.read(P.OUT/'capability-base.json')};letter=[tok.encode(t,add_special_tokens=False)[0] for t in 'ABCDE']
        for group in ['native','coordinate']:
            edited=P.read(P.OUT/f'capability-{group}.json');changed=[r for r in edited if r['correct']!=base[r['id']]['correct']]
            for row in changed:
                index=qidx[row['id']];chunk=qs[(index//4)*4:(index//4+1)*4];target=next(q for q in chunk if q['id']==row['id']);chunk=[target]+[q for q in chunk if q['id']!=row['id']];tokens=[]
                for q in chunk:
                    text=q['question']+'\n'+'\n'.join(f'{l}. {a}' for l,a in zip('ABCDE',q['choices']))+'\n请只回答正确选项字母。'
                    tokens.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
                batch={k:v.cuda() for k,v in tok.pad({'input_ids':tokens},padding=True,return_tensors='pt').items()};ids=batch['input_ids'][0].tolist();p=len(ids)-1
                for condition,units in [('base',[]),('edited',groups[group]['units'])]:
                    item={'id':f'cap:{group}:{row["id"]}:{condition}','entity':row['id'],'scope':f'能力题复采 · {group} · {condition}','split':'capability_audit','token_ids':ids,'text':tok.decode(ids),'prompt':tok.decode(ids)}
                    annotation={'intervention':{'kind':'current-query native-unit ablation','units':units,'alpha':1.,'activation_is_intervened':bool(units)},'input_attention_mask':batch['attention_mask'][0].tolist(),
                        'readout_annotations':[{'position':p,'role':'assistant_boundary'}],'capability_audit':{'question_id':row['id'],'group':group,'condition':condition,'gate_eligible':row['gate_eligible'],'original_batch_shape_retained':True}}
                    hs=install(m,units,1.)
                    try:
                        with Recorder(m,[p],P.OUT/'operation-native'/digest(item['id'])[:20],annotation=annotation) as rec:
                            with torch.inference_mode():o=m(**batch,use_cache=False,output_hidden_states=True,logits_to_keep=1)
                    finally:
                        for h in hs:h.remove()
                    d,path=rec.finish(o,item,tok,prov['model_revision'],P.RUN+':capability-audit',prov);store.register(path);pred='ABCDE'[int(o.logits[0,-1,letter[:len(target['choices'])]].argmax())];expected=base[row['id']]['prediction'] if condition=='base' else row['prediction']
                    out.append({'run_id':d['run_id'],'sample_id':item['id'],'trace_id':d['trace_id'],'condition':condition,'group':group,'question_id':row['id'],'replay_prediction':pred,'original_prediction':expected,'prediction_consistent':pred==expected,'input_tokens':len(ids)})
        c=P.read(P.OUT/'corpus.json')
        for item in [r for r in c['items'] if r['family']=='location' and r['family_index']==288]:
            for condition,units in [('base',[]),('native',groups['native']['units']),('coordinate',groups['coordinate']['units'])]:
                copy={**item,'id':f'operation:{item["id"]}:{condition}'};hs=install(m,units,1.)
                try:
                    with Recorder(m,item['readout_positions'],P.OUT/'operation-native'/digest(copy['id'])[:20],annotation={'intervention':{'kind':'current-query native-unit ablation','units':units,'alpha':1.,'activation_is_intervened':bool(units)},'response_profile_stage_indices':[0,1,2,3]}) as rec:
                        with torch.inference_mode():o=m(input_ids=torch.tensor([item['token_ids']]*4,device='cuda'),use_cache=False,output_hidden_states=True,logits_to_keep=1)
                finally:
                    for h in hs:h.remove()
                d,path=rec.finish(o,copy,tok,prov['model_revision'],P.RUN+':operation-audit',prov);store.register(path);out.append({'run_id':d['run_id'],'sample_id':copy['id'],'trace_id':d['trace_id'],'condition':condition,'input_tokens':len(item['token_ids'])})
        assert snapshot(P.MODEL)==prov['model_revision']
    result={'kind':'real','run_id':P.RUN,'rows':out,'provenance':prov,'script_sha256':file_sha(__file__),'scope':'audit replays, not new independent benchmark questions; post-ablation A may differ fromSiLU(G)V bydesign'}
    P.save(P.OUT/'operation-traces.json',result);status('operation_audit',traces=len(out),capability_replay_consistent=all(r.get('prediction_consistent',True) for r in out));print(json.dumps({'traces':len(out)}))
if __name__=='__main__':
    while not (P.OUT/'place-polarity/polarity-result.json').exists():time.sleep(5)
    main()
