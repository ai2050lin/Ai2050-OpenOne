"""Real CUDA gate and complete native module readouts for selected-entry roles."""
import sys,time,json,subprocess,shutil,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials as P
from shared.dist_protocol.mechanism_trace import GpuLease,Recorder as BaseRecorder,TraceStore,digest,file_sha,save_json
from shared.dist_protocol.natural_register_runner import snapshot
from deploy.research_library import Library

def status(stage,state='done',**kw):
    path=P.PUBLIC/'00-progress.research.json';d=P.read(path) if path.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':P.RUN,'title':'功能角色与词汇身份 · 等长单token选择的MLP研究','stages':{}}
    d['stages'][stage]={'state':state,'time':time.time(),**kw};d['updated_at']=time.time();P.save(path,d,False)
    Library(P.ROOT/'distributed_data/research_library').ingest(d,str(path.relative_to(P.ROOT)),kind='local_report');print(json.dumps({'stage':stage,'state':state,**kw},ensure_ascii=False),flush=True)

class Recorder(BaseRecorder):
    def finish(self,*a,**kw):
        d,p=super().finish(*a,**kw);d['causal_visibility']='First readout precedes the single differing active-key token andcannot encode selection. Final common-token readout follows key andboth records; later behavior question andanswer codes are not visible there.'
        d.pop('trace_id');d['trace_id']=digest(d);save_json(p,d);return d,p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['pilot','collect'],required=True);part=parser.parse_args().part
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    c=P.read(P.OUT/'corpus.json');pr=P.read(P.PUBLIC/'prereg.research.json');assert digest(c)==pr['corpus_sha256']
    prov={'git_sha':subprocess.check_output(['git','-c',f'safe.directory={P.ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),'git_dirty':True,'model_path':str(P.MODEL),'model_revision':c['model_snapshot'],'seed':P.SEED}
    if part=='collect':
        assert P.read(P.OUT/'pilot-result.json')['passed'];assert shutil.disk_usage(P.OUT).free>80*2**30
        files=['tests/glm5/native_mlp_function_materials.py','tests/glm5/native_mlp_function_discovery.py','tests/glm5/native_mlp_function_run.py','tests/glm5/native_mlp_function_analyze.py','shared/dist_protocol/mechanism_trace.py']
        P.save(P.OUT/'execution-freeze.json',{'kind':'real','run_id':P.RUN,'corpus_sha256':digest(c),'prereg_sha256':digest(pr),'dependencies':{n:file_sha(P.ROOT/n) for n in files},'provenance':prov})
    else:P.save(P.OUT/'pilot-freeze.json',{'script_sha256':file_sha(__file__),'prereg_sha256':digest(pr),'corpus_sha256':digest(c),'provenance':prov})
    with GpuLease():
        assert snapshot(P.MODEL)==c['model_snapshot'];torch.set_num_threads(4);torch.manual_seed(P.SEED);start=time.time()
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        choices={x:[tok.encode(s,add_special_tokens=False)[0] for s in [x,' '+x] if len(tok.encode(s,add_special_tokens=False))==1] for x in ['P','Q']};assert choices['P'] and choices['Q']
        items=[x for x in c['items'] if part=='collect' or x['split']=='discovery'];rows=[];bits=[];store=TraceStore()
        deep_ids={f'{family}:288:{value}' for family in P.FAMILIES for value in [0,1]}
        status('pilot' if part=='pilot' else 'collection','running',done=0,total=len(items))
        for i,item in enumerate(items):
            ids=torch.tensor([item['token_ids']],device='cuda')
            if part=='pilot':
                with torch.inference_mode():o=model(input_ids=ids,use_cache=False,logits_to_keep=1)
                trace_id=None
            else:
                folder=P.OUT/'native'/digest(item['id'])[:20]
                with Recorder(model,item['readout_positions'],folder,deep=item['id'] in deep_ids) as rec:
                    with torch.inference_mode():o=model(input_ids=ids,use_cache=False,output_hidden_states=True,logits_to_keep=1)
                d,p=rec.finish(o,item,tok,c['model_snapshot'],P.RUN,prov);store.register(p);trace_id=d['trace_id']
                assert max(v['attention_relative_l2'] for v in d['reconstruction'].values())<.05
            lp=o.logits[0,-1].float().log_softmax(-1);gap=float(lp[choices[item['expected']]].max()-lp[choices[item['contrast']]].max());top=int(lp.argmax())
            row={k:item[k] for k in ['id','pair_id','family','semantic','cue_form','frame','split','family_index','selected_key','positive_key','expected','contrast','input_tokens','readout_positions']}
            row.update(correct=gap>0,choice_gap=gap,first_top1=top in choices[item['expected']],top1_id=top,top1_text=tok.decode([top]),trace_id=trace_id);rows.append(row)
            if part=='collect':bits.append(o.logits[0,-1].contiguous().view(torch.uint16).cpu().numpy())
            del o
            if (i+1)%32==0:status('pilot' if part=='pilot' else 'collection','running',done=i+1,total=len(items),seconds=round(time.time()-start,1))
        result={'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'rows':rows,'provenance':prov,'seconds':time.time()-start,'snapshot_unchanged':snapshot(P.MODEL)==c['model_snapshot']}
        if part=='pilot':
            cells=[]
            for key in sorted(set((r['family'],r['cue_form'],r['frame'],r['semantic']) for r in rows)):
                rr=[r for r in rows if (r['family'],r['cue_form'],r['frame'],r['semantic'])==key];assert len(rr)==8
                cells.append({'family':key[0],'cue':key[1],'frame':key[2],'semantic':key[3],'n':len(rr),'correct':sum(r['correct'] for r in rr),'passed':sum(r['correct'] for r in rr)>=7})
            result.update(cells=cells,passed=all(x['passed'] for x in cells));P.save(P.OUT/'pilot-result.json',result);status('pilot','done' if result['passed'] else 'behavior_failed',passed=result['passed'],cells=cells)
        else:
            np.save(P.OUT/'answer-logits.npy',np.stack(bits));P.save(P.OUT/'observations.json',result);status('collection',total=len(rows),seconds=result['seconds'])
if __name__=='__main__':main()
