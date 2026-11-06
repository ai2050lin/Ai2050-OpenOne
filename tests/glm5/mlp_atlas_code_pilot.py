"""Discovery-only CUDA behavioral calibration; never inspect heldout activations."""
import sys,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials as P
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.natural_register_runner import snapshot
from deploy.research_library import Library

def status(stage,state='done',**kw):
    p=P.PUBLIC/'00-progress.research.json';d=P.read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':P.RUN,'title':'全MLP单元：事实、问法与答案代码解耦','stages':{}}
    d['stages'][stage]={'status':state,'time':time.time(),**kw};P.save(p,d,False);Library(P.ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')
    print(json.dumps({'stage':stage,'status':state,**kw},ensure_ascii=False),flush=True)

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    c=P.read(P.OUT/'corpus.json');pr=P.read(P.PUBLIC/'prereg.research.json');assert digest(c)==pr['corpus_sha256'];items=[r for r in c['items'] if r['background_index']<16];assert len(items)==384 and all(r['split']=='discovery' for r in items)
    prov={'git_sha':subprocess.check_output(['git','-c',f'safe.directory={P.ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),'git_dirty':True,'model_path':str(P.MODEL),'model_revision':c['model_snapshot'],'seed':P.SEED}
    P.save(P.OUT/'pilot-freeze.json',{'kind':'real','run_id':P.RUN,'corpus_sha256':digest(c),'prereg_sha256':digest(pr),'dependencies':{str(p.relative_to(P.ROOT)):file_sha(p) for p in [Path(__file__),Path(P.__file__)]},'provenance':prov})
    status('pilot','running',total=len(items),heldout_inputs_evaluated=0)
    with GpuLease():
        assert snapshot(P.MODEL)==c['model_snapshot'];torch.set_num_threads(4);torch.manual_seed(P.SEED);start=time.time();tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        choices={s:list({tok.encode(t,add_special_tokens=False)[0] for t in [s,' '+s] if len(tok.encode(t,add_special_tokens=False))==1}) for labels in c['codes'].values() for s in labels};assert all(choices.values());rows=[]
        for start_i in range(0,len(items),2):
            chunk=items[start_i:start_i+2]
            with torch.inference_mode():o=m(input_ids=torch.tensor([r['token_ids'] for r in chunk],device='cuda'),use_cache=False,logits_to_keep=1)
            lp=o.logits[:,-1].float().log_softmax(-1)
            for j,r in enumerate(chunk):
                gap=float(lp[j,choices[r['expected']]].max()-lp[j,choices[r['contrast']]].max());top=int(lp[j].argmax())
                rows.append({k:r[k] for k in ['id','background_index','code','mapping','fact','question','decision','frame','expected','contrast','split']}|{'correct':gap>0,'choice_gap':gap,'first_top1':top in choices[r['expected']],'top1_id':top,'top1_text':tok.decode([top])})
            del o,lp
            if (start_i+2)%48==0:status('pilot','running',done=start_i+2,total=len(items),seconds=time.time()-start)
        cells=[]
        for key in sorted({tuple(r[k] for k in ['code','mapping','fact','question','frame']) for r in rows}):
            rr=[r for r in rows if tuple(r[k] for k in ['code','mapping','fact','question','frame'])==key];assert len(rr)==8
            cells.append(dict(zip(['code','mapping','fact','question','frame'],key))|{'n':8,'correct':sum(r['correct'] for r in rr),'passed':sum(r['correct'] for r in rr)>=7})
        result={'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'rows':rows,'cells':cells,'passed':all(r['passed'] for r in cells),'seconds':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'snapshot_unchanged':snapshot(P.MODEL)==c['model_snapshot'],'provenance':prov}
        P.save(P.OUT/'pilot-result.json',result);status('pilot','done' if result['passed'] else 'behavior_failed',passed=result['passed'],failed_cells=sum(not c['passed'] for c in cells),total_cells=len(cells),seconds=result['seconds'])
        print(json.dumps({'passed':result['passed'],'cells':len(cells),'failed':sum(not c['passed'] for c in cells),'peak_GiB':result['peak_cuda_bytes']/2**30}),flush=True)
if __name__=='__main__':main()
