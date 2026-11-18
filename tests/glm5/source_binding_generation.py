"""Frozen small full-answer diagnostic; never replaces the main behavior gate."""
import argparse,copy,gc,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_runtime import Runtime
from source_binding_analysis import verify_code

def run(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    verify_code();D.admission(model,split);original=D.read(D.OUT/f'{model}-materials.json')['splits'][split][:64]
    rows=copy.deepcopy(original);new=[[] for _ in rows];done=[False]*len(rows);start=time.monotonic()
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);eos=rt.cfg.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        for step in range(4):
            _,lg=rt.forward(rows);tokens=lg.argmax(-1).cpu().tolist()
            for i,token in enumerate(tokens):
                if done[i]:continue
                new[i].append(token);rows[i]['tokens'].append(token);done[i]=token in eos
            del lg;gc.collect();torch.cuda.empty_cache();D.log(model+'/'+split+'/generation',step+1,4,start)
            if all(done):break
        results=[]
        for r,ids,stopped in zip(original,new,done):
            text=rt.tok.decode(ids,skip_special_tokens=True).strip()
            results.append({'id':r['id'],'tokens':ids,'text':text,'gold':r['gold'],'exact_answer':text==r['gold'],'terminated':stopped})
    D.save(f'{model}/{split}-generation.json',{'rows':results,'n':len(results),'exact_accuracy':sum(r['exact_answer'] for r in results)/len(results),
          'terminated_fraction':sum(r['terminated'] for r in results)/len(results),'max_new_tokens':4,'sampling':'first8 frozen families, greedy; descriptive only'},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--split',required=True);a=ap.parse_args();run(a.model,a.split)
