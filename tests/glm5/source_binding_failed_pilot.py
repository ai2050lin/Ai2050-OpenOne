"""Behavior-only diagnosis after failed pilot; never overrides admission."""
import copy,gc,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_runtime import Runtime

def run():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    model='glm9';D.checked(model);root=D.OUT/model;gate=D.read(root/'pilot-gate.json');assert not gate['passed']
    assert not list(root.glob('*-native-manifest.json'))
    original=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'][:128];rows=copy.deepcopy(original)
    baseline=D.read(root/'pilot-behavior.json')['rows'];new=[[] for _ in rows];done=[False]*len(rows);timer=time.monotonic()
    D.save('glm9/pilot-output-diagnostic-started.json',{'purpose':'behavior-only failure diagnosis; no internal arrays, no candidate extraction, no gate override',
        'n':128,'max_new_tokens':8,'code_sha256':D.sha(Path(__file__)),'failed_gate_sha256':D.sha(root/'pilot-gate.json')},model)
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);eos=rt.cfg.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        for step in range(8):
            _,logits=rt.forward(rows);tokens=logits.argmax(-1).cpu().tolist()
            if step==0:assert tokens==[r['top1'] for r in baseline],'First-step replay mismatch'
            for i,token in enumerate(tokens):
                if done[i]:continue
                new[i].append(token);rows[i]['tokens'].append(token);done[i]=token in eos
            del logits;gc.collect();torch.cuda.empty_cache();D.log('glm9/failed-pilot-output-diagnostic',step+1,8,timer)
            if all(done):break
        result=[]
        for r,ids,stop in zip(original,new,done):
            text=rt.tok.decode(ids,skip_special_tokens=True).strip();result.append({'id':r['id'],'text':text,'gold':r['gold'],'tokens':ids,
                'terminated':stop,'exact_code_only':text==r['gold'],'case_and_terminal_period_normalized':text.lower().rstrip('.')==r['gold'].lower()})
    D.save('glm9/pilot-output-diagnostic.json',{'rows':result,'n':len(result),'exact_code_only':sum(r['exact_code_only'] for r in result)/len(result),
        'case_and_terminal_period_normalized':sum(r['case_and_terminal_period_normalized'] for r in result)/len(result),
        'terminated_fraction':sum(r['terminated'] for r in result)/len(result),'first_step_replay_exact':True,
        'gate_override':False,'internal_arrays_written':False,'status':'failure diagnosis on already exposed pilot only'},model)

if __name__=='__main__':run()
