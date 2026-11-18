"""Stage routing: all behavior rows and gate committed before internal entry."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_runtime import Runtime

def behavior(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    pre=D.checked(model);folder=D.OUT/model;folder.mkdir(exist_ok=True)
    assert not (folder/f'{split}-behavior-started.json').exists(),'Behavior already started; explicit recovery/version required'
    if model=='glm9':
        gate=D.read(D.OUT/'qwen14/replication-decision.json');assert gate['glm_eligible'],'Qwen mechanism replication gate not passed'
    if split!='pilot':
        assert D.read(folder/'pilot-gate.json')['passed'],'Discovery-only pilot must pass before broader behavior'
    assert D.sha(D.OUT/f'{model}-materials.json')==pre['materials'][model]
    rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery' if split=='pilot' else split]
    if split=='pilot':rows=rows[:128]
    D.save(f'{model}/{split}-behavior-started.json',{'stage':'behavior_only','split':split,'n':len(rows),'internal_capture_enabled':False},model)
    start=time.monotonic();D.log(model+'/'+split+'/baseline',0,len(rows),start)
    assert snapshot(D.MODELS[model])==pre['models'][model]
    items=[];calibration=[]
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model)
        for chunk in range(0,len(rows),128):
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);h=rt.embed(ids)
            for b in range(rt.cfg.num_hidden_layers):
                layer=rt.block(b)
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));x=h[lo:hi];kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],
                        'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    y=layer(x,**kw)
                    if split=='pilot' and chunk==0 and lo==0 and b in [0,15,31,39]:
                        again=layer(x,**kw);equal=bool(torch.equal(y,again));assert equal
                        one=layer(x[:1],attention_mask=mask[:1],position_ids=pos[:1],position_embeddings=tuple(p[:1] for p in pe),use_cache=False)
                        err=float((y[:1].float()-one.float()).norm()/one.float().norm().clamp_min(1e-9));assert err<.02
                        calibration.append({'block':b,'same_shape_repeat_equal':equal,'batch1_vs8_relative_error':err})
                    h[lo:hi]=y
                del layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(model+'/'+split+'/baseline-block',chunk*40+(b+1)*len(rs),len(rows)*40,start,block=b)
            _,logits=rt.finish(h[:,-1]);items+=rt.score(logits,rs)
            del h,logits,ids,pos,pe,mask;gc.collect();torch.cuda.empty_cache()
            D.check_write(D.OUT,32*2**30,operation='来源绑定批次空间检查')
        scores=D.behavior(rows,items)
        D.save(f'{model}/{split}-behavior.json',{'rows':items,'complete':True,'n':len(rows),'calibration':calibration,
            'elapsed_seconds':time.monotonic()-start,'stage':'behavior_only','internal_arrays_written':False},model)
        gate={**scores,'stage':'behavior_complete','behavior_sha256':D.sha(folder/f'{split}-behavior.json'),
              'materials_sha256':D.sha(D.OUT/f'{model}-materials.json'),'plan_sha256':D.sha(D.OUT/'plan.json'),
              'model_snapshot':pre['models'][model]}
        D.save(f'{model}/{split}-gate.json',gate,model)
        D.log(model+'/'+split+'/gate',len(rows),len(rows),start,passed=scores['passed'],accuracy=scores['forced_accuracy'],top1=scores['top1_accuracy'])

def capture(model,split):
    proof=D.admission(model,split)  # Must run BEFORE importing capture or creating arrays/model.
    freeze=D.read(D.OUT/'internal-code-freeze.json')
    for name,value in freeze['code_sha256'].items():assert D.sha(D.ROOT/name)==value
    import source_binding_native
    source_binding_native.collect(model,split,proof)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['behavior','capture']);ap.add_argument('--model',choices=list(D.MODELS),required=True)
    ap.add_argument('--split',choices=['pilot','discovery','confirm','lockbox_frame'],required=True);a=ap.parse_args()
    globals()[a.stage](a.model,a.split)
