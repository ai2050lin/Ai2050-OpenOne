"""Frozen discovery selectors and held-out native component predictions."""
import argparse,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_native import unbits
from source_binding_runtime import Runtime

def load(model,split,name):return np.load(D.OUT/model/f'{split}-native/{name}.npy',mmap_mode='r')
def verify_code():
    f=D.read(D.OUT/'internal-code-freeze.json')
    assert f['plan_sha256']==D.sha(D.OUT/'plan.json')
    for p,h in f['code_sha256'].items():assert D.sha(D.ROOT/p)==h,p

def bootstrap(rows,values,key):
    a=np.asarray(values,dtype=float);families=sorted({r['family'] for r in rows})
    means=np.asarray([a[[i for i,r in enumerate(rows) if r['family']==f]].mean(0) for f in families])
    rg=D.rng(key);boot=means[rg.integers(0,len(means),(2000,len(means)))].mean(1)
    return np.quantile(boot,[.025,.975],axis=0).tolist()

def choose_heads(mass,targets,count=4):
    correct=mass[np.arange(len(mass)),:,:,targets]
    ratio=correct/np.maximum(mass[:,:,:,:4].sum(-1),1e-10)
    ranks=sorted([(float(ratio[:,b,h].mean()),float(correct[:,b,h].mean()),b,h)
                  for b in range(8,min(36,mass.shape[1])) for h in range(mass.shape[2])],reverse=True)
    result=[];used=set()
    for ratio,absolute,b,h in ranks:
        if b not in used:result.append({'block':b,'head':h,'record_ratio':ratio,'record_mass':absolute});used.add(b)
        if len(result)==count:break
    return result

def fit(model):
    verify_code();D.admission(model,'discovery');root=D.OUT/model;manifest=D.read(root/'discovery-native-manifest.json')
    assert manifest['base_replay']['passed'];rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery']
    cfg=D.read(D.OUT/f'{model}-identity.json')['config'];H=cfg['num_attention_heads'];KV=cfg['num_key_value_heads']
    masses=load(model,'discovery','source_mass');heads=choose_heads(masses,np.array([r['target_record'] for r in rows]))
    q=load(model,'discovery','query');k=load(model,'discovery','key');ctx=load(model,'discovery','head_context');length=manifest['length']
    for c in heads:
        b,h=c['block'],c['head'];kv=h//(H//KV);energy=np.zeros(128,np.float64);n=0
        for i,r in enumerate(rows):
            z=unbits(q[i,b,h])[None]*unbits(k[i,b,kv,-len(r['tokens']):]);energy+=(z*z).sum(0);n+=len(z)
        keep=np.argsort(energy,kind='stable')[-32:];random=D.rng(f'{model}/B{b}H{h}/randomQK').choice(128,32,replace=False)
        c.update(keep=sorted(keep.tolist()),random_keep=sorted(random.tolist()),energy=(energy/n).tolist(),
                 mean_context=unbits(ctx[:,b,h]).mean(0).tolist(),kv_head=kv)
    norms=load(model,'discovery','down-column-norm2');units=load(model,'discovery','units');rank=[];start=time.monotonic()
    for b in range(40):
        a=unbits(units[:,b]);var=a.var(0,dtype=np.float64);rank.append((float((var*norms[b]).sum()),b))
        if b%5==4:D.log(model+'/fit',b+1,40,start)
    mlps=[]
    for _,b in sorted(rank,reverse=True)[:2]:
        a=unbits(units[:,b]);score=a.var(0,dtype=np.float64)*norms[b];selected=np.argsort(score,kind='stable')[-256:]
        random=D.rng(f'{model}/B{b}/randomMLP').choice(a.shape[1],256,replace=False)
        # Match per-unit RMS contribution on discovery, no held-out information.
        size=np.sqrt((a*a).mean(0)*norms[b]);available=set(range(a.shape[1]))-set(selected.tolist());matched=[]
        for j in sorted(selected.tolist(),key=lambda j:size[j],reverse=True):
            ix=min(available,key=lambda k:(abs(float(size[k]-size[j])),k));matched.append(ix);available.remove(ix)
        ratio=float(np.linalg.norm(size[matched])/max(np.linalg.norm(size[selected]),1e-9))
        mlps.append({'block':b,'selected':sorted(selected.tolist()),'random':sorted(random.tolist()),'matched':sorted(matched),
             'mean_units':a.mean(0).tolist(),'mean_output':unbits(load(model,'discovery','mlp_output')[:,b]).mean(0).tolist(),
             'matched_rms_ratio':ratio,'matched_control_valid':abs(ratio-1)<=.10,'selection_score':float(score.sum())})
    D.save(f'{model}/candidates.json',{'heads':heads,'mlps':mlps,'discovery_manifest_sha256':D.sha(root/'discovery-native-manifest.json'),
          'internal_code_sha256':D.sha(D.OUT/'internal-code-freeze.json'),'selection_scope':'discovery only, no heldout predictions observed'},model)

def sparse_context(q,k,v,keep,lengths):
    scores=np.einsum('ni,nti->nt',q[:,keep],k[:,:,keep],optimize=True)/np.sqrt(q.shape[1])
    valid=np.arange(k.shape[1])[None]>=k.shape[1]-np.asarray(lengths)[:,None];scores[~valid]=-np.inf
    p=np.exp(scores-scores.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    return np.einsum('nt,ntd->nd',p,v,optimize=True),p

def metrics(truth,pred,mean,rows,key):
    se=np.square(truth-pred).sum(1);base=np.square(truth-mean).sum(1);diff=base-se
    return {'mse':float(se.mean()),'mean_baseline_mse':float(base.mean()),'improvement':float(1-se.sum()/base.sum()),
       'paired_squared_error_gain_ci_family':bootstrap(rows,diff,key),'whole_component_relative_error':float(np.sqrt(se.sum()/np.square(truth).sum()))}

def evaluate(model,split):
    import torch
    verify_code();D.admission(model,split);root=D.OUT/model;candidates=D.read(root/'candidates.json')
    manifest=D.read(root/f'{split}-native-manifest.json');assert manifest['base_replay']['passed']
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];lengths=[len(r['tokens']) for r in rows]
    rt=Runtime(model);result=[];saved={};start=time.monotonic()
    with torch.inference_mode():
        for c in candidates['heads']:
            b,h=c['block'],c['head'];bi=manifest['head_blocks'].index(b);tag=f'B{b}H{h}'
            q=unbits(load(model,split,'query')[:,bi,h]);k=unbits(load(model,split,'key')[:,bi,c['kv_head']]);v=unbits(load(model,split,'vproj')[:,bi,c['kv_head']])
            actual=unbits(load(model,split,'head_context')[:,bi,h]);w=rt.weight(f'model.layers.{b}.self_attn.o_proj.weight','cpu').float().numpy()[:,h*128:(h+1)*128]
            truth=actual@w.T;mean=np.asarray(c['mean_context'])@w.T;entry={'kind':'head','id':tag,'block':b,'head':h,'methods':{}}
            for name,keep in [('selected32',c['keep']),('random32',c['random_keep']),('full128',list(range(128)))]:
                context,p=sparse_context(q,k,v,keep,lengths);pred=context@w.T;record_mass=np.zeros((len(rows),4))
                for i,r in enumerate(rows):
                    for j in range(4):record_mass[i,j]=p[i,np.array(r['records'][str(j)]['span'])+len(p[i])-lengths[i]].sum()
                hit=record_mass.argmax(1)==np.array([r['target_record'] for r in rows]);m=metrics(truth,pred,mean,rows,f'{model}/{split}/{tag}/{name}')
                m.update(binding_record_accuracy=float(hit.mean()),binding_ci_family=bootstrap(rows,hit,f'{tag}/binding/{name}'))
                entry['methods'][name]=m;saved[f'{tag}_{name}_write']=pred;saved[f'{tag}_{name}_record_mass']=record_mass
            entry['prediction_gate']=entry['methods']['selected32']['improvement']>=.10 and entry['methods']['selected32']['binding_record_accuracy']>=.65
            result.append(entry);saved[f'{tag}_actual']=truth
            D.log(model+'/'+split+'/prediction',len(result),6,start)
        for c in candidates['mlps']:
            b=c['block'];bi=manifest['mlp_blocks'].index(b);tag=f'B{b}MLP';truth=unbits(load(model,split,'mlp_output')[:,bi]);a=unbits(load(model,split,'units')[:,bi])
            mean=np.asarray(c['mean_output'],np.float32);amean=np.asarray(c['mean_units'],np.float32);w=rt.weight(f'model.layers.{b}.mlp.down_proj.weight','cpu').float().numpy()
            entry={'kind':'mlp','id':tag,'block':b,'methods':{}}
            for name,indices in [('selected256',c['selected']),('random256',c['random']),('matched256',c['matched'])]:
                pred=mean+(a[:,indices]-amean[indices])@w[:,indices].T
                entry['methods'][name]=metrics(truth,pred,mean,rows,f'{model}/{split}/{tag}/{name}');saved[f'{tag}_{name}_write']=pred
            entry['prediction_gate']=entry['methods']['selected256']['improvement']>=.10
            entry['matched_control_valid']=c['matched_control_valid'];result.append(entry);saved[f'{tag}_actual']=truth
            D.log(model+'/'+split+'/prediction',len(result),6,start)
    path=root/f'{split}-predictions.npz';assert not path.exists();np.savez(path,**saved)
    D.save(f'{model}/{split}-prediction.json',{'candidates':result,'predictions_sha256':D.sha(path),'candidates_sha256':D.sha(root/'candidates.json'),
           'native_manifest_sha256':D.sha(root/f'{split}-native-manifest.json'),'inputs':'actual same-layer native Q/K/V and MLP G/V; partial operator, not text-only prediction'},model)

def replication():
    verify_code();root=D.OUT/'qwen14';panels=['confirm','lockbox_frame'];checks={}
    for split in panels:
        gp=root/f'{split}-gate.json';mp=root/f'{split}-native-manifest.json';pp=root/f'{split}-prediction.json'
        checks[split]={'behavior':gp.exists() and D.read(gp)['passed'],'replay':mp.exists() and D.read(mp)['base_replay']['passed'],
                       'candidates':{x['id']:x['prediction_gate'] for x in D.read(pp)['candidates']} if pp.exists() else {}}
    common=set(checks[panels[0]]['candidates'])&set(checks[panels[1]]['candidates'])
    good=[k for k in common if all(checks[s]['candidates'][k] for s in panels)]
    ok=all(v['behavior'] and v['replay'] for v in checks.values()) and bool(good)
    D.save('qwen14/replication-decision.json',{'glm_eligible':ok,'checks':checks,'common_passing_candidates':good,'rule':'frozen plan; no retuning after confirmation'},'qwen14')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate','replication']);ap.add_argument('--model',default='qwen14');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='replication':replication()
    elif a.stage=='fit':fit(a.model)
    else:evaluate(a.model,a.split)
