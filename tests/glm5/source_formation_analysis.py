"""Frozen forecasts from earlier native coordinates, with explicit information budgets."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_binding_native_v2 import unbits
from source_formation_behavior import Runtime

METHODS=['native','random','embedding']
def load(model,split,name):return np.load(D.OUT/model/f'{split}-native/{name}.npy',mmap_mode='r')
def close(*arrays):
    for a in arrays:
        if hasattr(a,'_mmap') and a._mmap is not None:a._mmap.close()

def ridge(x,y):
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-5)
    a=np.c_[np.ones(len(x)),(x-mu)/sd].astype(np.float64)
    penalty=np.eye(a.shape[1]);penalty[0,0]=0
    coef=np.linalg.solve(a.T@a+penalty,a.T@y)
    return {'center':mu.astype('f'),'scale':sd.astype('f'),'coef':coef.astype('f')}
def predict(x,params):return params['coef'][0]+((x-params['center'])/params['scale'])@params['coef'][1:]
def coordinates(x,y,count=128):
    z=(x-x.mean(0))/np.maximum(x.std(0),1e-5);target=y-y.mean(0)
    cov=z.T@target/max(len(x)-1,1);score=np.square(cov).sum(1)
    return np.sort(np.argsort(score,kind='stable')[-count:]),score
def save_fit(store,prefix,x,y,idx):
    store[prefix+'_idx']=np.array(idx,np.int64)
    for k,v in ridge(x[:,idx],y).items():store[prefix+'_'+k]=v
def predict_fit(store,prefix,x):
    return predict(x[...,store[prefix+'_idx']],{k:store[prefix+'_'+k] for k in ['center','scale','coef']})

def bootstrap(rows,x,key):
    families=sorted({r['family'] for r in rows});means=np.array([np.asarray(x)[[i for i,r in enumerate(rows) if r['family']==f]].mean(0) for f in families])
    boot=means[D.rng(key).integers(0,len(means),(2000,len(means)))].mean(1)
    return np.quantile(boot,[.025,.975],axis=0).tolist()
def metric(rows,truth,pred,mean,key):
    se=np.square(truth-pred).sum(1);base=np.square(truth-mean).sum(1);gain=base-se
    return {'improvement':float(1-se.sum()/max(base.sum(),1e-12)),'MSE':float(se.mean()),
        'mean_baseline_MSE':float(base.mean()),'relative_error':float(np.sqrt(se.sum()/max(np.square(truth).sum(),1e-12))),
        'paired_SE_gain_ci_family':bootstrap(rows,gain,key)}
def softmax(x):
    p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
def relation_features(rows,length,interaction):
    result=[]
    for r in rows:
        seq=sorted(range(6),key=lambda j:r['records'][str(j)]['span'][0])
        order={j:i for i,j in enumerate(seq)}
        for j in range(6):
            sm=float(j//2==r['source']);em=float(j%2==r['entity']);idx=r['records'][str(j)]['span']
            f=[sm,em,order[j]/5,np.mean(idx)/len(r['tokens']),len(idx)/len(r['tokens']),float(r['distant']),float(r['label']=='tag')]
            if interaction:f.append(sm*em)
            result.append(f)
    return np.array(result,np.float32)

def fit(model):
    from source_formation_native import verify
    import torch
    verify();D.admission(model,'discovery');root=D.OUT/model;manifest=D.read(root/'discovery-native-manifest.json');assert manifest['replay']['passed']
    rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'];n=len(rows);timer=time.monotonic()
    mass=load(model,'discovery','mass');q=load(model,'discovery','qpre');k=load(model,'discovery','kpre');h=load(model,'discovery','early')
    cfg=D.read(D.OUT/f'{model}-identity.json')['config'];H=cfg['num_attention_heads'];KV=cfg['num_key_value_heads']
    heads=[]
    gold=np.array([r['target_record'] for r in rows])
    for bi,b in enumerate(D.BLOCKS):
        actual=mass[:,bi];correct=np.array([actual[i,:,g] for i,g in enumerate(gold)])
        ratio=correct/np.maximum(actual.sum(-1),1e-9)
        order=sorted(range(H),key=lambda j:(ratio[:,j].mean(),correct[:,j].mean(),-j),reverse=True);head=order[0]
        heads.append({'block':b,'head':head,'kv_head':head//(H//KV),'discovery_record_ratio':float(ratio[:,head].mean()),'discovery_record_mass':float(correct[:,head].mean())})
    xq=unbits(h[:,-1]);yq=np.concatenate([unbits(q[:,bi,c['head']]) for bi,c in enumerate(heads)],axis=1)
    xk=[];yk=[];tokenids=[]
    for i,r in enumerate(rows):
        ps=np.unique(np.linspace(0,len(r['tokens'])-1,min(24,len(r['tokens'])),dtype=int));ix=ps+manifest['length']-len(r['tokens'])
        xk.append(unbits(h[i,ix]));yk.append(np.concatenate([unbits(k[i,bi,c['kv_head'],ix]) for bi,c in enumerate(heads)],axis=1));tokenids.extend(r['tokens'][p] for p in ps)
    xk=np.concatenate(xk);yk=np.concatenate(yk)
    close(h,q,k,mass);D.log(model+'/formation-fit',1,8,timer,key_training_tokens=len(xk))
    rt=Runtime(model);w=rt.weight('model.embed_tokens.weight','cpu')
    eq=w[torch.tensor([r['tokens'][-1] for r in rows])].float().numpy()
    ek=w[torch.tensor(tokenids)].float().numpy();del w;gc.collect()
    params={};selection={}
    for kind,x,y,embed in [('q',xq,yq,eq),('k',xk,yk,ek)]:
        selected,score=coordinates(x,y);random=D.rng(model+'/'+kind+'/random').choice(x.shape[1],128,replace=False)
        identity,escore=coordinates(embed,y)
        for method,xx,idx in [('native',x,selected),('random',x,random),('embedding',embed,identity)]:save_fit(params,kind+'_'+method,xx,y,idx)
        params[kind+'_mean']=y.mean(0);selection[kind]={'native_coordinates':selected.tolist(),'random_coordinates':sorted(random.tolist()),'embedding_coordinates':identity.tolist()}
        D.log(model+'/formation-fit',2+(kind=='k'),8,timer,kind=kind)
    del xq,xk,yq,yk,eq,ek;gc.collect()
    ctx=load(model,'discovery','head_context');mass=load(model,'discovery','mass')
    for bi,c in enumerate(heads):
        params[f'B{c["block"]}_mean_context']=unbits(ctx[:,bi,c['head']]).mean(0)
        target=np.log(np.maximum(mass[:,bi,c['head']],1e-12));target-=target.mean(1,keepdims=True)
        for interaction in [False,True]:
            name=f'B{c["block"]}_relation_{int(interaction)}'
            for key,val in ridge(relation_features(rows,manifest['length'],interaction),target.reshape(-1,1)).items():params[name+'_'+key]=val
    close(ctx,mass)
    units=load(model,'discovery','units');norm=load(model,'discovery','down-norm2');mlps=[]
    for bi,b in enumerate([38,39]):
        a=unbits(units[:,bi]);score=a.var(0)*norm[bi];selected=np.sort(np.argsort(score,kind='stable')[-256:])
        available=np.setdiff1d(np.arange(a.shape[1]),selected);random=np.sort(D.rng(model+f'/mlp{b}').choice(available,256,replace=False))
        mlps.append({'block':b,'selected':selected.tolist(),'random':random.tolist()});params[f'B{b}_mean_units']=a.mean(0)
    close(units,norm)
    path=root/'formation-fit.npz';assert not path.exists();np.savez(path,**params)
    D.save(f'{model}/formation-fit.json',{'heads':heads,'mlps':mlps,'selection':selection,'key_fit_tokens':len(tokenids),
        'freeze_sha256':D.sha(D.OUT/'internal-freeze.json'),'fit_sha256':D.sha(path),'discovery_manifest_sha256':D.sha(root/'discovery-native-manifest.json'),
        'methods':METHODS,'QK_inputs':'hs20 earlier native coordinates or embedding only. No actual later Q/K in predictions.',
        'origin':'New discovery only; fixed block window informed by Phase2906.'},model)
    D.log(model+'/formation-fit',8,8,timer)

def rotary(rt,rows,L):
    import torch
    ids=torch.zeros((len(rows),L),dtype=torch.long)
    for i,r in enumerate(rows):ids[i,-len(r['tokens']):]=torch.arange(len(r['tokens']))
    rot=rt.Rotary(rt.cfg,device='cpu');cos,sin=rot(torch.zeros((len(rows),L,1),dtype=torch.bfloat16),ids)
    return cos,sin
def rotate_qk(rt,q,k,pe):
    import torch
    # Emulate the original BF16 RoPE boundary. Learned predictions are rounded at this boundary.
    qt=torch.from_numpy(np.ascontiguousarray(q)).bfloat16()[:,None,None]
    kt=torch.from_numpy(np.ascontiguousarray(k)).bfloat16()[:,None]
    qrot,_=rt.mod.apply_rotary_pos_emb(qt,qt,pe[0][:,-1:],pe[1][:,-1:])
    _,krot=rt.mod.apply_rotary_pos_emb(kt,kt,*pe)
    return qrot[:,0,0].float().numpy(),krot[:,0].float().numpy()
def attention(rt,q,k,v,pe,lengths,past_only=False):
    qr,kr=rotate_qk(rt,q,k,pe);score=np.einsum('nd,ntd->nt',qr,kr,optimize=True)/np.sqrt(q.shape[-1])
    valid=np.arange(k.shape[1])[None]>=k.shape[1]-np.array(lengths)[:,None]
    if past_only:valid[:,-1]=False
    score[~valid]=-np.inf;p=softmax(score);ctx=np.einsum('nt,ntd->nd',p,v,optimize=True)
    return ctx,p

def evaluate(model,split):
    from source_formation_native import verify
    import torch
    verify();D.admission(model,split);root=D.OUT/model;meta=D.read(root/'formation-fit.json');params=np.load(root/'formation-fit.npz')
    assert meta['fit_sha256']==D.sha(root/'formation-fit.npz')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];manifest=D.read(root/f'{split}-native-manifest.json')
    assert manifest['replay']['passed'];n=len(rows);L=manifest['length'];rt=Runtime(model);timer=time.monotonic()
    q=load(model,split,'qpre');k=load(model,split,'kpre');v=load(model,split,'vproj');h=load(model,split,'early');ctx=load(model,split,'head_context');pa=load(model,split,'attention')
    weights={}
    for bi,c in enumerate(meta['heads']):
        b,head=c['block'],c['head'];w=rt.weight(f'model.layers.{b}.self_attn.o_proj.weight','cpu')
        weights[bi]=w[:,head*rt.hd:(head+1)*rt.hd].float().numpy().copy();del w
    # Gather only the embedding columns used by the identity controls.
    emb=rt.weight('model.embed_tokens.weight','cpu');eq=emb[:,params['q_embedding_idx']].float().numpy().copy();ek=emb[:,params['k_embedding_idx']].float().numpy().copy();del emb;gc.collect()
    stored={};results=[]
    for bi,c in enumerate(meta['heads']):
        tag=f'B{c["block"]}H{c["head"]}';truth=unbits(ctx[:,bi,c['head']])@weights[bi].T
        mean=params[f'B{c["block"]}_mean_context']@weights[bi].T
        stored[tag+'_truth']=truth
        predictions={m:[] for m in METHODS+['full_native','native_past','native_past_shuffled']};qpred={m:[] for m in METHODS};kerror={m:[] for m in METHODS};kbase=[];prob={m:[] for m in predictions}
        actualp=np.array(pa[:,bi,c['head']]);actualq=unbits(q[:,bi,c['head']]);meanq=params['q_mean'][bi*rt.hd:(bi+1)*rt.hd]
        for lo in range(0,n,32):
            hi=min(lo+32,n);rs=rows[lo:hi];x=unbits(h[lo:hi]);lengths=[len(r['tokens']) for r in rs];pe=rotary(rt,rs,L)
            actualk=unbits(k[lo:hi,bi,c['kv_head']]);values=unbits(v[lo:hi,bi,c['kv_head']])
            mean_k=params['k_mean'][bi*rt.hd:(bi+1)*rt.hd]
            valid=np.arange(L)[None]>=L-np.array(lengths)[:,None];kbase.extend(np.sum(np.square(actualk-mean_k)*valid[:,:,None],axis=(1,2)).tolist())
            ids=np.zeros((len(rs),L),np.int64)
            for i,r in enumerate(rs):ids[i,-len(r['tokens']):]=r['tokens']
            for method in METHODS:
                if method=='embedding':
                    qx=eq[ids[:,-1]];kx=ek[ids]
                    predq=predict(qx,{key:params['q_embedding_'+key] for key in ['center','scale','coef']})
                    predk=predict(kx,{key:params['k_embedding_'+key] for key in ['center','scale','coef']})
                else:
                    predq=predict_fit(params,'q_'+method,x[:,-1]);predk=predict_fit(params,'k_'+method,x)
                predq=predq[:,bi*rt.hd:(bi+1)*rt.hd];predk=predk[:,:,bi*rt.hd:(bi+1)*rt.hd]
                qpred[method].append(predq);kerror[method].extend(np.sum(np.square(actualk-predk)*valid[:,:,None],axis=(1,2)).tolist())
                context,p=attention(rt,predq,predk,values,pe,lengths);predictions[method].append(context@weights[bi].T);prob[method].append(p)
                if method=='native':
                    past,pp=attention(rt,predq,predk,values,pe,lengths,True);predictions['native_past'].append(past@weights[bi].T);prob['native_past'].append(pp)
                    shuffled=values.copy()
                    for i,r in enumerate(rs):
                        for rec in range(6):
                            dest=np.array(r['records'][str(rec)]['span'])+L-len(r['tokens'])
                            source=np.array(r['records'][str((rec+1)%6)]['span'])+L-len(r['tokens'])
                            take=source[np.linspace(0,len(source)-1,len(dest)).astype(int)]
                            shuffled[i,dest]=values[i,take]
                    sc=np.einsum('nt,ntd->nd',pp,shuffled,optimize=True)
                    predictions['native_past_shuffled'].append(sc@weights[bi].T);prob['native_past_shuffled'].append(pp)
            full,p=attention(rt,actualq[lo:hi],actualk,values,pe,lengths);predictions['full_native'].append(full@weights[bi].T);prob['full_native'].append(p)
            del x,actualk,values,pe;gc.collect()
        entry={'id':tag,**c,'methods':{},'inputs':'Early hs20 for forecast Q/K, actual same-layer V for local write; native_past excludes final-query V.'}
        for method in predictions:
            pred=np.concatenate(predictions[method]);p=np.concatenate(prob[method]);stored[tag+'_'+method]=pred
            m=metric(rows,truth,pred,mean,f'{model}/{split}/{tag}/{method}')
            masses=np.zeros((n,6))
            for i,r in enumerate(rows):
                for j in range(6):masses[i,j]=p[i,np.array(r['records'][str(j)]['span'])+L-len(r['tokens'])].sum()
            m['record_accuracy']=float(np.mean(masses.argmax(1)==np.array([r['target_record'] for r in rows])))
            m['attention_KL']=float(np.mean(np.sum(actualp*(np.log(np.maximum(actualp,1e-12))-np.log(np.maximum(p,1e-12))),axis=1)))
            if method in METHODS:
                pq=np.concatenate(qpred[method]);m['query']=metric(rows,actualq,pq,meanq,f'{tag}/{method}/query')
                m['key_improvement']=float(1-np.sum(kerror[method])/max(np.sum(kbase),1e-12))
            entry['methods'][method]=m;stored[tag+'_'+method+'_record_mass']=masses
        entry['passes']=entry['methods']['native']['improvement']>=.1 and entry['methods']['native']['MSE']<entry['methods']['embedding']['MSE']
        assert entry['methods']['full_native']['relative_error']<.02,'Original QKV/WO replay failed'
        # The external conditional model predicts only relative six-record mass.
        mass=load(model,split,'mass');target=np.array(mass[:,bi,c['head']]);close(mass);conditional=target/np.maximum(target.sum(1,keepdims=True),1e-12)
        entry['relation']={}
        for interaction in [False,True]:
            key=f'B{c["block"]}_relation_{int(interaction)}'
            score=predict(relation_features(rows,L,interaction),{x:params[key+'_'+x] for x in ['center','scale','coef']}).reshape(n,6);p=softmax(score)
            entry['relation']['interaction' if interaction else 'additive']={'conditional_KL':float(np.mean(np.sum(conditional*(np.log(np.maximum(conditional,1e-12))-np.log(np.maximum(p,1e-12))),1))),
                'record_accuracy':float(np.mean(p.argmax(1)==np.array([r['target_record'] for r in rows]))),'feature_count':8 if interaction else 7}
        results.append(entry);D.log(model+'/'+split+'/formation',bi+1,4,timer)
    close(q,k,v,h,ctx,pa);path=root/f'{split}-formation-predictions.npz';assert not path.exists();np.savez(path,**stored)
    D.save(f'{model}/{split}-formation.json',{'candidates':results,'predictions_sha256':D.sha(path),'fit_sha256':meta['fit_sha256'],
        'manifest_sha256':D.sha(root/f'{split}-native-manifest.json'),'status':'Frozen local QK forecasts; full_native is algebra calibration only',
        'elapsed_seconds':time.monotonic()-timer},model)

def replication():
    root=D.OUT/'qwen14';panels=['confirm','lockbox_frame'];checks={}
    for s in panels:
        checks[s]={'behavior':D.read(root/f'{s}-gate.json')['passed'],'replay':D.read(root/f'{s}-native-manifest.json')['replay']['passed'],
            'passing':[x['id'] for x in D.read(root/f'{s}-formation.json')['candidates'] if x['passes']]}
    common=set(checks[panels[0]]['passing'])&set(checks[panels[1]]['passing'])
    D.save('qwen14/replication-decision.json',{'glm_eligible':bool(common) and all(x['behavior'] and x['replay'] for x in checks.values()),'checks':checks,'common_passing':sorted(common),'rule':'Frozen earlier-state formation criterion, both independent panels'},'qwen14')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate','replication']);ap.add_argument('--model');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='replication':replication()
    elif a.stage=='fit':fit(a.model)
    else:evaluate(a.model,a.split)
