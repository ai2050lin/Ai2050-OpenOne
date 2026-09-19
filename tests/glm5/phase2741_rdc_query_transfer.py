"""Paired EN/ZH/code query response transfer, not cross-modal isomorphism."""
import argparse
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'transfer'
DIRECTIONS=[('python','en'),('en','python'),('zh','en'),('en_reordered','en')]
NAMES=['identity','query_only','affine','query_conditioned','shuffled_pair']

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    rows=gzread(PRIOR/'program_material.json.gz');assert len(rows)==768
    probes=read(BASE/'probes/protocol.json')['probes'];answer=next(i for i,p in enumerate(probes) if p['text']=='\nAnswer:')
    value={'timestamp':stamp(),'source':snapshot(__file__),'expressions':768,'semantic_groups':192,'queries':100,
      'directions':[list(p) for p in DIRECTIONS],'candidates':NAMES,'lambdas':[.0001,.01,1.,100.],
      'fit':'Allnative2560coordinates: affine [1,sourceResponse], query-conditioned [1,sourceResponse,queryOnlyResponse]. Ridge using trainsemanticgroups and60trainqueries only; intercept unpenalized; validationgroups/20queries selectlambda.',
      'inputs':'Observed source-expression query response and knownquery-only response. This is paired representation transfer, not an early-prefix prediction; observed target-expression response is forbidden input.',
      'shuffled_control':'Cyclically mispair source semanticgroups in training, retaining the same query; same3coefficient decoder and target rows.',
      'holdouts':'Original program semantic train96,validation32,test32,mixed_holdout32. All four expressions stay together; current vocabulary overlaps. Unseenquery contextual targets are never fit.',
      'injection':'Bounded prospective test on all32mixed-holdout ENexpressions with the fixed Answer-query suffix. Native versus one-shot codeidentity versus code-to-EN query-conditioned mapping of the recorded matching Python response. Subsequent generation uses each branch own history; no gold inputs. 1024token cap.',
      'injection_query_index':answer,'injection_branches':['native','code_identity','mapped_code'],'max_new_tokens':1024,
      'meaning':'Natural language and Python are both text. An approximate coordinate mapping is not a manifold isomorphism, proof of semantic purity, or AGI substrate.',
      'gates':'Report heldout error versus identity/query-only/shuffle and injection outcomes even if mapping fails, clearly treating failed-map deployment as stress-test not validated repair.'}
    compressed(OUT/'material.json.gz',rows);immutable(p,value);return value,rows

def capture():
    import torch
    from rdc_query_dynamic import QueryEngine
    protocol,rows=freeze()
    if (OUT/'capture_result.json').exists():return
    start=time.monotonic();guard(550*1024**2);model=None
    try:
      model,tok=load('qwen4',OUT);engine=QueryEngine(model);records=[]
      with torch.inference_mode():
        for i,row in enumerate(rows):
            sid=row['sample_id'];fp=OUT/'fields'/f'{sid}.npz';cp=OUT/'commits'/f'{sid}.json'
            if cp.exists():
                r=read(cp);assert sha(fp)==r['sha256'];records.append(r);continue
            pre=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True)
            post,stats=engine.run(pre.past_key_values,verify=i<4);npz(fp,postnorm=post,full_vocabulary_statistics=stats,unqueried_postnorm=bits(pre.last_hidden_state[0,-1]))
            r={'sample_id':sid,'source_group':row['source_group'],'representation':row['representation'],'split':row['split'],
              'queries':100,'sha256':sha(fp),'native_cached_fixed_query':True};save(cp,r);records.append(r);del pre
            if (i+1)%24==0:print('QUERY_TRANSFER_CAPTURE',i+1,768,round(time.monotonic()-start,1),flush=True);guard()
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'expressions':len(records),'semantic_groups':192,'query_endpoints':len(records)*100,
          'seconds':time.monotonic()-start,'scope':'Observed paired fixed-query responses from768previously frozen programs, not768independent semantics and not free-generated reasoning.'}
        save(OUT/'capture_result.json',result);ledger('paired_expression_native_query_capture',result['seconds']);print('QUERY_TRANSFER_CAPTURE_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

def fit():
    if (OUT/'fit_result.json').exists():return
    assert read(OUT/'capture_result.json')['all_passed'];start=time.monotonic();protocol,rows=freeze();probes=read(BASE/'probes/protocol.json')['probes']
    qi={s:np.array([i for i,p in enumerate(probes) if p['split']==s]) for s in ['train_query','validation_query','unseen_query']}
    with np.load(BASE/'prototypes/qwen4.npz') as z:prototype=unbits(z['postnorm']).astype(float)
    lookup=defaultdict(dict);arrays={}
    for r in rows:
        lookup[r['source_group']][r['representation']]=r
        with np.load(OUT/'fields'/f"{r['sample_id']}.npz") as z:arrays[r['sample_id']]=unbits(z['postnorm']).astype(np.float32)
    bysplit={s:sorted(g for g,rs in lookup.items() if next(iter(rs.values()))['split']==s) for s in sorted({r['split'] for r in rows})}
    print('TRANSFER_FIT_SPLITS',{k:len(v) for k,v in bysplit.items()},flush=True)
    validation_key='validation' if 'validation' in bysplit else 'val'
    metrics=[];coeffs={};selected_all={}
    def features(g,src,shuffle,ix):
        actual=shuffle.get(g,g);x=arrays[lookup[actual][src]['sample_id']][ix].astype(float)
        return np.stack([np.ones_like(x),x,prototype[ix]],-1)
    for src,dst in DIRECTIONS:
        direction=src+'_to_'+dst;order=bysplit['train'];shuffle={g:order[(i+1)%len(order)] for i,g in enumerate(order)};fitted={}
        for candidate in ['affine','query_conditioned','shuffled_pair']:
            width=2 if candidate=='affine' else 3;mapping=shuffle if candidate=='shuffled_pair' else {}
            aa=np.zeros((2560,width,width));bb=np.zeros((2560,width));count=0
            for g in order:
                x=features(g,src,mapping,qi['train_query'])[...,:width];y=arrays[lookup[g][dst]['sample_id']][qi['train_query']].astype(float)
                aa+=np.einsum('qdi,qdj->dij',x,x,optimize=True);bb+=np.einsum('qdi,qd->di',x,y,optimize=True);count+=len(y)
            xm=aa[:,0]/count;xs=np.sqrt(np.maximum(np.diagonal(aa,axis1=-2,axis2=-1)/count-xm*xm,1e-10));xm[:,0]=0;xs[:,0]=1
            tr=np.zeros_like(aa)
            for j in range(width):tr[:,j,j]=1/xs[:,j]
            for j in range(1,width):tr[:,0,j]=-xm[:,j]/xs[:,j]
            a=np.einsum('dji,djk,dkl->dil',tr,aa,tr,optimize=True)/count;b=np.einsum('di,dij->dj',bb,tr,optimize=True)/count
            betas=[];val=[]
            for lam in protocol['lambdas']:
                coef=np.linalg.solve(a+np.diag([0]+[lam]*(width-1)),b[...,None])[...,0];beta=np.einsum('dij,dj->di',tr,coef,optimize=True);betas.append(beta)
                losses=[]
                for g in bysplit[validation_key]:
                    x=features(g,src,{},qi['validation_query'])[...,:width];y=arrays[lookup[g][dst]['sample_id']][qi['validation_query']]
                    losses.append(float(((np.einsum('qdi,di->qd',x,beta,optimize=True)-y)**2).mean()))
                val.append(float(np.mean(losses)))
            li=int(np.argmin(val));fitted[candidate]=betas[li];coeffs[direction+'__'+candidate]=betas[li]
            selected_all[direction+'__'+candidate]={'lambda':protocol['lambdas'][li],'validation_mse':val,'training_endpoints':count}
        for split,gg in bysplit.items():
          for g in gg:
            x=features(g,src,{},np.arange(100));y=arrays[lookup[g][dst]['sample_id']].astype(float)
            pred=[x[...,1],prototype]+[np.einsum('qdi,di->qd',x[...,:fitted[c].shape[-1]],fitted[c],optimize=True) for c in ['affine','query_conditioned','shuffled_pair']]
            for qs,ix in qi.items():metrics.append({'direction':direction,'source_group':g,'split':split,'query_split':qs,
              'mse':[float(((v[ix]-y[ix])**2).mean()) for v in pred]})
        print('QUERY_TRANSFER_FIT',direction,flush=True)
    npz(OUT/'mapping.npz',**coeffs);compressed(OUT/'fit_metrics.json.gz',metrics);summaries=[]
    for direction in sorted({r['direction'] for r in metrics}):
      for split in bysplit:
       for qs in qi:
        rr=[r for r in metrics if (r['direction'],r['split'],r['query_split'])==(direction,split,qs)];groups=[r['source_group'] for r in rr]
        summaries.append({'direction':direction,'split':split,'query_split':qs,'groups':len(rr),
          'MSE':{n:clustered([r['mse'][i] for r in rr],groups) for i,n in enumerate(NAMES)},
          'control_minus_query_conditioned':{NAMES[i]:clustered([r['mse'][i]-r['mse'][3] for r in rr],groups) for i in [0,1,2,4]}})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'selected':selected_all,'summary':summaries,'seconds':time.monotonic()-start,
      'scope':'Approximate directed coordinate regressions evaluated on frozen semantic/query holdouts. No invertibility, topology preservation, unique semantics, or multi-step sufficiency has been established.'}
    save(OUT/'fit_result.json',result);ledger('paired_expression_full_coordinate_mapping',result['seconds']);print('QUERY_TRANSFER_FIT_DONE',result['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','capture','fit']);a=p.parse_args()
    if a.stage=='freeze':print('TRANSFER_PROTOCOL',freeze()[0])
    elif a.stage=='capture':capture()
    else:fit()
