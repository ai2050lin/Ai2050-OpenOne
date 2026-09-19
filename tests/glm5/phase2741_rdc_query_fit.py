"""Full-coordinate diagonal decoder competition; frozen grouped/query holdouts."""
from collections import defaultdict
from rdc_query_common import *
from phase2741_rdc_query_rules import OUT,NAMES,freeze

def rows_and_queries():
    ids=set(freeze()['sample_ids'])
    rows=[r for r in gzread(BASE/'material/natural.json.gz') if r['sample_id'] in ids]
    probes=read(BASE/'probes/protocol.json')['probes']
    qi={s:np.array([i for i,q in enumerate(probes) if q['split']==s]) for s in ['train_query','validation_query','unseen_query']}
    return rows,probes,qi

def packet(row,pr,qi=None):
    sid=row['sample_id']
    with np.load(BASE/'capture/fields'/f'{sid}.npz') as z:
        h=unbits(z['prefix_layers'][12]).astype(float)
        targets=np.stack([unbits(z['query_H12_H24_rawH36'][:,1]),unbits(z['query_H12_H24_rawH36'][:,2]),unbits(z['postnorm'])]).astype(float)
    with np.load(OUT/'features'/f'{sid}.npz') as z:cand=unbits(z['candidate_H13']).astype(float)
    if qi is not None:targets=targets[:,qi];cand=cand[:,qi];pr=pr[qi]
    def feature(c):return np.stack([np.ones_like(pr),np.broadcast_to(h,pr.shape),pr,cand[c]],-1)
    return feature,targets

def predict(x,beta):return np.einsum('qdi,tdi->tqd',x,beta,optimize=True)

def main():
    if (OUT/'fit_result.json').exists():print('QUERY_RULE_FIT_ALREADY_COMPLETE');return
    assert read(OUT/'capture_result.json')['all_passed']
    start=time.monotonic();protocol=freeze();rows,probes,qi=rows_and_queries()
    with np.load(BASE/'prototypes/qwen4.npz') as z:pr=np.stack([unbits(z[f'p{i}_H13'][-1]) for i in range(100)]).astype(float)
    a=np.zeros((5,2560,4,4));b=np.zeros((5,3,2560,4));y_sum=np.zeros((3,2560));y2_sum=np.zeros_like(y_sum);count=0
    for i,row in enumerate(rows):
        if row['split']!='train':continue
        fx,y=packet(row,pr,qi['train_query']);count+=y.shape[1];y_sum+=y.sum(1);y2_sum+=(y*y).sum(1)
        for c in range(5):
            x=fx(c);a[c]+=np.einsum('qdi,qdj->dij',x,x,optimize=True)
            b[c]+=np.einsum('qdi,tqd->tdi',x,y,optimize=True)
        if (i+1)%24==0:print('QUERY_FIT_MOMENTS',i+1,len(rows),flush=True)
    mean=y_sum/count;var=np.maximum(y2_sum/count-mean*mean,1e-10)
    xm=a[:,:,0,:]/count;xs=np.sqrt(np.maximum(np.diagonal(a,axis1=-2,axis2=-1)/count-xm*xm,1e-10));xm[:,:,0]=0;xs[:,:,0]=1
    # x_standard = x_raw @ transform, a different native-coordinate4x4 transform per coordinate.
    tr=np.zeros_like(a)
    for j in range(4):tr[:,:,j,j]=1/xs[:,:,j]
    for j in range(1,4):tr[:,:,0,j]=-xm[:,:,j]/xs[:,:,j]
    aa=np.einsum('cdji,cdjk,cdkl->cdil',tr,a,tr,optimize=True)/count
    bb=np.einsum('ctdi,cdij->ctdj',b,tr,optimize=True)/count
    betas=[]
    for lam in protocol['lambdas']:
        reg=np.diag([0,lam,lam,lam]);coef=np.linalg.solve(aa[:,None]+reg,bb[...,None])[...,0]
        raw=np.einsum('cdij,ctdj->ctdi',tr,coef,optimize=True);betas.append(raw)
    betas=np.stack(betas);validation=np.zeros((len(betas),5,3));vcount=0
    for row in rows:
        if row['split']!='validation':continue
        fx,y=packet(row,pr,qi['validation_query']);vcount+=1
        for c in range(5):
            x=fx(c)
            for li in range(len(betas)):validation[li,c]+=((predict(x,betas[li,c])-y)**2).mean((1,2))
    selected=validation.argmin(0);beta=np.stack([np.stack([betas[selected[c,t],c,t] for t in range(3)]) for c in range(5)])
    npz(OUT/'decoder.npz',beta=beta,train_mean=mean,train_variance=var,validation_mse=validation/vcount,
      selected_lambda_index=selected,all_beta=betas,standardization_mean=xm,standardization_scale=xs)
    metrics=[];coordinate_sse=defaultdict(lambda:np.zeros((5,3,2560)));coordinate_count=defaultdict(int)
    for i,row in enumerate(rows):
        fx,y=packet(row,pr);pred=np.stack([predict(fx(c),beta[c]) for c in range(5)])
        err=(pred-y[None])**2;den=(y-mean[:,None])**2
        for qs,ix in qi.items():
            mse=err[:,:,ix].mean((2,3));baseline=den[:,ix].mean((1,2));key=row['split']+'/'+qs
            coordinate_sse[key]+=err[:,:,ix].sum(2);coordinate_count[key]+=len(ix)
            metrics.append({'sample_id':row['sample_id'],'source_group':row['source_group'],'cohort':row['cohort'],'source_split':row['split'],
              'query_split':qs,'mse':mse.tolist(),'train_mean_mse':baseline.tolist(),
              'relative_to_train_mean':(mse/np.maximum(baseline[None],1e-12)).tolist()})
        if (i+1)%24==0:print('QUERY_FIT_EVALUATION',i+1,len(rows),flush=True)
    compressed(OUT/'metrics.json.gz',metrics)
    npz(OUT/'all_coordinate_errors.npz',**{k.replace('/','__'):v/coordinate_count[k] for k,v in coordinate_sse.items()})
    summaries=[]
    for ss in ['train','validation','test']:
      for qs in qi:
        rs=[r for r in metrics if r['source_split']==ss and r['query_split']==qs]
        for cohort in ['all','gum','ewt','cmrc']:
            rr=[r for r in rs if cohort=='all' or r['cohort']==cohort];groups=[r['source_group'] for r in rr]
            for t,target in enumerate(protocol['targets']):
                value={'source_split':ss,'query_split':qs,'cohort':cohort,'target':target,'prefixes':len(rr),'candidate_mse':{},'paired_improvement':{}}
                for c,name in enumerate(NAMES):value['candidate_mse'][name]=clustered([r['mse'][c][t] for r in rr],groups)
                for c in range(4):
                    value['paired_improvement'][NAMES[c]+'_minus_ordered']=clustered([r['mse'][c][t]-r['mse'][4][t] for r in rr],groups)
                summaries.append(value)
    held=next(r for r in summaries if r['source_split']=='test' and r['query_split']=='unseen_query' and r['cohort']=='all' and r['target']=='postnorm')
    supported=all(held['paired_improvement'][k+'_minus_ordered']['interval95'][0]>0 for k in ['uniform','shuffled_values'])
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'training_endpoints':count,'validation_prefixes':vcount,'panel_prefixes':len(rows),
      'coefficients_per_candidate_per_target':2560*4,'selected_lambdas':[[protocol['lambdas'][j] for j in r] for r in selected.tolist()],
      'summaries':summaries,'selection_gate_passed':supported,'selection_gate':held,'decoder_sha256':sha(OUT/'decoder.npz'),
      'seconds':time.monotonic()-start,'limitations':['Same nominal coefficient budget does not equal effective rank.','Per-coordinate decoder cannot learn new cross-coordinate couplings; native feature already mixes all original coordinates.',
      'Query prototypes have seen all known query strings, but unseen-query contextual targets were never fit.','Source clusters may have multiple overlapping within-split windows.','Good prediction does not identify unique semantics or sufficient autoregressive state.']}
    save(OUT/'fit_result.json',result);ledger('full_coordinate_query_rule_fit',result['seconds']);print('QUERY_RULE_FIT_DONE',result['seconds'],'ordered_gate',supported,flush=True)

if __name__=='__main__':main()
