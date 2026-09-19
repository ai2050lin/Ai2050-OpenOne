"""Independent-document confirmation and explicitly stronger-input diagnostic."""
import argparse
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'followup'

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    required=['atlas/result.json','events/result.json','rules/fit_result.json','rules/vocabulary_result.json','transfer/injection/result.json','formation/result.json']
    required += ['scale/'+m+'/result.json' for m in ['qwen4','qwen14','glm4']]
    required += ['late/'+b+'/result.json' for b in ['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']]
    for f in required:assert (BASE/f).exists(),('Finish integrated preceding stage first',f)
    candidates=gzread(BASE/'material/followup_candidates.json.gz');main=gzread(BASE/'material/natural.json.gz');maingroups={r['source_group'] for r in main};rows=[]
    selected_by_cohort={};available={}
    for c in ['gum','ewt','cmrc']:
        pool=sorted([r for r in candidates if r['cohort']==c],key=lambda r:rank('independent/'+r['source_id']));unique={}
        for r in pool:unique.setdefault(r['source_group'],r)
        selected_by_cohort[c]=list(unique.values());available[c]=len(unique)
    quotas={c:min(32,len(rr)) for c,rr in selected_by_cohort.items()}
    while sum(quotas.values())<96:
        before=sum(quotas.values())
        for c in ['ewt','cmrc','gum']:
            if sum(quotas.values())<96 and quotas[c]<available[c]:quotas[c]+=1
        assert sum(quotas.values())>before,('Fewer than96independent reserved documents',available)
    assert quotas=={'gum':19,'ewt':39,'cmrc':38},quotas
    for c in ['gum','ewt','cmrc']:
        chosen=selected_by_cohort[c][:quotas[c]]
        for r in chosen:
            x={k:v for k,v in r.items() if k not in ['tokens','full_context']}
            x.update(sample_id='q2743_'+rank(c+'/'+r['source_id'])[:20],split='confirmation',kind='natural',capture_mode='followup_query',
              novelty='Reserved-document, frozen after main-stage evidence and before any current capture; exact input and source group do not overlap the main10000windows.')
            assert x['source_group'] not in maingroups;rows.append(x)
    mainfit=read(BASE/'rules/fit_result.json')
    value={'timestamp':stamp(),'source':snapshot(__file__),'same_goal':True,'source_ids':[r['sample_id'] for r in rows],'sources':96,'independent_documents':96,'queries':100,
      'reason':'Main prefix-only query-rule uncertainty remains informative regardless of pass/fail; test frozen rules on reserved source documents and compare an explicitly later-information diagnostic.',
      'actual_documents_by_cohort':quotas,'reserved_documents_available':available,
      'pre_native_material_correction':'Initial32percohort assertion failed before material freeze or any new model execution:400GUMwindows came from19documents. Preserve independent documents and total96, allocate the13missing slots round-robin to EWT/CMRC. Actual19/39/38mixture is reported; not three balanced32document cohorts. Failure receipt and old executed source are retained in followup_recovery.',
      'main_ordered_gate_passed':mainfit['selection_gate_passed'],'unchanged_main_decoder_sha256':sha(BASE/'rules/decoder.npz'),
      'oracle_diagnostic':'Separate4column coordinate decoder [1,prefixH12,queryOnlyH13,ACTUAL queriedH12] fitted on original train-source/train-query only; validation selectslambda. Has additional observed-query information, so it is NOT an admissible competitor for prefix-only prediction.',
      'prediction_inputs':'All five original candidates remain unchanged and use original prefix-only features. No fresh source/query target refits; 20unseen queries remain separate.',
      'retention':'Every actual final/query-intermediate coordinate, prefixalllayeranchors and earlyKV retained; candidate feature arrays streamed except first3referencefixtures. Full original parameter/prototype recipe remains executable.',
      'budget':'Same existing6h/12GiB ceiling; admission requires measured remaining resources before capture; no automatic extension beyond the declared new96source bundle.',
      'required_evidence':[{ 'path':f,'sha256':sha(BASE/f)} for f in required]}
    compressed(OUT/'material.json.gz',rows);immutable(p,value);return value,rows

def oracle_fit():
    from phase2741_rdc_query_fit import rows_and_queries
    out=OUT/'oracle';file=out/'result.json'
    if file.exists():return
    start=time.monotonic();rows,probes,qi=rows_and_queries()
    with np.load(BASE/'prototypes/qwen4.npz') as z:pr=np.stack([unbits(z[f'p{i}_H13'][-1]) for i in range(100)]).astype(float)
    def build(r,ix):
        with np.load(BASE/'capture/fields'/f"{r['sample_id']}.npz") as z:
            prefix=unbits(z['prefix_layers'][12]).astype(float);later=unbits(z['query_H12_H24_rawH36']).astype(float)
            y=np.stack([later[:,1],later[:,2],unbits(z['postnorm'])]).astype(float)
        x=np.stack([np.ones_like(pr),np.broadcast_to(prefix,pr.shape),pr,later[:,0]],-1);return x[ix],y[:,ix]
    a=np.zeros((2560,4,4));b=np.zeros((3,2560,4));count=0
    for r in rows:
        if r['split']!='train':continue
        x,y=build(r,qi['train_query']);a+=np.einsum('qdi,qdj->dij',x,x,optimize=True);b+=np.einsum('qdi,tqd->tdi',x,y,optimize=True);count+=len(x)
    xm=a[:,0]/count;xs=np.sqrt(np.maximum(np.diagonal(a,axis1=-2,axis2=-1)/count-xm*xm,1e-10));xm[:,0]=0;xs[:,0]=1;tr=np.zeros_like(a)
    for j in range(4):tr[:,j,j]=1/xs[:,j]
    for j in range(1,4):tr[:,0,j]=-xm[:,j]/xs[:,j]
    aa=np.einsum('dji,djk,dkl->dil',tr,a,tr,optimize=True)/count;bb=np.einsum('tdi,dij->tdj',b,tr,optimize=True)/count
    lambdas=[.0001,.01,1.,100.];betas=[];val=np.zeros((4,3))
    for lam in lambdas:
        coef=np.linalg.solve(aa[None]+np.diag([0,lam,lam,lam]),bb[...,None])[...,0];betas.append(np.einsum('dij,tdj->tdi',tr,coef,optimize=True))
    for r in rows:
        if r['split']!='validation':continue
        x,y=build(r,qi['validation_query'])
        for j,beta in enumerate(betas):val[j]+=((np.einsum('qdi,tdi->tqd',x,beta,optimize=True)-y)**2).mean((1,2))
    selected=val.argmin(0);beta=np.stack([betas[selected[t]][t] for t in range(3)]);npz(out/'decoder.npz',beta=beta,selected_lambda_index=selected,validation_mse_sums=val)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'training_endpoints':count,'selected_lambdas':[lambdas[i] for i in selected],
      'additional_information':'Actual queriedH12 enters this diagnostic. It predicts only later layers24/36/postnorm, not that query state or futurequery construction.',
      'decoder_sha256':sha(out/'decoder.npz'),'seconds':time.monotonic()-start}
    save(file,result);ledger('later_information_diagnostic_fit',result['seconds']);print('LATER_INFORMATION_DIAGNOSTIC_FIT',result['seconds'],flush=True)

def run():
    import torch
    from phase2741_rdc_query_rules import prototype_native,rule_features,tensor
    from phase2741_rdc_query_fit import predict
    protocol,rows=freeze();oracle_fit()
    if (OUT/'result.json').exists():return
    guard(512*1024**2);used=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'));remaining=read(BASE/'resources.json')['compute_ceiling_seconds']-used
    assert remaining>800,('Insufficient recorded budget for frozen followup and final audit',remaining)
    immutable(OUT/'admission.json',{'timestamp':stamp(),'recorded_remaining_seconds':remaining,'minimum_reserved_seconds':800,'expected_extra_bytes':512*1024**2,
      'same_authorized_goal':True,'independent_document_bundle':True,'actual_run_admitted':True})
    start=time.monotonic();model=None;handle=None
    try:
      model,tok=load('qwen4',OUT);probes=read(BASE/'probes/protocol.json')['probes'];groups=defaultdict(list);data={}
      for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
      handle=model.model.layers[-1].register_forward_hook(lambda m,a,o:data.__setitem__('raw36',o.detach()))
      with np.load(BASE/'prototypes/qwen4.npz') as z:pr={k:z[k].copy() for k in z.files if k!='logprobs'};reference=torch.tensor(z['logprobs'],device='cuda')
      with np.load(BASE/'rules/decoder.npz') as z:beta=z['beta'].copy();trainmean=z['train_mean'].copy()
      with np.load(OUT/'oracle/decoder.npz') as z:oracle=z['beta'].copy()
      proto13=np.stack([unbits(pr[f'p{i}_H13'][-1]) for i in range(100)]).astype(float);metrics=[]
      with torch.inference_mode():
        protos,checks=prototype_native(model,pr,probes)
        for ri,row in enumerate(rows):
            sid=row['sample_id'];cp=OUT/'capture/commits'/f'{sid}.json';fp=OUT/'capture/fields'/f'{sid}.npz'
            if cp.exists():metrics.extend(read(cp)['metrics']);continue
            data.clear();pre=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True,output_hidden_states=True);cache=pre.past_key_values
            hs=list(pre.hidden_states[:-1])+[data['raw36']];prefix=np.stack([bits(h[0,-1]) for h in hs]);original_cache=cache_id(cache)
            arrays={'prefix_layers':prefix,'prefix_postnorm':bits(pre.last_hidden_state[0,-1]),'prefix_H12_sources':bits(pre.hidden_states[12][0]),
              'prefix_block12_keys':bits(cache.layers[12].keys[0]),'prefix_block12_values':bits(cache.layers[12].values[0])}
            post=np.zeros((100,2560),np.uint16);later=np.zeros((100,3,2560),np.uint16);statistics=np.zeros((100,4))
            for _,indices in sorted(groups.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16];own=clone_cache(cache,model.config,len(batch));data.clear()
                o=model.model(input_ids=torch.tensor([probes[q]['token_ids'] for q in batch],device='cuda'),past_key_values=own,use_cache=True,output_hidden_states=True)
                h=o.last_hidden_state[:,-1];lp=model.lm_head(h).float().double().log_softmax(-1);p=lp.exp();ref=reference[batch]
                post[batch]=bits(h);later[batch]=np.stack([bits(o.hidden_states[12][:,-1]),bits(o.hidden_states[24][:,-1]),bits(data['raw36'][:,-1])],1)
                statistics[batch]=torch.stack([-(p*lp).sum(-1),(p*(lp-ref)).sum(-1),(ref.exp()*(ref-lp)).sum(-1),lp.argmax(-1).double()],-1).cpu().numpy();del own,o,h,lp,p,ref
            assert original_cache==cache_id(cache)
            cand,merge=rule_features(model,protos,pr,cache.layers[12].keys[0].repeat_interleave(4,dim=0),cache.layers[12].values[0].repeat_interleave(4,dim=0),verifymerge=ri<3)
            target=np.stack([unbits(later[:,1]),unbits(later[:,2]),unbits(post)]).astype(float);feature=[];ph=unbits(prefix[12]).astype(float)
            for c in range(5):feature.append(np.stack([np.ones_like(proto13),np.broadcast_to(ph,proto13.shape),proto13,unbits(cand[c])],-1))
            ox=np.stack([np.ones_like(proto13),np.broadcast_to(ph,proto13.shape),proto13,unbits(later[:,0])],-1)
            predictions=np.stack([predict(feature[c],beta[c]) for c in range(5)]+[predict(ox,oracle)]);sse=(predictions-target[None])**2;kl=np.zeros((6,100))
            for _,indices in sorted(groups.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16];lp=model.lm_head(torch.tensor(target[2,batch],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                for c in range(6):
                    pl=model.lm_head(torch.tensor(predictions[c,2,batch],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1);kl[c,batch]=(lp.exp()*(lp-pl)).sum(-1).cpu().numpy()
            arrays.update(postnorm=post,query_H12_H24_rawH36=later,full_vocabulary_statistics=statistics,all_coordinate_prediction_MSE=sse.mean(2),all_query_prediction_KL=kl)
            if ri<3:arrays['candidate_H13']=cand
            npz(fp,**arrays);rr=[]
            for qs in ['train_query','validation_query','unseen_query']:
                ix=[i for i,p in enumerate(probes) if p['split']==qs];rr.append({'sample_id':sid,'source_group':row['source_group'],'cohort':row['cohort'],'query_split':qs,
                  'mse':sse[:,:,ix].mean((2,3)).tolist(),'KL':kl[:,ix].mean(1).tolist(),'train_mean_mse':((target[:,ix]-trainmean[:,None])**2).mean((1,2)).tolist()})
            record={'timestamp':stamp(),'sample_id':sid,'sha256':sha(fp),'metrics':rr,'all_prefix_KV_unchanged':True,'merge_checks':merge,'statistics_columns':['entropy','KL_to_query_only','reverse_KL','argmax']}
            save(cp,record);metrics.extend(rr);del pre,cache,hs,arrays,post,later,cand,sse,predictions;guard()
            if (ri+1)%8==0:print('INDEPENDENT_QUERY_CONFIRMATION',ri+1,96,round(time.monotonic()-start,1),flush=True)
        names=['query_only','uniform','quadratic','shuffled_values','ordered_softmax','ADDITIONAL_observed_queryH12'];summary=[]
        for qs in ['train_query','validation_query','unseen_query']:
          for cohort in ['all','gum','ewt','cmrc']:
            rr=[r for r in metrics if r['query_split']==qs and (cohort=='all' or r['cohort']==cohort)];gg=[r['source_group'] for r in rr]
            summary.append({'query_split':qs,'cohort':cohort,'documents':len(rr),'postnorm_MSE':{name:clustered([r['mse'][c][2] for r in rr],gg) for c,name in enumerate(names)},
              'full_vocab_KL':{name:clustered([r['KL'][c] for r in rr],gg) for c,name in enumerate(names)},
              'same_input_control_minus_ordered_MSE':{names[c]:clustered([r['mse'][c][2]-r['mse'][4][2] for r in rr],gg) for c in range(4)},
              'additional_query_information_MSE_reduction':clustered([r['mse'][4][2]-r['mse'][5][2] for r in rr],gg)})
        compressed(OUT/'metrics.json.gz',metrics);assert sha(BASE/'rules/decoder.npz')==protocol['unchanged_main_decoder_sha256']
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'same_goal_automatically_executed':True,'sources':96,'source_documents':96,'query_endpoints':9600,
          'all_original_decoders_unchanged':True,'query_prototype_checks':checks,'summary':summary,'seconds':time.monotonic()-start,
          'scope':'Independent main-document holdout with frozen prefix-only rules. The additional observed-queryH12 diagnostic is a different information condition, not evidence that a forbidden future input was deployable earlier.'}
        save(OUT/'result.json',result);ledger('independent_query_same_goal_confirmation',result['seconds']);print('QUERY_FOLLOWUP_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT,start,exc);raise
    finally:
        if handle is not None:handle.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','oracle','run']);a=p.parse_args()
    if a.stage=='freeze':print('FOLLOWUP_PROTOCOL',freeze()[0])
    elif a.stage=='oracle':oracle_fit()
    else:run()
