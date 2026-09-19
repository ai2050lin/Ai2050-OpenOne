"""Full-vocabulary response tests on native reachable, current-token-matched pairs."""
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'pairs'

def main():
    import torch
    if (OUT/'result.json').exists():return
    packet=read(BASE/'atlas/state_pairs.json');pairs=packet['pairs'];probes=read(BASE/'probes/protocol.json')['probes'];groups=defaultdict(list)
    for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
    start=time.monotonic();model=None
    try:
      model,tok=load('qwen4',OUT);records=[]
      with torch.inference_mode():
        for i,pair in enumerate(pairs):
            a=BASE/'capture/fields'/f"{pair['center']}.npz";b=BASE/'capture/fields'/f"{pair['other']}.npz"
            with np.load(a) as z:ha=unbits(z['postnorm']);currenta=unbits(z['prefix_postnorm'])
            with np.load(b) as z:hb=unbits(z['postnorm']);currentb=unbits(z['prefix_postnorm'])
            stats=np.zeros((100,3))
            la=model.lm_head(torch.tensor(currenta,device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
            lb=model.lm_head(torch.tensor(currentb,device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
            current=float(.5*((la.exp()*(la-lb)).sum()+(lb.exp()*(lb-la)).sum()))
            for _,indices in sorted(groups.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16]
                la=model.lm_head(torch.tensor(ha[batch],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                lb=model.lm_head(torch.tensor(hb[batch],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                da=(la.exp()*(la-lb)).sum(-1);db=(lb.exp()*(lb-la)).sum(-1)
                stats[batch]=torch.stack([da,db,(la.argmax(-1)==lb.argmax(-1)).double()],-1).cpu().numpy()
            fp=OUT/'fields'/f"{pair['center']}_{pair['kind']}.npz";npz(fp,full_vocabulary_pair_statistics=stats)
            records.append(pair|{'current_symmetric_KL':current,'mean_query_symmetric_KL':float(stats[:,:2].mean()),
              'finite_mean_KL_below005':float(stats[:,:2].mean())<.05,'query_argmax_agreement':float(stats[:,2].mean()),
              'mean_query_KL_by_family':{family:float(stats[[j for j,p in enumerate(probes) if p['family']==family],:2].mean()) for family in sorted({p['family'] for p in probes})},
              'archive_sha256':sha(fp)})
            if (i+1)%12==0:print('QUERY_REACHABLE_PAIR',i+1,len(pairs),round(time.monotonic()-start,1),flush=True)
        save(OUT/'records.json',records);paired=[]
        for c in sorted({r['center'] for r in records}):
            near=next(r for r in records if r['center']==c and r['kind']=='near');far=next(r for r in records if r['center']==c and r['kind']=='far')
            paired.append({'center':c,'source_group':near['center_group'],'current_near_minus_far':near['current_symmetric_KL']-far['current_symmetric_KL'],
              'query_near_minus_far':near['mean_query_symmetric_KL']-far['mean_query_symmetric_KL']})
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'pairs':len(records),'centers':len(paired),'unmatched_centers':packet['missing_centers'],
          'current_near_minus_far':clustered([r['current_near_minus_far'] for r in paired],[r['source_group'] for r in paired]),
          'query_near_minus_far':clustered([r['query_near_minus_far'] for r in paired],[r['source_group'] for r in paired]),
          'finite_mean_KL_below005':sum(r['finite_mean_KL_below005'] for r in records),'seconds':time.monotonic()-start,
          'scope':'Full151936vocabulary and all100fixedqueries. Current states are similar, not equal. MeanKLthreshold applies only to this finite suite and does not imply all-query or multi-step state equivalence. Pair endpoints overlap, so center-cluster intervals can still understate dependence from shared neighbors.'}
        save(OUT/'result.json',result);ledger('reachable_native_full_vocabulary_query_pairs',result['seconds']);print('QUERY_PAIRS_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
