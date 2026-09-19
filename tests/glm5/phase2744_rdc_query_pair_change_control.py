"""Aggregate previously recorded paired forecasts against the zero-change baseline."""
from phase2744_rdc_query_identifiability import *


def main():
    start=time.monotonic();assert read(OUT/'queue/status.json')['all_passed']
    material=gzread(OUT/'material.json.gz');probes=read(BASE/'probes/protocol.json')['probes']
    metrics={r['pair_id']:r for r in gzread(OUT/'relations/frozen_relation_change_metrics.json.gz')}
    names=['query_only','uniform','quadratic','shuffled_values','ordered_softmax'];records=[]
    for pair in material['pairs']:
        with np.load(OUT/'relations/native/fields'/f"{pair['sample_ids'][0]}.npz") as z:a=unbits(z['postnorm']).astype(float)
        with np.load(OUT/'relations/native/fields'/f"{pair['sample_ids'][1]}.npz") as z:b=unbits(z['postnorm']).astype(float)
        r=metrics[pair['pair_id']];actual=((b-a)**2).mean(-1)
        assert abs(actual.mean()-r['actual_query_relation_displacement_MSE'])<1e-12
        for split in ['train_query','validation_query','unseen_query']:
            ids=[i for i,p in enumerate(probes) if p['split']==split]
            records.append({'pair_id':pair['pair_id'],'source_group':pair['source_group'],'family':r['family'],'language':r['language'],
              'query_split':split,'zero_change_MSE':float(actual[ids].mean()),'frozen_candidate_change_MSE':r['query_split_displacement_MSE'][split]})
    reports=[]
    for split in ['train_query','validation_query','unseen_query']:
      for family in ['all']+FAMILIES:
        rows=[r for r in records if r['query_split']==split and (family=='all' or r['family']==family)];groups=[r['source_group'] for r in rows]
        baseline=np.array([r['zero_change_MSE'] for r in rows]);errors=np.array([r['frozen_candidate_change_MSE'] for r in rows])
        reports.append({'query_split':split,'family':family,'pairs':len(rows),'zero_change_MSE':clustered(baseline,groups),
          'candidate_change_MSE':{name:clustered(errors[:,i],groups) for i,name in enumerate(names)},
          'zero_minus_candidate_change_MSE':{name:clustered(baseline-errors[:,i],groups) for i,name in enumerate(names)},
          'control_minus_ordered_change_MSE':{names[i]:clustered(errors[:,i]-errors[:,4],groups) for i in range(4)}})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'records':records,'reports':reports,
      'design_status':'Unblinded supplemental aggregation after the main2744result; the paired forecast errors were already specified and saved during native capture. No predictor changed or new native sample collected.',
      'formula':'For the same question and token multiset, delta=h_B-h_A; error=mean_d,q[(hhat_B-hhat_A)-delta]^2. The zero-change predictor has error=mean_d,q[delta^2]. Positive zero-minus-candidate error is incremental prediction of the actual pair change.',
      'scope':'All2560coordinates, all160pairedworlds; query splits remain frozen. Actual response difference is a statistical target, not a transported activation or uniquely identified semantic direction. This does not determine which ordered cue the predictor uses.',
      'required_evidence_sha256':{'protocol':sha(OUT/'protocol.json'),'main_analysis':sha(OUT/'analysis/result.json'),
        'frozen_pair_metrics':sha(OUT/'relations/frozen_relation_change_metrics.json.gz'),'decoder':sha(BASE/'rules/decoder.npz')},
      'seconds':time.monotonic()-start}
    save(OUT/'analysis/pair_change_control.json',result);ledger('identity_paired_change_zero_baseline_audit',result['seconds'])
    print('PAIRED_CHANGE_CONTROL',json.dumps([r for r in reports if r['query_split']=='unseen_query' and r['family']=='all']),flush=True)


if __name__=='__main__':main()
