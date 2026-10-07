"""Conservative sensitivity: cluster repeated entity/connector identities across families.
The original source-group bootstrap is retained, never silently replaced.
"""
import json
import numpy as np
from phase2751_trusted_rebuild import OUT,write


def clustered(rows,fn,key):
    blocks={}
    for r in rows:blocks.setdefault(key(r),[]).append(fn(r))
    values=[np.asarray(v) for v in blocks.values()];rng=np.random.default_rng(2751031)
    draws=[float(np.median(np.concatenate([values[i] for i in rng.integers(0,len(values),len(values))]))) for _ in range(4000)]
    return dict(median=float(np.median(np.concatenate(values))),ci95=np.quantile(draws,[.025,.975]).tolist(),
                n_identity_blocks=len(values),n_rows=len(rows),method='Identity-block percentile bootstrap, all task families/conditions in a block resampled together')


def main():
    results={}
    for side in ['4B','14B','length_control/4B']:
        folder=OUT/side
        comp=json.loads((folder/'composition.json').read_text(encoding='utf-8'))['groups']
        repair=json.loads((folder/'measurement_repair.json').read_text(encoding='utf-8'))['rows']
        vals={}
        for split in sorted(set(r['split'] for r in comp)):
            rows=[r for r in comp if r['split']==split]
            key=lambda r:int(r['source_family'].split('_')[-1])
            vals[split]=dict(calibration_minus_additive=clustered(rows,lambda r:r['relative_errors']['dev_mean_interaction'][-1]-r['relative_errors']['additive'][-1],key),
                additive_minus_negation=clustered(rows,lambda r:r['relative_errors']['additive'][-1]-r['relative_errors']['negation'][-1],key))
        rep={}
        for split in sorted(set(r['split'] for r in repair)):
            rows=[r for r in repair if r['split']==split]
            key=lambda r:int(r['group'].split('_')[-1] if r['split']=='legacy' else r['group'].split('_')[1])
            rep[split]=clustered(rows,lambda r:r['corrected']['relative_error']-r['legacy_wrong']['relative_error'],key)
        results[side]=dict(composition=vals,repair=rep)
    write(OUT/'bootstrap_sensitivity.json',dict(results=results,
        reason='Semantic groups reuse the same entity index across task families; legacy groups share connective index. Identity-level resampling is more conservative.',
        limits='Still fixed small template/task-family set; partner names can overlap identity blocks in primary corpus. Not a population-level generalization test.'))
    print(json.dumps({s:{k:v['calibration_minus_additive'] for k,v in r['composition'].items() if k!='development'} for s,r in results.items()}))


if __name__=='__main__':main()
