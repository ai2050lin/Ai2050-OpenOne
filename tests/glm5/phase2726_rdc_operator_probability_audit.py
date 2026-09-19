"""Paired full-vocabulary comparisons and source-cluster uncertainty from actual commits."""
from rdc_operator_common import *


def main():
    start=time.monotonic();result=[]
    for stage in ('validation','confirmation'):
        rr=[r for p in sorted((BASE/'compiled'/stage/'commits').glob('*.json')) for r in read(p)['rows']]
        indexed={(r['sample_id'],r['anchor'],r['name']):r for r in rr}
        keys=sorted({(r['sample_id'],r['anchor']) for r in rr})
        for block in (6,16,34,'joint'):
            prefix='joint' if block=='joint' else f'L{block}';comparisons=[]
            for sid,anchor in keys:
                g=indexed[(sid,anchor,prefix+'_global')];s=indexed[(sid,anchor,prefix+'_selected')]
                comparisons.append({'sample_id':sid,'anchor':anchor,'source_group':s['source_group'],
                    'KL_gain_global_minus_selected':g['raw_KL']-s['raw_KL'],
                    'NLL_gain_global_minus_selected':g['approx_next_NLL']-s['approx_next_NLL'],
                    'argmax_gain_selected_minus_global':int(s['argmax_agreement'])-int(g['argmax_agreement'])})
            result.append({'stage':stage,'block':block,'queries':len(keys),
                **{k:{'anchor_mean':float(np.mean([r[k] for r in comparisons])),
                    'source_cluster':clustered([r[k] for r in comparisons],[r['source_group'] for r in comparisons])} for k in ('KL_gain_global_minus_selected','NLL_gain_global_minus_selected','argmax_gain_selected_minus_global')},
                'fraction_anchors_lower_KL_selected':float(np.mean([r['KL_gain_global_minus_selected']>0 for r in comparisons]))})
    paths=[read(p) for p in sorted((BASE/'compiled/autonomous').glob('*.json'))]
    autonomous=[]
    for a,b in [('joint_global','native'),('joint_selected','native'),('joint_selected','joint_global')]:
        difference=[p['branches'][a]['repeated_4gram_fraction']-p['branches'][b]['repeated_4gram_fraction'] for p in paths]
        autonomous.append({'branch_a':a,'branch_b':b,'meaning':'repeated4gram(a)-repeated4gram(b); positive means more repeats in a',
            'source_mean':float(np.mean(difference)),**clustered(difference,[p['source_group'] for p in paths])})
    report={'timestamp':stamp(),'source':snapshot(Path(__file__)),'paired_probability':result,'paired_autonomous_repetition':autonomous,
        'scope':'Paired original source/anchor comparisons, cluster resampling conditional on the frozen fit. Cluster means weight articles equally, whereas published anchor means weight each anchor equally. Intervals are descriptive bootstrap intervals, not correction for all post-hoc hypotheses. Native EOS was absent in all64 baseline48-token continuations, so no automatic stop-failure claim.'}
    save(BASE/'compiled/paired_audit.json',report);ledger('paired_probability_and_history_audit',time.monotonic()-start)
    print('PAIRED_COMPILE_AUDIT',[(r['block'],r['KL_gain_global_minus_selected']) for r in result if r['stage']=='confirmation'],flush=True)


if __name__=='__main__':main()
