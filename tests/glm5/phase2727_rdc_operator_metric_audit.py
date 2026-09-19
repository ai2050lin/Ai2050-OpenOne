"""Output-sensitive composition: paired source evidence, numerical error and transfer limits."""
from rdc_operator_common import *


def main():
    start=time.monotonic();out=BASE/'metric_followup';summary=read(out/'result.json')
    commits=[read(p) for p in sorted((out/'commits').glob('*.json'))];rows_=[r for c in commits for r in c['rows']]
    indexed={(r['id'],r['name']):r for r in rows_};comparisons=[];metric_quality=[];local_pairs=[]
    for scope in ('natural_reanalysis','QA_query_transfer'):
        ids=sorted({r['id'] for r in rows_ if r['scope']==scope})
        for reference in ('joint_global','joint_selected'):
            rr=[]
            for q in ids:
                baseline=indexed[(q,reference)];hybrid=indexed[(q,'output_selected_hybrid')]
                rr.append({'source_group':baseline['source_group'],'gain':baseline['exact_KL']-hybrid['exact_KL'],
                    'argmax_gain':int(hybrid['argmax_agreement'])-int(baseline['argmax_agreement'])})
            comparisons.append({'scope':scope,'reference':reference,'candidate':'output_selected_hybrid','queries':len(rr),
                'anchor_mean_KL_gain':float(np.mean([r['gain'] for r in rr])),
                'fraction_queries_KL_improved':float(np.mean([r['gain']>0 for r in rr])),
                'KL_gain_article_cluster':clustered([r['gain'] for r in rr],[r['source_group'] for r in rr]),
                'argmax_gain_article_cluster':clustered([r['argmax_gain'] for r in rr],[r['source_group'] for r in rr])})
        for name in ('joint_global','joint_selected','output_selected_hybrid'):
            rr=[r for r in rows_ if r['scope']==scope and r['name']==name]
            relative=[abs(r['endpoint_Fisher_half_variance']-r['exact_KL'])/max(r['exact_KL'],1e-12) for r in rr]
            metric_quality.append({'scope':scope,'name':name,'queries':len(rr),
                'endpoint_quadratic_relative_error_quantiles':dict(zip(('median','p90','p99','max'),np.quantile(relative,[.5,.9,.99,1]).tolist())),
                'max_adaptive_integral_absolute_error':max(r['adaptive_absolute_error'] for r in rr),
                'adaptive_node_counts':dict(__import__('collections').Counter(r['adaptive_nodes'] for r in rr))})
        for b in (6,16,34):
            pairs=[]
            for c in commits:
                local=c['same_prefix_local']
                if c['row']['scope']!=scope:continue
                g=next(r for r in local if r['block']==b and r['name']=='global');s=next(r for r in local if r['block']==b and r['name']=='selected')
                pairs.append({'source_group':g['source_group'],'gain':g['relative_MSE']-s['relative_MSE']})
            local_pairs.append({'scope':scope,'block':b,'queries':len(pairs),'mean_same_prefix_local_relative_MSE_gain':float(np.mean([r['gain'] for r in pairs])),
                'article_cluster':clustered([r['gain'] for r in pairs],[r['source_group'] for r in pairs])})
    base_paths={p.stem:read(p) for p in (BASE/'compiled/autonomous').glob('*.json')}
    paths=[read(p) for p in sorted((out/'autonomous').glob('*.json'))];auto=[]
    for name in ('native','joint_global','joint_selected'):
        d=[p['repeated_4gram_fraction']-base_paths[p['sample_id']]['branches'][name]['repeated_4gram_fraction'] for p in paths]
        auto.append({'reference':name,'candidate':'output_selected_hybrid','meaning':'hybrid repeated4gram minus reference repeated4gram',
            'source_mean':float(np.mean(d)),**clustered(d,[p['source_group'] for p in paths])})
    replay=[]
    old_compiled={p.stem:read(p) for p in (BASE/'compiled/confirmation/commits').glob('*.json')}
    for c in commits:
        if c['row']['scope']!='natural_reanalysis':continue
        anchor=int(c['row']['id'].rsplit('_a',1)[1]);old=old_compiled[c['row']['sample_id']]['rows']
        for name in ('joint_global','joint_selected'):
            now=next(r for r in c['rows'] if r['name']==name);before=next(r for r in old if r['anchor']==anchor and r['name']==name)
            difference=abs(now['exact_KL']-before['raw_KL']);assert difference<1e-4,(now['id'],name,difference)
            replay.append(difference)
    inversions=[];inversion_counts=[];tolerance=1e-5
    for b in (6,16,34):
        count=0;locally_better=0
        for c in commits:
            if c['row']['scope']!='natural_reanalysis':continue
            count+=1;anchor=int(c['row']['id'].rsplit('_a',1)[1]);old=old_compiled[c['row']['sample_id']]['rows']
            lg=next(r for r in c['same_prefix_local'] if r['block']==b and r['name']=='global')
            ls=next(r for r in c['same_prefix_local'] if r['block']==b and r['name']=='selected')
            pg=next(r for r in old if r['anchor']==anchor and r['name']==f'L{b}_global')
            ps=next(r for r in old if r['anchor']==anchor and r['name']==f'L{b}_selected')
            gain=lg['relative_MSE']-ls['relative_MSE'];worsening=ps['raw_KL']-pg['raw_KL']
            locally_better+=int(gain>tolerance)
            if gain>tolerance and worsening>tolerance:
                inversions.append({'block':b,'sample_id':c['row']['sample_id'],'anchor':anchor,'source_group':c['row']['source_group'],
                    'language':c['row']['language'],'actual_prefix_token_IDs':c['row']['ids'],
                    'same_prefix_global_local_relative_MSE':lg['relative_MSE'],'same_prefix_selected_local_relative_MSE':ls['relative_MSE'],
                    'single_block_global_full_vocab_KL':pg['raw_KL'],'single_block_selected_full_vocab_KL':ps['raw_KL'],
                    'local_relative_MSE_gain':gain,'KL_worsening':worsening})
        selected=[r for r in inversions if r['block']==b]
        inversion_counts.append({'block':b,'queries':count,'locally_better_queries':locally_better,'local_better_but_probability_worse_queries':len(selected),
            'source_groups':len({r['source_group'] for r in selected}),'tolerance_in_each_metric':tolerance,
            'first_canonical_example':selected[0] if selected else None})
    compressed(out/'all_same_prefix_rank_inversions.json.gz',inversions)
    behavior_groups=[]
    for matched in (True,False):
        for name in ('joint_global','joint_selected','output_selected_hybrid'):
            rr=[r for r in rows_ if r['scope']=='QA_query_transfer' and r['name']==name and r['native_answer_string_match']==matched]
            if rr:
                behavior_groups.append({'native_answer_string_match':matched,'name':name,'queries':len(rr),'mean_KL':float(np.mean([r['exact_KL'] for r in rr])),
                    'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in rr])),
                    'KL_article_cluster':clustered([r['exact_KL'] for r in rr],[r['source_group'] for r in rr])})
    outcome={'timestamp':stamp(),'source':snapshot(Path(__file__)),'paired_hybrid':comparisons,'same_prefix_local_pairs':local_pairs,'QA_behavior_groups':behavior_groups,
        'same_prefix_rank_inversions':inversion_counts,
        'metric_quality':metric_quality,'autonomous_repetition_pairs':auto,
        'prior_compiled_replay_comparisons':len(replay),'prior_FP32_vs_new_FP64_max_KL_difference':max(replay),
        'limits':'Frozen validation-based hybrid choice; article bootstrap conditional on the fit, without multiplicity correction. Source-level weighted means differ from article-equal means. Natural confirmation is re-analysis; QA boundary transfer not new source/semantic holdout. Endpoint Fisher is a local approximation and the complete variance integral is a known identity.'}
    save(out/'paired_audit.json',outcome);ledger('output_metric_paired_replay_and_uncertainty',time.monotonic()-start)
    print('OUTPUT_METRIC_PAIRED_AUDIT_COMPLETE',comparisons,flush=True)


if __name__=='__main__':main()
