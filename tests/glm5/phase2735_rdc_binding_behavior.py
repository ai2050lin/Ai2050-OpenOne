"""Finite matched behavior and cross-model relation-space audits, not semantic judging."""
from collections import Counter,defaultdict
from rdc_binding_common import *

def records(folder):return [read(p) for p in sorted(folder.glob('*.json'))]

def summarize(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r.get(k,'') for k in keys)].append(r)
    result=[]
    for key,rr in groups.items():
        parsed=[r for r in rr if r.get('conservative_final_correct') is not None]
        r=dict(zip(keys,key));r.update(rows=len(rr),semantic_cases=len({r['source_group'] for r in rr}),
          strict_answer_accuracy=float(np.mean([s['exact_correct'] for s in rr])),
          conservative_parse_coverage=len(parsed)/len(rr),
          conservative_answer_accuracy_among_parsed=float(np.mean([s['conservative_final_correct'] for s in parsed])) if parsed else None,
          conservative_correct_fraction_of_all=float(sum(bool(s.get('conservative_final_correct')) for s in rr)/len(rr)),
          conservative_stopped_correct_fraction_of_all=float(sum(bool(s.get('conservative_final_correct')) and s['eos'] for s in rr)/len(rr)),
          conservative_parsed_but_censored_fraction=float(sum(s.get('conservative_final_digit') is not None and s['token_limit_censored'] for s in rr)/len(rr)),
          eos_fraction=float(np.mean([s['eos'] for s in rr])),censored_fraction=float(np.mean([s['token_limit_censored'] for s in rr])),
          mean_generated_tokens=float(np.mean([len(s['generated_ids']) for s in rr])))
        evaluation=[s for s in rr if s.get('initial_evaluation')]
        if evaluation:
            r['native_first_target_nll']=float(np.mean([s['initial_evaluation']['native_first_target_nll'] for s in evaluation]))
            r['native_first_argmax_accuracy']=float(np.mean([s['initial_evaluation']['first_target_argmax_correct'] for s in evaluation]))
            for name in ('content_nll','format_nll','digit_probability_mass'):
                r[name]=float(np.mean([s['initial_evaluation'][name] for s in evaluation]))
            r['conditional_digit_accuracy']=float(np.mean([s['initial_evaluation']['conditional_digit_argmax']==s['initial_evaluation']['target_id'] for s in evaluation]))
        result.append(r)
    return result

def rsa():
    from phase2734_rdc_binding_scale import material
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    natural,program=material();rows=natural+program;cohorts=[r['cohort'] for r in rows]
    representations={};model_info={}
    for model in ('qwen4','qwen14','glm4'):
        runtime=read(BASE/'scale'/model/'runtime.json');early=runtime['early'];collected={'early_query':[],'last_mlp':[]}
        for row in rows:
            with np.load(BASE/'scale'/model/'fields'/f'{row["sample_id"]}.npz') as z:
                collected['early_query'].append(unbits(z['H'])[early,-1]);collected['last_mlp'].append(unbits(z['mlp'])[-1])
        model_info[model]={k:runtime[k] for k in ('width','units','depth','early','dtype','quantized')}
        for name,values in collected.items():
            raw=np.array(values,dtype=np.float64)
            for mode in ('raw','cohort_centered'):
                a=raw.copy()
                if mode=='cohort_centered':
                    for c in sorted(set(cohorts)):
                        ix=[i for i,x in enumerate(cohorts) if x==c];a[ix]-=a[ix].mean(0)
                a/=np.linalg.norm(a,axis=1,keepdims=True).clip(1e-12);gram=a@a.T
                representations[model,name,mode]=gram
                path=BASE/'scale/relational_gram'/f'{model}_{name}_{mode}.npz'
                if not path.exists():npz(path,gram=gram)
    result=[];tri=np.triu_indices(len(rows),1);rng=np.random.default_rng(2734)
    coarse_strata=[]
    for r in rows:
        coarse_strata.append((r['cohort'],r['split']) if r['kind']=='natural' else (r['cohort'],r['family'],r['depth']))
    coarse_groups=[[i for i,key in enumerate(coarse_strata) if key==group] for group in sorted(set(coarse_strata))]
    def corr(a,b):
        a=a-a.mean();b=b-b.mean();return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)).clip(1e-12))
    for a,b in [('qwen4','qwen14'),('qwen4','glm4'),('qwen14','glm4')]:
      for name in ('early_query','last_mlp'):
       for mode in ('raw','cohort_centered'):
        left=representations[a,name,mode][tri];right=representations[b,name,mode];point=corr(left,right[tri]);null=[];coarse_null=[]
        for repeat in range(300):
            perm=np.arange(len(rows))
            for cohort in sorted(set(cohorts)):
                ix=np.array([i for i,c in enumerate(cohorts) if c==cohort]);perm[ix]=rng.permutation(ix)
            null.append(corr(left,right[perm][:,perm][tri]))
            perm=np.arange(len(rows))
            for ix in coarse_groups:perm[ix]=rng.permutation(ix)
            coarse_null.append(corr(left,right[perm][:,perm][tri]))
        result.append({'model_a':a,'model_b':b,'field':name,'normalization':mode,'source_rows':len(rows),'pair_entries':len(left),
          'relation_matrix_pearson':point,'within_cohort_identity_permutation_mean':float(np.mean(null)),
          'permutation_interval95':np.quantile(null,[.025,.975]).tolist(),'one_sided_permutation_tail':float((1+sum(x>=point for x in null))/301),
          'family_depth_preserving_permutation_mean':float(np.mean(coarse_null)),
          'family_depth_preserving_interval95':np.quantile(coarse_null,[.025,.975]).tolist(),
          'family_depth_preserving_one_sided_tail':float((1+sum(x>=point for x in coarse_null))/301),
          'scope':'Exploratory full-coordinate relation matrices;300permutations each forwithincohort andwithincohort/family/depth (natural usescohort/split). Correlated pair entries are not independent. Shared lexical/template/target information remains.'})
    return {'models':model_info,'comparisons':result,'stronger_null_stratum_sizes':list(map(len,coarse_groups)),
      'limits':['Native widths and relative early depths differ; no coordinate mapping or physical isomorphism is inferred.',
        'Same material/template/length/target can produce relation agreement. Cohort centering and cohort/family/depth-preserving shuffles remain limited controls.',
        'All validation/test rows participate in this descriptive representation audit; it is not an independent train-only forecast.']}

def rsa_report():
    """Independent CPU stage, reusable after the serial model captures complete."""
    assert read(BASE/'scale/suite_result.json')['all_passed']
    path=BASE/'scale/relational_analysis.json'
    inputs={model:{name:sha(BASE/'scale'/model/name) for name in ('result.json','runtime.json')}
      for model in ('qwen4','qwen14','glm4')}
    if path.exists():
        cached=read(path);assert cached['input_sha256']==inputs
        assert cached['source']['sha256']==sha(Path(__file__))
        return cached['analysis']
    start=time.monotonic();analysis=rsa();seconds=time.monotonic()-start
    save(path,{'timestamp':stamp(),'source':snapshot(Path(__file__)),'input_sha256':inputs,
      'analysis':analysis,'seconds':seconds,'scope':'Descriptive cross-model relations, not train-only prediction or semantic isomorphism. Captured array identities are separately checked by the final full-array audit.'})
    ledger('binding_cross_model_relations_CPU',seconds)
    print('BINDING_CROSS_MODEL_RELATIONS_COMPLETE',len(analysis['comparisons']),flush=True)
    return analysis


def main():
    start=time.monotonic()
    assert (BASE/'format_content/autonomous/result.json').exists()
    native=records(BASE/'format_content/native_commits');autonomous=[r for folder in (BASE/'format_content/autonomous/commits').iterdir() if folder.is_dir() for r in records(folder)]
    baseline={r['sample_id']:r for r in autonomous if r['branch']=='native'};paired=[]
    for branch in sorted({r['branch'] for r in autonomous if r['branch']!='native'}):
      for split in sorted({r['split'] for r in autonomous if r['branch']==branch}):
        rr=[r for r in autonomous if r['branch']==branch and r['split']==split]
        for rep in ['pooled']+['en','zh','python','en_reordered']:
            use=rr if rep=='pooled' else [r for r in rr if r['representation']==rep]
            if not use:continue
            changes=[float(r['exact_correct'])-float(baseline[r['sample_id']]['exact_correct']) for r in use]
            nll=[r['initial_evaluation']['native_first_target_nll']-baseline[r['sample_id']]['initial_evaluation']['native_first_target_nll'] for r in use]
            paired.append({'branch':branch,'split':split,'representation':rep,'rows':len(use),
              'strict_answer_change':float(np.mean(changes)),'strict_answer_cluster':clustered(changes,[r['source_group'] for r in use]),
              'native_initial_nll_change':float(np.mean(nll)),'nll_cluster':clustered(nll,[r['source_group'] for r in use]),
              'changed_token_trajectories':sum(r['generated_ids']!=baseline[r['sample_id']]['generated_ids'] for r in use),
              'native_strict_correct':sum(baseline[r['sample_id']]['exact_correct'] for r in use),'branch_strict_correct':sum(r['exact_correct'] for r in use)})
    shorter=[]
    for r in native:
        p=BASE/'capture/commits'/f'{r["sample_id"]}.json'
        if not p.exists():continue
        old=read(p);shorter.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'representation':r['representation'],'family':r['family'],
          'old8_strict':old['exact_correct'],'new128_strict':r['exact_correct'],'old8_censored':old['token_limit_censored'],
          'new128_censored':r['token_limit_censored'],'long_parsed':r['conservative_final_digit'],'long_parsed_correct':r['conservative_final_correct'],
          'prefix_exact':r['original8token_prefix_exact']})
    binding=[r for f in (BASE/'binding_live/commits').iterdir() if f.is_dir() for r in records(f)];bn={r['sample_id']:r for r in binding if r['branch']=='native'};bs=[]
    for branch in sorted({r['branch'] for r in binding}):
      for cohort in ('gum','ewt'):
        rr=[r for r in binding if r['branch']==branch and r['cohort']==cohort];prefix=[]
        for r in rr:
            a=r['generated_ids'];b=bn[r['sample_id']]['generated_ids'];length=0
            for u,v in zip(a,b):
                if u!=v:break
                length+=1
            prefix.append(length)
        report={'branch':branch,'cohort':cohort,'rows':len(rr),'changed_trajectories':sum(r['generated_ids']!=bn[r['sample_id']]['generated_ids'] for r in rr),
          'mean_shared_native_token_prefix':float(np.mean(prefix)),'eos_fraction':float(np.mean([r['eos'] for r in rr])),
          'mean_repeated_4gram_fraction':float(np.mean([r['repeated_4gram_fraction'] for r in rr]))}
        delta=[r['initial_evaluation']['native_first_target_nll']-bn[r['sample_id']]['initial_evaluation']['native_first_target_nll'] for r in rr]
        report.update(first_observed_next_token_nll=float(np.mean([r['initial_evaluation']['native_first_target_nll'] for r in rr])),
          first_observed_next_token_accuracy=float(np.mean([r['initial_evaluation']['first_target_argmax_correct'] for r in rr])),
          first_next_token_nll_change=float(np.mean(delta)),first_next_token_nll_cluster=clustered(delta,[r['source_group'] for r in rr]),
          evaluation_target_text_counts=dict(Counter(r['first_evaluation_target_text'] for r in rr)),
          target_scoring_scope='Actual final material token after last anchor, often punctuation. Evaluation-only label; no claim that this is a full semantic-quality score.')
        if branch.startswith('frozen_binding'):
            report['first_state_relative_squared_error']=float(np.mean([r['state_error_first'] for r in rr]))
            later=[r['state_error_after_first'] for r in rr if r['state_error_after_first'] is not None]
            report['later_own_history_state_relative_squared_error']=float(np.mean(later)) if later else None
        bs.append(report)
    # The offline capture evaluated the full material, while live generation starts
    # at its last anchor (one token shorter). Do not assume bitwise equivalence.
    natural=gzread(BASE/'natural_confirmation.json.gz');lookup={r['sample_id']:(i,r) for i,r in enumerate(natural)}
    execution=[]
    def relative_vector_difference(a,b):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
    for block,kernel in ((16,'role_position_pair'),(35,'source_mean')):
        with np.load(BASE/'confirmation'/f'b{block}_{kernel}.npz') as z:prior=z['prediction']
        for row in binding:
            if row['branch']!=f'frozen_binding{block}':continue
            sid=row['sample_id'];i,material_row=lookup[sid]
            with np.load(BASE/'binding_live/fields'/row['branch']/f'{sid}.npz') as live, np.load(BASE/'capture/natural'/f'{sid}.npz') as capture:
                q=unbits(capture['H'])[12,-1];q=q/max(float(np.sqrt(np.mean(q*q))),1e-8)
                execution.append({'sample_id':sid,'block':block,'full_material_tokens':len(material_row['prompt_ids']),
                  'live_prefill_tokens':len(row['prompt_ids']),
                  'query_RMS_relative_L2_difference':relative_vector_difference(live['query'][0],q),
                  'native_MLP_relative_L2_difference':relative_vector_difference(live['native_mlp'][0],unbits(capture[f'L{block}_mlp'])[-1]),
                  'frozen_prediction_relative_L2_difference':relative_vector_difference(live['predicted_mlp'][0],prior[i]),
                  'scope':'Observed numerical comparison only: full-window/no-cache capture versus one-token-shorter cached prefill, plus kernel execution shape. Not an isolated attribution experiment. Live RSE uses same-forward nativeMLP, not the old captured reference.'})
    scales={};target_lengths={}
    for model in ('qwen4','qwen14','glm4'):
        rr=[r for r in records(BASE/'scale'/model/'commits') if 'generated' in r]
        scales[model]=summarize(rr,['split','representation']);target_lengths[model]=dict(Counter(len(r['native_target_ids']) for r in rr))
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'long_native':summarize(native,['split','representation']),
      'long_native_by_family':summarize(native,['split','family','representation']),
      'short_to_long':shorter,'autonomous':summarize(autonomous,['branch','split','representation']),
      'autonomous_paired':paired,'binding_natural':bs,'binding_initial_execution_comparison':execution,'scale':scales,'scale_target_token_lengths':target_lengths,
      'cross_model_relations':rsa_report(),'seconds':time.monotonic()-start,
      'limits':['Conservative final-answer parsing abstains unless an isolated digit or explicitly marked finalanswer is present; report coverage, not arbitrary last-number correctness.',
        'A parsed answer-like suffix at a censored token limit can still be provisional; stopped-correct and parsed-but-censored fractions are recorded separately.',
        'Natural continuation changes are not semantic-quality scores. No LLM judge or invented manual correctness labels.',
        'Paired old autonomous32program rows represent8semantic cases, new16rows represent4cases; small group counts constrain intervals.',
        'Global directions .02 and per-case Beta .10 are distinct predefined intervention groups, not identical-step method rankings.']}
    save(BASE/'analysis/behavior.json',result);ledger('binding_behavior_and_RSA',result['seconds'])
    print('BINDING_BEHAVIOR_COMPLETE',len(autonomous),len(native),len(binding),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--rsa-only',action='store_true');args=parser.parse_args()
    rsa_report() if args.rsa_only else main()
