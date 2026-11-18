"""Post-execution summaries, confidence intervals and explicit failure accounting."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_analysis import bootstrap

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(c-r),float(c+r)]

def behavioral(model,split):
    root=D.OUT/model;rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    data=D.read(root/f'{split}-behavior.json')['rows'];gate=D.read(root/f'{split}-gate.json')
    misses=[];categories={};arrays={}
    for key in ['forced_correct','normalized_correct','strict_correct','top1_correct']:
        x=np.array([a[key] and (a['terminated'] if key in ['normalized_correct','strict_correct'] else True) for a in data],float)
        arrays[key]={'accuracy':float(x.mean()),'family_bootstrap95':bootstrap(rows,x,f'{model}/{split}/{key}'),
            'wilson_descriptive_independent_rows':wilson(int(x.sum()),len(x))}
    for r,a in zip(rows,data):
        if a['normalized_correct'] and a['terminated']:continue
        text=D.normalized(a['generated_text'])
        if not a['terminated']:category='max_token_budget_unterminated'
        elif text in [D.normalized(v) for v in r['values']]:
            j=[D.normalized(v) for v in r['values']].index(text)
            category='wrong_entity_same_source' if j//2==r['source'] else 'wrong_source_same_entity' if j%2==r['entity'] else 'wrong_source_and_entity'
        elif any(name.casefold() in text for name in r['names']):category='entity_name_or_labeled_answer'
        else:category='other_output'
        categories[category]=categories.get(category,0)+1
        misses.append({'id':r['id'],'gold':r['gold'],'text':a['generated_text'],'category':category,'source':r['source'],'entity':r['entity'],'frame':r['frame']})
    return {'gate':{k:v for k,v in gate.items() if k!='provenance'},'estimates':arrays,'failures':misses,'failure_categories':categories,
        'CI_scope':'Family bootstrap respects within-family12 correlated queries; Wilson row intervals descriptive only. Neither makes fixed point gate a confidence-bound test.'}

def formation_comparisons(model,split):
    root=D.OUT/model;rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    report=D.read(root/f'{split}-formation.json');stored=np.load(root/f'{split}-formation-predictions.npz')
    comparisons={}
    for c in report['candidates']:
        tag=c['id'];truth=stored[tag+'_truth'];native=np.square(truth-stored[tag+'_native']).sum(1)
        entry={}
        for baseline in ['random','embedding','native_past_shuffled']:
            se=np.square(truth-stored[tag+'_'+baseline]).sum(1);gain=se-native
            entry[baseline]={'native_MSE_gain':float(gain.mean()),'gain_ci_family':bootstrap(rows,gain,f'{model}/{split}/{tag}/versus{baseline}')}
        se_past=np.square(truth-stored[tag+'_native_past']).sum(1)
        se_shuffle=np.square(truth-stored[tag+'_native_past_shuffled']).sum(1)
        gain=se_shuffle-se_past
        entry['matched_past_V_shuffle']={'unshuffled_past_MSE_gain':float(gain.mean()),'gain_ci_family':bootstrap(rows,gain,f'{model}/{split}/{tag}/matchedPastShuffle'),
            'scope':'Both exclude final-query V. Entire record V segments are moved, including contextual source/entity information; not a value-only causal intervention.'}
        comparisons[tag]=entry
    stored.close()
    return {'results':report,'paired_comparisons':comparisons}

def causal_comparisons(model,split):
    root=D.OUT/model;rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];c=D.read(root/f'{split}-causal.json')
    comparisons={}
    for kind,one,two in [('head','head_correct','head_other'),('mlp','mlp_selected','mlp_random')]:
        a=c['rows'][one];b=c['rows'][two]
        margin=np.array([x['gold_margin']-y['gold_margin'] for x,y in zip(a,b)])
        accuracy=np.array([float(x['forced_correct'])-float(y['forced_correct']) for x,y in zip(a,b)])
        groups={}
        for x in c['local']:
            if x['kind']==kind:groups.setdefault((x['chunk'],x['offset'],x['block']),{})[x['arm']]=x
        errors=[];targets=[];scales={one:[],two:[]};failed_rows=set();byblock={}
        for group in groups.values():
            aa,bb=group[one],group[two];assert aa['target_norm']==bb['target_norm']
            targets.extend(aa['target_norm'])
            errors.extend((np.abs(np.array(aa['actual_norm'])-bb['actual_norm'])/np.maximum(aa['target_norm'],1e-8)).tolist())
            for arm in [one,two]:scales[arm].extend(group[arm]['scale'])
            for arm in [one,two]:
                obs=group[arm];bad=np.abs(np.array(obs['actual_norm'])-obs['target_norm'])/np.maximum(obs['target_norm'],1e-8)>.05
                failed_rows.update(obs['chunk']+obs['offset']+int(i) for i in np.flatnonzero(bad))
                byblock.setdefault(str(obs['block'])+'/'+arm,[]).extend((~bad).tolist())
        comparisons[kind]={'margin_difference':float(margin.mean()),'margin_difference_ci_family':bootstrap(rows,margin,f'{model}/{split}/{kind}/pairMargin'),
            'accuracy_difference':float(accuracy.mean()),'accuracy_difference_ci_family':bootstrap(rows,accuracy,f'{model}/{split}/{kind}/pairAcc'),
            'actual_pair_relative_norm_error_max':max(errors),'actual_pair_relative_norm_error_p95':float(np.quantile(errors,.95)),
            'mean_common_target_norm':float(np.mean(targets)),'zero_target_fraction':float(np.mean(np.array(targets)==0)),
            'mean_scales':{a:float(np.mean(s)) for a,s in scales.items()},'frozen_matching_admitted':c['norm_matching'][kind]['admitted']}
        comparisons[kind]['matching_failed_rows']={'n':len(failed_rows),'ids':[rows[i]['id'] for i in sorted(failed_rows)],
            'contribution_to_full_sample_mean_margin_difference':float(margin[list(failed_rows)].sum()/len(rows)),'all_rows_retained':True,
            'scope':'Descriptive accounting of allowed or disallowed numerical failures, not a new filtered estimate or replacement gate.'}
        comparisons[kind]['matching_by_block_arm']={key:{'n':len(v),'fraction_within_5pct':float(np.mean(v))} for key,v in byblock.items()}
        comparisons[kind]['interpretation_status']='admitted matched-dose group comparison' if c['norm_matching'][kind]['admitted'] else 'FAILED matching gate; descriptive execution only, no matched-specificity conclusion'
    return {'summary':c['summary'],'matching':c['norm_matching'],'paired_comparisons':comparisons,'checks':c['checks']}

def resources():
    progress=[json.loads(line) for line in (D.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    processes={}
    for r in progress:
        p=processes.setdefault(r['pid'],{'elapsed':0,'cuda':False,'stage':r['stage']})
        if '/gpu_wait' not in r['stage']:p['elapsed']=max(p['elapsed'],r.get('elapsed_seconds',0))
        p['cuda'] |= r.get('cuda_peak',0)>0
    return {'max_cuda_GiB':max(r.get('cuda_peak',0) for r in progress)/2**30,
        'max_shared_MiB':max(r.get('shared_gpu_bytes') or 0 for r in progress)/2**20,
        'max_dedicated_DXGI_GiB':max(r.get('dedicated_gpu_bytes') or 0 for r in progress)/2**30,
        'min_available_RAM_GiB':min(r['ram_available_bytes'] for r in progress)/2**30,
        'logged_processes':processes,
        'time_caution':'Within campaign multiple stages reuse a PID and reset elapsed timers. Sum per-artifact elapsed_seconds for execution time; per-PID maxima are not cumulative GPU minutes.',
        'artifact_bytes':sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file())}

def rollout_comparisons(model,split):
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    data=D.read(D.OUT/model/f'{split}-update-output.json')['rows'];results={}
    for baseline in ['mean','state128','shuffled_content']:
        one=data['routed_content'];two=data[baseline]
        acc=np.array([float(a['forced_correct'])-float(b['forced_correct']) for a,b in zip(one,two)])
        kl=np.array([b['KL']-a['KL'] for a,b in zip(one,two)])
        results[baseline]={'routed_accuracy_gain':float(acc.mean()),'accuracy_gain_ci_family':bootstrap(rows,acc,f'{model}/{split}/rollout/{baseline}/acc'),
            'routed_KL_reduction':float(kl.mean()),'KL_reduction_ci_family':bootstrap(rows,kl,f'{model}/{split}/rollout/{baseline}/kl')}
    return {'comparisons':results,'scope':'Descriptive uncertainty for already frozen methods; all models conditional on real prefix V for routed variants. Unequal information budget versus state128.'}

def report():
    assert D.read(D.OUT/'campaign-complete.json')['completed']
    assert D.read(D.OUT/'supplement-complete.json')['completed']
    models={}
    for model in D.MODELS:
        root=D.OUT/model;panels={}
        for split in D.COUNTS:
            if not (root/f'{split}-behavior.json').exists():continue
            result={'behavior':behavioral(model,split)}
            if (root/f'{split}-formation.json').exists():result['formation']=formation_comparisons(model,split)
            if (root/f'{split}-causal.json').exists():result['causal']=causal_comparisons(model,split)
            if (root/f'{split}-update-output.json').exists():result['rollout_comparisons']=rollout_comparisons(model,split)
            for suffix in ['update','update-output','native-parameters','content-controls','error-split','matching-numerics','native39-matching-numerics','matching-coordinate-lineage']:
                if (root/f'{split}-{suffix}.json').exists():result[suffix]=D.read(root/f'{split}-{suffix}.json')
            panels[split]=result
        required=['confirm','lockbox_frame'];complete=all('formation' in panels.get(s,{}) for s in required)
        common=sorted(set.intersection(*[{c['id'] for c in panels[s]['formation']['results']['candidates'] if c['passes']} for s in required])) if complete else []
        models[model]={'panels':panels,'model_snapshot':D.read(D.OUT/f'{model}-identity.json')['model_snapshot'],
            'primary_outcome':{'both_required_panels_completed':complete,'common_passing_heads':common,'at_least_one_frozen_candidate_replicated':complete and bool(common),
                'scope':'Within-model frozen operational criterion. Cross-model coordinate correspondence, causal equivalence and formal evidence promotion are not implied.'}}
    D.save('report.json',{'models':models,'resources':resources(),'replication_decision':D.read(D.OUT/'qwen14/replication-decision.json'),
        'inference_scope':'Earlier-state conditional prediction, local original-model interventions and conditional recurrence. Not a complete text-to-model surrogate or universal language law.',
        'exposure':'All new panel outcomes now exposed; follow-up selection requires new heldout families. Layer window informed by prior Phase2906.',
        'formal_evidence_promotion':0})
    print('REPORT_COMPLETE')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['report']);ap.parse_args();report()
