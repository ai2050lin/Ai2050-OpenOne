"""Paired native formation, own-history controls, and matched-model evidence."""
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'analysis'


def behavior(mode,branches):
    root=BASE/'late' if mode=='late' else BASE/'transfer/injection'
    records={}
    for b in branches:
        p=root/b/'commits' if mode=='late' else root/'commits'/b
        records[b]={r['sample_id']:r for f in sorted(p.glob('*.json')) if (r:=read(f))}
    native=records['native'];assert all(set(rr)==set(native) for rr in records.values())
    detail=[];summary=[]
    for branch,rr in records.items():
      for sid,r in rr.items():
        n=native[sid];s=r['answer_scoring'];ns=n['answer_scoring'];fire=r.get('intervention');fire=fire if fire is not None else {}
        detail.append({'mode':mode,'branch':branch,'sample_id':sid,'source_group':r['source_group'],'representation':r['representation'],
          'correct_and_stopped':int(s['parsed_and_stopped_correct']),'native_correct_and_stopped':int(ns['parsed_and_stopped_correct']),
          'success_delta':int(s['parsed_and_stopped_correct'])-int(ns['parsed_and_stopped_correct']),
          'tokens':len(r['generated_ids']),'native_tokens':len(n['generated_ids']),'token_delta':len(r['generated_ids'])-len(n['generated_ids']),
          'parsed':s['conservative_final_answer'] is not None,'EOS':s['EOS'],'censored':s['censored'],
          'parsed_wrong':s['conservative_final_answer'] is not None and not s['conservative_final_correct'],
          'new_scorer_gain_over_primary':int(s['parsed_and_stopped_correct'])-int(s['primary']['parsed_and_stopped_correct']),
          'trigger_step':fire.get('step'),'first_divergence':r.get('first_token_divergence_from_native',r.get('first_divergence')),
          'own_history_next_KV':r.get('own_history_next_KV_comparison'),
          'all_same_current_history_KV_exact':fire.get('all_same_history_KV_bit_equal',r.get('cache_audit',{}).get('same_current_history_KV_exact'))})
      for rep in ['all','en','python','zh','en_reordered']:
        rows=[r for r in detail if r['branch']==branch and (rep=='all' or r['representation']==rep)]
        if not rows:continue
        groups=[r['source_group'] for r in rows];matched=[r['own_history_next_KV'] for r in rows if r['own_history_next_KV'] is not None]
        summary.append({'mode':mode,'branch':branch,'representation':rep,'expressions':len(rows),'source_groups':len(set(groups)),
          'correct_and_stopped':sum(r['correct_and_stopped'] for r in rows),'parsed':sum(r['parsed'] for r in rows),
          'unparsed_EOS':sum(r['EOS'] and not r['parsed'] for r in rows),'censored':sum(r['censored'] for r in rows),
          'parsed_wrong':sum(r['parsed_wrong'] for r in rows),'mean_tokens':float(np.mean([r['tokens'] for r in rows])),
          'paired_success_delta_vs_native':clustered([r['success_delta'] for r in rows],groups),
          'paired_token_delta_vs_native':clustered([r['token_delta'] for r in rows],groups),
          'scoring_only_gain_over_primary':sum(r['new_scorer_gain_over_primary'] for r in rows),
          'triggered':sum(r['trigger_step'] is not None for r in rows),
          'next_KV_compared':len(matched),'next_KV_changed':sum(not r['own_history_cache_equal'] for r in matched),
          'next_KV_after_changed_trigger_token':sum(r['new_token_at_trigger_differs'] for r in matched)})
    return detail,summary


def formation():
    result=read(BASE/'formation/result.json');p=read(BASE/'formation/protocol.json');material=gzread(BASE/'formation/material.json.gz');panel=material['panel']
    with np.load(BASE/'formation/native_baseline.npz') as z:native=z['loss'].copy()
    with np.load(BASE/'formation/bridge_baseline.npz') as z:bridge=z['loss'].copy()
    pairs=[];numerics=[]
    for kind,cohort in [('natural_content','all')]+sorted({(r['kind'],r['cohort']) for r in panel}):
        ix=[i for i,r in enumerate(panel) if r['kind']==kind and (cohort=='all' or r['cohort']==cohort)]
        numerics.append({'kind':kind,'cohort':cohort,'examples':len(ix),'bridge_minus_native_NLL':clustered(bridge[ix]-native[ix],[panel[i]['source_group'] for i in ix])})
    for seed in p['seeds']:
      for checkpoint in [*p['checkpoints'],'deployed_BF16']:
        filename=f'checkpoint{checkpoint}.npz' if isinstance(checkpoint,int) else checkpoint+'.npz'
        with np.load(BASE/'formation'/f'natural_target_{seed}'/filename) as z:a=z['loss'].copy();at=z['argmax'].copy()
        with np.load(BASE/'formation'/f'within_cohort_permuted_target_{seed}'/filename) as z:b=z['loss'].copy();bt=z['argmax'].copy()
        for kind,cohort in [('natural_content','all')]+sorted({(r['kind'],r['cohort']) for r in panel}):
            ix=[i for i,r in enumerate(panel) if r['kind']==kind and (cohort=='all' or r['cohort']==cohort)];gs=[panel[i]['source_group'] for i in ix]
            base=native if checkpoint=='deployed_BF16' else bridge;gold=np.array([panel[i]['target'] for i in ix])
            pairs.append({'seed':seed,'checkpoint':checkpoint,'kind':kind,'cohort':cohort,'examples':len(ix),
              'natural_minus_permuted_NLL':clustered(a[ix]-b[ix],gs),'natural_minus_baseline_NLL':clustered(a[ix]-base[ix],gs),
              'permuted_minus_baseline_NLL':clustered(b[ix]-base[ix],gs),
              'natural_argmax_accuracy':float(np.mean(at[ix]==gold)),'permuted_argmax_accuracy':float(np.mean(bt[ix]==gold))})
    return {'executed_runs':len(result['runs']),'training_pool_examples':len(material['train']),'draws_per_run':p['steps']*p['batch_examples'],
      'distinct_drawn_examples_per_run':[{'seed':r['seed'],'condition':r['condition'],'examples':r['distinct_drawn_examples']} for r in result['runs']],
      'actual_parameter_displacements':[{k:r[k] for k in ['condition','seed','delta_FP32_norm','delta_native_BF16_norm']} for r in result['runs']],
      'displacement_control_limit':'Matched0.02per-step gradient norm does not match the32-step cumulative displacement or the actualBF16deployment norm; different gradient-direction coherence can contribute to target-condition differences. No norm-matched interpolation control was executed.',
      'bridge_numerical_baseline':numerics,'paired_reports':pairs,
      'scope':'Natural target versus within-cohort permuted targets, two fixed draw orders. Each32-step run draws128example occurrences from the576example pool; not all576examples are trained in each run. Native pretraining history and multi-step reasoning chain formation are not reconstructed.'}


def scale():
    names=['qwen4','qwen14','glm4'];source=read(BASE/'scale/protocol.json')['source_ids'];commits={}
    for model in names:commits[model]={r['sample_id']:r for p in (BASE/'scale'/model/'commits').glob('*.json') if (r:=read(p))}
    common=[sid for sid in source if all(sid in commits[m] for m in names)];assert len(common)>=16
    summaries=[];grams={};paired_stats={}
    for model in names:
        gram=np.zeros((100,100));effects=[]
        for sid in common:
            with np.load(BASE/'scale'/model/'fields'/f'{sid}.npz') as z:h=unbits(z['postnorm']).astype(float);st=z['full_vocabulary_statistics']
            h-=h.mean(0);gram+=h@h.T/h.shape[1];effects.append(st[:,1])
        gram/=len(common);grams[model]=gram;paired_stats[model]=np.stack(effects)
        summaries.append({'model':model,'matched_source_documents':len(common),
          'native_query_KL_to_standalone':clustered(np.mean(effects,1),[commits[model][sid]['source_group'] for sid in common]),
          'actual_width':read(BASE/'scale'/model/'result.json')['width']})
    geometry=[];upper=np.triu_indices(100,1)
    for i,a in enumerate(names):
      for b in names[i+1:]:
        A=grams[a];B=grams[b];aa=A[upper];bb=B[upper]
        geometry.append({'models':[a,b],'matched_queries':100,'offdiagonal_entries':len(aa),
          'centered_Gram_entry_correlation':float(np.corrcoef(aa,bb)[0,1]),
          'unit_Frobenius_Gram_difference':float(np.linalg.norm(A/np.linalg.norm(A)-B/np.linalg.norm(B))),
          'scope':'Within-model all-coordinate Gram, same fixedquery indices and matching source texts. Entries are dependent and not independent trials; no native coordinate alignment or universal isometry inference.'})
    npz(OUT/'matched_model_query_geometry.npz',**{m+'__centered_Gram':g for m,g in grams.items()},**{m+'__all_query_KL':v for m,v in paired_stats.items()})
    return {'common_source_ids':common,'matched_documents':len(common),'within_model_summaries':summaries,'descriptive_between_model_Gram_comparisons':geometry,
      'scope':'OriginalBF16 models with their own tokenization, Q/K rotation and native dimensions. Architecture, training, size, tokenizer and execution residency are not isolated.'}


def main():
    out=OUT/'phase2742.json'
    if out.exists():return
    start=time.monotonic();req=['formation/result.json','transfer/injection/result.json']
    branches=['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']
    req += ['late/'+b+'/result.json' for b in branches]+['scale/'+m+'/result.json' for m in ['qwen4','qwen14','glm4']]
    for path in req:assert read(BASE/path)['all_passed']
    ld,ls=behavior('late',branches);idetail,isummary=behavior('injection',['native','code_identity','mapped_code'])
    compressed(OUT/'all_own_history_behavior_metrics.json.gz',ld+idetail)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'formation':formation(),'late':ls,'injection':isummary,'scale':scale(),
      'mapping_qualification':read(BASE/'transfer/injection/mapping_qualification.json'),
      'native_batch_shape_controls':read(BASE/'late/native/result.json')['B1_shape_controls'],
      'limitations':['Correct terminal value and EOS are separate from reasoning-chain correctness.',
        'A shorter response can be premature or still correct; high vocabulary entropy alone does not identify unnecessary explanation.',
        'Uniform category bias cannot reorder current digit probabilities; future token and cache changes can still alter later computation.',
        'Scorer coverage changes are measurements, not model ability gains; primary, prior-secondary and frozen prospective records all remain available.',
        'Intervals compare source-cluster means conditional on these models, prompts, draws and frozen rules. They do not quantify training-population or model-family uncertainty.',
        'Natural/permuted labels share draws, step count and per-step norm, not equal cumulative native parameter displacement; target-content and update-direction/magnitude explanations are not fully isolated.',
        'Mapping injection receives an observed matching code-response vector and only changes the initial readout; it does not transplant a complete autoregressive state.'],
      'seconds':time.monotonic()-start}
    save(out,result);ledger('actual_formation_history_and_scale_synthesis',result['seconds']);print('QUERY_PHASE2742_ANALYSIS',result['seconds'],flush=True)


if __name__=='__main__':main()
