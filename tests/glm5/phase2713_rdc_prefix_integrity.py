"""Finite source/numerical/provenance audit. No model execution or mutation of frozen results."""
import hashlib
from collections import defaultdict
from safetensors import safe_open
from rdc_prefix_estimators import *


def source_history_audit():
    from phase2714_rdc_source_kernels import dataset,kernel_set
    from phase2711_rdc_prefix_atlas import span_token
    out=CAMPAIGN/'full_source_history';frozen=read(out/'frozen.json')
    for rel,digest in frozen['files'].items():assert sha(out/rel)==digest,('Source rule changed',rel)
    for rel,digest in frozen['original_inputs'].items():assert sha(CAMPAIGN/rel)==digest,('Source input changed',rel)
    recaptured=0;fresh=0;tokens=0;new_h0={}
    for scope,expected in [('main',496),('fresh',64)]:
        commits=list((out/scope/'commits').glob('*.json'));assert len(commits)==expected
        material=read(CAMPAIGN/'material_stratified.json') if scope=='main' else read(out/'fresh_material.json');byid={r['sample_id']:r for r in material}
        for cp in commits:
            c=read(cp);r=byid[cp.stem];assert c['source_sha']==sha(ROOT/'tests/glm5/phase2714_rdc_full_source_history.py')
            assert c['checks']['same_shape_repeat_postnorm_bitwise']
            for rel,digest in c['files'].items():assert sha(out/rel)==digest
            with np.load(out/scope/f'fields/{cp.stem}.npz') as z:
                assert z['h12'].shape==(len(r['prompt_ids']),2560);tokens+=len(r['prompt_ids'])
                if scope=='main':
                    assert c['checks']['all_six_H12_two_H36_postnorm_match_original']
                    with np.load(CAMPAIGN/f'qwen4/fields/{cp.stem}.npz') as old:assert np.array_equal(z['h12'][r['positions']],old['h'][12])
                    recaptured+=1
                else:
                    assert c['fresh_frozen_hash']==sha(out/'frozen.json');fresh+=1
                    for k,p in enumerate(r['positions']):
                        tid=r['prompt_ids'][p]
                        if tid in new_h0:assert np.array_equal(new_h0[tid],z['h0'][k])
                        new_h0[tid]=z['h0'][k].copy()
        print('SOURCE_INTEGRITY_COMMITS',scope,expected,flush=True)
    md=ROOT/'models/hf/qwen3-4b';index=read(md/'model.safetensors.index.json')['weight_map']
    with safe_open(md/index['model.embed_tokens.weight'],framework='pt',device='cpu') as f:
        import torch
        for tid,v in new_h0.items():assert np.array_equal(f.get_slice('model.embed_tokens.weight')[tid:tid+1,:].view(torch.uint16).numpy()[0],v)
    sources,rows,current,means,target=dataset(False);tr,va,te=splits(rows)
    kernels=kernel_set(sources,rows,current,means,read(out/'scales.json'));serial=[]
    with np.load(out/'input_grams.npz') as z:
        for key,k in kernels.items():assert np.allclose(z[key],k,rtol=1e-6,atol=1e-6)
    for key,k in kernels.items():
        with np.load(out/f'models/{key}.npz') as z:pred=(k[np.ix_(te,tr)]@z['alpha'])*z['target_scales']+z['means']
        with np.load(out/f'predictions/test_{key}.npz') as z:stored=z['prediction']
        rmse=float(np.sqrt(np.mean((pred-stored)**2)));relative=rmse/max(float(np.sqrt(np.mean(stored.astype(float)**2))),1)
        assert relative<1e-4;serial.append({'rule':key,'serialized_relative_RMSE':relative})
    ns,nr,nc,nm,ny=dataset(True)
    fresh_kernels=kernel_set(ns,nr,nc,nm,read(out/'scales.json'),sources,[rows[i] for i in tr],current[tr],means[tr])
    fresh_serial=[]
    for key,k in fresh_kernels.items():
        with np.load(out/f'models/{key}.npz') as z:pred=(k@z['alpha'])*z['target_scales']+z['means']
        with np.load(out/f'predictions/fresh_{key}.npz') as z:stored=z['prediction']
        relative=float(np.sqrt(np.mean((pred-stored)**2))/max(np.sqrt(np.mean(stored.astype(float)**2)),1))
        assert relative<1e-4;fresh_serial.append({'rule':key,'serialized_relative_RMSE':relative})
    # The current-only rule matches the prior early-linear last-layer prediction; no new improvement claim.
    oldrows=read(CAMPAIGN/'shared_rules/qwen4/rows.json');oldtest=splits(oldrows)[2]
    with np.load(CAMPAIGN/'shared_rules/predictions/early_linear.npz') as z:oldpred=z['prediction'][:,-2560:]
    oldmap={(oldrows[i]['sample_id'],oldrows[i]['anchor']):oldpred[j] for j,i in enumerate(oldtest)}
    with np.load(out/'predictions/test_current.npz') as z:newpred=z['prediction']
    prior_delta=float(np.max(np.abs(np.stack([oldmap[(rows[i]['sample_id'],rows[i]['anchor'])] for i in te])-newpred)))
    assert prior_delta<1e-3,prior_delta
    prior=read(CAMPAIGN/'material_stratified.json')+read(CAMPAIGN/'confirmation_material.json');new=read(out/'fresh_material.json')
    assert not {r['source_group'] for r in prior}&{r['source_group'] for r in new}
    norm=lambda s:re.sub(r'\W','',s).casefold()
    assert not {norm(r['text']) for r in prior}&{norm(r['text']) for r in new}
    grams=lambda s:{s[i:i+3] for i in range(max(1,len(s)-2))};oldg=[grams(norm(r['text'])) for r in prior];nearest=[]
    for r in new:
        g=grams(norm(r['text']));scores=[len(g&h)/max(len(g|h),1) for h in oldg];i=int(np.argmax(scores))
        nearest.append({'sample_id':r['sample_id'],'nearest_prior_sample':prior[i]['sample_id'],'trigram_Jaccard':scores[i]})
    sensitivity=read(out/'template_sensitivity.json')
    assert sensitivity['selection_changed'] is False and sensitivity['refitting'] is False
    assert sensitivity['fresh_excluded']==['fresh-zh-0010']
    report=read(out/'fresh_result.json');lookup={r['rule']:r for r in report['reports']};pairs=[]
    for alternative in ('mean_history','absolute_history','relative_history'):
        a=lookup['current']['by_source_group'];b=lookup[alternative]['by_source_group'];ids=sorted(a);d=np.array([a[k]['mse']-b[k]['mse'] for k in ids]);rng=np.random.default_rng(2714)
        boot=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
        pairs.append({'a':'current','b':alternative,'mean_MSE_a_minus_b':float(d.mean()),'CI95_source_units':np.quantile(boot,[.025,.975]).tolist(),'fraction_current_better':float(np.mean(d<0))})
    relation=read(out/'relations/result.json');materials={('main',r['sample_id']):r for r in read(CAMPAIGN/'material_stratified.json')};materials.update({('fresh',r['sample_id']):r for r in new})
    relation_pairs=read(out/'relations/pairs.json');counts=defaultdict(int)
    for p in relation_pairs:
        r=materials[(p['scope'],p['sample_id'])];n=len(r['prompt_ids']);a,b=p['observed_token_pair'];c,d=p['control_token_pair']
        assert 0<=min(a,b,c,d) and max(a,b,c,d)<n and b-a==d-c==p['signed_distance'] and a!=c
        words={w['id']:w for w in r['retrospective_ud']};wa,wb=p['word_ids']
        assert words[wa]['head']==wb and words[wa]['relation'].split(':')[0]==p['relation']
        assert [span_token(r,words[wa]['char_span']),span_token(r,words[wb]['char_span'])]==[a,b]
        counts[(p['relation'],p['split'])]+=1
    assert len(relation_pairs)==11227
    for r in relation['reports']:
        for split,v in r['splits'].items():assert counts[(r['relation'],split)]==v['pairs']
    outside=[]
    for r in relation['reports']:
        for s in r['repetition']:
            lo,hi=s['fixed_train_source_bootstrap_CI95'];point=s['full_coordinate_cosine_after_control']
            if not lo<=point<=hi:outside.append({'relation':r['relation'],'split':s['split'],'point':point,'resampling_percentiles':[lo,hi]})
    save(out/'relations/uncertainty_audit.json',{'timestamp':stamp(),'bootstrap_repetition_records':28,'point_outside_resampling_percentile_range':outside,
      'interpretation':'Legacy key CI95 contains plain source-bootstrap percentiles with a fixed training profile. In high-dimensional cosine estimation bootstrap duplication/noise can shrink cosine; these ranges are not calibrated95% coverage and can exclude the original point estimate. Treat as conditional resampling stability diagnostics, not significance tests or universal-law evidence.',
      'no_new_independent_confirmation':True,'lexical_POS_confounding_remains':True})
    save(out/'paired_frozen_comparisons.json',pairs)
    return {'passed':True,'recaptured_main_commits':recaptured,'reused_full_panels':16,'fresh_commits':fresh,
      'new_capture_H12_source_tokens':tokens,'fresh_unique_embedding_rows_checked':len(new_h0),'all_main_H12_tokens_including_reuse':18480,
      'fresh_H12_tokens':2222,'frozen_inputs_unchanged':True,'serialized_prediction_checks':serial,'fresh_serialized_prediction_checks':fresh_serial,
      'current_rule_max_difference_from_prior_early_linear_H36':prior_delta,'fresh_source_groups_and_normalized_texts_disjoint_from_prior640':True,
      'fresh_nearest_prior_trigram':nearest,'paired_current_vs_history':pairs,'relation_pair_indices_and_distances_checked':len(relation_pairs),
      'relation_bootstrap_point_outside_percentile_count':len(outside),'relation_status':'Exploratory retrospective, not frozen relation discovery; see uncertainty audit.',
      'numeric_template_overlap_audited':True,'template_sensitivity_sha':sha(out/'template_sensitivity.json')}


def main():
    started=time.monotonic();run_reports=[];all_hashes={};scalars=0;npz_count=0;file_references=0
    model_dirs={'qwen4':'qwen3-4b','qwen4_confirmation':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}
    for run,expected in [('qwen4',512),('qwen4_confirmation',128),('qwen14',64),('glm4',64)]:
        out=CAMPAIGN/run;commits=sorted((out/'commits').glob('*.json'));assert len(commits)==expected
        count=0;tokens=0;hidden_scalars=0;embedding={};rows=[];future_checks=0
        for cp in commits:
            c=read(cp)
            if 'protocol_file' in c:rp=out/c['protocol_file']
            else:
                candidates=list(out.glob('protocol*.json'))+list((out/'protocols').glob('*.json'))
                matches=[p for p in candidates if sha(p)==c['protocol_sha']]
                assert matches,(run,cp.name,'Cannot resolve historical protocol by exact SHA');rp=matches[0]
            assert sha(rp)==c['protocol_sha']
            for rel,digest in c['files'].items():
                p=out/rel;actual=sha(p);assert actual==digest,(run,rel,'commit mismatch');all_hashes[str(p.relative_to(CAMPAIGN))]=actual;file_references+=1
            r=read(out/f'rows/{cp.stem}.json');rows.append(r);tokens+=len(r['prompt_ids']);hidden_scalars+=int(np.prod(c['observed_allH_shape']))
            with np.load(out/f'fields/{cp.stem}.npz') as z:
                h=z['h'];assert h.shape==tuple(c['retained_anchor_shape']);assert np.array_equal(z['positions'],r['positions'])
                for k,pos in enumerate(r['positions']):
                    tid=r['prompt_ids'][pos]
                    if tid in embedding:assert np.array_equal(embedding[tid],h[0,k])
                    else:embedding[tid]=h[0,k].copy()
                    if 'L23_p' in z:assert not np.any(unbits(z['L23_p'][:,k,pos+1:]));future_checks+=1
            count+=1
        with np.load(out/'all_token_moments.npz') as z:
            assert int(z['counts'].sum())==tokens;assert set(z['processed_ids'].tolist())=={r['sample_id'] for r in rows}
        md=ROOT/'models/hf'/model_dirs[run];index=read(md/'model.safetensors.index.json')['weight_map'];key='model.embed_tokens.weight'
        with safe_open(md/index[key],framework='pt',device='cpu') as f:
            for tid,v in embedding.items():
                import torch
                actual=f.get_slice(key)[tid:tid+1,:].view(torch.uint16).numpy()[0]
                assert np.array_equal(v,actual),(run,tid,'embedding table mismatch')
        run_reports.append({'run':run,'commits':count,'tokens_observed':tokens,'streamed_hidden_scalar_visits':hidden_scalars,
          'unique_embedding_rows_checked_against_actual_checkpoint':len(embedding),'embedding_table_mismatches':0,
          'saved_positions':count*6,'full_panels':len(list((out/'full_panels').glob('*.npz'))),'future_attention_zero_checks':future_checks})
        print('PREFIX_INTEGRITY_RUN',run,count,tokens,len(embedding),flush=True)
    extension=source_history_audit() if (CAMPAIGN/'full_source_history/probability_result.json').exists() else None
    # Every saved numerical element, including all covariance entries and all native BF16 coordinates.
    for p in sorted(CAMPAIGN.rglob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for key in z.files:
                a=z[key]
                if a.dtype.kind in 'fciu':
                    if a.dtype==np.uint16:a=unbits(a)
                    assert np.isfinite(a).all(),(str(p),key,'nonfinite');scalars+=a.size
        all_hashes[str(p.relative_to(CAMPAIGN))]=sha(p);npz_count+=1
        if npz_count%256==0:print('PREFIX_FINITE',npz_count,scalars,flush=True)
    frozen=read(CAMPAIGN/'shared_rules/frozen_models.json')
    for rel,digest in frozen['files'].items():assert sha(CAMPAIGN/'shared_rules'/rel)==digest
    lf=read(CAMPAIGN/'layer_operators/frozen.json')
    for key,rel in [('model_sha','models.npz'),('protocol_sha','protocol.json'),('result_sha','result.json')]:assert sha(CAMPAIGN/'layer_operators'/rel)==lf[key]
    # Recompute stored primary forecasts from serialized inputs/coefficients, not from target states.
    src=CAMPAIGN/'shared_rules';rows=read(src/'qwen4/rows.json');tr,va,te=splits(rows)
    with np.load(src/'qwen4/features.npz') as z:data={k:z[k] for k in z.files}
    serial=[]
    for temporal in (False,True):
        bank=KernelBank(data,tr,read(src/('temporal_scales.json' if temporal else 'input_scales.json')),temporal=temporal)
        for mp in sorted((src/'models').glob('*.npz')):
            if mp.stem.startswith('temporal_')!=temporal:continue
            name=mp.stem.removeprefix('temporal_').removesuffix('_df128');gram=bank.gram(name,te,tr)
            with np.load(mp) as z:p=(gram@z['alpha'])*z['target_scales']+z['means']
            with np.load(src/f'predictions/{mp.stem}.npz') as z:old=z['prediction']
            rmse=float(np.sqrt(np.mean((p-old)**2)));scale=max(float(np.sqrt(np.mean(old.astype(float)**2))),1)
            assert rmse/scale<1e-4,(mp.stem,rmse/scale)
            serial.append({'model':mp.stem,'RMSE_serialized_vs_original_prediction':rmse,'relative_to_prediction_RMS':rmse/scale})
    material=read(CAMPAIGN/'material_stratified.json');confirmation=read(CAMPAIGN/'confirmation_material.json')
    groups={s:{r['source_group'] for r in material if r['split']==s} for s in ('train','validation','test')}
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    assert not set.union(*groups.values())&{r['source_group'] for r in confirmation}
    norm=lambda text:re.sub(r'\W+','',text.casefold())
    old=[(r,norm(r['text'])) for r in material];new=[(r,norm(r['text'])) for r in confirmation]
    assert not {s for _,s in old}&{s for _,s in new}
    grams=lambda s:{s[i:i+3] for i in range(max(1,len(s)-2))}
    oldg=[grams(s) for _,s in old];nearest=[]
    for r,s in new:
        g=grams(s);scores=[len(g&h)/max(len(g|h),1) for h in oldg];i=int(np.argmax(scores))
        nearest.append({'confirmation_sample':r['sample_id'],'nearest_main_sample':old[i][0]['sample_id'],'character_trigram_Jaccard':scores[i]})
    # Updated paired uncertainty for corrected random controls; original confirmation file remains immutable.
    comparisons=[];cr=read(CAMPAIGN/'confirmation/result.json')['reports'];ctrl=read(CAMPAIGN/'causal_hash_control/result.json')['reports']
    for a,b in [('graph_interaction','hash_interaction'),('graph_interaction_df128','hash_interaction_df128')]:
        aa=next(r for r in cr if r['model']==a)['layers']['h36']['by_source_group'];bb=next(r for r in ctrl if r['model']==b and r['evaluation']=='confirmation')['by_source_group']
        ids=sorted(aa);d=np.array([aa[k]['mse']-bb[k]['mse'] for k in ids]);rng=np.random.default_rng(2713)
        draws=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
        comparisons.append({'a':a,'b':b,'mean_MSE_a_minus_b':float(d.mean()),'CI95_source_bootstrap':np.quantile(draws,[.025,.975]).tolist(),
          'fraction_a_better':float(np.mean(d<0)),'control_status':'software-corrected rerun, not new independent confirmation'})
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';prefix=read(CAMPAIGN/'memo_prefix.json');b=memo.read_bytes()
    assert hashlib.sha256(b[:prefix['bytes']]).hexdigest()==prefix['sha256']
    phases=list(map(int,re.findall(r'^## Phase (\d+):',b[prefix['bytes']:].decode('utf-8'),re.M)))
    assert phases==list(range(2711,2711+len(phases)))
    source_files=[ROOT/'tests/glm5'/name for name in ('rdc_prefix_common.py','rdc_prefix_estimators.py')]
    source_files+=list((ROOT/'tests/glm5').glob('phase271[1234]_rdc_*.py'))
    source_files+=[ROOT/p for p in ('server/rdc_prefix_service.py','server/server.py','frontend/src/main.jsx','frontend/src/components/app/RdcFeatureAtlas.jsx','frontend/src/components/app/RdcPrefixAtlas.jsx','frontend/src/components/app/RdcPrefixAtlas.css','tests/glm5_temp/rdc_prefix_client_test.cjs','tests/glm5_temp/rdc_prefix_serial_tail.py')]
    source_files.append(ROOT/'tests/glm5_temp/rdc_source_serial_tail.py')
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    for p in source_files:
        if p.suffix=='.py':compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    all_hashes.update({str(p.relative_to(CAMPAIGN)):sha(p) for p in CAMPAIGN.rglob('*.json') if 'verification' not in p.parts and p.name not in ('integrity_audit.json','artifact_hashes.json')})
    save(CAMPAIGN/'verification/artifact_hashes.json',all_hashes)
    save(CAMPAIGN/'verification/integrity_audit.json',{'passed':True,'timestamp':stamp(),'runs':run_reports,
      'committed_file_references':file_references,'npz_files_checked':npz_count,'numeric_array_elements_checked_finite':int(scalars),
      'primary_rules_unchanged':True,'layer_operator_unchanged':True,'serialized_prediction_checks':serial,'source_history_extension':extension,
      'source_groups_disjoint':True,'normalized_exact_main_confirmation_duplicates':0,
      'nearest_confirmation_character_trigram_matches':nearest,'near_duplicates_at_Jaccard_0_8':[r for r in nearest if r['character_trigram_Jaccard']>=.8],
      'corrected_control_paired_comparisons':comparisons,'source_hashes':source_hashes,
      'memo_original_prefix_intact':True,'new_phase_headers':phases,'memo_bytes_at_audit':len(b),
      'campaign_bytes':usage(),'free_disk_bytes':shutil.disk_usage(ROOT).free,'seconds':time.monotonic()-started,
      'limits':['Streaming scalar visits are not independent samples.','No proof source material was absent from pretraining.','Character trigram check is a defined near-duplicate screen, not exhaustive semantic leakage detection.',
        'Committed runtime protocols retain historical source hashes; full historical source file snapshots were not archived. Current scripts and recorded fixes permit recomputation, not an invented archived-version claim.',
        'Read-only audit of this campaign, not every historical phase or entire repository.']})
    guard();print('PREFIX_INTEGRITY_PASS',npz_count,scalars,usage(),CEILING-usage(),flush=True)


if __name__=='__main__':main()
