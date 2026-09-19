"""Paired descriptive/clustered deployment and scale results; no new model fitting."""
from rdc_law_common import *


def main():
    start=time.monotonic();p=read(BASE/'deployment/protocol.json')
    assert read(BASE/'deployment/result.json')['trajectories']==672
    assert read(BASE/'own_history/result.json')['trajectories']==96
    branches={b:{r['sample_id']:r for file in (BASE/'deployment/rollouts'/b).glob('*.json') for r in [read(file)]} for b in p['rollout_branches']}
    native=branches['native'];strata=[];paired=[];firsttoken=[]
    for branch,data in branches.items():
      for cohort in ('all_natural','all_QA','gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa'):
        rr=[r for r in data.values() if r['cohort']==cohort or (cohort=='all_natural' and r['kind']=='natural') or (cohort=='all_QA' and r['kind']=='QA')]
        assert rr
        result={'branch':branch,'group':cohort,'rows':len(rr),'sources':len({r['source_group'] for r in rr}),
            'mean_new_tokens':float(np.mean([len(r['generated_ids']) for r in rr])),
            'EOS_fraction':float(np.mean([r['stopped_by_native_EOS'] for r in rr])),
            'repeated4gram':float(np.mean([r['repeated_4gram_fraction'] for r in rr]))}
        if rr[0]['kind']=='QA':
            result.update(EM=float(np.mean([r['normalized_full_EM'] for r in rr])),F1=float(np.mean([r['answer_F1'] for r in rr])))
            if branch!='native':
                for metric in ('normalized_full_EM','answer_F1'):
                    delta=[float(r[metric])-float(native[r['sample_id']][metric]) for r in rr]
                    paired.append({'branch':branch,'group':cohort,'metric':metric,'mean_paired_change':float(np.mean(delta)),
                        'source_cluster':clustered(delta,[r['source_group'] for r in rr]),
                        'improved_cases':sum(x>0 for x in delta),'worsened_cases':sum(x<0 for x in delta),
                        'scope':'Pairedsamequestions, originalanswers/fullnormalizedstrings. Smallcohorts and sampleconditional bootstrap; not a general languageability effect.'})
        else:
            if branch!='native':
                delta=[native[r['sample_id']]['steps'][0]['given_original_next_token_logprob']-r['steps'][0]['given_original_next_token_logprob'] for r in rr]
                firsttoken.append({'branch':branch,'group':cohort,'queries':len(rr),'BF16_NLL_change_vs_native':clustered(delta,[r['source_group'] for r in rr]),
                    'scope':'Singleoriginalnexttoken givenONLYas scoringlabel, sameactualprefillshape; positiveNLLchange is worse. Model stillgenerates ownargmax, not goldtoken.'})
        strata.append(result)
    scale=[];scale_paired=[];scale_behavior={};alignment=[];scale_visibility=[];visibility_counts=[];visible_paired=[]
    meta=gzread(BASE/'material.json.gz')+gzread(BASE/'confirmation_material.json.gz');at={r['sample_id']:r for r in meta}
    fresh=[at[s] for s in p['scale_ids'] if at[s]['split']=='confirmation']
    # Owncharacteralignedanchors keep same row/anchor order, but tokenizer/context
    # boundaries differ; paired errors compare routes WITHIN each model only.
    groups=[r['source_group'] for r in fresh for _ in range(3)]
    for model in ('qwen4','qwen14','glm4'):
        root=BASE/'scale'/model;r=read(root/'result.json');scale.append(r)
        rows=[read(f) for f in (root/'rows').glob('*.json')]
        alignment.append({'model':model,'rows':len(rows),'actual_natural_tokens':sum(len(r['prompt_ids']) for r in rows),
            'exact_character_endpoints':sum(sum(r['exact_char_endpoint']) for r in rows),'anchor_endpoints':3*len(rows),
            'scope':'Differenttokenizers may end strictlybefore the Q4characterendpoint; thosepositions are retained and labeled, not silently claimed perfectlyaligned.'})
        rowmap={r['sample_id']:r for r in rows};visible=[]
        for original in fresh:
            observed=rowmap[original['sample_id']]
            for anchor,position in enumerate(observed['positions']):
                end=observed['offsets'][position][1]
                types={edge['type'][3:] for edge in original.get('retrospective_graph',[])
                    if edge.get('type','').startswith('ud:') and 'available_after_token' in edge
                    and original['token_offsets'][edge['available_after_token']][1]<=end}
                pairs=[pair for pair in ('obj+advcl','nsubj:pass+obl') if all(t in types for t in pair.split('+'))]
                assert set(pairs)<=set(original['held_relation_combinations'])
                visible.append(bool(pairs))
                scale_visibility.append({'model':model,'sample_id':original['sample_id'],'source_group':original['source_group'],
                    'anchor_index':anchor,'own_token_position':position,'available_character_end':end,
                    'window_declared_pairs':original['held_relation_combinations'],'endpoint_visible_pairs':pairs})
        visible=np.asarray(visible,dtype=bool)
        visibility_counts.append({'model':model,'confirmation_queries':len(visible),'prefix_endpoint_visible_queries':int(visible.sum()),
            'window_declared_queries':sum(bool(r['held_relation_combinations']) for r in fresh)*3,
            'scope':'Posthoc boundary audit using each own-tokenizer actual character endpoint. Full-sentence gold labels remain retrospective, not online features or verified semantic composition.'})
        predictions={}
        for decoder in ('direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product'):
            with np.load(root/'confirmation_predictions'/f'{decoder}.npz') as z:predictions[decoder]=z['relative_MSE']
        for decoder,errors in predictions.items():
            delta=predictions['direct_mlp']-errors
            scale_paired.append({'model':model,'decoder':decoder,'queries':len(errors),'gain_over_own_validation_selected_direct':clustered(delta,groups),
                'scope':'Validationselects eachdecoder kernel/df before ownconfirmation. Different targetdimensionalities are declared; not a universal capacity identity. MainQ4subset was alreadyobserved, so Q4scale is reanalysis.'})
            visible_paired.append({'model':model,'decoder':decoder,'queries':int(visible.sum()),
                'gain_over_own_validation_selected_direct':clustered(delta[visible],[g for g,v in zip(groups,visible) if v]),
                'scope':'Same frozen predictions, retrospective endpoint-visible subset only; no refitting or new independent confirmation.'})
        scale_behavior[model]={r['sample_id']:r for f in (root/'qa/commits').glob('*.json') for r in [read(f)]}
    cross=[]
    for model in ('qwen14','glm4'):
        for cohort in ('all','squad_qa','cmrc_qa','hotpot_qa'):
            rr=[r for r in scale_behavior[model].values() if cohort=='all' or r['cohort']==cohort]
            delta=[float(r['answer_F1'])-float(scale_behavior['qwen4'][r['sample_id']]['answer_F1']) for r in rr]
            cross.append({'model':model,'cohort':cohort,'questions':len(rr),'paired_F1_change_vs_qwen4':float(np.mean(delta)),
                'source_cluster':clustered(delta,[r['source_group'] for r in rr]),
                'identical_decoded_answers':sum(r['generated_text']==scale_behavior['qwen4'][r['sample_id']]['generated_text'] for r in rr),
                'scope':'Matchedoriginalquestions, ownnativechat/tokenizer, common32token cap. Modelsize,training,architecture andtokenization not experimentally separated.'})
    compressed(BASE/'scale/combination_visibility_catalog.json.gz',scale_visibility)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'strata':strata,'paired_QA':paired,'paired_natural_first_token':firsttoken,'scale_alignment':alignment,
        'scale_combination_visibility':visibility_counts,'scale_prefix_visible_decoder_paired':visible_paired,
        'scale_decoder_paired':scale_paired,'scale_QA_comparisons':cross,
        'scale_class_scope':'Scale predictortraining is naturaltext only: the reused task_conditioned kernel class variable distinguishesEnglish/Chinese, not QA/natural or knowledge/reasoning/syntax. Main4B classes instead use knownlanguage xnatural/QA.',
        'deployment_sha':sha(BASE/'deployment/result.json'),'own_history_sha':sha(BASE/'own_history/result.json'),
        'scale_result_sha':{m:sha(BASE/'scale'/m/'result.json') for m in ('qwen4','qwen14','glm4')},
        'case_selection':'All deployedcases; no exclusion for nativeerrors, repeatedoutput, earlyEOS or cappedanswers.',
        'limits':['MainQA48tokens and matchedscaleQA32tokens are separate experiments; rawEM must not be pooled acrossdifferentcaps.',
            'Autonomous rollout errors depend on selectedhistory; samehistory companion is diagnostic and not an online feature.',
            'Allfour trainingstates retained, no bestseed selection. Independentformulation/longerdepth generalization remains untested.'],
        'seconds':time.monotonic()-start}
    save(BASE/'deployment/paired_analysis.json',result);ledger('deployment_scale_paired_analysis',result['seconds'])
    print('LAW_DEPLOYMENT_PAIRED_ANALYSIS_COMPLETE',len(strata),len(paired),len(scale_paired),flush=True)


if __name__=='__main__':main()
