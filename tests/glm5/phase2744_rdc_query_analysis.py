"""Report identifiability, actual parameter deployment, and calibration separately."""
from phase2744_rdc_query_identifiability import *


def calibration(material):
    rows=material['natural'];ci=[i for i,r in enumerate(rows) if r['split']=='calibration'];ti=[i for i,r in enumerate(rows) if r['split']=='prospective_natural']
    grids={};observations={};selections={};losses={}
    groups=sorted({rows[i]['source_group'] for i in ci})
    for variant in VARIANTS:
        root=OUT/'calibration'/variant
        with np.load(root/'all_fields.npz') as z:g=z['temperature_prior_mixture_NLL'].copy()
        grids[variant]=g;observations[variant]=gzread(root/'observations.json.gz')
        assert [r['sample_id'] for r in observations[variant]]==[r['sample_id'] for r in rows]
        objective=np.mean([g[[i for i in ci if rows[i]['source_group']==group]].mean(0) for group in groups],0)
        idx=np.unravel_index(objective.argmin(),objective.shape);it=int(objective[:,0].argmin());ia=int(objective[2].argmin())
        selections[variant]={'joint_temperature':TEMPERATURES[idx[0]],'joint_prior_mixture':MIXTURES[idx[1]],'joint_index':list(map(int,idx)),
          'temperature_only':TEMPERATURES[it],'prior_only_alpha':MIXTURES[ia],'calibration_documents':len(groups),'selection_objective':'Equal document-cluster mean calibration NLL'}
        losses[variant]={'raw':g[:,2,0],'joint_calibrated':g[:,idx[0],idx[1]],'temperature_only':g[:,it,0],'prior_only':g[:,2,ia]}
    reports=[]
    for cohort in ['all','gum','ewt','cmrc']:
        ix=[i for i in ti if cohort=='all' or rows[i]['cohort']==cohort];gs=[rows[i]['source_group'] for i in ix]
        for variant in VARIANTS:
            item={'variant':variant,'cohort':cohort,'positions':len(ix),'documents':len(set(gs)),
              'raw_NLL':clustered(losses[variant]['raw'][ix],gs),'joint_calibrated_NLL':clustered(losses[variant]['joint_calibrated'][ix],gs),
              'raw_minus_original_raw':clustered((losses[variant]['raw']-losses['native']['raw'])[ix],gs),
              'joint_minus_original_joint':clustered((losses[variant]['joint_calibrated']-losses['native']['joint_calibrated'])[ix],gs),
              'original_temperature_only_minus_original_raw':clustered((losses['native']['temperature_only']-losses['native']['raw'])[ix],gs),
              'original_prior_only_minus_original_raw':clustered((losses['native']['prior_only']-losses['native']['raw'])[ix],gs),
              'original_joint_minus_original_raw':clustered((losses['native']['joint_calibrated']-losses['native']['raw'])[ix],gs),
              'raw_entropy':clustered([observations[variant][i]['entropy'] for i in ix],gs),
              'raw_argmax_accuracy':float(np.mean([observations[variant][i]['argmax_correct'] for i in ix]))}
            reports.append(item)
    paired=[]
    for seed in [2742,2743]:
      for kind in ['raw','joint_calibrated']:
        a=losses[f'natural_target_{seed}'][kind];b=losses[f'within_cohort_permuted_target_{seed}'][kind]
        paired.append({'seed':seed,'loss':kind,'natural_minus_permuted':clustered((a-b)[ti],[rows[i]['source_group'] for i in ti])})
    npz(OUT/'analysis/calibrated_natural_losses.npz',**{v+'__'+k:a for v,entry in losses.items() for k,a in entry.items()})
    return {'selections':selections,'prospective_reports':reports,'paired_training_target_comparisons':paired,
      'scope':'Calibration selected only on separate validation documents. Newly evaluated content positions come from96reserved documents; same documents were used for another query task in2743. Calibrators are probability-scoring alternatives, not new native parameters and not deployed during reported free generation.'}


def relations(material):
    rows=material['controlled'];lookup={r['sample_id']:r for r in rows};reports=[];pairs=[];rules=[];coordinate=defaultdict(list)
    qs=read(BASE/'probes/protocol.json')['probes'];unseen=[i for i,q in enumerate(qs) if q['split']=='unseen_query']
    names=['query_only','uniform','quadratic','shuffled_values','ordered_softmax'];native={};allrecords={}
    for variant in VARIANTS:
        rr={r['sample_id']:r for p in (OUT/'relations'/variant/'commits').glob('*.json') if (r:=read(p))};assert len(rr)==320;allrecords[variant]=rr
        if variant=='native':native=rr
        for pair in material['pairs']:
            a,b=[rr[sid] for sid in pair['sample_ids']];meta=lookup[a['sample_id']];orientation=1 if meta['truth'] else -1
            yes=(a['actual_current']['binary_conditional_yes_probability']-b['actual_current']['binary_conditional_yes_probability'])*orientation
            with np.load(OUT/'relations'/variant/'fields'/f"{a['sample_id']}.npz") as z:ha=unbits(z['matched_subset_postnorm']).astype(float)
            with np.load(OUT/'relations'/variant/'fields'/f"{b['sample_id']}.npz") as z:hb=unbits(z['matched_subset_postnorm']).astype(float)
            pairs.append({'variant':variant,'pair_id':pair['pair_id'],'source_group':pair['source_group'],'family':meta['family'],'language':meta['language'],
              'answer_aligned_yes_probability_separation':yes,'all_matched_query_coordinate_displacement_MSE':float(np.mean((ha-hb)**2)),
              'both_first_argmax_answers_correct':bool(a['actual_current']['first_argmax_correct'] and b['actual_current']['first_argmax_correct'])})
        for family in ['all']+FAMILIES:
            pr=[r for r in pairs if r['variant']==variant and (family=='all' or r['family']==family)];gs=[r['source_group'] for r in pr]
            reports.append({'variant':variant,'family':family,'token_matched_pairs':len(pr),'groups':len(set(gs)),
              'answer_aligned_yes_probability_separation':clustered([r['answer_aligned_yes_probability_separation'] for r in pr],gs),
              'mean_query_displacement_MSE':clustered([r['all_matched_query_coordinate_displacement_MSE'] for r in pr],gs),
              'both_first_argmax_answers_correct':sum(r['both_first_argmax_answers_correct'] for r in pr)})
    for r in rows:
        with np.load(OUT/'relations/native/fields'/f"{r['sample_id']}.npz") as z:
            mse=z['frozen_rule_all_query_MSE'][:,unseen].mean(1);kl=z['frozen_rule_all_query_KL'][:,unseen].mean(1)
        rules.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'family':r['family'],'MSE':mse.tolist(),'KL':kl.tolist()})
    rule_summary=[]
    for family in ['all']+FAMILIES:
        rr=[r for r in rules if family=='all' or r['family']==family];gs=[r['source_group'] for r in rr]
        item={'family':family,'expressions':len(rr),'MSE':{n:clustered([r['MSE'][c] for r in rr],gs) for c,n in enumerate(names)},
          'KL':{n:clustered([r['KL'][c] for r in rr],gs) for c,n in enumerate(names)},
          'control_minus_ordered_MSE':{names[c]:clustered([r['MSE'][c]-r['MSE'][4] for r in rr],gs) for c in range(4)},
          'control_minus_ordered_KL':{names[c]:clustered([r['KL'][c]-r['KL'][4] for r in rr],gs) for c in range(4)}}
        rule_summary.append(item)
    # Every actual gate/up/activation unit is compared; no data-selected unit subset.
    for variant in VARIANTS[1:]:
      for family in FAMILIES:
        selected=[r for r in rows if r['family']==family];means={}
        names=[f'L{block}_{field}' for block in [16,35] for field in ['gate_proj','up_proj','activation']]
        acc={name:np.zeros(9728) for name in names}
        for row in selected:
            with np.load(OUT/'relations'/variant/'fields'/f"{row['sample_id']}.npz") as za,np.load(OUT/'relations/native/fields'/f"{row['sample_id']}.npz") as zb:
                for name in names:acc[name]+=(unbits(za[name]).astype(float)-unbits(zb[name]).astype(float))**2
        for name in names:means[variant+'__'+family+'__'+name+'__all_unit_MSE']=acc[name]/len(selected)
        coordinate.update({k:v for k,v in means.items()})
    npz(OUT/'analysis/all_native_unit_training_changes.npz',**coordinate);compressed(OUT/'analysis/relational_pairs.json.gz',pairs)
    compressed(OUT/'analysis/all_frozen_rule_metrics.json.gz',rules)
    return {'matched_pair_reports':reports,'frozen_rule_unseen_query_reports':rule_summary,
      'frozen_rule_relation_change_metrics':gzread(OUT/'relations/frozen_relation_change_metrics.json.gz'),
      'scope':'Exact full-token histogram and fixed question isolate order/assignment from a bag-only explanation. They do not exclude token-position heuristics, all lexical mechanisms or all sequence-sensitive shallow alternatives. Controlled recipe semantics remain external labels, not named internal neurons.'}


def behavior(material):
    rows={r['sample_id']:r for r in material['controlled']};groups=defaultdict(dict);detail=[];reports=[]
    for variant in VARIANTS:
        groups[variant]={r['sample_id']:r for p in (OUT/'behavior'/variant/'commits').glob('*.json') if (r:=read(p))};assert len(groups[variant])==320
    for variant,rr in groups.items():
      for family in ['all']+FAMILIES:
        selected=[r for r in rr.values() if family=='all' or r['family']==family];gs=[r['source_group'] for r in selected]
        actual=[r['answer_scoring'] for r in selected];native=[groups['native'][r['sample_id']] for r in selected]
        pair_ids=sorted({r['pair_id'] for r in selected});both=[]
        for pair_id in pair_ids:
            pair=[r for r in selected if r['pair_id']==pair_id];assert len(pair)==2
            both.append(int(all(r['answer_scoring']['parsed_and_stopped_correct'] for r in pair)))
        reports.append({'variant':variant,'family':family,'expressions':len(selected),'groups':len(set(gs)),
          'correct_and_stopped':sum(r['parsed_and_stopped_correct'] for r in actual),'parsed':sum(r['conservative_final_answer'] is not None for r in actual),
          'unparsed_EOS':sum(r['EOS'] and r['conservative_final_answer'] is None for r in actual),'censored':sum(r['censored'] for r in actual),
          'parsed_wrong':sum(r['conservative_final_answer'] is not None and not r['conservative_final_correct'] for r in actual),
          'paired_token_matched_both_correct':sum(both),'pair_count':len(pair_ids),
          'scoring_wrapper_only_gain':sum(int(r['parsed_and_stopped_correct'])-int(r['prior_identity_grammar']['parsed_and_stopped_correct']) for r in actual),
          'mean_tokens':float(np.mean([len(r['generated_ids']) for r in selected])),
          'paired_success_change_vs_native':clustered([int(r['answer_scoring']['parsed_and_stopped_correct'])-int(n['answer_scoring']['parsed_and_stopped_correct']) for r,n in zip(selected,native)],gs),
          'paired_token_change_vs_native':clustered([len(r['generated_ids'])-len(n['generated_ids']) for r,n in zip(selected,native)],gs),
          'first_token_B1_B8_disagreements':sum(r['initial_shape_control']['B8_initial_token_id']!=r['initial_shape_control']['B1_initial_token_id'] for r in selected)})
    # The first case in every family is predeclared; no success/failure cherry-pick.
    examples=[]
    for family in FAMILIES:
        selected=[r for r in material['controlled'] if r['family']==family and r['case']==0 and r['language']=='en']
        examples.append({'family':family,'material':selected,'actual_outputs':{v:[groups[v][r['sample_id']] for r in selected] for v in VARIANTS}})
    return {'summary':reports,'predeclared_real_examples':examples,'scope':'Conservative explicit terminal yes/no and EOS; no post-outcome scoring extension and no complete reasoning-chain grading. Calibrators are not used to generate these responses.'}


def main():
    out=OUT/'analysis/result.json'
    if out.exists():return
    start=time.monotonic();version=snapshot(__file__);protocol,material=freeze()
    for stage in ['relations','calibration','behavior']:assert read(OUT/stage/'result.json')['all_passed']
    assert read(OUT/'preflight.json')['all_passed']
    result={'timestamp':stamp(),'source':version,'all_passed':True,'phase':2744,'same_goal_automatically_executed':True,
      'calibration':calibration(material),'relations':relations(material),'behavior':behavior(material),
      'common_phenomenon':'Exact-word-bag pairs differ only in ordering/assignment while target relations reverse; natural label training and shuffled-label training can both change output confidence.',
      'candidate_rules':'Frozen prefix-query rule, answer-aligned pair sensitivity, and held-validation scalar temperature/prior-mixture controls; separate information conditions and tests.',
      'native_parameter_structure':'Four actual originalblock16 fullgate/up/down deltas deployed BF16 with originalformation-loss reconstruction; all9728units atblocks16/35 and fullselected-layer coordinates.',
      'unseen_composition_prediction':'New paired relation combinations,20unseenqueries for unchanged predictors, and newly evaluated natural content positions on96reserved documents. Vocabulary/contextingredients have known historical exposure.',
      'training_formation':'Actual2742continuedlearning transferred to these new observations; no new training and no historicalpretraining reconstruction. Training-target dependence and calibration are measured as competing explanations.',
      'additional_identification_limit':'The four learned checkpoints also differ in cumulativeFP32/BF16parameter displacement despite equal per-step norms. Scalar calibration does not replace a norm-matched parameter-direction control; residual differences are not uniquely attributable to semantic labels.',
      'seconds':time.monotonic()-start}
    save(out,result);ledger('identity_calibration_relation_synthesis',result['seconds']);print('IDENTITY_ANALYSIS_DONE',result['seconds'],flush=True)


if __name__=='__main__':main()
