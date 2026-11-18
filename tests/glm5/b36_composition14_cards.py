"""Typed local feature records; describes finished tests without new selection."""
import numpy as np
import b36_composition14 as B
import b36_composition14_analysis as A
import b36_composition14_tail as T


def run():
    B.configure();T.admit()
    rows,_=A.load('discovery');tail=B.read(B.OUT/'confirm-tail-report.json')
    admission=B.read(B.OUT/'confirm-admissibility.json');group=B.group()
    fields={k:A.field('discovery',36,k)[:,group] for k in 'GVA'}
    cards=[]
    for j,u in enumerate(group):
        stats=[]
        for key in sorted({(r['branch'],r['mode'],r['title']) for r in rows}):
            ix=[i for i,r in enumerate(rows) if (r['branch'],r['mode'],r['title'])==key]
            values={k:dict(mean=float(x[ix,j].mean()),sd=float(x[ix,j].std()),min=float(x[ix,j].min()),max=float(x[ix,j].max())) for k,x in fields.items()}
            stats.append(dict(branch=key[0],mode=key[1],title=key[2],n=len(ix),GVA=values))
        exemplars=np.argsort(np.abs(fields['A'][:,j]))[-2:][::-1]
        cards.append(dict(feature_id=f'{B.SNAPSHOT}:B36:down_input:U{u}',method='native_scalar_unit_from_previously_frozen_group',block=36,unit=u,
            boundary='SiLU(gate_proj(post_attention_norm))*up_proj(post_attention_norm), final answer-prompt token',
            discovery_statistics=stats,typical_contexts=[dict(row_id=rows[i]['id'],prompt=rows[i]['prompt'],A=float(fields['A'][i,j])) for i in exemplars],
            typical_context_selection='Two largest absolute discovery A values, descriptive; not evidence of semantic selectivity.',
            unembedding_readout='Conditioned local derivative of final correct-minus-baseline-competitor logit, through B37..B39; no direct static semantic label.',
            output_linearization=[r for r in tail['unit_linearized_attribution'] if r['unit']==u],
            causal_result='Member was included in group interventions; individual intervention not performed.',
            upstream='Actual B36 gate_proj/up_proj and post-attention normalized residual; tensor elements indexed by unit.',
            downstream='Actual B36 down_proj[:,unit], residual h37 and conditional B37..B39 propagation.',
            counterexamples='Failed program composition strata and group-level control failures retained in linked reports.',
            evidence_level='native_parameter_source_and_observation_plus_approximate_local_attribution',
            natural_corpus_statistics='not_collected_in_this_stage',independent_replication=False))
    group_card=dict(feature_id=f'{B.SNAPSHOT}:B36:group32:previous_parameter_selection',indices=group,
        origin=str(B.PREVIOUS/'frozen-selection.json'),origin_sha256=B.file_sha(B.PREVIOUS/'frozen-selection.json'),
        evidence_scope=admission['eligible_strata'],individual_member_necessity='not_tested',shared_semantics='not_established',
        matched_primary=tail['primary_matched'],local_prediction=tail['counterfactual_prediction'],path_clamps=tail['path_clamps'],
        title_controls=tail['title_control'],norm_control_qualification=tail['norm_control_qualification'],
        formation_of_higher_order_features='not_admitted_due_to_program_composition_behavior_failure',
        formal_promotion=False)
    if (B.OUT/'format-tail-report.json').exists():
        fmt=B.read(B.OUT/'format-tail-report.json')
        group_card['answer_code_control']=dict(panel='XYZ, same20 families/branch, title0',primary=fmt['primary_matched'],
            warning='ABC output effect does not establish code-invariant semantics; inspect same-family paired reversals in path-output-report.json.',independent_confirmation=False)
    evidence={n:B.file_sha(B.OUT/n) for n in ['selection.json','confirm-admissibility.json','confirm-tail-report.json','confirm-state-report.json','execution-freeze.json']}
    if (B.OUT/'format-tail-report.json').exists():evidence['format-tail-report.json']=B.file_sha(B.OUT/'format-tail-report.json')
    B.save('feature-cards.json',dict(kind='local_scoped_feature_records',group=group_card,members=cards,evidence=evidence,source_sha256=B.file_sha(__file__)))
    edges=[
        dict(source='external_source_title_and_sequence',target='B36_group32',type='discovery_observation',evidence='feature-cards.json',limitation='No unique semantic assignment.'),
        dict(source='B36_group32',target='h37',type='native_parameter_source',formula='sum_j A36[j]*down_proj36[:,j]',evidence='model-identity.json',limitation='Original BF16 matmul rounding retained.'),
        dict(source='h37',target='B39_roots_GVA',type='heldout_conditional_prediction',evidence='confirm-state-report.json',limitation='Measured h37 input, unequal information budgets; pass/fail per stratum.'),
        dict(source='h37_edit',target='B39_roots_and_h40_edit',type='prospective_local_counterfactual_prediction',formula='J39 J38 J37 delta_h37',evidence='confirm-tail-report.json',limitation='Uses natural downstream states and original parameters; not text-only prediction.'),
        dict(source='B36_group32',target='output',type='scoped_original_model_group_intervention',evidence='confirm-tail-report.json',limitation='Matched/norm controls and finite dose; no member-level necessity.'),
        dict(source='B37_B39_attention_MLP_and_roots',target='output',type='scoped_path_clamp_and_root_restoration',evidence='confirm-tail-report.json',limitation='Not unique mediation, and restoration can introduce its own effect.')]
    B.save('mechanism-graph.json',dict(kind='typed_local_evidence_graph',edges=edges,model_snapshot_id=B.SNAPSHOT,admitted_strata=admission['eligible_strata'],
        failed_claims=['Reliable execution of unobserved composed programs','Shared abstract meaning of all32 members','Emergence of a more complex semantic feature','Text-only atlas prediction'],
        evidence=evidence,layout_is_not_model_geometry=True,source_sha256=B.file_sha(__file__)))
    print('Saved32 member cards, group card and6 typed edges',flush=True)


if __name__=='__main__':run()
