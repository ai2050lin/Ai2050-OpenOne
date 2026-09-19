"""Compare error in native coordinate and output probability spaces on frozen tests."""
from rdc_query_common import *


def main():
    out=BASE/'analysis/phase2741.json'
    if out.exists():return
    start=time.monotonic();fit=read(BASE/'rules/fit_result.json');vocab=read(BASE/'rules/vocabulary_result.json')
    transfer=read(BASE/'transfer/fit_result.json');tv=read(BASE/'transfer/vocabulary_result.json');pairs=read(BASE/'pairs/result.json')
    assert all(r['all_passed'] for r in [fit,vocab,transfer,tv,pairs])
    h=fit['selection_gate'];v=next(r for r in vocab['summary'] if r['cohort']=='all' and r['query_split']=='unseen_query')
    gates={'MSE_ordered_vs_uniform_and_shuffle':fit['selection_gate_passed'],
      'KL_ordered_vs_uniform_and_shuffle':all(v['paired_KL_control_minus_ordered'][c]['interval95'][0]>0 for c in ['uniform','shuffled_values'])}
    cross=[]
    for r in transfer['summary']:
        if r['split'] not in ['test','mixed_holdout'] or r['query_split']!='unseen_query':continue
        vv=next(v for v in tv['summary'] if all(v[k]==r[k] for k in ['direction','split','query_split']))
        cross.append({'direction':r['direction'],'split':r['split'],'MSE':r['MSE'],'KL':vv['KL'],
          'MSE_mapping_beats_identity_and_shuffle':all(r['control_minus_query_conditioned'][c]['interval95'][0]>0 for c in ['identity','shuffled_pair']),
          'KL_mapping_beats_identity_and_shuffle':all(vv['control_minus_mapped'][c]['interval95'][0]>0 for c in ['identity','shuffled_pair'])})
    records=read(BASE/'pairs/records.json');near=[r for r in records if r['kind']=='near']
    nearest_example=min(near,key=lambda r:r['current_cosine_distance']) if near else None
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'primary_coordinate_test':h,'primary_vocabulary_test':v,
      'prespecified_candidate_qualification':gates,'cross_expression_unseen_queries':cross,
      'native_reachable_pairs':pairs,'closest_actual_current_state_pair_descriptive_example':nearest_example,
      'common_phenomenon':'The same known suffix can produce different full-coordinate and full-vocabulary responses after different natural prefixes.',
      'candidate_rule':'Declared context-free query prototypes read all native prefix keys/values atblock12; matched per-coordinate decoders predict three later targets.',
      'native_parameter_structure':'Actual Q/K projection and normalization, Q/K-only RoPE, allheads, W_O and nativeMLP12. Source/value shuffle isolates one component, not all possible identity/position confounding.',
      'unseen_composition_prediction':'Heldout source documents and contextual targets of20unseen queries are separate. Program transfer uses paired source responses, a stronger information condition than early-prefix prediction.',
      'training_formation_evidence':'Decoder and mapping fitting are external algorithm training, not native parameter formation. Native continuation training is separately tested in2742; a causal formation theory of these rules is not established.',
      'limitations':['No abstract relation role has been identified solely by predictive success.',
        'The query-only decoder still sees prefixH12; only its additional native feature is query-only. Candidate names do not mean absence of all prefix information.',
        'Equal nominal4coefficients per native coordinate do not guarantee equal effective rank.',
        'All-coordinates retained does not make diagonal regression universally expressive; no new cross-coordinate coupling is learnt by that decoder.',
        'The two primary metrics can disagree; raw coordinate proximity does not entail similar full-vocabulary probabilities.',
        'Multiple subgroup intervals are exploratory and not multiplicity-adjusted. The primary source/unseen-query comparisons were frozen.',
        'Finite-query agreement is not all-future equivalence. Near/far pair endpoints overlap and shared-neighbor dependence remains.',
        'Pair-protocol wording correction: the actual selected current-state distance is ordinary cosine after L2 normalization, NOT mean-centered cosine. The frozen code and pair IDs are unchanged; do not interpret the mistaken adjective as the performed calculation.',
        'EN/ZH/Python are all text; approximate transfer does not imply invertibility, manifold topology, or cross-modal equivalence.'],
      'seconds':time.monotonic()-start}
    save(out,result);ledger('query_prediction_and_transfer_synthesis',result['seconds']);print('QUERY_PHASE2741_ANALYSIS',gates,flush=True)


if __name__=='__main__':main()
