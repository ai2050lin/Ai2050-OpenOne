"""Same-input general/native-constrained query forecasts with available past KV."""
from collections import Counter
from rdc_construction_common import *

OUT=BASE/'phase2746/history_prediction'


def freeze():
    if (OUT/'protocol.json').exists():return read(OUT/'protocol.json'),gzread(OUT/'material.json.gz')
    source=gzread(BASE/'phase2746/runtime/material.json.gz')
    records={r['sample_id']:r for r in gzread(BASE/'phase2746/runtime/records.json.gz')}
    rows=[]
    for row in source:
        original=records[row['sample_id']]
        for step in range(3):
            r={k:row[k] for k in ['sample_id','source_group','family','kind','split','language','original_text']}
            r.update(point_id=row['sample_id']+'_t'+str(step),step=step,
                prompt_ids=row['prompt_ids']+original['generated_ids'][:step],
                field_path=original['field_path'],field_sha256=original['field_sha256'],
                next_native_token_id=original['generated_ids'][step],
                source_scope='Existing exposed discovery material; heldout means excluded from this fit/selection, not previously unobserved language.')
            if row['kind']=='controlled':
                r.update({k:row[k] for k in ['pair_id','world','case','target']})
            rows.append(r)
    assert len(rows)==2688
    compressed(OUT/'material.json.gz',rows)
    protocol={'timestamp':stamp(),'source':snapshot(__file__),'phase':2746,'model':'qwen4',
        'question':'Does explicitly available high-layer past KV support transferable current-query construction, and do native parameter constraints improve over a same-input general map?',
        'source_rows':896,'native_history_points':2688,'steps':[0,1,2],
        'split_points':dict(Counter(r['split'] for r in rows)),
        'available_inputs':'Current original embedding H0, current original H12, every past-token native K/V atblock35, actual current position. Previous tokens have already passed all layers in autoregressive inference. No current H35/H36, current35Q/K/V, emitted current token or gold is a predictor input.',
        'state_boundary':'H12 is after blocks0..11. KV35 excludes the current query token; prefix cache plus strictly earlier appended entries reconstructs past-only KV.',
        'features':'X=[H0,H12,B35(H12;pastKV35,position)] keeps all3times2560native coordinates. Third vector is an available-input native-block candidate, not the true future H36. A matched key/value-source-pair permutation feature uses the same H0/H12 and keys but permuted past values.',
        'shared_inputs':'General H36 map, predicted-H35/native-block map and general-Q35 override all receive exactly the same X and available past KV. Native Q and general Q branches share predicted H35, current K/V derived from it, past K/V, downstream native MLP and finalnorm.',
        'fit_targets':['H35','H36','Q35_before_RoPE_all32heads128components'],
        'prediction_routes':['direct_complete_coordinate_H36','predicted_H35_native_block35',
            'predicted_H35_general_Q35_native_remainder','predicted_H35_general_Q35_native_head_RMS_native_remainder',
            'no_fit_native_block35_on_H12'],
        'controls':['same_X_training_target_correspondence_shuffled_within_family_language_step',
            'same_available_information_source_value_pair_permutation_feature',
            'training_group_mean_target','native_complete_block_with_actual_H35_oracle_for_numerical_floor_only'],
        'ridge':'Weighted full-coordinate linear ridge via complete dual eigensystem; all eigencomponents retained, no top-k/PCA. Per-coordinate feature mean and RMS scale estimated on train only, scale floor1e-8. Targets centered using train weights.',
        'weights':'Equal8families, equal source_group within family, equal rows/steps within group for fitting and validation selection. Report raw natural/controlled and all8family source-group intervals separately.',
        'lambdas':[.001,.01,.1,1.0],
        'selection':'Validation only; select by complete native-postnorm MSE per route. Full vocabulary KL/argmax agreement and semantic-group relation-pair changes are scored separately, not chosen on test. Fit target dimension/effective ridge degrees of freedom are explicit, not all parameter counts are asserted identical.',
        'precision':'Original BF16 scalar parameters represented FP32 for exact same-valued native-block candidate mathematics; cached native BF16 K/V/cos/sin promoted FP32. Report actual-input smooth reconstruction floor separately. Native behavior remains the original BF16 record.',
        'source_shuffle':'Hash-frozen permutation of past values only; no token/answer labels used. This is an algorithm control, not asserted reachable natural model state.',
        'general_Q_RMS':'For nonzero original head gain gamma, normalize qhat/gamma to unit head RMS then restore gamma; exact zero-gain coordinates stay zero. An explicit magnitude control, not a new semantic law.',
        'confirmation':'Freeze chosen rules before excluding every explicitly inventoried discovery/source group for new natural confirmation. New combinations and source/wording exclusions must be enumerated, not inferred from random split.',
        'own_history_followup':'Frozen predictors need their own H0/H12/historyKV branch for autonomous deployment; saved native-history scoring is not self-fed model capability.',
        'scope':'A candidate propagation/readout algorithm with native parameter constraints, not a full original-model rerun as extraction and not an all-future sufficient-state theorem.',
        'material_sha256':sha(OUT/'material.json.gz')}
    immutable(OUT/'protocol.json',protocol);return protocol,rows


if __name__=='__main__':
    p,r=freeze();print('HISTORY_PREDICTION_FROZEN',len(r),p['split_points'],flush=True)
