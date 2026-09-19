"""Explicit same-goal assessment and empirical full-stage resource admission."""
from rdc_update_common import *


def main():
    assert (BASE/'long_answers/result.json').exists(),'Finish the current authorized diagnostic before assessing continuation'
    result_paths=['fresh_graph/result.json','middle_training/result.json','scale/qwen4/result.json',
      'scale/qwen14/result.json','scale/glm4/result.json','scale_batch/qwen4/result.json']
    reference_seconds={p:float(read(BASE/p)['seconds']) for p in result_paths}
    areas=['fresh_graph','middle_training','scale','scale_batch']
    reference_bytes={area:sum(p.stat().st_size for p in (BASE/area).rglob('*') if p.is_file()) for area in areas}
    resources=read(BASE/'resources.json');booked=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'))
    used=usage();time_reserve=256;byte_reserve=64*1024**2
    # This is an engineering admission estimate, not a scientific lower bound.
    # A full new stage retains fresh source fields, all four trained increments,
    # frozen controls and all three native-model replication sets.
    seconds_estimate=1.2*sum(reference_seconds.values())
    bytes_estimate=1.2*sum(reference_bytes.values())
    seconds_available=max(0,resources['compute_ceiling_seconds']-booked-time_reserve)
    bytes_available=max(0,resources['result_ceiling_bytes']-used-byte_reserve)
    compute_fit=seconds_estimate<=seconds_available;storage_fit=bytes_estimate<=bytes_available
    result={'timestamp':stamp(),'source':snapshot(__file__),'same_long_term_goal':True,
      'executed_automatic_same_goal_followup':[2739],
      'candidate_stage':'Naturally occurring directed event/role streams with pre-outcome token-identity, position and length controls; prefix-only full-coordinate prediction, four actual32-step training branches, then sequential originalBF16 replication on all three local models.',
      'scientific_reason':'The current source feature has condition-dependent predictive value, but lexical/position identifiability, decoder confounds and ordered predictive sufficiency remain open. New independent natural event materials are needed; repeating the same template would not resolve them.',
      'reference_profile':'Equal-size components actually executed in this campaign:128fresh natural windows with frozen prediction controls; four32-step native training branches;128materials and36own-history generations per native model plus the Q4matched-shape capture. This is a declared full-stage planning reference, not already collected future data.',
      'reference_result_sha256':{p:sha(BASE/p) for p in result_paths},
      'reference_measured_seconds':reference_seconds,'reference_retained_bytes':reference_bytes,
      'planning_multiplier':1.2,'estimated_full_stage_seconds':seconds_estimate,'estimated_full_stage_additional_bytes':bytes_estimate,
      'current_booked_seconds':booked,'current_result_bytes':used,
      'current_compute_ceiling_seconds':resources['compute_ceiling_seconds'],'current_result_ceiling_bytes':resources['result_ceiling_bytes'],
      'remaining_seconds_after_finalization_reserve':seconds_available,'remaining_bytes_after_finalization_reserve':bytes_available,
      'finalization_reserve_seconds':time_reserve,'finalization_reserve_bytes':byte_reserve,
      'compute_admitted':compute_fit,'storage_admitted':storage_fit,'full_next_stage_admitted':compute_fit and storage_fit,
      'budget_entirely_exhausted_claimed':False,
      'limits':'Observed reference plus20%planning margin is not a physical minimum, exact future runtime or proof that no smaller pilot fits. The fresh result timer excludes its separate collection/fitting stages, making the compute reference incomplete; storage alone remains an independent gate. Do not delete currently queryable evidence to force admission or silently open a new unlimited budget.',
      'decision':'Continue only if both full-stage gates pass; otherwise finish the current finite deliverable and retain an explicit unexecuted next-stage protocol.',
      'original_or_user_data_removed':[]}
    save(BASE/'next_stage_admission.json',result)
    print('NEXT_STAGE_ADMISSION',result['full_next_stage_admitted'],'compute',compute_fit,'storage',storage_fit,
      'estimated_seconds',seconds_estimate,'available_seconds',seconds_available,
      'estimated_GiB',bytes_estimate/2**30,'available_GiB',bytes_available/2**30,flush=True)


if __name__=='__main__':main()
