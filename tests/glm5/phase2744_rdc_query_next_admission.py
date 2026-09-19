"""Measured whole-stage continuation assessment after actual identification work."""
from phase2744_rdc_query_identifiability import *


def main():
    assert read(OUT/'queue/status.json')['all_passed'];assert read(OUT/'analysis/result.json')['all_passed']
    material=gzread(OUT/'material.json.gz');n=len(material['controlled']);source_ids=read(BASE/'scale/protocol.json')['source_ids'][:3]
    costs=[]
    for model in ['qwen4','qwen14','glm4']:
        records=[read(BASE/'scale'/model/'commits'/f'{sid}.json') for sid in source_ids]
        files=[BASE/'scale'/model/'fields'/f'{sid}.npz' for sid in source_ids]
        costs.append({'model':model,'reference_sources':source_ids,'observed_mean_seconds_with_full_Q_and_attention':float(np.mean([r['seconds'] for r in records])),
          'observed_mean_bytes_with_full_Q_and_attention':float(np.mean([p.stat().st_size for p in files]))})
    required_seconds=1.2*n*sum(r['observed_mean_seconds_with_full_Q_and_attention'] for r in costs)
    required_bytes=int(1.2*n*sum(r['observed_mean_bytes_with_full_Q_and_attention'] for r in costs))
    resources=read(BASE/'resources.json');used=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'));size=usage()
    remain_seconds=resources['compute_ceiling_seconds']-used-200;remain_bytes=resources['result_ceiling_bytes']-size-32*1024**2
    time_gate=required_seconds<remain_seconds;storage_gate=required_bytes<remain_bytes
    result={'timestamp':stamp(),'source':snapshot(__file__),'same_authorized_goal':True,'automatically_executed_same_goal_phases':[2743,2744],
      'proposed_phase':2745,'scientific_question':'Which context-dependent early-layer query-construction relationships survive strict token-matched role changes, output-calibration controls and both larger native architectures?',
      'whole_stage_scope':'All320strictpaired expressions,100frozenqueries,allnativeQcoordinates and all-source attention for each model,with full laterresponse and available-prefix prediction comparison; native token-histogram matching rechecked pertokenizer. Add cumulative-parameter-displacement-matched direction controls for actualQ4natural/permuted training on natural content and paired relation decisions. All three model jobs serial, noquantization.',
      'required_information':'The existing scale test retained fullQ/attention for only3sources permodel. This proposal extends that missing query-construction object across the complete identity-controlled panel, not merely a larger-model headline score.',
      'cost_reference':costs,'controlled_material_rows':n,'reference_estimated_capture_seconds':required_seconds,'reference_estimated_capture_bytes':required_bytes,
      'remaining_seconds_after_audit_reserve':remain_seconds,'remaining_result_bytes_after_delivery_reserve':remain_bytes,
      'time_gate':time_gate,'storage_gate':storage_gate,'complete_stage_admitted':time_gate and storage_gate,
      'estimate_scope':'Measured native100query/Q/attention source costs scaled to declared320panel and20%margin. Not a physical lower bound or exact forecast; length/architecture and detailed further algorithm costs can differ. No claim that every small pilot is impossible.',
      'retention_boundary':'Current fields remain live-client-queryable and needed evidence. No original checkpoint or retained field is deleted to force another stage inside a fixed budget.',
      'resource_scope':'Configured12GiB result ceiling and21600recorded-script-second ceiling, not total disk capacity or total wall-clock. A failed whole-stage gate is not proof that language mechanism research is solved.'}
    save(BASE/'continuation_after_2744.json',result);print('AFTER_IDENTITY_ADMISSION',result,flush=True)


if __name__=='__main__':main()
