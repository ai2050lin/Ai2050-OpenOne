"""Full-coordinate descriptive synthesis; no linguistic causality from bright plots."""
import argparse
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'analysis'


def main(refresh=False):
    path=OUT/'phase2740.json'
    if path.exists() and not refresh:return
    previous_report=None
    if path.exists():
        previous_report=read(path);archive=OUT/'provenance_history'/('phase2740_'+sha(path)+'.json');save(archive,previous_report)
    start=time.monotonic();source_at_start=snapshot(__file__)
    atlas=read(BASE/'atlas/result.json');events=read(BASE/'events/result.json')
    assert atlas['all_passed'] and events['all_passed']
    probes=read(BASE/'probes/protocol.json')['probes'];paths=read(BASE/'events/paths_index.json')
    trajectories=gzread(BASE/'events/material.json.gz')['trajectories'];pair_records=[];time_records=[];coordinate_arrays={}
    for item in paths:
      with np.load(BASE/item['path']) as z:
        for block in [16,35]:
            g=z[f'L{block}_source_gate_read'].astype(float);u=z[f'L{block}_source_up_read'].astype(float)
            sig=z[f'L{block}_sigmoid_gate'].astype(float);G=(g*g).sum(0);U=(u*u).sum(0);C=(g*u).sum(0)
            # Exact all-pair energy through full source factors, without materializing S*S*9728.
            total=sig*sig*G*U;anti=.5*sig*sig*np.maximum(G*U-C*C,0)
            direct=sig*g.sum(0)*u.sum(0);native=z[f'L{block}_activation'].astype(float)
            rem=z[f'L{block}_activation_rounding_remainder'].astype(float)
            pair=z[f'L{block}_ordered_pair_all_unit_sum'].astype(float)
            error=float(np.linalg.norm(direct+rem-native)/max(np.linalg.norm(native),1e-12))
            assert error<1e-5,(item['path'],block,error)
            key=Path(item['path']).stem+f'__L{block}'
            coordinate_arrays[key+'__all_unit_pair_energy']=total
            coordinate_arrays[key+'__all_unit_antisymmetric_energy']=anti
            pair_records.append({'path':item['path'],'sample_id':item['sample_id'],'mode':item['mode'],'block':block,
              'visible_sources':len(g)-1,'MLP_units':len(sig),'antisymmetric_all_unit_energy_fraction':float(anti.sum()/max(total.sum(),1e-12)),
              'all_unit_pair_energy':float(total.sum()),'activation_reconstruction_with_remainder_relative_error':error,
              'aggregated_pair_matrix_direction_fraction':float(np.sum(((pair-pair.T)/2)**2)/max(np.sum(pair*pair),1e-12)),
              'complete_double_sum_antisymmetry_relative_residual':float(abs(np.sum(pair-pair.T))/max(np.sum(abs(pair)),1e-12))})
    for item in trajectories:
        sid=item['row']['sample_id'];commit=read(BASE/'events/commits'/f'{sid}.json')
        with np.load(BASE/'events/fields'/f'{sid}.npz') as z:
            h=unbits(z['H']).astype(float);q=unbits(z['dynamic_query_postnorm']).astype(float);st=z['dynamic_full_vocabulary_statistics'];steps=z['steps']
        dh=np.diff(h,axis=0);dq=np.diff(q,axis=0)
        coordinate_arrays[sid+'__all_layer_native_coordinate_displacement_MSE']=(dh*dh).mean(0)
        coordinate_arrays[sid+'__all_query_native_coordinate_displacement_MSE']=(dq*dq).mean(0)
        for j in range(1,len(steps)):
            previous=int(steps[j-1]);now=int(steps[j]);in_interval=[e for e in item['events'] if previous<e['emitted_token_step']<=now]
            time_records.append({'sample_id':sid,'source_group':item['row']['source_group'],'representation':item['row']['representation'],
              'previous_step':previous,'step':now,'emitted_tokens_between':now-previous,'output_text_event_count':len(in_interval),
              'has_explicit_variable_annotation':any(e['type']=='explicit_variable_value' for e in in_interval),
              'has_terminal_marker_annotation':any(e['type']=='terminal_marker' for e in in_interval),
              'per_layer_all_coordinate_MSE':np.mean(dh[j-1]**2,-1).tolist(),
              'per_query_all_coordinate_MSE':np.mean(dq[j-1]**2,-1).tolist(),
              'per_query_KL_to_standalone_change':(st[j,:,1]-st[j-1,:,1]).tolist(),
              'native_current_entropy_before':commit['steps'][previous]['entropy'],'native_current_entropy_after':commit['steps'][now]['entropy']})
    npz(OUT/'phase2740_all_coordinate_analysis.npz',**coordinate_arrays)
    compressed(OUT/'phase2740_generation_intervals.json.gz',time_records)
    summaries=[]
    for mode in sorted({r['mode'] for r in pair_records}):
      for block in [16,35]:
        rr=[r for r in pair_records if r['mode']==mode and r['block']==block]
        summaries.append({'mode':mode,'block':block,'path_records':len(rr),
          'unit_resolved_antisymmetric_energy_fraction_min':min(r['antisymmetric_all_unit_energy_fraction'] for r in rr),
          'unit_resolved_antisymmetric_energy_fraction_max':max(r['antisymmetric_all_unit_energy_fraction'] for r in rr),
          'max_native_activation_reconstruction_error':max(r['activation_reconstruction_with_remainder_relative_error'] for r in rr)})
    example_row=trajectories[0];sid=example_row['row']['sample_id'];native=read(ROOT/example_row['native_record'])
    result={'timestamp':stamp(),'source':source_at_start,'all_passed':True,'natural_prefixes':atlas['natural_prefixes'],
      'query_endpoints':atlas['query_endpoints'],'source_documents':atlas['source_documents'],'native_path_block_records':len(pair_records),
      'directed_path_records':pair_records,'directed_path_summary':summaries,'generation_intervals':len(time_records),
      'real_example':{'sample_id':sid,'prompt':example_row['row'].get('prompt',example_row['row'].get('text')),
        'generated_text':native['generated_text'],'events':example_row['events'],'anchors':example_row['anchors']},
      'supplemental_attachment_corrections':[
        {'claim':'Language families are proved to be pretrained fixed query templates and pure semantic structure cannot exist',
         'status':'unsupported_universal_generalization',
         'correction':'Conditional query responses are a useful experimental object. Current finite observations and restricted continuation training establish neither the unique historical mechanism nor the nonexistence of every more abstract semantic correspondence.'},
        {'claim':'Arbitrary-length logic prediction and universal hallucination repair can serve as this finite campaign completion criterion',
         'status':'replace_with_bounded_verifiable_tests',
         'correction':'Report fixed query coverage, source/query holdouts, declared token caps, explicit failure cases and actual output/cache effects. These finite tests cannot verify a universal arbitrary-length claim.'}],
      'meaning':['All-native-coordinate displacement describes response changes, not a semantic distance invariant.',
        'All source pairs retain branch identity. Their antisymmetric energy is a factor-level statistic, not a percentage of semantic direction.',
        'The source allocation conditions on already observed attention weights, gate sigmoid and RMS denominator. It does not separately allocate the causal effects of changing source keys, query construction or normalization.',
        'Source aggregation over both ordered indices cancels antisymmetric contributions; downstream benefit must be tested separately.',
        'Output-text event windows differ in token length and content; associations are descriptive and cannot establish an internal symbolic executor.',
        'The original32trajectories represent8semantic groups. Their exact replay adds numerical provenance, not32new independent behavior confirmations.'],
      'seconds':time.monotonic()-start}
    if previous_report is not None:
        compared=['natural_prefixes','query_endpoints','source_documents','native_path_block_records','directed_path_records','directed_path_summary','generation_intervals','real_example','meaning']
        assert all(result[k]==previous_report[k] for k in compared)
        result['provenance_refresh']={'previous_result_archive':str(archive.relative_to(BASE)),
          'reason':'A metadata-only supplemental-claim edit overlapped the first analysis process startup. That run snapshotted source at completion, so its source label did not accurately identify the pre-edit executed version. Preserve the original report and recompute with source frozen at startup.',
          'all_previous_numeric_and_descriptive_results_exact':True,'compared_fields':compared,
          'new_content':'Two supplemental attachment boundary statements; no new sample, no changed observation and no replaced hypothesis test.'}
    save(path,result);ledger('full_coordinate_directed_event_analysis',result['seconds']);print('QUERY_PHASE2740_ANALYSIS',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true');arg=p.parse_args()
    if arg.refresh and not (OUT/'provenance_history/refresh_failure_001.json').exists():
        receipt={'timestamp':stamp(),'error':'TypeError: int object is not subscriptable; previous report variable reused as a generation-step integer in the first metadata refresh assertion.',
          'observed_process_wall_seconds':9.889527900000001,'timing_source':'Actual exec_command completion reported in the task conversation; not a reconstructed internal training timer.',
          'scope':'CPU descriptive analysis refresh only; no native model execution or original data modified. The old report was preserved and not replaced.'}
        save(OUT/'provenance_history/refresh_failure_001.json',receipt);ledger('failed_analysis_metadata_refresh',receipt['observed_process_wall_seconds'])
    process_start=time.monotonic()
    try:main(arg.refresh)
    except Exception as exc:failure(OUT/'provenance_history',process_start,exc);raise
