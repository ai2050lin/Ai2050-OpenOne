"""Full-vocabulary mapping audits and all six single-readout own-history paths."""
import argparse
from collections import defaultdict
from rdc_formation_common import *
from rdc_formation_readout import checked_arrays
from phase2747_rdc_transfer_readout import METRICS, BIASES
from phase2747_rdc_transfer_prepare import NAMES
from phase2747_rdc_program_own_history import BRANCHES


def readout_analysis():
    start = time.monotonic(); folder = OUT/'transfer/analysis'
    if (folder/'result.json').exists(): return
    result = read(OUT/'transfer/readout_result.json'); assert result['all_passed']
    protocol = read(OUT/'transfer/readout_protocol.json'); qi = protocol['answer_query']
    frozen = gzread(OUT/'transfer/records.json.gz')
    rows = {r['direction']+'/'+r['source_group']: r for r in frozen}
    assert len(rows) == 256
    records = []; checks = []; groups = defaultdict(list)
    for path in sorted((OUT/'transfer/readout_records').rglob('*.json')):
        r = read(path); a = checked_arrays(r['readout']); mm = a['metrics']
        assert mm.shape == (5,3,100,12) and a['baseline'].shape == (100,4)
        for index in [8,9,10]:
            assert np.array_equal(mm[:,:, :,index], np.repeat(mm[:,0:1,:,index],3,axis=1))
        assert np.all(mm[:,1,:,6] >= mm[:,0,:,6]-1e-14)
        original = rows[r['direction']+'/'+r['source_group']]
        gold = int(protocol['digit_tokens'][int(original['target_material']['target'])-1])
        for ci, candidate in enumerate(NAMES):
            for bi, bias in enumerate(BIASES):
                v = mm[ci,bi,qi]
                record = {k:r[k] for k in ['direction','source_group','split']}
                record.update(candidate=candidate,bias=bias,query=qi,
                    original_target_argmax_correct=bool(a['baseline'][qi,1] == gold),
                    original_target_gold_NLL=float(a['baseline'][qi,2]),
                    digit_rank_correct=bool(v[10] == gold),
                    **{m:float(v[i]) for i,m in enumerate(METRICS)})
                records.append(record)
                groups[(r['direction'],r['split'],candidate,bias)].append(record)
        checks.append({'direction':r['direction'],'source_group':r['source_group'],
            'all100_queries_conditional_digit_probability_margin_rank_invariant':True,
            'all500_paths_digit_mass_nondecreasing_under_digit_bias':True,
            'old_B16_vs_new_B1_argmax_mismatches':int(a['old_batch_vs_new_B1'][:,1].sum()),
            'old_B16_vs_new_B1_entropy_max_abs':float(abs(a['old_batch_vs_new_B1'][:,0]).max())})
    assert len(checks) == 256 and len(records) == 3840
    summary = []
    for key, rr in sorted(groups.items()):
        entry = dict(zip(['direction','split','candidate','bias'],key))
        entry.update(groups=len(rr),metrics={metric:clustered([float(v[metric]) for v in rr],[v['source_group'] for v in rr])
            for metric in ['original_target_argmax_correct','original_target_gold_NLL','digit_rank_correct',
                'complete_vocabulary_argmax_gold_correct','gold_digit_NLL','digit_mass','KL_target_to_prediction']})
        summary.append(entry)
    compressed(folder/'fixed_answer_records.json.gz',records)
    value = {'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'checks':checks,'summary':summary,'readout_result_sha256':sha(OUT/'transfer/readout_result.json'),
        'scope':'Previously frozen groups/mappings, actual observed source response. Fixed Answer query is answer-oriented; other99queries are response-prediction diagnostics, not99additional program questions.',
        'limits':['Uniform class-bias ranking invariance is algebraic, not evidence of a semantic module.',
            'Native baseline here reads stored B16 postnorm with B1 head; new B1 target-prefill own history is separately measured.',
            'Uncertainty unit is semantic group; frozen-fit and query-selection uncertainty are not included.'],
        'seconds':time.monotonic()-start}
    save(folder/'result.json',value);ledger('phase2747_program_readout_analysis',value['seconds'])
    print('FORMATION_PROGRAM_READOUT_ANALYSIS',len(checks),flush=True)


def own_analysis():
    start=time.monotonic();folder=OUT/'program_own_history/analysis'
    if (folder/'result.json').exists(): return
    run=read(OUT/'program_own_history/result.json');assert run['all_passed'] and run['trajectories']==192
    protocol=read(OUT/'program_own_history/protocol.json')
    targets={r['sample_id']:r for r in gzread(OUT/'followup/material.json.gz')['program_own_targets']}
    records={};checks=[];scope=[]
    transfer={r['source_group']:r for r in gzread(OUT/'transfer/records.json.gz') if r['direction']=='python_to_en'}
    for branch in BRANCHES:
        rr=[]
        for sid,row in targets.items():
            r=read(OUT/'program_own_history/records'/branch/(sid+'.json'));a=checked_arrays(r['field'])
            n=len(r['generated_ids']);assert a['generated_ids'].tolist()==r['generated_ids']
            assert a['native_postnorm_BF16'].shape==(n,2560) and a['first_final_all_hidden_BF16'].shape==(2,37,2560)
            assert a['statistics'].shape==(n,2)
            audit=r['cache_audit'];assert audit['readout_does_not_mutate_cache'] and audit['same_prefill_and_query_as_native']
            assert not audit['correct_answer_enters_mapping_or_bias'] and not audit['original_parameters_changed']
            assert r['answer_scoring']['scope'].startswith('Prospectively frozen')
            native=r if branch=='native' else records['native'][sid]
            if branch!='native':
                old=checked_arrays(native['field'])
                assert np.array_equal(a['first_final_all_hidden_BF16'][0],old['first_final_all_hidden_BF16'][0])
                assert np.array_equal(a['native_postnorm_BF16'][0],old['native_postnorm_BF16'][0])
                assert audit['original_query_cache_identity']==native['cache_audit']['original_query_cache_identity']
                source=transfer[r['source_group']];name=rank('python_to_en/'+r['source_group'])[:24]
                rd=read(OUT/'transfer/readout_records/python_to_en'/(name+'.json'))
                rm=checked_arrays(rd['readout'])['metrics']
                ci=0 if branch=='code_identity' else (4 if branch=='shuffled_map' else 3)
                bi={'mapped_digit_bias':1,'mapped_letter_bias':2}.get(branch,0)
                assert int(rm[ci,bi,protocol['query_index'],3])==r['generated_ids'][0]
            divergence=None if branch=='native' else next((i for i in range(max(n,len(native['generated_ids'])))
                if i>=n or i>=len(native['generated_ids']) or r['generated_ids'][i]!=native['generated_ids'][i]),None)
            assert divergence==r['first_divergence_from_native']
            if branch!='native' and r['generated_ids'][0]==native['generated_ids'][0]:
                assert divergence is None
                assert np.array_equal(a['native_postnorm_BF16'],old['native_postnorm_BF16'])
                assert np.array_equal(a['first_final_all_hidden_BF16'],old['first_final_all_hidden_BF16'])
                assert np.array_equal(a['statistics'][1:],old['statistics'][1:])
            rr.append({**r,'first_digit_correct':r['generated_ids'][0]==protocol['digit_ids'][int(r['target'])-1],
                'correct_change':int(r['answer_scoring']['parsed_and_stopped_correct'])-int(native['answer_scoring']['parsed_and_stopped_correct']),
                'EOS_change':int(r['answer_scoring']['EOS'])-int(native['answer_scoring']['EOS'])})
            checks.append({'branch':branch,'sample_id':sid,'all_array_shapes_and_ids_exact':True,
                'untouched_initial_native_hidden_and_KV':True,'mapped_first_argmax_matches_fullV_readout':branch!='native'})
        records[branch]={r['sample_id']:r for r in rr}
        for stratum in ['all']+sorted({str(r['depth']) for r in rr}):
            selected=rr if stratum=='all' else [r for r in rr if str(r['depth'])==stratum]
            scope.append({'branch':branch,'depth':stratum,'groups':len(selected),
                'correct_and_stopped':sum(r['answer_scoring']['parsed_and_stopped_correct'] for r in selected),
                'EOS':sum(r['answer_scoring']['EOS'] for r in selected),
                'censored':sum(r['answer_scoring']['censored'] for r in selected),
                'first_digit_correct':sum(r['first_digit_correct'] for r in selected),
                'same_complete_tokens_as_native':sum(r['first_divergence_from_native'] is None for r in selected),
                'paired_correct_change':clustered([r['correct_change'] for r in selected],[r['source_group'] for r in selected]),
                'paired_EOS_change':clustered([r['EOS_change'] for r in selected],[r['source_group'] for r in selected])})
    assert len(checks)==192
    compressed(folder/'audited_records.json.gz',[r for rr in records.values() for r in rr.values()])
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,'summary':scope,
        'pilot_replay_count':len(run['pilot_replay_checks']),'actual_generated_tokens':run['actual_generated_tokens'],
        'scope':'Single readout replacement; native initial computation and KV unchanged; all later tokens self-fed. Correctness/format/stopping remain separate.',
        'limits':['32 frozen heldout semantic groups only; six branches are paired interventions, not192independent examples.',
            'Observed Python response is additional information; not autonomous early-state extraction.',
            'First literal-digit correctness does not grade an answer that starts with whitespace or Markdown. The frozen complete-answer parser is separate.',
            'If the one-shot replacement preserves the first chosen token, unchanged native KV and parameters imply the same later deterministic computation; all such cases are bit-checked, not counted as distinct mechanisms.',
            'After first differing generated token, branches have different text histories.',
            'All failed mapping branches and cap-censored trajectories remain included.'],
        'seconds':time.monotonic()-start}
    save(folder/'result.json',value);ledger('phase2747_program_own_analysis',value['seconds'])
    print('FORMATION_PROGRAM_OWN_ANALYSIS',len(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['readout','own']);args=parser.parse_args()
    readout_analysis() if args.mode=='readout' else own_analysis()
