"""Independent CPU audit of completed scalar records; no model or phase finish."""
import sys, itertools, math, struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from phase2620_native_coordinate_contract import read,save,sha,RESULT

OUT=RESULT/'phase2689_native_qkv_scalar'

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    # Finite float32, round-to-nearest ties-to-even to BF16, independent of torch.
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',bits))[0]

def main():
    cfg=read(OUT/'protocol/frozen.json')
    before_hashes=read(OUT/'protocol/matrix_hashes_before.json')
    after_hashes=read(OUT/'analysis/matrix_hashes_after.json')
    assert len(before_hashes)==24 and before_hashes==after_hashes
    material=RESULT/'phase2686_independent_role_contract/material/initial.json'
    assert sha(material)==cfg['material_sha256']
    byid={r['case_id']:r for r in read(material)}
    originals={r['case_index']:r for r in read(RESULT/'phase2687_role_qkv_field/analysis/records.json')}
    unique={};groups={};natural_groups={};report_hashes={}
    count=0;changes=0;zeros=0;by_function={};argmax_records=[]
    for ii,cid in enumerate(cfg['case_ids']):
        path=OUT/f'analysis/case_{ii:03d}.json';rec=read(path);report_hashes[path.name]=sha(path)
        row=byid[cid];n=len(row['prompt_ids']);base=originals[row['case_index']]
        assert rec['case_id']==cid and rec['case_index']==row['case_index']
        assert rec['native_noop_exact'] and rec['restored_exact']
        assert rec['baseline_generated_ids']==base['generated_ids']
        expected=list(itertools.product(range(48),cfg['doses'],cfg['signs']))
        assert [(x['control_index'],x['dose'],x['sign']) for x in rec['conditions']]==expected
        for x in rec['conditions']:
            ctrl=cfg['controls'][x['control_index']]
            for a,b in [('projection_kind','kind'),('layer','layer'),('output_row','output_row'),('input_coordinate','input_coordinate'),('control','control')]:assert x[a]==ctrl[b]
            requested=x['sign']*x['dose']*ctrl['row_RMS']
            assert x['requested_delta']==requested
            native_before=ctrl['original_weight'];native_after=bf16(f32(f32(native_before)+f32(requested)))
            effective=native_after-native_before
            assert x['effective_delta']==effective and x['upstream_x_exact']
            key=(x['control_index'],x['dose'],x['sign'])
            value={'control_index':key[0],'dose':key[1],'sign':key[2],'kind':ctrl['kind'],'control':ctrl['control'],
                   'original':native_before,'changed':native_after,'effective_delta':effective,
                   'relative_abs_change':abs(effective/native_before),'crossed_sign':native_after*native_before<0}
            assert native_before!=0
            if key in unique:assert unique[key]==value
            else:unique[key]=value
            assert x['probability']['coordinates']==2*32*n and x['head_output']['coordinates']==8192
            for k in ('probability','head_output'):
                d=x[k];assert 0<=d['actual_changed_coordinates']<=d['coordinates']
                assert d['actual_L1']>=0 and d['ideal_L1']>=0 and d['prediction_error_L1']>=0
                assert d['prediction_error_max_abs']<=d['prediction_error_L1']+1e-12
                assert math.isclose(d['actual_L1'],0,abs_tol=0)==(d['actual_changed_coordinates']==0)
            if ctrl['kind']=='v':assert x['probability']['actual_L1']==x['probability']['ideal_L1']==x['probability']['prediction_error_L1']==0
            assert len(x['fixed_baseline_sequence_token_logprob_change'])==len(base['generated_ids'])
            assert math.isclose(sum(x['fixed_baseline_sequence_token_logprob_change']),x['fixed_baseline_sequence_logprob_change'],abs_tol=1e-10)
            name=f"{ctrl['kind']}/{ctrl['control']}/dose{x['dose']}"
            g=groups.setdefault(name,{'n':0,'effective_weight_zeros':0,'zero_full_probability_effects':0,'native_P_L1':0.,'ideal_P_error_L1':0.,
                'native_head_L1':0.,'ideal_head_error_L1':0.,'whole_vocabulary_L1':0.,'next_token_changes':0})
            g['n']+=1;g['effective_weight_zeros']+=effective==0;g['zero_full_probability_effects']+=x['all_vocabulary_probability_L1']==0
            g['native_P_L1']+=x['probability']['actual_L1'];g['ideal_P_error_L1']+=x['probability']['prediction_error_L1']
            g['native_head_L1']+=x['head_output']['actual_L1'];g['ideal_head_error_L1']+=x['head_output']['prediction_error_L1']
            g['whole_vocabulary_L1']+=x['all_vocabulary_probability_L1'];g['next_token_changes']+=x['baseline_next_id']!=x['changed_next_id']
            count+=1;zeros+=x['all_vocabulary_probability_L1']==0;changes+=x['baseline_next_id']!=x['changed_next_id']
            fg=by_function.setdefault(row['output_function'],{'n':0,'next_token_argmax_changes':0,'published_prefix_argmax_changes':0})
            fg['n']+=1
            if x['baseline_next_id']!=x['changed_next_id']:
                fg['next_token_argmax_changes']+=1;fg['published_prefix_argmax_changes']+=row['parameter_published']
                argmax_records.append({'case_id':cid,'case_index':row['case_index'],'control_index':x['control_index'],
                    'dose':x['dose'],'sign':x['sign'],'baseline_next_id':x['baseline_next_id'],'changed_next_id':x['changed_next_id'],
                    'natural_changed_path_measured':row['parameter_published']})
        natural=rec['natural_changed_generations']
        assert len(natural)==(192 if row['parameter_published'] else 0)
        if natural:assert [(x['control_index'],x['dose'],x['sign']) for x in natural]==expected
        for x in natural:
            ctrl=cfg['controls'][x['control_index']];name=f"{ctrl['kind']}/{ctrl['control']}/dose{x['dose']}"
            g=natural_groups.setdefault(name,{'n':0,'whole_generated_ids_changed':0,'whole_generated_text_changed':0,
                'eos_changed':0,'explicit_status_changed':0,'explicit_choice_changed':0})
            g['n']+=1;g['whole_generated_ids_changed']+=x['generated_ids']!=base['generated_ids']
            g['whole_generated_text_changed']+=x['generated']!=base['generated']
            assert len(x['generated_token_logprobs'])==len(x['generated_ids'])
            g['eos_changed']+=x['eos']!=base['eos']
            g['explicit_status_changed']+=x['explicit_final']['status']!=base['explicit_final']['status']
            g['explicit_choice_changed']+=x['explicit_final'].get('choice')!=base['explicit_final'].get('choice')
    summary=read(OUT/'analysis/final.json')['summary']
    assert groups==summary['aggregate_by_kind_control_dose']
    assert count==24576 and len(unique)==192 and sum(g['n'] for g in natural_groups.values())==3072
    ratios={}
    for name,g in groups.items():
        ratios[name]={'P_error_over_actual_L1':g['ideal_P_error_L1']/g['native_P_L1'] if g['native_P_L1'] else None,
                     'head_error_over_actual_L1':g['ideal_head_error_L1']/g['native_head_L1'] if g['native_head_L1'] else None}
    ranges={}
    for kind in ('q','k','v'):
        for ctrl in ('ordinary','low'):
            for dose in cfg['doses']:
                vals=[r for r in unique.values() if (r['kind'],r['control'],r['dose'])==(kind,ctrl,dose)]
                assert len(vals)==16
                ranges[f'{kind}/{ctrl}/dose{dose}']={'unique_edits':16,'relative_abs_min':min(r['relative_abs_change'] for r in vals),
                    'relative_abs_max':max(r['relative_abs_change'] for r in vals),'crossed_sign':sum(r['crossed_sign'] for r in vals)}
    result={'all_independent_checks_passed':True,'actual_conditions':count,'unique_parameter_dose_sign_edits':len(unique),
        'all24_full_matrix_hashes_restored':True,
        'full_vocabulary_zero_effects':zeros,'next_token_argmax_changes':changes,'recomputed_groups':groups,
        'local_prediction_error_ratios':ratios,'relative_parameter_change_ranges':ranges,'unique_edits':list(unique.values()),
        'natural_generation_comparison':natural_groups,'source_records_sha256':report_hashes,'code_sha256':sha(Path(__file__)),
        'next_token_changes_by_function':by_function,'next_token_argmax_records':argmax_records,
        'limits':['This audits durable measured records, not a fresh model replication.',
            'Absolute rowRMS doses do not equalize relative changes; low and ordinary also differ in coordinate selection.',
            'Native output zero or global sign failure is not semantic irrelevance; Q/K prediction discrepancy cannot be assigned only to BF16 without a numerical control.',
            'Whole generated ID/text difference is not a semantic improvement or failure. Fixed256 next-token argmax and natural cache generation are separate protocols.',
            '3072 natural trajectories reuse16prefixes;24576scalarconditions reuse128prefixes and192uniqueedits.',
            'No per-case changed raw P arrays were recorded; aggregate coordinate maps must be labelled aggregate.']}
    save(OUT/'analysis/independent_condition_audit.json',result)
    print({k:result[k] for k in ('all_independent_checks_passed','actual_conditions','unique_parameter_dose_sign_edits','full_vocabulary_zero_effects','next_token_argmax_changes')},flush=True)
    print('Relative ranges',ranges,flush=True)
    print('Natural generation',natural_groups,flush=True)
    print('Argmax functions',by_function,flush=True)

if __name__=='__main__':main()
