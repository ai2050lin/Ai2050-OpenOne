"""Independent material/objective accounting and scalar-factor recomputation."""
import math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2661_sequence_coordinate_delivery import OUT,MATERIAL,BF,FP,Q14


def main():
    rows=read(MATERIAL/'material/cases.json');material={r['case_index']:r for r in rows};fp=read(FP/'analysis/records.json')
    cond=read(RESULT/'phase2659_sequence_scalar_validation/analysis/conditions.json');checks={};by={}
    checks['truth_and_mapping_8192']=all(r['statement_truth']==(r['target_index']==r['probe_index']) and r['question_affirmative']==(r['statement_truth']!=bool(r['polarity'])) and r['expected_yes']==(r['question_affirmative']!=bool(r['mapping'])) and r['target']==r['common_readout_words'][0 if r['expected_yes'] else 1] for r in rows)
    checks['fp_case_set_exact']=set(r['case_index'] for r in fp)==set(r['case_index'] for r in rows if r['fp_selected'])
    for r in rows:
        key=tuple(r[k] for k in ('family','language','unit','form','target_index','mention_order'));by.setdefault(key,[]).append(r)
    checks['1024_bodies_eight_query_cells']=len(by)==1024 and all(len(rr)==8 and len({r['body'] for r in rr})==1 and len({(r['probe_index'],r['polarity'],r['mapping']) for r in rr})==8 for rr in by.values())
    checks['256_exact_teacher_forced_branches']=all(b['input_ids']==material[r['case_index']]['prompt_ids']+material[r['case_index']]['canonical_answer_ids'][i] and b['target_ids']==material[r['case_index']]['canonical_answer_ids'][i]+[material[r['case_index']]['eos_token_id']] and b['prediction_positions']==list(range(len(material[r['case_index']]['prompt_ids'])-1,len(material[r['case_index']]['prompt_ids'])+len(material[r['case_index']]['canonical_answer_ids'][i]))) for r in fp for i,b in enumerate(r['branches']))
    scalar=[r for r in cond if r['kind']=='single_weight'];ids=sorted({r['case_index'] for r in scalar});errors=[];control_errors=[]
    for ci in ids:
        rr=[r for r in scalar if r['case_index']==ci];T=len(material[ci]['prompt_ids']);cache={}
        with np.load(FP/f'field/case_{ci:04d}.npz',allow_pickle=False) as z:
            for r in rr:
                key=r['layer'],r['j'],r['k']
                if key not in cache:
                    l,j,k=key;terms=[z[f'{label}__L{l}_v_x'][:,k].astype('float64')*z[f'{label}__L{l}_v_g'][:,j].astype('float64') for label in ('Y','N')]
                    cache[key]=[math.fsum(terms[0])-math.fsum(terms[1]),float(terms[0][T-1]-terms[1][T-1]),float(terms[0][-1]-terms[1][-1])]
                delta=r['target_weight']-r['original_weight'];assert delta==r['actual_delta']
                errors.append(abs(delta*cache[key][0]-r['predicted']))
                control_errors.extend([abs(delta*cache[key][1]-r['prompt_last_only']),abs(delta*cache[key][2]-r['branch_last_only'])])
    checks['2048_scalar_predictions_independently_recomputed']=len(scalar)==2048 and len(ids)==128 and max(errors)<1e-12
    checks['4096_wrong_control_values_recomputed']=len(control_errors)==4096 and max(control_errors)<1e-12
    checks['all_finite_change_content_plus_EOS']=max(abs(r['effect']-r['first_token_effect']-r['eos_effect']) for r in scalar)<1e-5
    q14=read(Q14/'material/cases.json');checks['256_same_text_crossmodel']=len(q14)==256 and all(r['text']==material[r['case_index']]['text'] and r['target']==material[r['case_index']]['target'] for r in q14)
    checks['Qwen14_no_index_alignment']=read(Q14/'protocol/model.json')['dimensions']=={'hidden':5120,'layers':40,'mlp':17408}
    flips=[{k:r[k] for k in ('case_index','case_id','first_token_contrast','eos_contrast','contrast')} for r in fp if (r['contrast']>0)!=(r['first_token_contrast']>0)]
    report={'checks':checks,'all_checks_passed':all(checks.values()),'maximum_independent_scalar_reduction_difference':max(errors),
        'maximum_control_difference':max(control_errors),'first_complete_rank_flips':flips,
        'scope':'Only records/formulas/token coverage audited; no new natural inference, statistical independence or semantic mechanism claim.'}
    save(OUT/'analysis/scientific_checks.json',report);assert report['all_checks_passed'];print(json.dumps({k:v for k,v in report.items() if k!='first_complete_rank_flips'},ensure_ascii=True))


if __name__=='__main__':main()
