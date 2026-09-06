"""Independent truth, complete score coverage and physical-scalar accounting."""
import math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2669_symmetric_multitoken_delivery import OUT,SHORT,FP,BF,Q14,MAPS,CONTRACT
from phase2666_multitoken_parameter_engine import PARTS


def main():
    rows=read(SHORT/'material/cases.json');multi=read(FP/'material/cases.json');material={r['case_index']:r for r in multi};fp=read(FP/'analysis/records.json')
    cond=read(RESULT/'phase2667_multitoken_scalar_validation/analysis/conditions.json');checks={}
    checks['truth_question_mapping_8192']=len(rows)==8192 and all(r['statement_truth']==(r['target_index']==r['probe_index']) and r['question_affirmative']==(r['statement_truth']!=bool(r['polarity'])) and r['expected_yes']==(r['question_affirmative']!=bool(r['mapping'])) for r in rows)
    checks['frozen_mask_hash']=sha(CONTRACT/'maps/frozen_masks.npz')==read(CONTRACT/'protocol/frozen.json')['frozen_mask_sha256']
    pairs={}
    for r in rows:pairs.setdefault(tuple(r[k] for k in ('family','language','unit','form','target_index','mention_order','probe_index','polarity')),[]).append(r)
    checks['4096_exact_length_matched_mapping_pairs']=len(pairs)==4096 and all(len(rr)==2 and len({len(r['prompt_ids']) for r in rr})==1 and len({r['expected_yes'] for r in rr})==2 for rr in pairs.values())
    checks['256_fp_case_set']=len(fp)==256 and {r['case_index'] for r in fp}==set(material)
    checks['all_answers_four_tokens_not_long_generation']=all(len(a)==4 for r in multi for a in r['canonical_answer_ids'])
    checks['512_exact_teacher_forced_branches']=all(b['input_ids']==material[r['case_index']]['prompt_ids']+material[r['case_index']]['canonical_answer_ids'][i] and b['target_ids']==material[r['case_index']]['canonical_answer_ids'][i]+[material[r['case_index']]['eos_token_id']] and b['prediction_positions']==list(range(len(material[r['case_index']]['prompt_ids'])-1,len(material[r['case_index']]['prompt_ids'])+4)) and b['categories']==material[r['case_index']]['answer_token_categories'][i] and len(b['categories'])==5 and b['categories'][-1]=='eos' for r in fp for i,b in enumerate(r['branches']))
    checks['part_logprob_exact_coverage']=all(abs(sum(b['part_logprobs'].values())-b['total_logprob'])<1e-9 and all(abs(sum(lp for lp,cat in zip(b['logprobs'],b['categories']) if cat==p)-b['part_logprobs'][p])<1e-9 for p in PARTS) for r in fp for b in r['branches'])
    checks['shared_first_token_not_answer_contrast']=all(abs(r['branches'][0]['logprobs'][0]-r['branches'][1]['logprobs'][0])<1e-9 for r in fp)
    scalar=[r for r in cond if r['kind']=='single_weight'];ids=sorted({r['case_index'] for r in scalar});errors=[];control_errors=[]
    for ci in ids:
        rr=[r for r in scalar if r['case_index']==ci];T=len(material[ci]['prompt_ids']);cache={}
        with np.load(FP/f'field/case_{ci:04d}.npz',allow_pickle=False) as z:
            for r in rr:
                key=r['layer'],r['j'],r['k']
                if key not in cache:
                    l,j,k=key;cache[key]={}
                    for part in ('all',)+PARTS:
                        suffix='' if part=='all' else '_'+part
                        terms=[z[f'{label}__L{l}_v_x'][:,k].astype('float64')*z[f'{label}__L{l}_v_g{suffix}'][:,j].astype('float64') for label in ('Y','N')]
                        cache[key][part]=[math.fsum(terms[0])-math.fsum(terms[1]),float(terms[0][T-1]-terms[1][T-1]),float(terms[0][-1]-terms[1][-1])]
                delta=r['target_weight']-r['original_weight'];assert delta==r['actual_delta']
                for p in ('all',)+PARTS:errors.append(abs(delta*cache[key][p][0]-(r['predicted'] if p=='all' else r['parts'][p]['predicted'])))
                control_errors.extend([abs(delta*cache[key]['all'][1]-r['prompt_last_only']),abs(delta*cache[key]['all'][2]-r['branch_last_only'])])
    checks['8192_scalar_and_part_predictions_recomputed']=len(scalar)==2048 and len(ids)==128 and len(errors)==8192 and max(errors)<1e-12
    checks['4096_wrong_control_values_recomputed']=len(control_errors)==4096 and max(control_errors)<1e-12
    checks['2048_finite_effect_part_accounting']=max(abs(r['effect']-sum(r['parts'][p]['effect'] for p in PARTS)) for r in scalar)<1e-5
    restoration=read(RESULT/'phase2667_multitoken_scalar_validation/analysis/restoration.json');checks['all_four_weight_matrices_restored']=restoration['before']==restoration['after'] and not restoration['disk_model_changed']
    q14=read(Q14/'material/cases.json');checks['1024_q14_both_styles']=len(q14)==1024 and {r['style'] for r in q14}=={0,1}
    checks['Qwen14_no_index_alignment']=read(Q14/'protocol/model.json')['dimensions']=={'hidden':5120,'layers':40,'mlp':17408}
    with np.load(MAPS/'maps/confirmed_masks.npz') as z:survivors={k:np.argwhere(z[k]).tolist() for k in z.files}
    checks['two_H_five_MLP_midlayer_candidates']=len(survivors['h'])==2 and len(survivors['mlp'])==5
    report={'checks':checks,'all_checks_passed':all(checks.values()),'maximum_independent_scalar_difference':max(errors),'maximum_control_difference':max(control_errors),'survivors':survivors,
        'qa_erratum':'First independent audit appended a second EOS category to metadata that already included EOS. Corrected the audit expectation, not experimental records or results; all five targets and category partition are checked explicitly.',
        'scope':'Exact arithmetic/material accounting, not evidence for independent samples, long answer success or semantic closure. Padding support was not exercised because all actual answers have equal four-token length.'}
    save(OUT/'analysis/scientific_checks.json',report);print(json.dumps(report,ensure_ascii=True),flush=True);assert report['all_checks_passed']


if __name__=='__main__':main()
