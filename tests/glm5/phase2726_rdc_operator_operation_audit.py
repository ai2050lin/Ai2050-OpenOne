"""Answer agreement versus complete internal/probability responses under natural reordering."""
from collections import defaultdict
from rdc_operator_common import *
from rdc_operator_qa import normalize_answer


def main():
    start=time.monotonic();out=BASE/'operations';result=read(out/'result.json')
    records=[read(p) for p in sorted((out/'commits').glob('*.json'))];assert len(records)==64
    moments={};groups=defaultdict(list);rows_=[]
    for r in records:
        with np.load(out/'fields'/f'{r["question_id"]}.npz') as z:new=unbits(z['H']).astype(float)
        with np.load(BASE/'qa/qwen4/confirmation/fields'/f'{r["question_id"]}.npz') as z:old=unbits(z['H']).astype(float)
        relative=np.mean((new-old)**2,1)/np.maximum(np.mean(old*old,1),1e-20)
        assert np.allclose(relative,r['per_layer_full_coordinate_relative_MSE'],rtol=1e-10,atol=1e-12)
        old_unit=old/np.maximum(np.sqrt(np.mean(old*old,1,keepdims=True)),1e-12)
        new_unit=new/np.maximum(np.sqrt(np.mean(new*new,1,keepdims=True)),1e-12)
        same=normalize_answer(r['generated_text'],r['language'])==normalize_answer(r['context_first']['generated_text'],r['language'])
        both=r['normalized_full_EM'] and r['context_first']['normalized_full_EM']
        keys=['same_normalized_answer' if same else 'different_normalized_answer',
            'both_gold_string_match' if both else 'not_both_gold_string_match']
        entry={'question_id':r['question_id'],'source_group':r['source_group'],'language':r['language'],
            'same_normalized_answer':same,'both_gold_string_match':bool(both),
            'same_last_query_token_ID':r['same_final_prompt_token_id'],
            'raw_H36_relative_MSE':float(relative[-1]),'RMS_normalized_H36_mean_squared_difference':float(np.mean((new_unit[-1]-old_unit[-1])**2)),
            **r['full_query_probability_comparison']}
        rows_.append(entry)
        for key in keys:
            if key not in moments:moments[key]=np.zeros((4,37,2560),dtype=np.float64)
            moments[key]+=np.stack([old,new,(new-old)**2,(new_unit-old_unit)**2])
            groups[key].append(entry)
    summaries={}
    for key,rr in groups.items():
        moments[key]/=len(rr);moments[key][2:]=np.sqrt(np.maximum(moments[key][2:],0))
        summaries[key]={'questions':len(rr),'article_groups':len({r['source_group'] for r in rr}),
            'mean_full_vocab_query_KL':float(np.mean([r['KL_context_first_to_question_first'] for r in rr])),
            'KL_article_cluster':clustered([r['KL_context_first_to_question_first'] for r in rr],[r['source_group'] for r in rr]),
            'mean_raw_H36_relative_MSE':float(np.mean([r['raw_H36_relative_MSE'] for r in rr])),
            'mean_RMS_normalized_H36_squared_difference':float(np.mean([r['RMS_normalized_H36_mean_squared_difference'] for r in rr])),
            'query_argmax_agreement':float(np.mean([r['query_head_argmax_agrees'] for r in rr])),
            'same_last_query_token_ID':sum(r['same_last_query_token_ID'] for r in rr)}
    npz(out/'answer_agreement_complete_coordinate_moments.npz',**moments)
    save(out/'response_agreement_audit.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'questions':64,'groups':summaries,'all_rows':rows_,
        'moment_axes':['old_raw_mean/new_raw_mean/raw_difference_RMS/row_RMS_normalized_difference_RMS','H0..H36','every2560nativecoordinate'],
        'group_definition':'Two overlapping descriptive partitions: normalized generated-answer equality, and both answers matching any original gold string. Each question occurs in one group of each partition.',
        'scope':'Post-hoc response stratification, not an online predictor or a newly untouched hypothesis test. Same decoded answer does not assert equal probabilities, hidden states or semantic computation; mismatch is not automatic semantic error. Order also changes positions and available cross-token history. No difference vector is transported into a model.'})
    ledger('natural_order_answer_agreement_full_coordinate_audit',time.monotonic()-start);guard()
    print('NATURAL_ORDER_RESPONSE_AUDIT_PASS',summaries,flush=True)


if __name__=='__main__':main()
