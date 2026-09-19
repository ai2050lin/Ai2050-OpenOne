"""Original complete-answer ability, first-branch alignment and fixed examples."""
import argparse
from collections import Counter,defaultdict
from rdc_question_common import *


def main(key,confirmation=False):
    start=time.monotonic()
    scope='confirmation'if confirmation else'nonconfirmation'
    folder=Path('behavior')/key/scope
    final=OUT/folder/'result.json'
    if final.exists():
        assert read(final)['source']['sha256']==sha(__file__)
        print('NATURAL_NATIVE_BEHAVIOR_ALREADY_COMPLETE',key,scope,flush=True);return
    native=read(OUT/f'native/{key}/{scope}/result.json');assert native['all_passed']
    contract,_,rows,groups=material(key,confirmation)
    selection=read(OUT/'fit'/key/'validation_selection.json')
    by_id={r['question_id']:r for r in rows}
    original=[];group_records=[]
    for pointer in native['group_receipts']:
        group=read(OUT/pointer)
        if group['split']=='train':continue
        questions=group['questions']
        assert all('history'in q and 'teacher'in q for q in questions)
        sequences=[q['history']['generated_ids']for q in questions]
        shortest=min(map(len,sequences));common=0
        while common<shortest and len({v[common]for v in sequences})==1:common+=1
        equal=all(v==sequences[0]for v in sequences)
        group_record={'group_id':group['group_id'],'cohort':group['cohort'],'split':group['split'],
            'questions':[q['question_id']for q in questions],
            'exact_and_stopped_questions':sum(q['history']['score']['whole_response_exact_and_stopped']for q in questions),
            'all4generated_sequences_identical':equal,'common_generated_prefix_tokens':common,
            'first_divergent_generated_index':None if equal else common,
            'first_divergent_scope':'Zero-based generated-token index; teacher labels are not used to identify the original output branch.'}
        group_records.append(group_record)
        original.extend(questions)
    summaries=[]
    for split in sorted({q['split']for q in original}):
        for cohort in ['drop','quoref']:
            qq=[q for q in original if q['split']==split and q['cohort']==cohort]
            gg=[g for g in group_records if g['split']==split and g['cohort']==cohort]
            outcomes={'exact_and_stopped':sum(q['history']['score']['whole_response_exact_and_stopped']for q in qq),
                'exact_content_ignoring_stop':sum(q['history']['score']['whole_response_normalized_exact']for q in qq),
                'natural_EOS':sum(q['history']['native_EOS']for q in qq),
                'cap_censored':sum(q['history']['censored']for q in qq),
                'strict_JSON_array':sum(q['history']['score']['strict_nonempty_json_string_array']for q in qq)}
            summaries.append({'split':split,'cohort':cohort,'questions':len(qq),'contexts':len(gg),
                'counts':outcomes,'fractions':{k:v/len(qq)for k,v in outcomes.items()},
                'generated_tokens':sum(len(q['history']['generated_ids'])for q in qq),
                'teacher_tokens':sum(q['teacher']['tokens']for q in qq),
                'mean_complete_teacher_answer_NLL':float(np.mean([q['teacher']['mean_token_NLL']for q in qq])),
                'first_generated_ID_counts':dict(Counter(q['history']['generated_ids'][0]for q in qq)),
                'within_context_correct_question_count_histogram':dict(Counter(g['exact_and_stopped_questions']for g in gg)),
                'within_context_first_generated_divergence_histogram':dict(Counter('all_same'if g['all4generated_sequences_identical']else str(g['first_divergent_generated_index'])for g in gg))})
    qmap={q['question_id']:q for q in original}
    examples=[]
    full=set(contract['capture']['full_history_context_ids'])
    for split in sorted({g['split']for g in group_records}):
        for cohort in ['drop','quoref']:
            selected=min((g for g in group_records if g['split']==split and g['cohort']==cohort and g['group_id']in full),key=lambda g:g['group_id'])
            source=by_id[selected['questions'][0]]
            examples.append({'group_id':selected['group_id'],'cohort':cohort,'split':split,'passage':source['passage'],
                'selection':'Lexicographically first context among pre-native-declared full-history IDs for each cohort/split; no outcome selection.',
                'questions':[{'question_id':qid,'question':by_id[qid]['question'],
                    'all_complete_accepted_annotations':by_id[qid]['answer_annotations'],
                    'original_output':qmap[qid]['history']['generated_text'],
                    'original_generated_ids':qmap[qid]['history']['generated_ids'],
                    'original_scoring':qmap[qid]['history']['score'],
                    'original_history_field':qmap[qid]['history']['field'],
                    'first_prefix_field':qmap[qid]['field']}for qid in selected['questions']]})
    # Native correctness strata are descriptive, not a new split or fit input.
    strata=[]
    if not confirmation:
        ev=read(OUT/'fit'/key/selection['primary_rule']/'evaluation.json')
        for split in ['validation','diagnostic']:
            record=next(r for r in ev['evaluations']if r['kind']=='selected'and r['split']==split and r['target']=='postnorm')
            ordered=ev[split]
            ref=record['field'];assert sha(ROOT/ref['path'])==ref['sha256']
            with np.load(ROOT/ref['path'])as z:
                for cohort in ['drop','quoref']:
                    for passed in [False,True]:
                        take=np.array([r['cohort']==cohort and bool(qmap[r['question_id']]['history']['score']['whole_response_exact_and_stopped'])==passed for r in ordered])
                        strata.append({'split':split,'cohort':cohort,'conservative_native_exact_and_stopped':passed,'questions':int(take.sum()),
                            'primary_absolute_MSE':float(z['absolute_MSE_by_question'][take].mean())if take.any()else None,
                            'primary_within_MSE':float(z['within_MSE_by_question'][take].mean())if take.any()else None,
                            'zero_change_MSE':float(z['zero_change_MSE_by_question'][take].mean())if take.any()else None,
                            'scope':'Post-outcome descriptive stratum, questions still centered within their original4questioncontext. Nonmatch does not establish reasoning failure.'})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,'scope':scope,
        'native_result_sha256':sha(OUT/f'native/{key}/{scope}/result.json'),
        'fit_selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
        'summaries':summaries,'context_branch_records':group_records,'fixed_examples':examples,'native_outcome_strata':strata,
        'interpretation':'Corpus schema types are external labels. Complete conservative text equality differs from officialQuoref/DROPmetrics and general semantic correctness. Teacherlikelihood is given-history scoring, not natural answer accuracy. First-format-token agreement is not whole-answer agreement.',
        'seconds':time.monotonic()-start}
    immutable(final,result)
    print('NATURAL_NATIVE_BEHAVIOR_COMPLETE',key,scope,len(original),flush=True)
    for row in summaries:print('NATURAL_NATIVE_BEHAVIOR',row['split'],row['cohort'],row['questions'],row['counts'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    parser.add_argument('--confirmation',action='store_true');args=parser.parse_args();main(args.model,args.confirmation)
