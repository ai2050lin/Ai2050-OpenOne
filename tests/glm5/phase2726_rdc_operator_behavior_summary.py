"""Retrospective native QA cohorts, retaining every layer and native coordinate."""
from collections import defaultdict
from rdc_operator_common import *


def summarize(model, records):
    out=BASE/'behavior'/model
    if (out/'result.json').exists():return read(out/'result.json')
    sums={};counts=defaultdict(int);energy=defaultdict(list);examples=[]
    for r in records:
        scope=r['_scope'];path=BASE/'qa'/model/scope/'fields'/f'{r["question_id"]}.npz'
        with np.load(path) as z:
            h=unbits(z['H']).astype(float)
        rms=np.maximum(np.sqrt(np.mean(h*h,axis=1)),1e-12)
        group=r['language']+('_answer_match' if r['normalized_full_EM'] else '_answer_nonmatch')
        packet=np.stack([h,h*h,h/rms[:,None],(h/rms[:,None])**2])
        if group not in sums:sums[group]=np.zeros_like(packet)
        sums[group]+=packet;counts[group]+=1;energy[group].append(rms*rms)
        examples.append({'question_id':r['question_id'],'scope':scope,'sample_id':r['sample_id'],
            'source_group':r['source_group'],'language':r['language'],'question_type':r['question_type'],
            'group':group,'question':r['question'],'generated_text':r['generated_text'],
            'reference_answers':[a['text'] for a in r['answers']],
            'normalized_full_EM':r['normalized_full_EM'],'answer_F1':r['answer_F1'],
            'prompt_tokens':len(r['prompt_ids']),'generation_tokens':len(r['generated_ids']),
            'last_prompt_token_ID':r['prompt_ids'][-1],'stopped_by_native_EOS':r['stopped_by_native_EOS'],
            'raw_query_field':str(path.relative_to(BASE))})
    npz(out/'full_native_response_moments.npz',**{g:v/counts[g] for g,v in sums.items()})
    compressed(out/'all_qa_rows.json.gz',examples)
    groups={}
    for key in sorted(counts):
        rr=[r for r in examples if r['group']==key]
        groups[key]={'count':counts[key],'mean_layer_energy':np.mean(energy[key],axis=0).tolist(),
            'mean_prompt_tokens':float(np.mean([r['prompt_tokens'] for r in rr])),
            'mean_generated_tokens':float(np.mean([r['generation_tokens'] for r in rr])),
            'mean_F1':float(np.mean([r['answer_F1'] for r in rr])),
            'native_EOS':sum(r['stopped_by_native_EOS'] for r in rr),
            'question_type_counts':dict(__import__('collections').Counter(r['question_type'] for r in rr)),
            'last_query_ID_counts':dict(__import__('collections').Counter(r['last_prompt_token_ID'] for r in rr))}
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':model,'questions':len(records),
        'groups':groups,'moment_axes':['raw_mean/raw_second_moment/RMS_normalized_mean/RMS_normalized_second_moment','native_layer_boundary','every_native_residual_coordinate'],
        'limits':'Answer string matching is an operational behavior label, not semantic correctness certification. These posterior cohorts differ in language, type, context length and potentially query-token ID. Descriptive response contrasts are not causal reasoning directions; neither outcome label enters any predictor.'}
    save(out/'result.json',result);return result


def main():
    start=time.monotonic();all_records={};results={}
    for model in ('qwen4','qwen14','glm4'):
        paths=[BASE/'qa'/model/s/'result.json' for s in (('main','confirmation') if model=='qwen4' else ('main',))]
        assert all(p.exists() for p in paths),('Native model cohort still incomplete',model)
        records=[]
        for p in paths:
            rr=[dict(read(q),_scope=p.parent.name) for q in sorted((p.parent/'commits').glob('*.json'))]
            assert len(rr)==read(p)['sources']
            records.extend(rr)
        all_records[model]={r['question_id']:r for r in records};results[model]=summarize(model,records)
    shared=set.intersection(*(set(v) for v in all_records.values()));assert len(shared)==64
    matched={};pairs=[]
    for model,rr in all_records.items():
        values=[rr[q] for q in sorted(shared)]
        matched[model]={'questions':len(values),'EM':float(np.mean([r['normalized_full_EM'] for r in values])),
            'F1':float(np.mean([r['answer_F1'] for r in values])),
            'EOS':float(np.mean([r['stopped_by_native_EOS'] for r in values])),
            'groups':{g:{'n':len(vv),'EM':float(np.mean([r['normalized_full_EM'] for r in vv])),
                'F1':float(np.mean([r['answer_F1'] for r in vv]))} for g in sorted({r['language']+'/'+r['question_type'] for r in values})
                for vv in [[r for r in values if r['language']+'/'+r['question_type']==g]]}}
    for a,b in [('qwen14','qwen4'),('glm4','qwen4'),('qwen14','glm4')]:
        av,bv=all_records[a],all_records[b]
        for q in shared:
            assert av[q]['question']==bv[q]['question'] and av[q]['answers']==bv[q]['answers']
        pairs.append({'model_a':a,'model_b':b,'metric':'paired F1(a)-F1(b)',
            **clustered([av[q]['answer_F1']-bv[q]['answer_F1'] for q in sorted(shared)],[av[q]['source_group'] for q in sorted(shared)])})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'models':results,'matched64':matched,'paired_F1':pairs,
        'matched_question_ids':sorted(shared),'limits':'Matched questions, original complete contexts and gold; native chat/tokenizers/architectures differ. Finite length-bounded reading-comprehension subset, not an official benchmark or a controlled scaling-law experiment.'}
    save(BASE/'behavior/result.json',result);ledger('behavior_qualified_full_native_fields',time.monotonic()-start)
    print('NATIVE_BEHAVIOR_SUMMARY_COMPLETE',matched,flush=True)


if __name__=='__main__':main()
