"""Human-question/span hyperedges and complete native query responses; no gold labels enter inference."""
import re
from collections import Counter, defaultdict
from rdc_operator_common import *


def token_overlap(offsets, spans):
    return sorted({i for i,(a,b) in enumerate(offsets) if b>a and any(b>s and a<e for s,e in spans)})


def main():
    out=BASE/'qa_atlas'
    if (out/'result.json').exists():
        return
    start=time.monotonic()
    selected=gzread(BASE/'qa_balanced_material.json.gz')+gzread(BASE/'qa_multihop_material.json.gz')
    selected={r['question_id']:r for r in selected}
    records,profiles,counts=[],{},Counter()
    for cp in sorted((BASE/'qa/qwen4/main/commits').glob('*.json')):
        response=read(cp)
        row=selected[response['question_id']]
        actual=response['actual_prompt'];base=actual.find(row['full_context']);assert base>=0
        question=actual.rfind(row['question']);assert question>=0
        answer_spans=[]
        for answer in row['answers']:
            if 'answer_start' in answer:
                s=base+answer['answer_start'];assert actual[s:s+len(answer['text'])]==answer['text'];answer_spans.append([s,s+len(answer['text'])])
            elif answer['text'].casefold() not in ('yes','no'):
                answer_spans.extend([[base+m.start(),base+m.end()] for m in re.finditer(re.escape(answer['text']),row['full_context'],re.I)])
        support_spans=[]
        cursor=base
        for title,sentences in row.get('context_paragraphs',[]):
            assert actual[cursor:cursor+len(title)]==title
            cursor+=len(title)+1
            for i,sentence in enumerate(sentences):
                assert actual[cursor:cursor+len(sentence)]==sentence
                if [title,i] in row['supporting_facts'] or (title,i) in row['supporting_facts']:
                    support_spans.append([cursor,cursor+len(sentence)])
                cursor+=len(sentence)
            cursor+=2
        offsets=response['token_offsets']
        groups={'question':token_overlap(offsets,[[question,question+len(row['question'])]]),
                'context':token_overlap(offsets,[[base,base+len(row['full_context'])]]),
                'gold_answer_occurrences':token_overlap(offsets,answer_spans),
                'gold_support_sentences':token_overlap(offsets,support_spans)}
        answer_tokens=set(groups['gold_answer_occurrences'])
        controls=[]
        ids=response['prompt_ids']
        for p in sorted(answer_tokens):
            candidates=[i for i in groups['context'] if i not in answer_tokens and ids[i]==ids[p]]
            if candidates:
                controls.append({'answer_position':p,'same_ID_control_position':min(candidates,key=lambda i:(abs(i-p),i)),
                                 'token_id':ids[p]})
        with np.load(BASE/'qa/qwen4/main/fields'/f'{row["question_id"]}.npz') as z:
            h=unbits(z['H']).astype(float)
            key=row['language']+'/'+row['question_type']+'/'+('gold_string_agreement' if response['normalized_full_EM'] else 'gold_string_mismatch')
            if key not in profiles:
                profiles[key]=np.zeros((2,*h.shape),float)
            profiles[key][0]+=h;profiles[key][1]+=h*h;counts[key]+=1
            blocks={}
            for b in (6,16,34):
                attention=unbits(z[f'L{b}_attention_sources']).astype(float)
                blocks[str(b)]={name:{'source_positions':len(pos),'mean_head_mass':float(attention[:,pos].sum(1).mean()) if pos else None,
                    'mass_per_source_position':float(attention[:,pos].mean()) if pos else None} for name,pos in groups.items()}
                blocks[str(b)]['same_ID_attention_pairs']={'pairs':len(controls),
                    'mean_answer_minus_control':float(np.mean([attention[:,p['answer_position']]-attention[:,p['same_ID_control_position']] for p in controls])) if controls else None}
        records.append({k:response[k] for k in ('question_id','sample_id','source_group','language','question_type','question','answers','generated_text','normalized_full_EM','answer_F1')} |
            {'hyperedge':{'context_char_span':[base,base+len(row['full_context'])],'question_char_span':[question,question+len(row['question'])],
                'answer_char_spans':answer_spans,'support_char_spans':support_spans,'token_groups':groups,'same_ID_answer_controls':controls},
             'native_source_attention':blocks,
             'meaning':'Source annotation-to-native-query link; support/answer labels are retrospective, attention mass is not a proof of causal use or complete multihop reasoning.'})
    assert len(records)==288
    compressed(out/'hyperedges.json.gz',records)
    npz(out/'all_query_coordinate_profiles.npz',**{k:v/counts[k] for k,v in profiles.items()})
    save(out/'profile_index.json',{'counts':dict(counts),'axes':['mean/mean_square','layer0..36','coordinate0..2559'],
        'missing_label_note':'String mismatches can be correct paraphrases, added prepositions, numeric variants or actual mistakes; they are not all semantic failures.'})
    matching=[]
    for langtype in sorted({r['language']+'/'+r['question_type'] for r in records}):
        rr=[r for r in records if r['language']+'/'+r['question_type']==langtype]
        matching.append({'type':langtype,'questions':len(rr),'answer_token_pairs':sum(len(r['hyperedge']['same_ID_answer_controls']) for r in rr),
            'answer_attention_paired_by_block':{str(b):clustered([r['native_source_attention'][str(b)]['same_ID_attention_pairs']['mean_answer_minus_control'] for r in rr
                if r['native_source_attention'][str(b)]['same_ID_attention_pairs']['pairs']],
                [r['source_group'] for r in rr if r['native_source_attention'][str(b)]['same_ID_attention_pairs']['pairs']]) for b in (6,16,34)}})
    # Explicit scoring audit; original immutable response and aggregate scores are not rewritten.
    audited=[]
    for record in records:
        if record['question'] in ['How many people were on the U.S. fact-finding team?', 'Why was Dinuzulu kaCetshwayo imprisoned on the island?',
                                  'With whom did Chopin go to London with in 1837?', 'How wide was the widest tornado ever?']:
            audited.append({k:record[k] for k in ('question_id','question','answers','generated_text','normalized_full_EM')} |
                {'note': 'Literal/normalized string metrics are not a complete semantic adjudication. Inspect original passage and equivalence; preserve original scores.'})
    save(out/'scoring_audit.json',{'timestamp':stamp(),'actual_examples':audited,
        'explicit_cases': {'three_vs_3':'Same integer, yet the original normalizer does not convert number words; normalized-EM false is not evidence of failed counting.',
            'For_leading':'An added preposition before the full annotated reason fails whole-string EM but can preserve its meaning.',
            'sentence_final_period':'Makes literal span match false while punctuation-normalized EM stays true; no requirement forbade a final period.',
            '4.2km_vs2.6miles':'Potential rounded unit-equivalent answer; do not classify as a semantic error from string mismatch alone. Full passage is available for review.'},
        'conclusion':'Retain high-specificity gold-string agreement as an eligibility stratum, but label remaining cases mismatch/unknown rather than definite comprehension failure.'})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'questions':len(records),'hyperedges':len(records),
        'questions_with_answer_span':sum(bool(r['hyperedge']['answer_char_spans']) for r in records),
        'questions_with_support_sentences':sum(bool(r['hyperedge']['support_char_spans']) for r in records),
        'questions_with_same_ID_answer_control':sum(bool(r['hyperedge']['same_ID_answer_controls']) for r in records),
        'matched_attention':matching,'all_coordinates':[37,2560],
        'scope':'New external graph consists of verified source/question/answer-span/support-sentence hyperedges, linked to native full-coordinate last-query responses and complete source attention. Not a gold syntax graph or proof of the model actually executing every annotated hop.',
        'native_behavior':'Gold-string agreement is measured, broader semantic correctness is incomplete; literal format and actual EOS remain separate.',
        'historical_graph_link':'Prior UD/GUM entities, discourse and syntax remain queryable at /rdc-joint; they are not silently transplanted as annotations of this new corpus.'}
    save(out/'result.json',result);ledger('QA_hypergraph_all_coordinate_atlas',time.monotonic()-start,questions=len(records))
    print('QA_HYPERGRAPH_COMPLETE', {k:result[k] for k in ('questions','questions_with_answer_span','questions_with_support_sentences','questions_with_same_ID_answer_control')},flush=True)


if __name__=='__main__':
    main()
