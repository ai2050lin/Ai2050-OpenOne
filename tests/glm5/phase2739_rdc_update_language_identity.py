"""Post-observation exact token/position controls using unchanged full fields."""
from collections import Counter
from rdc_update_common import *


def main():
    start=time.monotonic();out=BASE/'language_identity';rows=gzread(BASE/'language_material.json.gz')
    lookup={(r['source_group'],r['language'],r['answer_style']):i for i,r in enumerate(rows)}
    alignment=[]
    for r in rows:
        p=r['anchors'][0];end=r['token_offsets'][p][1];limit=r['body_end_character']
        assert end<=limit
        assert p==max(i for i,(a,b) in enumerate(r['token_offsets']) if b>a and b<=limit)
        alignment.append({'sample_id':r['sample_id'],'family':r['family'],'language':r['language'],'style':r['answer_style'],
          'anchor':p,'token_id':r['prompt_ids'][p],'token_text':r['text'][r['token_offsets'][p][0]:end],
          'declared_body_end':limit,'actual_token_end':end,'exact_character_endpoint':end==limit,'unconsumed_body_suffix':r['text'][end:limit]})
    compressed(out/'body_anchor_alignment.json.gz',alignment);comparisons=[];evidence={}
    for layer in (0,6,12,17,24,36):
      for anchor in (0,1):
        path=BASE/'language_analysis'/f'all_pair_cosine_H{layer}_anchor{anchor}.npz';evidence[str(path.relative_to(BASE))]=sha(path)
        with np.load(path) as z:cos=z['cosine']
        for lang,style in (('zh','direct'),('en','explain'),('zh','explain')):
          for family in sorted({r['family'] for r in rows}):
            for control in ('family_truth','plus_current_target_token','plus_current_target_token_and_exact_position'):
                delta=[];groups=[];matched_counts=[];unavailable=[]
                for i,r in enumerate(rows):
                    if (r['family'],r['language'],r['answer_style'],r['split'])!=(family,'en','direct','language_test'):continue
                    j=lookup[r['source_group'],lang,style];target=rows[j];p=target['anchors'][anchor];tid=target['prompt_ids'][p]
                    others=[]
                    for k,s in enumerate(rows):
                        if s['source_group']==r['source_group'] or (s['family'],s['language'],s['answer_style'],s['split'],s['truth'])!=(family,lang,style,r['split'],r['truth']):continue
                        if control!='family_truth' and s['prompt_ids'][s['anchors'][anchor]]!=tid:continue
                        if control=='plus_current_target_token_and_exact_position' and s['anchors'][anchor]!=p:continue
                        others.append(k)
                    if not others:unavailable.append(r['source_group']);continue
                    delta.append(float(cos[i,j]-cos[i,others].mean()));groups.append(r['source_group']);matched_counts.append(len(others))
                comparisons.append({'layer':layer,'anchor':anchor,'family':family,'target_language':lang,'target_style':style,'control':control,
                  'available_pairs':len(delta),'unavailable_groups':unavailable,'comparison_count_per_pair':matched_counts,
                  'same_group_advantage':clustered(delta,groups) if delta else None})
    result={'timestamp':stamp(),'source':snapshot(__file__),'rows':640,'semantic_groups':160,
      'exact_body_character_endpoints':sum(r['exact_character_endpoint'] for r in alignment),
      'omitted_suffix_counts':dict(Counter(r['unconsumed_body_suffix'] for r in alignment)),
      'comparisons':comparisons,'unchanged_cosine_archives_sha256':evidence,'material_sha256':sha(BASE/'language_material.json.gz'),
      'seconds':time.monotonic()-start,
      'correction':'Body anchor means last fully contained tokenizer token, not always the literal final body character. A punctuation-plus-newline token can straddle the boundary, leaving an ordinary word as the query token.',
      'scope':'Post-observation identity/position sensitivity audit on the same held language groups, NOT new independent confirmation or predictor selection. Matching may leave few or zero controls; all coverage is reported. These controls do not exhaust lexical context, templates, or semantic confounding.'}
    save(out/'result.json',result);ledger('language_current_token_and_position_audit',result['seconds'])
    print('LANGUAGE_IDENTITY_AUDIT',result['exact_body_character_endpoints'],result['omitted_suffix_counts'],flush=True)


if __name__=='__main__':main()
