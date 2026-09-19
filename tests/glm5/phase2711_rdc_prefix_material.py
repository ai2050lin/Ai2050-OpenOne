"""Frozen naturally occurring bilingual units, explicit source/license/group provenance."""
import argparse
import urllib.request
from collections import Counter
from rdc_prefix_common import *

URLS = {'en':'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-{part}.conllu',
        'zh':'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/r2.15/zh_gsd-ud-{part}.conllu'}


def download(language,part):
    path=CAMPAIGN/f'sources/{language}_{part}.conllu'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(urllib.request.urlopen(URLS[language].format(part=part),timeout=60).read())
    return path


def parse(path,language):
    for block in path.read_text(encoding='utf-8').strip().split('\n\n'):
        lines=block.splitlines();meta={};words=[]
        for line in lines:
            if line.startswith('# ') and ' = ' in line:
                k,v=line[2:].split(' = ',1);meta[k]=v
            elif not line.startswith('#'):
                cols=line.split('\t')
                if len(cols)==10 and cols[0].isdigit():
                    words.append(dict(id=int(cols[0]),form=cols[1],lemma=cols[2],upos=cols[3],head=int(cols[6]),relation=cols[7]))
        if not meta.get('text') or len(words)<10:continue
        sid=meta.get('sent_id','');genre=sid.split('-')[0] if language=='en' else 'wikipedia'
        group=sid.rsplit('-',1)[0] if language=='en' else 'content-'+re.sub(r'\W','',meta['text']).casefold()[:60]
        yield {'source_sentence_id':sid,'source_group':language+'/'+group,'genre':genre,
          'language':language,'text':meta['text'],'retrospective_ud':words,
          'source_has_document_id':language=='en','ud_scope':'full-sentence gold annotations; never prediction inputs'}


def aligned_words(text,words):
    cursor=0;out=[]
    for word in words:
        start=text.find(word['form'],cursor)
        if start<0:
            out.append(dict(word,char_span=None));continue
        end=start+len(word['form']);out.append(dict(word,char_span=[start,end]));cursor=end
    return out


def stratify():
    """Correct a material-only preflight issue; retain original selection manifest."""
    rows=read(CAMPAIGN/'material.json')
    for language in ('en','zh'):
        rr=[r for r in rows if r['language']==language]
        buckets={g:[r for r in rr if r['genre']==g] for g in sorted({r['genre'] for r in rr})}
        def allot(total,capacities):
            weights={g:total*v/sum(capacities.values()) for g,v in capacities.items()}
            result={g:int(v) for g,v in weights.items()}
            for g in sorted(weights,key=lambda g:(-(weights[g]-result[g]),g))[:total-sum(result.values())]:result[g]+=1
            return result
        train=allot(160,{g:len(v) for g,v in buckets.items()})
        valid=allot(48,{g:len(v)-train[g] for g,v in buckets.items()})
        for g,group in buckets.items():
            for i,r in enumerate(group):
                r['initial_split_before_stratification']=r['split']
                r['split']='train' if i<train[g] else 'validation' if i<train[g]+valid[g] else 'test'
    immutable(CAMPAIGN/'material_stratified.json',rows)
    counts=Counter((r['language'],r['genre'],r['split']) for r in rows)
    for language,genre in {(r['language'],r['genre']) for r in rows}:
        assert all(counts[(language,genre,s)]>0 for s in ('train','validation','test'))
    immutable(CAMPAIGN/'stratification_audit.json',{'passed':True,'reason':'Material-only preflight found round-robin exhaustion put rarer English genres entirely in training. Before any forward observations, allocate split counts within each language/genre instead. Original material.json is retained for audit, material_stratified.json is authoritative.',
      'original_sha':sha(CAMPAIGN/'material.json'),'authoritative_sha':sha(CAMPAIGN/'material_stratified.json'),
      'strata':{'/'.join(k):v for k,v in counts.items()},'split_counts':dict(Counter(r['split'] for r in rows)),
      'model_outcomes_used':False})
    print('STRATIFICATION_FROZEN',dict(Counter(r['split'] for r in rows)),flush=True)


def select(tok,part,n,excluded):
    pool=[];manifest=[]
    for language in ('en','zh'):
        path=download(language,part);manifest.append({'path':str(path.relative_to(CAMPAIGN)),'url':URLS[language].format(part=part),'sha256':sha(path)})
        seen=set();eligible=[]
        for r in parse(path,language):
            normalized=re.sub(r'\W','',r['text']).casefold()
            if normalized in excluded or r['source_group'] in seen:continue
            enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
            if not 24<=len(enc['input_ids'])<=96:continue
            seen.add(r['source_group']);r['normalized_identity']=normalized
            r.update(prompt_ids=enc['input_ids'],token_offsets=enc['offset_mapping'],tokens=tok.convert_ids_to_tokens(enc['input_ids']))
            eligible.append(r)
        # Stable content-hash rank within genre, then round-robin genres; no model outcomes.
        buckets={}
        for r in eligible:buckets.setdefault(r['genre'],[]).append(r)
        for rows in buckets.values():rows.sort(key=lambda r:__import__('hashlib').sha256(r['source_group'].encode()).hexdigest())
        selected=[]
        for k in range(max(map(len,buckets.values()))):
            for genre in sorted(buckets):
                if k<len(buckets[genre]):selected.append(buckets[genre][k])
            if len(selected)>=n:break
        assert len(selected)>=n,(language,len(selected),n)
        for i,r in enumerate(selected[:n]):
            r['split']='confirmation' if part=='test' else ('train' if i<160 else 'validation' if i<208 else 'test')
            r['sample_id']=f'{part}-{language}-{i:04d}'
            count=len(r['prompt_ids']);anchors=[max(3,count//3),min(count-4,2*count//3)]
            r['anchors']=anchors;r['positions']=[a+d for a in anchors for d in (0,1,2)]
            r['full_panel']=part=='train' and i<8
            r['retrospective_ud']=aligned_words(r['text'],r['retrospective_ud'])
            r['anchor_graphs']=[prefix_graph(r['text'][:r['token_offsets'][p][1]],p,language) for p in r['positions']]
            r['source_path']=str((CAMPAIGN/f'sources/{language}_{part}.conllu').relative_to(CAMPAIGN))
            pool.append(r)
    # Interleave languages for balanced pilots and deterministic memory/cost measurements.
    pool.sort(key=lambda r:(int(r['sample_id'].rsplit('-',1)[-1]),r['language']))
    return pool,manifest


def main():
    from transformers import AutoTokenizer
    freeze();guard(100*1024**2)
    if (CAMPAIGN/'material.json').exists():
        stratify();return
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    main,manifest=select(tok,'train',256,set())
    confirmation,extra=select(tok,'test',64,{r['normalized_identity'] for r in main})
    assert len(main)==512 and len(confirmation)==128
    assert len({r['source_group'] for r in main})==512
    main_groups={r['source_group'] for r in main}
    assert not main_groups.intersection(r['source_group'] for r in confirmation)
    immutable(CAMPAIGN/'material.json',main);immutable(CAMPAIGN/'confirmation_material.json',confirmation)
    immutable(CAMPAIGN/'sources/manifest.json',{'files':manifest+extra,'version':'UD r2.15',
      'license':'UD annotation CC BY-SA 4.0; respect underlying source-text terms. Research excerpts retain treebank and sentence IDs.',
      'documentation':['https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15','https://github.com/UniversalDependencies/UD_Chinese-GSD/tree/r2.15'],
      'limitations':['English web genres versus Chinese Wikipedia: language and genre confounded.',
        'English document IDs disjoint; Chinese lacks verified document IDs, exact/normalized identity and conservative prefix-group separation only.',
        'Single natural sentences, 24–96 model tokens; not full documents, arbitrary long context or proven coreference annotations.',
        'Official held-out material is not proof it was unseen during original LLM pretraining.']})
    counts=Counter((r['language'],r['genre'],r['split']) for r in main)
    save(CAMPAIGN/'material_audit.json',{'passed':True,'main':512,'confirmation':128,'model_tokens':sum(len(r['prompt_ids']) for r in main),
      'split_counts':dict(Counter(r['split'] for r in main)),'strata':{'/'.join(k):v for k,v in counts.items()},
      'max_tokens':max(len(r['prompt_ids']) for r in main),'min_tokens':min(len(r['prompt_ids']) for r in main),
      'main_sha':sha(CAMPAIGN/'material.json'),'confirmation_sha':sha(CAMPAIGN/'confirmation_material.json'),
      'selection':'No model responses. Shared grouping and fixed quantile anchors. Confirmation features/states not analyzed until algorithm freeze.'})
    save(CAMPAIGN/'legacy_index.json',{'purpose':'reference-only linked controlled/long-generation families, not newly sampled natural units or joint training',
      'runs':[{'run_id':r,'root':str((PRIOR/r).relative_to(ROOT)),'result_sha':sha(PRIOR/r/'result.json') if (PRIOR/r/'result.json').exists() else None}
        for r in ('i_factorial','k_long','l_aligned_qwen4','l_aligned_qwen14','l_aligned_glm4','m_order','o_generalization','p_token_conditioned')]})
    print('MATERIAL_FROZEN',len(main),len(confirmation),dict(Counter(r['genre'] for r in main)),flush=True)
    stratify()


if __name__=='__main__':main()
