"""Natural multi-genre language, document-isolated roles and reserved relation combinations.

Gold UD/support edges are analysis labels ONLY. Inference uses actual token prefixes.
No fabricated narrative is substituted for an original corpus passage.
"""
import re
import urllib.request
from collections import Counter, defaultdict
from rdc_law_common import *
from phase2711_rdc_prefix_material import aligned_words
from phase2719_rdc_joint_material import merge, document_annotations, annotate_window
from phase2724_rdc_operator_material import normalize, question_type
from rdc_operator_qa import prompt

HELD_PAIRS = [('obj', 'advcl'), ('nsubj:pass', 'obl')]


def parse_ud(text, bank, part):
    result = []
    document = None
    word_index = sentence_index = 0
    for block in text.replace('\r\n', '\n').strip().split('\n\n'):
        meta, words = {}, []
        for line in block.splitlines():
            if line.startswith('# ') and ' = ' in line:
                k, v = line[2:].split(' = ', 1)
                meta[k] = v
            elif not line.startswith('#'):
                cols = line.split('\t')
                if len(cols) == 10 and cols[0].isdigit():
                    words.append(dict(id=int(cols[0]), form=cols[1], lemma=cols[2], upos=cols[3], xpos=cols[4],
                        features=cols[5], head=int(cols[6]), relation=cols[7], enhanced=cols[8], misc=cols[9]))
        if not words or not meta.get('text'):
            continue
        sid = meta.get('sent_id', '')
        current = meta.get('newdoc id') or sid.rsplit('-', 1)[0]
        if current != document:
            document, word_index, sentence_index = current, 0, 0
        sentence_index += 1
        for w in words:
            word_index += 1
            w.update(doc_word_index=word_index, sentence_id=sid)
        genre = sid.split('_')[1] if bank == 'gum' else sid.split('-')[0].split('_')[0]
        result.append({'source_sentence_id': sid, 'source_group': f'en/{bank}/{document}', 'document_id': document,
            'source_has_document_id': True, 'document_sentence_index': sentence_index, 'source_key': bank+'_'+part,
            'language': 'en', 'treebank': bank, 'official_part': part, 'genre': genre, 'text': meta['text'],
            'retrospective_ud': aligned_words(meta['text'], words), 'component_ids': [sid], 'source_metadata': meta})
    return result


def archive_ud(bank, part, manifest):
    if bank == 'gum':
        path = JOINT / 'sources' / f'gum_{part}.conllu.gz'
        url = f'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.15/en_gum-ud-{part}.conllu'
    else:
        path = BASE / 'sources' / f'ewt_{part}.conllu.gz'
        url = f'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-{part}.conllu'
        if not path.exists():
            raw = urllib.request.urlopen(url, timeout=45).read()
            assert 100_000 < len(raw) < 30*1024**2
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, mtime=0))
    raw = gzip.decompress(path.read_bytes())
    manifest.append({'path': str(path.relative_to(ROOT)), 'archive_sha': sha(path), 'raw_sha': hashlib.sha256(raw).hexdigest(),
        'source': bank, 'official_part': part, 'url': url, 'version': 'UD r2.15',
        'license': 'GUM CC BY-NC-SA4.0 with underlying text terms' if bank == 'gum' else 'EWT annotations/database CC BY-SA4.0, underlying texts retain original authors rights'})
    return parse_ud(raw.decode('utf-8'), bank, part)


def encode(tok, row):
    enc = tok(row['text'], add_special_tokens=False, return_offsets_mapping=True)
    return dict(row, prompt_ids=enc['input_ids'], token_offsets=enc['offset_mapping'], tokens=tok.convert_ids_to_tokens(enc['input_ids']))


def natural_candidates(tok, bank, part, manifest):
    sentences = archive_ud(bank, part, manifest)
    annotations = document_annotations(sentences) if bank == 'gum' else {}
    result = []
    for i, first in enumerate(sentences):
        collected = []
        for r in sentences[i:i+6]:
            if r['source_group'] != first['source_group']:
                break
            collected.append(r)
            joined = merge(collected)
            joined.update(source_key=bank+'_'+part)
            enc = encode(tok, joined)
            n = len(enc['prompt_ids'])
            if n > 192:
                break
            if n < 48:
                continue
            enc['kind'] = 'natural'
            enc['source_id'] = bank+'/'+enc['source_sentence_id']
            # Reuse the actual annotated graph, including nonlocal document links when present.
            annotate_window(enc, annotations)
            types = {w['relation'] for w in enc['retrospective_ud']}
            held = ['+'.join(p) for p in HELD_PAIRS if all(t in types for t in p)]
            enc['held_relation_combinations'] = held
            enc['relation_types'] = sorted(types)
            enc['operation_signature'] = '|'.join(sorted(types))
            result.append(enc)
            break
    return result


def select_grouped(candidates, number, used_texts, used_components, group_limit=8):
    """Balanced round-robin among native documents; no output-driven sampling."""
    bygroup = defaultdict(list)
    for r in candidates:
        bygroup[r['source_group']].append(r)
    for rr in bygroup.values():
        rr.sort(key=lambda r: rank(r['source_id']))
    groups = sorted(bygroup, key=rank)
    selected = []
    progress = True
    counts = Counter()
    while len(selected) < number and progress:
        progress = False
        for g in groups:
            if counts[g] >= group_limit:
                continue
            row = next((r for r in bygroup[g] if normalize(r['text']) not in used_texts and not set(r.get('component_ids', [])) & used_components), None)
            if row is None:
                continue
            selected.append(row)
            used_texts.add(normalize(row['text']))
            used_components.update(row.get('component_ids', []))
            counts[g] += 1
            progress = True
            if len(selected) == number:
                break
    return selected


def encyclopedia_pool(tok, name, manifest):
    path = OPERATOR / 'sources' / (name+'.json.gz')
    data = gzread(path)
    manifest.append({'path': str(path.relative_to(ROOT)), 'archive_sha': sha(path), 'source': name,
        'license': 'CC BY-SA4.0; original Wikipedia/title attribution retained', 'reused_original_source_not_new_download': True})
    lang = 'en' if name.startswith('squad') else 'zh'
    result = []
    for ai, article in enumerate(data['data']):
        title = str(article.get('title') or article.get('id') or f'unnamed-{name}-{ai}')
        for pi, p in enumerate(article['paragraphs']):
            enc = tok(p['context'], add_special_tokens=False, return_offsets_mapping=True)
            n = len(enc['input_ids'])
            if not 64 <= n <= 768:
                continue
            end = len(p['context']) if n <= 192 else enc['offset_mapping'][191][1]
            window = p['context'][:end]
            we = tok(window, add_special_tokens=False, return_offsets_mapping=True)
            while len(we['input_ids']) > 192:
                window = window[:we['offset_mapping'][-1][0]]
                we = tok(window, add_special_tokens=False, return_offsets_mapping=True)
            qs = []
            for q in p['qas']:
                answers = [a for a in q.get('answers', []) if p['context'][a['answer_start']:a['answer_start']+len(a['text'])] == a['text'] and 0 < len(tok(a['text'], add_special_tokens=False)['input_ids']) <= 24]
                if answers:
                    qs.append({'question_id': q['id'], 'question': q['question'], 'answers': answers, 'question_type': question_type(q['question'], lang)})
            if not qs:
                continue
            result.append(encode(tok, {'kind': 'natural', 'text': window, 'full_context': p['context'],
                'window_char_span': [0,len(window)], 'full_context_tokens_Q4': n, 'source_key': name,
                'source_id': f'{name}-{ai:04d}-{pi:03d}', 'source_group': lang+'/'+normalize(title),
                'title': title, 'language': lang, 'genre': 'encyclopedia', 'official_part': name.split('_')[-1],
                'annotations': qs, 'retrospective_ud': [], 'graph': [], 'held_relation_combinations': [],
                'relation_types': [], 'operation_signature': 'unparsed_encyclopedia', 'component_ids': [f'{name}-{ai:04d}-{pi:03d}']}))
    return result


def make_qa(tok, row, q):
    r = {**row, **q, 'kind': 'QA'}
    r['text'] = prompt(r)
    r = encode(tok, r)
    # Explicit chat wrapping is part of the recorded input; no hidden think/answer tokens added.
    wrapped = tok.apply_chat_template([{'role': 'user', 'content': r['text']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    r['user_message'] = r['text']
    r['text'] = wrapped
    r = encode(tok, r)
    r['source_id'] += '/qa/' + q['question_id']
    r['graph'] = [{'type': 'question_requests:'+q['question_type'], 'source': 'question', 'target': 'provided_context', 'scope': 'human_question_type_not_proof_of_computation'}]
    r['operation_signature'] = 'QA/'+q['question_type']
    r['held_relation_combinations'] = []
    return r


def main():
    if (BASE / 'material_audit.json').exists():
        print('LAW_MATERIAL_ALREADY_FROZEN', flush=True)
        return
    from transformers import AutoTokenizer
    assert (BASE/'plan.json').exists()
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b', local_files_only=True, use_fast=True)
    manifest, mainrows, freshrows, eligibility = [], [], [], {}
    used_texts, used_components, assigned_groups = set(), set(), {}
    old_natural = gzread(OPERATOR/'material.json.gz')
    old_groups = {r['source_group'] for r in old_natural}
    old_texts = {normalize(r['text']) for r in old_natural}
    for p in (JOINT/'material.json.gz', JOINT/'fresh_material.json.gz'):
        rr = gzread(p)
        old_groups.update(r['source_group'] for r in rr)
        old_texts.update(normalize(r['text']) for r in rr)
    # Avoid repeating the all-encyclopedia main study. EWT adds five genuinely different natural genres.
    for bank in ('gum', 'ewt'):
        pools = {part:natural_candidates(tok,bank,part,manifest) for part in ('train','dev','test')}
        groups = sorted({r['source_group'] for r in pools['train']}, key=rank)
        ntest = max(1,round(.2*len(groups)))
        testgroups = set(groups[:ntest])
        for split, number, pool in [
            ('train',160,[r for r in pools['train'] if r['source_group'] not in testgroups]),
            ('validation',32,pools['dev']),
            ('test',64,[r for r in pools['train'] if r['source_group'] in testgroups]),
        ]:
            eligible = [r for r in pool if not r['held_relation_combinations'] and normalize(r['text']) not in old_texts]
            chosen = select_grouped(eligible,number,used_texts,used_components)
            eligibility[bank+'/'+split] = {'eligible':len(eligible),'selected':len(chosen),'requested':number,'source_groups':len({r['source_group'] for r in chosen})}
            for r in chosen:
                r.update(split=split, cohort=bank, novelty='new text versus recent2724/2719materials; document history recorded separately')
                mainrows.append(r)
                assigned_groups.setdefault(r['source_group'],split)
                assert assigned_groups[r['source_group']] == split
        for condition, n in [('unseen_combination',32),('new_source_same_components',32)]:
            pool = [r for r in pools['test'] if bool(r['held_relation_combinations']) == (condition=='unseen_combination')
                and normalize(r['text']) not in old_texts and r['source_group'] not in assigned_groups]
            chosen = select_grouped(pool,n,used_texts,used_components)
            eligibility[bank+'/confirmation/'+condition] = {'eligible':len(pool),'selected':len(chosen),'requested':n,'source_groups':len({r['source_group'] for r in chosen})}
            for r in chosen:
                r.update(split='confirmation',cohort=bank,novelty=condition)
                freshrows.append(r)
        for r in freshrows:
            assigned_groups.setdefault(r['source_group'],'confirmation')
    encycl = {name:encyclopedia_pool(tok,name,manifest) for name in ('squad_train','squad_dev','cmrc_train','cmrc_dev')}
    cmrc_train = [r for r in encycl['cmrc_train'] if r['source_group'] not in old_groups]
    groups = sorted({r['source_group'] for r in cmrc_train},key=rank)
    group_split = {g:('validation' if i%10==0 else 'test' if i%10 in (1,2) else 'train') for i,g in enumerate(groups)}
    for split,n in [('train',160),('validation',32),('test',64),('confirmation',64)]:
        pool = ([r for r in cmrc_train if group_split[r['source_group']]==split] if split!='confirmation' else
                [r for r in encycl['cmrc_dev'] if r['source_group'] not in old_groups|set(assigned_groups)])
        chosen = select_grouped(pool,n,used_texts,used_components,group_limit=1)
        eligibility['cmrc/'+split]={'eligible':len(pool),'selected':len(chosen),'requested':n,'source_groups':len(chosen)}
        for r in chosen:
            r.update(split=split,cohort='cmrc',novelty='new article versus recent2724/2719materials')
            (freshrows if split=='confirmation' else mainrows).append(r)
            assigned_groups[r['source_group']]=split
    # Human questions: all-new question IDs, global context-document exclusion and type balancing.
    old_qa = gzread(OPERATOR/'qa_balanced_material.json.gz') + gzread(OPERATOR/'qa_multihop_material.json.gz')
    old_qids = {r['question_id'] for r in old_qa}
    for bank in ('squad','cmrc'):
        candidates = []
        for r in encycl[bank+'_train']:
            if r['source_group'] in assigned_groups:
                continue
            for q in r['annotations']:
                if q['question_id'] not in old_qids:
                    candidates.append((r,q))
        bytype = defaultdict(list)
        for r,q in candidates:
            bytype[q['question_type']].append((r,q))
        for values in bytype.values():
            values.sort(key=lambda x:rank(x[1]['question_id']))
        for split,n in [('train',32),('validation',16),('test',16)]:
            chosen=[]
            while len(chosen)<n:
                progress=False
                for typ in sorted(bytype):
                    pair=next(((r,q) for r,q in bytype[typ] if r['source_group'] not in assigned_groups),None)
                    if pair is None:continue
                    r,q=pair
                    item=make_qa(tok,r,q)
                    if len(item['prompt_ids'])>1100:
                        bytype[typ].remove(pair)
                        progress=True
                        continue
                    item.update(split=split,cohort=bank+'_qa',novelty='new human question; prior article exposure recorded')
                    chosen.append(item);assigned_groups[r['source_group']]=split;progress=True
                    if len(chosen)==n:break
                if not progress:break
            mainrows.extend(chosen)
            eligibility[bank+'_qa/'+split]={'selected':len(chosen),'requested':n,'source_groups':len(chosen)}
    # Original Hotpot distractor context arrays; available count can be less than10.
    import pyarrow.parquet as pq
    hp_path=OPERATOR/'sources/hotpot_distractor_validation_hf.parquet'
    hp=pq.read_table(hp_path).to_pylist()
    manifest.append({'path':str(hp_path.relative_to(ROOT)),'archive_sha':sha(hp_path),'source':'HotpotQA distractor validation',
        'license':'CC BY-SA4.0','attribution':'Yang et al.2018; https://hotpotqa.github.io/'})
    blocked_titles={normalize(r['title']).replace('_','') for r in old_natural if r['language']=='en'}
    blocked_titles.update(normalize(t).replace('_','') for r in old_qa if isinstance(r['title'],list) for t in r['title'])
    blocked_titles.update(normalize(r['title']).replace('_','') for r in mainrows if r['language']=='en' and isinstance(r.get('title'),str))
    counts=Counter()
    for split,n in [('train',32),('validation',16),('test',16)]:
        chosen=[]
        for typ in ('bridge','comparison'):
            for r in sorted((r for r in hp if r['type']==typ),key=lambda x:rank(x['id'])):
                titles={normalize(t).replace('_','') for t in r['context']['title']}
                if titles & blocked_titles or r['id'] in old_qids:continue
                context='\n\n'.join(t+'\n'+''.join(s) for t,s in zip(r['context']['title'],r['context']['sentences']))
                context_n=len(tok(context,add_special_tokens=False)['input_ids'])
                if not 256<=context_n<=1200 or len(tok(r['answer'],add_special_tokens=False)['input_ids'])>24:continue
                item={'kind':'QA','text':context,'full_context':context,'source_id':'hotpot/'+r['id'],'source_group':'hotpot/'+r['id'],
                    'title':r['context']['title'],'language':'en','genre':'multihop_reading','source_key':'hotpot_distractor_validation',
                    'context_paragraphs':[[t,s] for t,s in zip(r['context']['title'],r['context']['sentences'])],
                    'context_document_groups':['en/'+t for t in sorted(titles)],
                    'supporting_facts':[[t,i] for t,i in zip(r['supporting_facts']['title'],r['supporting_facts']['sent_id'])],
                    'retrospective_ud':[],'relation_types':[],'component_ids':[r['id']]}
                item=make_qa(tok,item,{'question_id':r['id'],'question':r['question'],'answers':[{'text':r['answer']}],'question_type':typ})
                item.update(split=split,cohort='hotpot_qa',novelty='new question and all context titles disjoint from prior selected Hotpot/natural English titles')
                chosen.append(item);blocked_titles.update(titles);counts[typ]+=1
                if sum(x['question_type']==typ for x in chosen)==n//2:break
        mainrows.extend(chosen)
        eligibility['hotpot_qa/'+split]={'selected':len(chosen),'requested':n,'source_groups':len(chosen)}
    complete=all(v['selected']==v['requested'] for v in eligibility.values())
    # Freeze even eligibility failures honestly; do not silently change requested counts.
    save(BASE/'material_eligibility.json',{'timestamp':stamp(),'counts':eligibility,'passed':complete,'source':snapshot(Path(__file__))})
    if not complete:
        raise RuntimeError('Material eligibility below frozen target; inspect before revising, no model has run')
    assert len(mainrows)==960 and len(freshrows)==192
    allrows=mainrows+freshrows
    all_context_sets={}
    for i,r in enumerate(allrows):
        r['sample_id']=f'{r["split"]}-{r["cohort"]}-{i:04d}'
        r['prior_document_in_recent_campaigns']=r['source_group'] in old_groups
        r['anchors']=[len(r['prompt_ids'])-1] if r['kind']=='QA' else sorted(set([len(r['prompt_ids'])//3,2*len(r['prompt_ids'])//3,len(r['prompt_ids'])-2]))
        r['future_target_policy']='Actual next corpus token for natural anchors; independent QA answer evaluation, answer tokens never predictor inputs.'
        r['context_document_groups']=r.get('context_document_groups',[r['source_group']])
        for group in r['context_document_groups']:
            all_context_sets.setdefault(group,r['split'])
            assert all_context_sets[group]==r['split'],('Cross-split document leakage',group)
    assert not any(r['held_relation_combinations'] for r in mainrows if r['kind']=='natural')
    primitives=Counter(t for r in mainrows if r['split']=='train' for t in r.get('relation_types',[]))
    assert all(primitives[t]>0 for pair in HELD_PAIRS for t in pair),primitives
    compressed(BASE/'material.json.gz',mainrows)
    compressed(BASE/'confirmation_material.json.gz',freshrows)
    # Freeze materials before confirmation outputs, but do not expose them to fitting.
    save(BASE/'material_audit.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'main_rows':len(mainrows),'confirmation_rows':len(freshrows),
        'main_tokens_Q4':sum(len(r['prompt_ids']) for r in mainrows),'main_anchors':sum(len(r['anchors']) for r in mainrows),
        'counts':dict(Counter(r['split']+'/'+r['cohort'] for r in allrows)),
        'genres':dict(Counter(r['genre'] for r in allrows)),'languages':dict(Counter(r['language'] for r in allrows)),
        'held_relation_pairs':HELD_PAIRS,'held_pair_in_main_natural':0,'fit_primitive_counts':dict(primitives),
        'confirmation_pair_counts':dict(Counter(t for r in freshrows for t in r['held_relation_combinations'])),
        'article_groups':dict(Counter(all_context_sets.values())),
        'prior_document_overlap':dict(Counter(r['split'] for r in allrows if r['prior_document_in_recent_campaigns'])),
        'sources':manifest,'material_sha':sha(BASE/'material.json.gz'),'confirmation_sha':sha(BASE/'confirmation_material.json.gz'),
        'plan_refinement_before_model_outputs':'Natural SQuAD cohort replaced by EWT to expand beyond encyclopedia and enable independent document/source tests: all490 SQuAD1.1 train/dev titles already occurred in Phase2724. SQuAD remains as new human QA, never called previously unseen article. Main counts unchanged. Hotpot retains ALL available original contexts, not an assumed fixed10.',
        'gold_scope':'UD dependency relations and document annotations describe corpus structure retrospectively, not guaranteed semantic operations or gold online conditions. Held pair means co-occurrence of two dependency types in a natural window, not a proven algebraic composition of mental operations.',
        'limits':['No pretraining exclusion claim.','Only recent2724/2719 exact windows checked for historical exposure; older project phases can contain related text.',
            'No assertion that syntactic types capture all logical scopes.','English two corpora, Chinese encyclopedia; language and genre not factorially balanced.',
            'QA models may have seen source facts during pretraining.','Full-context QA differs in length and instruction structure from raw natural continuations.'],
        'representative_full_fields':[r['sample_id'] for cohort in sorted({r['cohort'] for r in allrows}) for r in [s for s in allrows if s['cohort']==cohort][:2]],
        'seconds':time.monotonic()-start})
    ledger('material_freeze',time.monotonic()-start)
    print('LAW_MATERIAL_FROZEN',len(mainrows),len(freshrows),Counter(r['genre'] for r in allrows),flush=True)


if __name__=='__main__':
    main()
