"""Natural bilingual passages, article-held-out partitions, and human-gold QA references."""
import hashlib
import re
import urllib.request
from collections import Counter, defaultdict
from rdc_operator_common import *

SOURCES = {
    'squad_train': ('en', 'train', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'),
    'squad_dev': ('en', 'dev', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'),
    'cmrc_train': ('zh', 'train', 'https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_train.json'),
    'cmrc_dev': ('zh', 'dev', 'https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json'),
}


def normalize(s):
    return re.sub(r'\s+', '', s.casefold())


def fetch(name, spec):
    path = BASE / 'sources' / (name + '.json.gz')
    if path.exists():
        raw = gzip.decompress(path.read_bytes())
    else:
        request = urllib.request.Request(spec[2], headers={'User-Agent': 'RDC-research-source-audit/1.0'})
        raw = urllib.request.urlopen(request, timeout=45).read()
        assert len(raw) < 64 * 1024**2
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(raw, mtime=0))
    return json.loads(raw), {'name': name, 'url': spec[2], 'language': spec[0], 'official_split': spec[1],
        'sha256_raw': hashlib.sha256(raw).hexdigest(), 'sha256_archive': sha(path), 'raw_bytes': len(raw),
        'license': 'CC BY-SA 4.0; underlying Wikipedia attribution remains with source titles and dataset authors',
        'archive': str(path.relative_to(BASE)), 'download_identity': 'Exact downloaded bytes frozen; mutable upstream URL is not a version identifier.'}


def old_texts():
    paths = [PRIOR / 'material.json.gz', PRIOR / 'fresh_material.json.gz',
             PRIOR / 'extension/tail_confirmation/material.json.gz',
             RESULT / 'rdc_relation_dynamics_20260910/material.json',
             RESULT / 'rdc_relation_dynamics_20260910/fresh_material.json',
             RESULT / 'rdc_prefix_atlas_20260910/material_stratified.json',
             RESULT / 'rdc_prefix_atlas_20260910/confirmation_material.json',
             RESULT / 'rdc_prefix_atlas_20260910/full_source_history/fresh_material.json']
    out, evidence = [], []
    for p in paths:
        if not p.exists():
            continue
        data = gzread(p) if p.suffix == '.gz' else read(p)
        out.extend(normalize(r['text']) for r in data)
        evidence.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'records': len(data)})
    return out, evidence


def question_type(text, lang):
    text = text.casefold().strip()
    if lang == 'en':
        for name, pat in [('cause', r'\bwhy\b'), ('manner', r'\bhow\b(?!\s+(?:many|much|long|old))'),
                          ('quantity', r'\bhow\s+(?:many|much|long|old)\b'), ('time', r'\bwhen\b|what year'),
                          ('place', r'\bwhere\b'), ('person', r'\bwho\b|\bwhose\b')]:
            if re.search(pat, text):
                return name
    else:
        for name, pat in [('cause', '为什么|为何|原因'), ('manner', '如何|怎样|怎么'), ('quantity', '多少|几[个次岁]'),
                          ('time', '何时|什么时候|哪年|哪一年'), ('place', '哪里|何处|哪[个一]?地|哪[个一]?国'), ('person', '谁|哪[个位]人')]:
            if re.search(pat, text):
                return name
    return 'other_fact'


def lexical_features(text):
    """Explicit observable cues, not semantic gold or automatically inferred relations."""
    return {'cause': bool(re.search(r'\b(because|therefore|thus|hence|since)\b|因为|因此|由于|所以', text, re.I)),
            'contrast': bool(re.search(r'\b(but|however|although|despite|whereas)\b|但是|然而|虽然|尽管', text, re.I)),
            'negation': bool(re.search(r'\b(not|never|no|neither|without)\b|没有|并非|不是|未能|不能', text, re.I)),
            'reference': bool(re.search(r'\b(he|she|they|their|it|its|these|those)\b|他们|她们|它们|其|该', text, re.I))}


def main():
    if (BASE / 'material_audit.json').exists():
        print('OPERATOR_MATERIAL_ALREADY_FROZEN', flush=True)
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/hf/qwen3-4b', local_files_only=True, use_fast=True)
    corpus, manifest = {}, []
    for name, spec in SOURCES.items():
        data, item = fetch(name, spec)
        corpus[name] = data
        manifest.append(item)
        print('SOURCE_READY', name, len(data['data']), item['raw_bytes'], flush=True)
    old, old_manifest = old_texts()
    old_full = set(old)
    # Every old48-character substring is indexed. New windows scan all48-character substrings.
    shingles = {s[i:i+48] for s in old for i in range(max(0, len(s)-47))}
    rejected, raw_pool, groups_by_lang = Counter(), [], defaultdict(set)
    for name, data in corpus.items():
        lang, part, _ = SOURCES[name]
        for ai, article in enumerate(data['data']):
            title = str(article.get('title') or article.get('id') or f'unnamed-{name}-{ai}')
            group = lang + '/' + normalize(title)
            groups_by_lang[lang, part].add(group)
            for pi, paragraph in enumerate(article['paragraphs']):
                context = paragraph['context']
                fullenc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
                if not 64 <= len(fullenc['input_ids']) <= 768:
                    rejected['paragraph_length'] += 1
                    continue
                # Plain leading natural span; never choose a window around a gold answer.
                end = len(context) if len(fullenc['input_ids']) <= 192 else fullenc['offset_mapping'][191][1]
                text = context[:end]
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                while len(enc['input_ids']) > 192:
                    end = enc['offset_mapping'][-1][0]
                    text = context[:end]
                    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                if len(enc['input_ids']) < 64:
                    rejected['window_length'] += 1
                    continue
                nt = normalize(text)
                if nt in old_full or any(nt[i:i+48] in shingles for i in range(max(0, len(nt)-47))):
                    rejected['prior_exact_or48char_overlap'] += 1
                    continue
                qa = []
                for q in paragraph['qas']:
                    valid = []
                    for a in q.get('answers', []):
                        start = a['answer_start']
                        if context[start:start+len(a['text'])] == a['text'] and 0 < len(tok(a['text'], add_special_tokens=False)['input_ids']) <= 24:
                            valid.append(a)
                        else:
                            rejected['invalid_or_long_answer_annotation'] += 1
                    if valid:
                        qa.append({'question_id': q['id'], 'question': q['question'], 'answers': valid,
                                   'question_type': question_type(q['question'], lang)})
                if not qa:
                    rejected['no_valid_QA'] += 1
                    continue
                sid = f'{name}-{ai:04d}-{pi:03d}'
                raw_pool.append({'source_id': sid, 'source_key': name, 'language': lang, 'official_split': part,
                    'source_group': group, 'title': title, 'source_paragraph_index': pi, 'full_context': context,
                    'text': text, 'window_char_span': [0, end], 'full_context_tokens_Q4': len(fullenc['input_ids']),
                    'prompt_ids': enc['input_ids'], 'token_offsets': enc['offset_mapping'], 'tokens': tok.convert_ids_to_tokens(enc['input_ids']),
                    'annotations': qa, 'lexical_cues': lexical_features(text), 'article_id_available': 'title' in article or 'id' in article})
    # Official development articles are never assigned to fitting, including duplicated titles.
    labels = {}
    for lang in ('en', 'zh'):
        groups = sorted(groups_by_lang[lang, 'train'] - groups_by_lang[lang, 'dev'], key=rank)
        nval = max(1, round(.15 * len(groups)))
        for i, group in enumerate(groups):
            labels[group] = 'validation' if i < nval else 'test' if i < 2*nval else 'train'
        for group in groups_by_lang[lang, 'dev']:
            labels[group] = 'confirmation'
    pools = defaultdict(lambda: defaultdict(list))
    for row in raw_pool:
        if row['official_split'] == 'train' and labels[row['source_group']] == 'confirmation':
            rejected['train_dev_title_overlap'] += 1
            continue
        row['split'] = labels[row['source_group']]
        pools[row['language'], row['split']][row['source_group']].append(row)
    selected, seen_text, eligibility = [], set(), {}
    for lang in ('en', 'zh'):
      for split, count in [('train', 640), ('validation', 128), ('test', 128), ('confirmation', 128)]:
        pool = pools[lang, split]
        docs = sorted(pool, key=rank)
        for values in pool.values():
            values.sort(key=lambda r: rank(r['source_id']))
        got = []
        # Equal round-robin document coverage, at most8 passages per title, before any model outcome.
        for j in range(8):
            for doc in docs:
                if j >= len(pool[doc]):
                    continue
                row = pool[doc][j]
                nt = normalize(row['text'])
                if nt in seen_text:
                    rejected['new_duplicate_window'] += 1
                    continue
                seen_text.add(nt)
                row['sample_id'] = f'{split}-{lang}-o{len(got):04d}'
                n = len(row['prompt_ids'])
                row['anchors'] = [n//3, 2*n//3]
                row['prefix_cues'] = [lexical_features(row['text'][:row['token_offsets'][p][1]]) for p in row['anchors']]
                got.append(row)
                if len(got) == count:
                    break
            if len(got) == count:
                break
        eligibility[lang + '/' + split] = {'documents': len(docs), 'eligible_passages': sum(map(len, pool.values())), 'selected': len(got), 'requested': count}
        if len(got) != count:
            save(BASE / 'material_eligibility_failure.json', {'timestamp': stamp(), 'counts': eligibility, 'rejected': dict(rejected), 'no_model_outputs': True})
            raise RuntimeError(('Outcome-blind material eligibility; no counts silently changed', lang, split, len(got), count))
        selected.extend(got)
    assert len(selected) == 2048 and len({r['source_id'] for r in selected}) == 2048
    split_docs = {s: {r['source_group'] for r in selected if r['split'] == s} for s in ('train', 'validation', 'test', 'confirmation')}
    for a in split_docs:
        for b in split_docs:
            assert a == b or not split_docs[a] & split_docs[b]
    # QA candidates use exactly the same frozen articles, one original human question per passage.
    qa_rows = []
    for lang in ('en', 'zh'):
      for split, n in [('train', 32), ('validation', 16), ('test', 32), ('confirmation', 48)]:
        candidates = [r for r in selected if r['language'] == lang and r['split'] == split]
        used = set()
        for row in sorted(candidates, key=lambda r: rank('qa:' + r['source_id'])):
            if row['source_group'] in used:
                continue
            qs = sorted(row['annotations'], key=lambda q: rank(q['question_id']))
            q = qs[0]
            qa_rows.append({k: row[k] for k in ('sample_id', 'source_id', 'source_group', 'title', 'source_key', 'language', 'split', 'full_context')} | q)
            used.add(row['source_group'])
            if len(used) == n:
                break
        assert len(used) == n, ('Insufficient independent QA documents', lang, split, len(used), n)
    assert len(qa_rows) == 256
    representatives = [r['sample_id'] for lang in ('en', 'zh') for split in ('train', 'validation', 'test', 'confirmation')
                       for r in [s for s in selected if s['language'] == lang and s['split'] == split][:2]]
    compressed(BASE / 'material.json.gz', selected)
    compressed(BASE / 'qa_material.json.gz', qa_rows)
    save(BASE / 'sources/manifest.json', {'timestamp': stamp(), 'sources': manifest, 'old_material_exclusion': old_manifest,
        'attribution': ['SQuAD: Rajpurkar et al. 2016; https://rajpurkar.github.io/SQuAD-explorer/',
                        'CMRC2018: Cui et al. 2019; https://github.com/ymcui/cmrc2018'],
        'license_note': 'This is an adapted subset/window arrangement of CC BY-SA4.0 datasets; preserve source titles/question IDs and attribution. No official benchmark score is claimed.'})
    report = {'timestamp': stamp(), 'source': snapshot(Path(__file__)), 'sources': len(selected), 'QA_questions': len(qa_rows),
        'all_tokens_Q4': sum(len(r['prompt_ids']) for r in selected), 'eligibility': eligibility, 'rejected': dict(rejected),
        'split_documents': {s: len(d) for s, d in split_docs.items()}, 'representative_full_fields': representatives,
        'QA_type_counts': dict(Counter(r['language'] + '/' + r['question_type'] for r in qa_rows)),
        'cue_counts': {c: sum(r['lexical_cues'][c] for r in selected) for c in ('cause', 'contrast', 'negation', 'reference')},
        'material_sha': sha(BASE / 'material.json.gz'), 'QA_sha': sha(BASE / 'qa_material.json.gz'),
        'coverage': 'Natural Wikipedia paragraphs/windows, not templated assertions; human span QA is an independent behavior view. Historical UD/GUM graph identities are linked but not claimed newly annotated on these texts.',
        'limits': ['No guaranteed model-pretraining exclusion.', 'Article-title split plus exact/48-character old overlap control is not complete semantic-family deduplication.',
                   'Window may end before a sentence ends; char span and full paragraph are preserved. QA always receives the complete paragraph.',
                   'Lexical causal/contrast/reference cues are not gold logical or syntactic structure.', 'English and Chinese are independent corpora, not verified meaning-equivalent translations.',
                   'This material alone cannot establish formal multihop depth or grammar-tree correctness.']}
    save(BASE / 'material_audit.json', report)
    print('OPERATOR_MATERIAL_FROZEN', report, flush=True)


if __name__ == '__main__':
    main()
