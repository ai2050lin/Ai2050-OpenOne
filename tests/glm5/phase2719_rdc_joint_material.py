"""Independent natural windows with document-scoped entities/discourse and strict prefix boundaries."""
import re
import hashlib
import urllib.request
from collections import Counter, defaultdict
from rdc_joint_common import *
from rdc_relation_common import canonical, normalized, construction
from phase2711_rdc_prefix_material import aligned_words

SOURCES = {
    'gum_train': ('en', 'gum', 'train', 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.15/en_gum-ud-train.conllu'),
    'gum_dev': ('en', 'gum', 'dev', 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.15/en_gum-ud-dev.conllu'),
    'gum_test': ('en', 'gum', 'test', 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.15/en_gum-ud-test.conllu'),
    'pud_test': ('zh', 'pud', 'test', 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-PUD/r2.15/zh_pud-ud-test.conllu'),
}


def rank(value):
    return hashlib.sha256(('2719:' + value).encode('utf-8')).hexdigest()


def source_text(name, manifest):
    language, treebank, part, url = SOURCES[name]
    dest = BASE / 'sources' / (name + '.conllu.gz')
    if dest.exists():
        raw = gzip.decompress(dest.read_bytes())
    else:
        guard(8 * 1024**2)
        raw = urllib.request.urlopen(url, timeout=45).read()
        assert len(raw) < 24 * 1024**2
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    text = raw.decode('utf-8')
    assert '# sent_id = ' in text and '\t' in text
    manifest.append({'name': name, 'language': language, 'treebank': treebank, 'official_part': part,
        'url': url, 'version': 'UD r2.15', 'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
        'stored_path': str(dest.relative_to(BASE)), 'gzip_sha256': sha(dest),
        'license': 'CC BY-NC-SA4.0; retain individual underlying text terms' if treebank == 'gum' else 'CC BY-SA3.0; underlying text/translation terms as original treebank',
        'official_role': 'PUD reserved exclusively for confirmation; never used to train a relation parser or state predictor' if treebank == 'pud' else 'Official document splits preserved; additional train-document holdout for main test'})
    return parse(text, language, treebank, part)


def parse(text, language, treebank, part):
    out = []
    document = None
    doc_index = 0
    sentence_index = 0
    for block in text.replace('\r\n', '\n').strip().split('\n\n'):
        meta, words = {}, []
        for line in block.splitlines():
            if line.startswith('# ') and ' = ' in line:
                key, value = line[2:].split(' = ', 1)
                meta[key] = value
            elif not line.startswith('#'):
                c = line.split('\t')
                if len(c) == 10 and c[0].isdigit():
                    words.append({'id': int(c[0]), 'form': c[1], 'lemma': c[2], 'upos': c[3], 'xpos': c[4],
                                  'features': c[5], 'head': int(c[6]), 'relation': c[7], 'enhanced': c[8], 'misc': c[9]})
        if not words or not meta.get('text'):
            continue
        sid = meta.get('sent_id', '')
        stated_doc = meta.get('newdoc id')
        if treebank == 'gum':
            inferred_doc = sid.rsplit('-', 1)[0]
            this_document = stated_doc or inferred_doc
        else:
            # PUD IDs encode news/wiki and source language, not a reliable original article ID.
            this_document = sid
        if this_document != document:
            document, doc_index, sentence_index = this_document, 0, 0
        for word in words:
            doc_index += 1
            word['doc_word_index'] = doc_index
            word['sentence_id'] = sid
        sentence_index += 1
        group = f'{language}/{treebank}/{document}'
        genre = sid.split('_')[1] if treebank == 'gum' and len(sid.split('_')) > 2 else ('news' if sid.startswith('n') else 'wikipedia')
        out.append({'source_sentence_id': sid, 'source_group': group, 'source_has_document_id': treebank == 'gum',
            'document_id': document if treebank == 'gum' else None, 'document_sentence_index': sentence_index,
            'language': language, 'treebank': treebank, 'official_part': part, 'genre': genre,
            'text': meta['text'], 'retrospective_ud': aligned_words(meta['text'], words), 'component_ids': [sid],
            'source_metadata': {k: v for k, v in meta.items() if k != 'text'}})
    return out


def merge(sentences):
    assert sentences and len({r['source_group'] for r in sentences}) == 1
    result = dict(sentences[0])
    words, components, parts = [], [], []
    id_offset = char_offset = 0
    for row in sentences:
        for word in row['retrospective_ud']:
            w = dict(word)
            w['id'] += id_offset
            w['head'] = w['head'] + id_offset if w['head'] else 0
            w['char_span'] = [x + char_offset for x in w['char_span']] if w['char_span'] else None
            words.append(w)
        id_offset += max(w['id'] for w in row['retrospective_ud'])
        char_offset += len(row['text']) + 1
        parts.append(row['text'])
        components.extend(row['component_ids'])
    result.update(text=' '.join(parts), retrospective_ud=words, component_ids=components,
                  source_sentence_id='..'.join(components))
    return result


def misc_map(word):
    return dict(item.split('=', 1) for item in word['misc'].split('|') if '=' in item)


def document_annotations(sentences):
    """Parse documented GUM brackets and EDU starts; no inferred geometry or new gold labels."""
    groups = defaultdict(list)
    for r in sentences:
        groups[r['source_group']].extend(r['retrospective_ud'])
    output = {}
    for group, words in groups.items():
        mentions, stack, discourse, edu_starts, constructs = [], [], [], {}, []
        for word in words:
            wi = word['doc_word_index']
            misc = misc_map(word)
            for match in re.finditer(r'\((\d+)([^()]*)|(\d*)\)', misc.get('Entity', '')):
                if match[1] is not None:
                    attrs = match[2].lstrip('-')
                    stack.append({'entity_id': match[1], 'start_doc_word': wi, 'attributes': attrs,
                                  'entity_type': attrs.split('-')[0] if attrs else 'unknown'})
                elif stack:
                    entity_id = match[3]
                    choices = [i for i, value in enumerate(stack) if not entity_id or value['entity_id'] == entity_id]
                    if choices:
                        mention = stack.pop(choices[-1])
                        mention['end_doc_word'] = wi
                        mentions.append(mention)
            for item in misc.get('Discourse', '').split(';'):
                m = re.match(r'([^:]+):(\d+)(?:->(\d+))?:', item)
                if m:
                    edu_starts[int(m[2])] = wi
                    if m[3] is not None:
                        discourse.append({'type': 'discourse:' + m[1], 'dependent_edu': int(m[2]),
                                          'parent_edu': int(m[3]), 'raw_annotation': item})
            if 'Cxn' in misc:
                constructs.append({'doc_word': wi, 'type': misc['Cxn']})
        starts = sorted(edu_starts.items(), key=lambda item: item[1])
        edu_spans = {eid: [start, starts[i+1][1]-1 if i+1 < len(starts) else words[-1]['doc_word_index']]
                     for i, (eid, start) in enumerate(starts)}
        output[group] = {'mentions': sorted(mentions, key=lambda m: (m['end_doc_word'], m['start_doc_word'])),
                         'discourse': discourse, 'edu_spans': edu_spans, 'constructions': constructs,
                         'unclosed_mentions': len(stack)}
    return output


def endpoint(row, word):
    span = word.get('char_span')
    if not span:
        return None
    hits = [i for i, (a, b) in enumerate(row['token_offsets']) if b > a and b > span[0] and a < span[1]]
    return hits[-1] if hits else None


def annotate_window(row, annotations):
    words = row['retrospective_ud']
    by_doc = {w['doc_word_index']: w for w in words}
    by_id = {w['id']: w for w in words}
    graph = []
    for w in words:
        if w['head'] and w['head'] in by_id:
            a, b = endpoint(row, w), endpoint(row, by_id[w['head']])
            if a is not None and b is not None and a != b:
                graph.append({'type': 'ud:' + w['relation'], 'dependent_token': a, 'head_token': b,
                              'available_after_token': max(a, b), 'scope': 'retrospective_gold_not_online_input'})
    record = annotations.get(row['source_group'], {})
    mentions = [m for m in record.get('mentions', []) if m['start_doc_word'] in by_doc and m['end_doc_word'] in by_doc]
    last = {}
    retained_mentions = []
    for m in mentions:
        end = endpoint(row, by_doc[m['end_doc_word']])
        begin = by_doc[m['start_doc_word']]['char_span']
        finish = by_doc[m['end_doc_word']]['char_span']
        if end is None or not begin or not finish:
            continue
        r = {**m, 'end_token': end, 'char_span': [begin[0], finish[1]], 'text': row['text'][begin[0]:finish[1]]}
        eid = m['entity_id']
        if eid in last and last[eid]['end_token'] < end:
            link = 'predication' if '-pred' in m['attributes'] else 'anaphora' if '-ana' in m['attributes'] else 'coreference'
            graph.append({'type': 'entity:' + link, 'dependent_token': end, 'head_token': last[eid]['end_token'],
                          'available_after_token': end, 'entity_id': eid, 'entity_type': m['entity_type'],
                          'dependent_span': r['char_span'], 'head_span': last[eid]['char_span'],
                          'scope': 'retrospective_document_annotation_not_online_input'})
        last[eid] = r
        retained_mentions.append(r)
    unresolved = 0
    for d in record.get('discourse', []):
        a = record['edu_spans'].get(d['dependent_edu'])
        b = record['edu_spans'].get(d['parent_edu'])
        if not a or not b or not all(x in by_doc for x in a+b):
            if a and a[0] in by_doc:
                unresolved += 1
            continue
        u, v = endpoint(row, by_doc[a[1]]), endpoint(row, by_doc[b[1]])
        if u is None or v is None or u == v:
            continue
        graph.append({**d, 'dependent_token': u, 'head_token': v, 'dependent_doc_span': a, 'head_doc_span': b,
                      'available_after_token': max(u, v), 'scope': 'retrospective_complete_EDU_pair_not_online_input'})
    cxn = [c for c in record.get('constructions', []) if c['doc_word'] in by_doc]
    family = {'syntax'}
    if retained_mentions:
        family.add('entity_reference')
    if any(g['type'] == 'entity:predication' for g in graph):
        family.add('annotated_predication')
    if any(g['type'].startswith(('discourse:causal', 'discourse:contingency', 'discourse:explanation', 'discourse:purpose')) for g in graph):
        family.add('discourse_reason_relation')
    if any(g['type'].startswith('discourse:adversative') for g in graph):
        family.add('discourse_contrast')
    if '?' in row['text'] or '？' in row['text']:
        family.add('question_form')
    if any('Mood=Imp' in w['features'] for w in words) or row['genre'] == 'howto':
        family.add('instruction_form')
    if re.search(r'\b(?:not|never|no|without|neither)\b|不|未|没有|沒有|無|无', row['text'], flags=re.I):
        family.add('lexical_negation_cue')
    row.update(retrospective_graph=graph, entity_mentions=retained_mentions, annotated_constructions=cxn,
               unresolved_discourse_edges_outside_window=unresolved, language_mode_families=sorted(family),
               family_scope='Overlapping annotated/cue conditions, not proof of native knowledge or reasoning ability. Gold labels never silently enter forecast features.')
    return row


def old_identities():
    old = old_rows()
    for rel in ('material_stratified.json', 'confirmation_material.json', 'full_source_history/fresh_material.json'):
        old.extend(read(PREFIX / rel))
    texts, nums, skeletons, components = set(), set(), set(), set()
    for r in old:
        texts.add(normalized(r['text']))
        nums.add(canonical(r['text']))
        skeletons.add(construction(r))
        texts.update(r.get('normalized_texts', []))
        nums.update(r.get('numeric_families', []))
        skeletons.update(r.get('construction_families', []))
        components.update((r['language'], x) for x in r.get('component_ids', [r['source_sentence_id']]))
    return texts, nums, skeletons, components


def assign_gum_documents(sentences):
    genres = defaultdict(set)
    for r in sentences:
        genres[r['genre']].add(r['source_group'])
    labels = {}
    for genre, documents in genres.items():
        ordered = sorted(documents, key=rank)
        held = max(1, round(len(ordered) * .23)) if len(ordered) >= 4 else 0
        for i, doc in enumerate(ordered):
            labels[doc] = 'test' if i < held else 'train'
    return labels


def candidates(tok, sentences, labels, excluded, rejected):
    text_seen, num_seen, skeleton_seen, component_seen = excluded
    grouped = defaultdict(list)
    for r in sentences:
        grouped[r['source_group']].append(r)
    out = defaultdict(lambda: defaultdict(list))
    for group, units in grouped.items():
        label = labels[group]
        for i, first in enumerate(units):
            widths = [2, 3, 1] if first['treebank'] == 'gum' else [1]
            for width in widths:
                chunk = units[i:i+width]
                if len(chunk) != width:
                    continue
                row = merge(chunk)
                if any((row['language'], sid) in component_seen for sid in row['component_ids']):
                    rejected['old_component'] += 1
                    continue
                nt = {normalized(r['text']) for r in chunk} | {normalized(row['text'])}
                nu = {canonical(r['text']) for r in chunk} | {canonical(row['text'])}
                sk = {construction(r) for r in chunk} | {construction(row)}
                if nt & text_seen or nu & num_seen or sk & skeleton_seen:
                    rejected['old_text_numeric_or_skeleton'] += 1
                    continue
                if sum(w['char_span'] is None for w in row['retrospective_ud']) > .05 * len(row['retrospective_ud']):
                    rejected['word_alignment'] += 1
                    continue
                if row['treebank'] == 'gum' and (row['genre'] == 'reddit' or re.search(r'_{4,}', row['text'])):
                    rejected['redacted_or_reddit'] += 1
                    continue
                enc = tok(row['text'], add_special_tokens=False, return_offsets_mapping=True)
                if not 24 <= len(enc['input_ids']) <= 128:
                    rejected['token_length'] += 1
                    continue
                row.update(prompt_ids=enc['input_ids'], token_offsets=enc['offset_mapping'], tokens=tok.convert_ids_to_tokens(enc['input_ids']),
                           numeric_families=sorted(nu), normalized_texts=sorted(nt), construction_families=sorted(sk), split=label)
                out[label][group].append(row)
                break
    for groups in out.values():
        for vals in groups.values():
            vals.sort(key=lambda r: rank(r['source_sentence_id']))
    return out


def take(pool, number, excluded):
    text_seen, num_seen, skeleton_seen, component_seen = excluded
    selected = []
    groups = sorted(pool, key=rank)
    genres = defaultdict(list)
    for group in groups:
        genres[pool[group][0]['genre']].append(group)
    per_group = Counter()
    for round_index in range(8):
        for gi in range(max(map(len, genres.values()), default=0)):
            for genre in sorted(genres):
                if gi >= len(genres[genre]):
                    continue
                group = genres[genre][gi]
                for row in pool[group]:
                    if any((row['language'], sid) in component_seen for sid in row['component_ids']):
                        continue
                    if set(row['normalized_texts']) & text_seen or set(row['numeric_families']) & num_seen or set(row['construction_families']) & skeleton_seen:
                        continue
                    selected.append(row)
                    text_seen.update(row['normalized_texts'])
                    num_seen.update(row['numeric_families'])
                    skeleton_seen.update(row['construction_families'])
                    component_seen.update((row['language'], sid) for sid in row['component_ids'])
                    per_group[group] += 1
                    break
                if len(selected) == number:
                    return selected
    raise RuntimeError(('Insufficient outcome-blind material', number, len(selected), len(pool), dict(per_group)))


def main():
    guard(16 * 1024**2)
    if material_path().exists() and material_path(True).exists():
        print('JOINT_MATERIAL_ALREADY_FROZEN', flush=True)
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/hf/qwen3-4b', local_files_only=True, use_fast=True)
    manifest = []
    corpus = {name: source_text(name, manifest) for name in SOURCES}
    gsd = PREFIX / 'sources/zh_train.conllu'
    corpus['gsd_train'] = parse(gsd.read_text(encoding='utf-8'), 'zh', 'gsd', 'train')
    manifest.append({'name': 'gsd_train', 'language': 'zh', 'treebank': 'gsd', 'official_part': 'train',
                     'path': str(gsd.relative_to(ROOT)), 'raw_sha256': sha(gsd), 'reused_without_copy': True,
                     'version': 'UD r2.15', 'role': 'Only never-selected text/component/numeric/skeleton identities remain eligible'})
    annotations = {}
    for name, sentences in corpus.items():
        if name.startswith('gum'):
            annotations.update(document_annotations(sentences))
    excluded = old_identities()
    rejected = Counter()
    main_rows, fresh = [], []
    gum_labels = assign_gum_documents(corpus['gum_train'])
    labels = {'gum_train': gum_labels,
              'gum_dev': {r['source_group']: 'validation' for r in corpus['gum_dev']},
              'gum_test': {r['source_group']: 'confirmation' for r in corpus['gum_test']},
              'pud_test': {r['source_group']: 'confirmation' for r in corpus['pud_test']},
              'gsd_train': {r['source_group']: 'main' for r in corpus['gsd_train']}}
    requested = {'gum_train': {'train': 160, 'test': 48}, 'gum_dev': {'validation': 48},
                 'gum_test': {'confirmation': 128}, 'pud_test': {'confirmation': 128}, 'gsd_train': {'main': 256}}
    eligibility = {}
    for name in ('gum_train', 'gum_dev', 'gsd_train', 'gum_test', 'pud_test'):
        pool = candidates(tok, corpus[name], labels[name], excluded, rejected)
        eligibility[name] = {label: {'groups': len(groups), 'candidate_windows': sum(len(v) for v in groups.values())} for label, groups in pool.items()}
        for label, count in requested[name].items():
            selected = take(pool[label], count, excluded)
            if name == 'gsd_train':
                # All component/content families are already disjoint; assign deterministic exact counts.
                for i, row in enumerate(selected):
                    row['split'] = 'train' if i < 160 else 'validation' if i < 208 else 'test'
            for row in selected:
                n = len(row['prompt_ids'])
                anchors = [n//3, 2*n//3]
                row.update(anchors=anchors, positions=[0, 1, anchors[0], anchors[0]+1, anchors[1], anchors[1]+1],
                           material_kind='natural_adjacent_document_window' if len(row['component_ids']) > 1 else 'natural_sentence',
                           online_scope='Only actual prefix token IDs/positions and explicitly supplied earlier native state; full gold entities/discourse/Cxn are analysis labels, not forecast inputs.',
                           source_key=name)
                annotate_window(row, annotations)
                (fresh if label == 'confirmation' else main_rows).append(row)
    for collection in (main_rows, fresh):
        counters = Counter()
        for row in sorted(collection, key=lambda r: (r['language'], rank(r['source_sentence_id']))):
            label = row['split'] + '-' + row['language']
            row['sample_id'] = f'{label}-j{counters[label]:04d}'
            counters[label] += 1
        collection.sort(key=lambda r: (r['sample_id'][-4:], r['split'], r['language']))
    all_rows = main_rows + fresh
    split_groups = {s: {r['source_group'] for r in all_rows if r['split'] == s} for s in ('train', 'validation', 'test', 'confirmation')}
    for a in split_groups:
        for b in split_groups:
            if a != b:
                assert not split_groups[a] & split_groups[b]
    assert len(main_rows) == 512 and len(fresh) == 256
    assert Counter(r['split'] for r in main_rows) == {'train': 320, 'validation': 96, 'test': 96}
    assert len({r['sample_id'] for r in all_rows}) == 768
    compressed_json(material_path(), main_rows)
    compressed_json(material_path(True), fresh)
    save(BASE / 'sources/manifest.json', {'timestamp': stamp(), 'files': manifest,
        'documentation': ['https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.15/README.md',
                          'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-PUD/r2.15/README.md'],
        'pud_limit': 'Human-translated news/wiki sentences; not guaranteed adjacent or independent original articles. PUD used only for confirmation, never predictor training.',
        'gum_limit': 'Gold entities/discourse may depend on full document; only completed in-window spans enter retrospective comparison. They never become online known inputs by visibility alone.'})
    audit = {'timestamp': stamp(), 'passed': True, 'source': snapshot(Path(__file__)), 'main_units': len(main_rows), 'fresh_units': len(fresh),
             'main_tokens': sum(len(r['prompt_ids']) for r in main_rows), 'fresh_tokens': sum(len(r['prompt_ids']) for r in fresh),
             'split_units': dict(Counter(r['split'] for r in all_rows)), 'split_declared_groups': {k: len(v) for k, v in split_groups.items()},
             'strata': {'/'.join(k): v for k, v in Counter((r['split'], r['language'], r['genre']) for r in all_rows).items()},
             'language_modes': dict(Counter(f for r in all_rows for f in r['language_mode_families'])),
             'retrospective_relation_types': dict(Counter(g['type'] for r in all_rows for g in r['retrospective_graph'])),
             'entity_mentions': sum(len(r['entity_mentions']) for r in all_rows), 'constructions': sum(len(r['annotated_constructions']) for r in all_rows),
             'adjacent_windows': sum(len(r['component_ids']) > 1 for r in all_rows),
             'eligibility': eligibility, 'rejected': dict(rejected), 'material_sha': sha(material_path()), 'fresh_material_sha': sha(material_path(True)),
             'no_new_model_outputs_used': True,
             'limits': ['Not complete semantic/paraphrase family exclusion or model-pretraining exclusion.', 'GUM has multiple nonoverlapping windows per document; use document clusters for uncertainty.',
                        'Chinese GSD/PUD lack verified article groups; sentence/content groups are not guaranteed independent documents.',
                        'Fresh PUD differs from GSD in annotation, translation and domains; generalization failure cannot be attributed only to language.',
                        'Annotated predication/discourse relations do not measure native factual correctness or multistep reasoning success.']}
    save(BASE / 'material_audit.json', audit)
    save(BASE / 'material_index.json', {'main': [{k: r[k] for k in ('sample_id', 'split', 'language', 'genre', 'source_group', 'language_mode_families')} for r in main_rows],
                                      'fresh': [{k: r[k] for k in ('sample_id', 'split', 'language', 'genre', 'source_group', 'language_mode_families')} for r in fresh]})
    print('JOINT_MATERIAL_FROZEN', audit, 'bytes', usage(), flush=True)


if __name__ == '__main__':
    main()
