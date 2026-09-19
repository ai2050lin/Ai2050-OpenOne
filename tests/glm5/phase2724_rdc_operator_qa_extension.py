"""Outcome-blind breadth correction: balance original questions and add genuine annotated multi-hop data."""
import urllib.request
from collections import Counter, defaultdict
from rdc_operator_common import *
from phase2724_rdc_operator_material import normalize


def main():
    if (BASE / 'qa_extension.json').exists():
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/hf/qwen3-4b', local_files_only=True, use_fast=True)
    material = rows()
    balanced = []
    types = ['cause', 'manner', 'quantity', 'time', 'place', 'person', 'other_fact']
    for lang in ('en', 'zh'):
      for split, count in [('train', 32), ('validation', 16), ('test', 32), ('confirmation', 48)]:
        groups = defaultdict(list)
        for row in material:
            if row['language'] != lang or row['split'] != split:
                continue
            for q in row['annotations']:
                groups[q['question_type']].append((row, q))
        for candidates in groups.values():
            candidates.sort(key=lambda rq: rank('balanced:'+rq[1]['question_id']))
        used, got = set(), []
        while len(got) < count:
            progress = False
            for typ in types:
                candidate = next(((r, q) for r, q in groups[typ] if r['source_group'] not in used), None)
                if candidate is None:
                    continue
                row, q = candidate
                got.append({k: row[k] for k in ('sample_id', 'source_id', 'source_group', 'title', 'source_key', 'language', 'split', 'full_context')} | q)
                used.add(row['source_group'])
                progress = True
                if len(got) == count:
                    break
            assert progress
        balanced.extend(got)
    compressed(BASE / 'qa_balanced_material.json.gz', balanced)
    # A separate transfer set supplies genuine support annotations; question types/cues alone do not establish hops.
    url = 'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json'
    source = BASE / 'sources/hotpot_distractor_validation_hf.parquet'
    mirror_url = 'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet'
    expected_sha = 'c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6'
    if not source.exists():
        raw = urllib.request.urlopen(mirror_url, timeout=45).read()
        assert len(raw) == 27452575
        source.write_bytes(raw)
    assert sha(source) == expected_sha, 'Downloaded LFS identity must match independently observed HF tree entry'
    import pyarrow.parquet as pq
    converted = pq.read_table(source).to_pylist()
    source_data = [{**r, '_id': r['id'], 'context': list(zip(r['context']['title'], r['context']['sentences'])),
                    'supporting_facts': list(zip(r['supporting_facts']['title'], r['supporting_facts']['sent_id']))} for r in converted]
    used_titles = {normalize(r['title']).replace('_', '') for r in material if r['language'] == 'en'}
    selected, rejected = [], Counter()
    for typ in ('bridge', 'comparison'):
        got = 0
        for row in sorted([r for r in source_data if r['type'] == typ], key=lambda r: rank(r['_id'])):
            titles = {normalize(t).replace('_', '') for t, sentences in row['context']}
            if titles & used_titles:
                rejected['overlapping_context_title'] += 1
                continue
            context = '\n\n'.join(title + '\n' + ''.join(sentences) for title, sentences in row['context'])
            ntokens = len(tok(context, add_special_tokens=False)['input_ids'])
            if not 256 <= ntokens <= 1536 or len(tok(row['answer'], add_special_tokens=False)['input_ids']) > 24:
                rejected['context_or_answer_length'] += 1
                continue
            mapping = dict(row['context'])
            if not all(t in mapping and 0 <= i < len(mapping[t]) for t, i in row['supporting_facts']):
                rejected['invalid_support_annotation'] += 1
                continue
            selected.append({'sample_id': 'hotpot-' + row['_id'], 'source_id': row['_id'], 'source_group': 'hotpot/' + row['_id'],
                'source_key': 'hotpot_dev_distractor', 'language': 'en', 'split': 'external_transfer', 'title': [t for t, s in row['context']],
                'question_id': row['_id'], 'question': row['question'], 'answers': [{'text': row['answer']}], 'full_context': context,
                'context_paragraphs': row['context'], 'supporting_facts': row['supporting_facts'], 'question_type': typ,
                'level': row['level'], 'full_context_tokens_Q4': ntokens,
                'scope': 'Original10-paragraph distractor condition, not gold-support-only oracle. Support labels only for analysis, not supplied as input.'})
            used_titles.update(titles)
            got += 1
            if got == 64:
                break
        assert got == 64, ('Insufficient title-disjoint Hotpot examples', typ, got)
    compressed(BASE / 'qa_multihop_material.json.gz', selected)
    report = {'timestamp': stamp(), 'source': snapshot(Path(__file__)), 'no_native_QA_outputs_seen_at_revision': True,
        'original_random256_preserved_sha': sha(BASE / 'qa_material.json.gz'), 'balanced256_sha': sha(BASE / 'qa_balanced_material.json.gz'),
        'primary_QA_selection': 'Balanced256 replaces random256 for primary execution before any QA model result; original random material is retained, not falsely reported executed.',
        'reason': 'Original random question selection yielded only1English why-question. Rotate original human question types within each language/split, one title per stratum; no wording or answer invented.',
        'balanced_type_counts': dict(Counter(r['language'] + '/' + r['question_type'] for r in balanced)),
        'multihop128_sha': sha(BASE / 'qa_multihop_material.json.gz'), 'multihop_source_sha': sha(source), 'multihop_source_url': url,
        'download_fallback': {'actual_url': mirror_url, 'actual_format': 'Parquet in hotpotqa HF namespace; exact LFS SHA validated',
            'failure': 'Original HTTPS45s SSL handshake timeout and independent HTTP15s HEAD timeout; no bypass of TLS checks.',
            'conversion': 'id -> _id; context title/sentence parallel arrays and support title/sent_id arrays zipped back to published schema; original Parquet retained.',
            'original_JSON_byte_equality': 'Not verified because original host unavailable; do not claim a bitwise mirror of original JSON.'},
        'multihop_type_counts': dict(Counter(r['question_type'] for r in selected)), 'multihop_rejected': dict(rejected),
        'resource_change': '384primary4B QA sources rather than256; fixed48-token content budget, no change to21600second/6GiB campaign envelopes; large-model64 QA design unchanged.',
        'reference': 'Yang et al.2018 HotpotQA; https://hotpotqa.github.io/; dataset CC BY-SA4.0, source question IDs and paragraph titles retained.',
        'limits': ['Length-bounded subset is not an official benchmark score.', 'No proof every annotated multi-hop question requires every support sentence for this model.',
                   'Title-disjoint selection is not guaranteed absence from pretraining or semantic duplicate exclusion.', 'Longer Hotpot contexts differ in domain and execution shape from natural192-token windows.']}
    save(BASE / 'qa_extension.json', report)
    print('QA_BREADTH_FROZEN', report, flush=True)


if __name__ == '__main__':
    main()
