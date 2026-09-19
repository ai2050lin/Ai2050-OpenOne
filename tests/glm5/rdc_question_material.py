"""Pure source adapters for prospective natural multi-question experiments.

No download, model execution, result registration, or stage completion occurs
on import. Corpus schemas were inspected at their primary-source readers;
these adapters preserve original text/annotations instead of executing a
third-party dataset-loading script. Dataset goals are NOT mechanism labels.
"""
from collections import Counter, defaultdict
import hashlib
import json
import unicodedata
from urllib.parse import unquote, urlsplit


SOURCE_REFERENCES = {
    'quoref': {
        'publisher': 'https://huggingface.co/datasets/allenai/quoref',
        'schema': 'https://huggingface.co/datasets/allenai/quoref/blob/main/quoref.py',
        'data': 'https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip',
        'license': 'CC-BY-4.0',
    },
    'drop': {
        'publisher': 'https://github.com/allenai/allennlp-website/blob/master/drop.html',
        'schema': 'https://raw.githubusercontent.com/allenai/allennlp-reading-comprehension/master/allennlp_rc/dataset_readers/drop.py',
        'scoring': 'https://raw.githubusercontent.com/allenai/allennlp-reading-comprehension/master/allennlp_rc/eval/drop_eval.py',
        'data': 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip',
        'license': 'CC-BY-SA-4.0 (original publisher; derivative cards may differ)',
    },
}


def text_id(text):
    assert isinstance(text, str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical_title(value):
    """Conservative matching key only; never replace the actual model text."""
    return ' '.join(unicodedata.normalize('NFC', unquote(value)).replace('_', ' ').split()).casefold()


def article_identity(title, url):
    parts = urlsplit(url)
    if parts.hostname in {'en.wikipedia.org', 'en.m.wikipedia.org'} and parts.path.startswith('/wiki/'):
        return 'enwiki:' + canonical_title(parts.path[len('/wiki/'):]), 'publisher_wikipedia_url'
    if title.strip():
        return 'enwiki_title:' + canonical_title(title), 'publisher_title_without_verified_url'
    return None, 'missing_article_identity'


def quoref_rows(dataset, split, excluded_question_ids=()):
    """One row per human question; all spans in an annotation are required.

    answer_start offsets are checked against the untrimmed original context.
    The source SQuAD-like array is not interpreted as SQuAD-style alternatives.
    The complete raw question object is retained alongside checked spans.
    """
    assert isinstance(dataset['data'], list)
    seen = set()
    excluded_question_ids = set(excluded_question_ids)
    for article_number, article in enumerate(dataset['data']):
        title, url = article.get('title', ''), article.get('url', '')
        article_id, article_scope = article_identity(title, url)
        for paragraph_number, paragraph in enumerate(article['paragraphs']):
            passage = paragraph['context']
            assert isinstance(passage, str) and passage.strip()
            context_id = 'quoref_context_' + text_id(passage)
            for qa in paragraph['qas']:
                qid = str(qa['id'])
                if qid in excluded_question_ids:
                    continue
                assert qid not in seen, ('Repeated source question ID', split, qid)
                seen.add(qid)
                question = qa['question']
                assert isinstance(question, str) and question.strip()
                answer = qa['answers']
                assert isinstance(answer, list) and answer
                checked = []
                for span in answer:
                    begin, content = span['answer_start'], span['text']
                    assert isinstance(begin, int) and isinstance(content, str) and content
                    end = begin + len(content)
                    assert begin >= 0 and passage[begin:end] == content, ('Source answer offset mismatch', qid, begin, content)
                    checked.append({'start': begin, 'end_exclusive': end, 'text': content})
                yield {
                    'question_id': 'quoref:' + qid, 'original_question_id': qid,
                    'context_id': context_id, 'context_text_sha256': text_id(passage),
                    'article_id': article_id, 'article_identity_scope': article_scope,
                    'title': title, 'url': url, 'passage': passage, 'question': question,
                    'source_split': split, 'cohort': 'quoref', 'language': 'en',
                    'source_locator': {'article': article_number, 'paragraph': paragraph_number},
                    'answer_annotations': [{'required_spans': [s['text'] for s in checked],
                                            'type': 'span' if len(checked) == 1 else 'multiple_required_spans'}],
                    'answer_source_spans': checked, 'raw_question_annotation': qa,
                    'family_scope': 'Human reading-comprehension question from a coreference-oriented corpus; '
                                    'individual coreference chain not inferred from corpus membership.',
                }


def drop_answer(annotation):
    """Keep joint spans versus alternative complete annotations distinct."""
    number = annotation.get('number', '')
    spans = annotation.get('spans', [])
    date = annotation.get('date', {})
    assert isinstance(number, str) and isinstance(spans, list) and isinstance(date, dict)
    active = [bool(number), bool(spans), any(str(x).strip() for x in date.values())]
    assert sum(active) <= 1, ('Ambiguous source answer type', annotation)
    if not any(active):
        return None
    if number:
        return {'required_spans': [str(number)], 'type': 'number'}
    if spans:
        assert all(isinstance(x, str) and x.strip() for x in spans)
        return {'required_spans': list(spans), 'type': 'span' if len(spans) == 1 else 'multiple_required_spans'}
    return {'required_spans': [' '.join(str(date.get(k, '')) for k in ['day', 'month', 'year'])],
            'type': 'date', 'date_components': dict(date)}


def drop_rows(dataset, split, excluded_question_ids=()):
    """Official passage IDs do not by themselves establish article identity."""
    assert isinstance(dataset, dict)
    seen = set()
    excluded_question_ids = set(excluded_question_ids)
    for passage_id, context in dataset.items():
        passage = context['passage']
        assert isinstance(passage, str) and passage.strip()
        for qa in context['qa_pairs']:
            qid = str(qa['query_id'])
            if qid in excluded_question_ids:
                continue
            assert qid not in seen, ('Repeated source question ID', split, qid)
            seen.add(qid)
            question = qa['question']
            assert isinstance(question, str) and question.strip()
            annotations = []
            for raw in [qa['answer']] + list(qa.get('validated_answers', [])):
                answer = drop_answer(raw)
                if answer is not None and answer not in annotations:
                    annotations.append(answer)
            assert annotations, ('No nonempty source answer', qid)
            yield {
                'question_id': 'drop:' + qid, 'original_question_id': qid,
                'context_id': 'drop_context_' + text_id(passage), 'context_text_sha256': text_id(passage),
                'original_passage_id': str(passage_id), 'article_id': None,
                'article_identity_scope': 'Not provided by the original passage-keyed schema; no article-heldout claim.',
                'passage': passage, 'question': question, 'source_split': split,
                'cohort': 'drop', 'language': 'en', 'answer_annotations': annotations,
                'raw_question_annotation': qa,
                'family_scope': 'Human question from a discrete-reasoning-oriented corpus. '
                                'Number-valued answers alone do not prove arithmetic is necessary.',
            }


def exact_answer_signature(annotation):
    """Identity/dedup key, not an answer scorer or learned representation."""
    return tuple(sorted(' '.join(s.split()).casefold() for s in annotation['required_spans']))


def inventory(rows):
    """Count real eligible multi-question contexts before freezing a sample."""
    rows = list(rows)
    by_context = defaultdict(list)
    seen = {}
    for row in rows:
        qid = row['question_id']
        assert qid not in seen, ('Cross-file duplicate question ID; inspect before splitting', qid)
        seen[qid] = row
        by_context[row['context_id']].append(row)
    sizes = Counter(len(group) for group in by_context.values())
    distinct = Counter()
    for group in by_context.values():
        # An overlap between any complete accepted annotations makes the two
        # answers unsuitable as an unambiguous different-answer contrast.
        signatures = [{exact_answer_signature(a) for a in row['answer_annotations']} for row in group]
        distinct[sum(not signatures[i] & signatures[j]
                     for i in range(len(group)) for j in range(i))] += 1
    return {
        'questions': len(seen), 'exact_contexts': len(by_context),
        'known_articles': len({r['article_id'] for r in rows if r['article_id'] is not None}),
        'questions_per_exact_context_histogram': dict(sorted(sizes.items())),
        'nonoverlapping_answer_pairs_per_context_histogram': dict(sorted(distinct.items())),
        'answer_types': dict(Counter(a['type'] for row in rows for a in row['answer_annotations'])),
        'scope': 'Source qualification only; not yet sampled, split, tokenized, scored or run on any model.',
    }


def conservative_complete_answer(generated_text, accepted_annotations, eos_seen):
    """A deliberately explicit *new prospective* whole-response score.

    No leading/trailing arbitrary-number extraction, incomplete reasoning
    parsing, or revision of any Phase2747 scoring. JSON list form can express
    all required spans. A plain whole response is one span. Exact normalized
    text is reported separately from EOS and strict JSON formatting. This is
    not the official DROP F1 or a semantic equivalence oracle.
    """
    text = generated_text.strip()
    strict_json = False
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed and all(isinstance(v, str) for v in parsed):
            spans = parsed
            strict_json = True
        elif isinstance(parsed, str):
            spans = [parsed]
        else:
            spans = [text]
    except (ValueError, TypeError):
        spans = [text]
    def norm(value):
        return ' '.join(unicodedata.normalize('NFC', value).split()).casefold()
    candidate = Counter(norm(s) for s in spans)
    accepted = [Counter(norm(s) for s in a['required_spans']) for a in accepted_annotations]
    match = candidate in accepted
    return {'whole_response_normalized_exact': match, 'eos_seen': bool(eos_seen),
            'whole_response_exact_and_stopped': bool(match and eos_seen),
            'strict_nonempty_json_string_array': strict_json, 'response_spans': spans,
            'scope': 'Conservative complete-response text equality, not general semantic correctness; '
                     'a nonmatch is not automatically a reasoning error.'}
