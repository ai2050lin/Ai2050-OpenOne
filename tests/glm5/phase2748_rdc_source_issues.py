"""Audit ALL original annotation failures before any eligibility correction."""
import io
import json
import zipfile
from collections import Counter
from rdc_construction_common import *
from rdc_question_material import quoref_rows, drop_rows, drop_answer, text_id

OUT = BASE/'phase2748'


def main():
    assert read(OUT/'protocol.json')['status'] == 'frozen_before_model_observation'
    result_path = OUT/'sources/schema_audit.json'
    assert not result_path.exists(), 'Keep the first full schema audit unchanged'
    start = time.monotonic()
    issues = []
    manifests = []
    for cohort in ['quoref', 'drop']:
        downloaded = read(OUT/'sources'/(cohort+'_current.json'))
        archive_path = ROOT/downloaded['archive']['path']
        assert sha(archive_path) == downloaded['archive']['sha256']
        paths = ({'train': 'quoref-train-dev-v0.1/quoref-train-v0.1.json',
                  'dev': 'quoref-train-dev-v0.1/quoref-dev-v0.1.json'} if cohort == 'quoref' else
                 {'train': 'drop_dataset/drop_dataset_train.json', 'dev': 'drop_dataset/drop_dataset_dev.json'})
        counts = {}
        first_failures = []
        with zipfile.ZipFile(archive_path) as archive:
            for split, member in paths.items():
                with archive.open(member) as entry:
                    payload = json.load(io.TextIOWrapper(entry, encoding='utf-8'))
                parser = quoref_rows if cohort == 'quoref' else drop_rows
                try:
                    list(parser(payload, split))
                except Exception as exc:
                    first_failures.append({'split': split, 'error': str(exc), 'traceback': traceback.format_exc()})
                n = 0
                if cohort == 'quoref':
                    for ai, article in enumerate(payload['data']):
                        for pi, paragraph in enumerate(article['paragraphs']):
                            passage = paragraph['context']
                            for qa in paragraph['qas']:
                                n += 1
                                for si, span in enumerate(qa['answers']):
                                    begin, content = span['answer_start'], span['text']
                                    if begin < 0 or not content or passage[begin:begin+len(content)] != content:
                                        locations = []
                                        pos = passage.find(content) if content else -1
                                        while pos >= 0:
                                            locations.append(pos)
                                            pos = passage.find(content, pos+1)
                                        issues.append({'cohort': cohort, 'split': split, 'question_id': qa['id'],
                                            'issue': 'answer_span_offset_or_text_invalid',
                                            'source_locator': [ai, pi, si], 'title': article.get('title'),
                                            'context_sha256': text_id(passage), 'original_annotation': qa,
                                            'declared_start': begin, 'observed_substring': passage[begin:begin+len(content)],
                                            'exact_text_occurrence_offsets': locations,
                                            'context_excerpt_near_declared_offset': passage[max(0, begin-120):begin+len(content)+120],
                                            'action': 'Exclude entire question from primary eligibility; do not repair the source span or substitute an occurrence.'})
                else:
                    for passage_id, context in payload.items():
                        for qa in context['qa_pairs']:
                            n += 1
                            annotations = []
                            errors = []
                            for raw in [qa['answer']] + list(qa.get('validated_answers', [])):
                                try:
                                    answer = drop_answer(raw)
                                    if answer is not None:
                                        annotations.append(answer)
                                except Exception as exc:
                                    errors.append(str(exc))
                            if errors or not annotations:
                                issues.append({'cohort': cohort, 'split': split, 'question_id': qa['query_id'],
                                    'issue': 'ambiguous_answer_annotation' if errors else 'all_answer_annotations_empty',
                                    'original_passage_id': passage_id, 'context_sha256': text_id(context['passage']),
                                    'original_annotation': qa, 'errors': errors,
                                    'action': 'Exclude entire question from primary eligibility; keep exact original annotation and do not infer a gold answer.'})
                counts[split] = n
        manifests.append({'cohort': cohort, 'archive': downloaded['archive'], 'raw_questions_by_split': counts,
                          'first_strict_parser_failures': first_failures})
    counts = Counter((r['cohort'], r['split'], r['issue']) for r in issues)
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'strict_parser_source': snapshot(Path(__file__).with_name('rdc_question_material.py')),
        'all_original_questions_scanned': True, 'sources': manifests, 'issues': issues,
        'counts': [{'cohort': k[0], 'split': k[1], 'issue': k[2], 'records': v,
                    'distinct_questions': len({r['question_id'] for r in issues if (r['cohort'], r['split'], r['issue']) == k})}
                   for k, v in sorted(counts.items())],
        'seconds': time.monotonic()-start,
        'scope': 'Original annotation eligibility, before any native language experiment. A malformed/empty label is not a model failure. Original archives and rejected source records retained unchanged.'}
    immutable(result_path, value)
    print('ORIGINAL_QA_SCHEMA_AUDIT', value['counts'], flush=True)


if __name__ == '__main__':
    main()
