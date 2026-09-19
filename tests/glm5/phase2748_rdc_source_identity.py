"""Full original QA-ID multiplicity audit, retaining duplicate occurrences."""
import io
import json
import zipfile
from collections import defaultdict, Counter
from rdc_construction_common import *
from rdc_question_material import text_id

OUT = BASE/'phase2748'


def main():
    original = read(OUT/'sources/schema_audit.json')
    start = time.monotonic()
    duplicates = []
    exclusions = []
    for cohort in ['quoref', 'drop']:
        source = read(OUT/'sources'/(cohort+'_current.json'))
        archive_path = ROOT/source['archive']['path']
        assert sha(archive_path) == source['archive']['sha256']
        paths = ({'train': 'quoref-train-dev-v0.1/quoref-train-v0.1.json',
                  'dev': 'quoref-train-dev-v0.1/quoref-dev-v0.1.json'} if cohort == 'quoref' else
                 {'train': 'drop_dataset/drop_dataset_train.json', 'dev': 'drop_dataset/drop_dataset_dev.json'})
        occurrences = defaultdict(list)
        with zipfile.ZipFile(archive_path) as archive:
            for split, member in paths.items():
                with archive.open(member) as entry:
                    payload = json.load(io.TextIOWrapper(entry, encoding='utf-8'))
                if cohort == 'quoref':
                    iterator = ((qa['id'], [ai, pi, qi], paragraph['context'], qa)
                        for ai, article in enumerate(payload['data']) for pi, paragraph in enumerate(article['paragraphs'])
                        for qi, qa in enumerate(paragraph['qas']))
                else:
                    iterator = ((qa['query_id'], [passage_id, qi], context['passage'], qa)
                        for passage_id, context in payload.items() for qi, qa in enumerate(context['qa_pairs']))
                for qid, locator, passage, qa in iterator:
                    occurrences[qid].append({'split': split, 'source_locator': locator,
                        'context_sha256': text_id(passage), 'raw_question_annotation': qa})
        duplicated = {qid for qid, rows in occurrences.items() if len(rows) > 1}
        for qid in sorted(duplicated):
            rows = occurrences[qid]
            duplicates.append({'cohort': cohort, 'question_id': qid, 'occurrences': rows,
                'all_contexts_identical': len({r['context_sha256'] for r in rows}) == 1,
                'all_annotations_identical': all(r['raw_question_annotation'] == rows[0]['raw_question_annotation'] for r in rows),
                'action': 'Exclude all occurrences of this original ID from primary material; no arbitrary first-entry selection or ID renaming.'})
        for split in paths:
            bad_schema = {r['question_id'] for r in original['issues'] if r['cohort'] == cohort and r['split'] == split}
            excluded = bad_schema | duplicated
            entries = [(qid, row) for qid, rows in occurrences.items() if qid in excluded for row in rows if row['split'] == split]
            exclusions.append({'cohort': cohort, 'split': split,
                'question_ids': sorted({qid for qid, _ in entries}),
                'distinct_excluded_ids': len({qid for qid, _ in entries}),
                'excluded_original_occurrences': len(entries),
                'raw_question_occurrences': sum(r['split'] == split for rows in occurrences.values() for r in rows)})
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'prior_schema_audit_sha256': sha(OUT/'sources/schema_audit.json'),
        'all_original_question_occurrences_scanned': True, 'duplicate_ids': duplicates,
        'primary_exclusions': exclusions, 'seconds': time.monotonic()-start,
        'scope': 'Source IDs/annotations only, no native model outputs observed. All malformed, empty and duplicate-ID entries remain in original archives and audit records.'}
    immutable(OUT/'sources/identity_audit.json', value)
    print('ORIGINAL_QA_IDENTITY_AUDIT', dict(Counter(r['cohort'] for r in duplicates)),
          [{k: r[k] for k in ['cohort','split','distinct_excluded_ids','excluded_original_occurrences']} for r in exclusions], flush=True)


if __name__ == '__main__':
    main()
