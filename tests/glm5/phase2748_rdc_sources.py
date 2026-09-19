"""Gated acquisition/inventory of original published QA data, never code."""
import argparse
import io
import json
import urllib.request
import zipfile
from rdc_construction_common import *
from rdc_question_material import SOURCE_REFERENCES, quoref_rows, drop_rows, inventory


OUT = BASE/'phase2748'


def gate():
    prior = read(BASE/'phase2747/delivery_manifest.json')
    assert prior['phase2747_complete'] and prior['all_passed']
    protocol = read(OUT/'protocol.json')
    assert protocol['phase'] == 2748 and protocol['status'] == 'frozen_before_model_observation'
    assert protocol['previous_delivery_sha256'] == sha(BASE/'phase2747/delivery_manifest.json')
    storage = read(OUT/'storage.json')
    directory = OUT/'field_store'
    assert directory.resolve() == Path(storage['physical_directory']).resolve()
    guard()
    assert shutil.disk_usage(directory).free > 4*1024**3 + 256*1024**2
    return directory


def acquire(cohort, directory):
    """Only the two allowlisted public data archives; retain exact bytes."""
    reference = SOURCE_REFERENCES[cohort]
    source_dir = directory/'sources'
    source_dir.mkdir(parents=True, exist_ok=True)
    current = OUT/'sources'/(cohort+'_current.json')
    if current.exists():
        saved = read(current)
        archive = ROOT/saved['archive']['path']
        assert sha(archive) == saved['archive']['sha256']
        return archive, saved
    destination = source_dir/(cohort+'_'+str(time.time_ns())+'.zip')
    request = urllib.request.Request(reference['data'], headers={'User-Agent': 'RDC-local-research-source-audit/1.0'})
    expected = None
    with urllib.request.urlopen(request, timeout=45) as response:
        assert response.status == 200
        final_url = response.url
        assert final_url.startswith('https://')
        size = response.headers.get('Content-Length')
        expected = int(size) if size is not None else None
        if expected is not None:
            assert 0 < expected <= 64*1024**2
        count = 0
        # A failed attempt is kept under its unique name, not reused as valid.
        with destination.open('xb') as stream:
            while True:
                block = response.read(1024**2)
                if not block:
                    break
                count += len(block)
                assert count <= 64*1024**2, 'Unexpected public archive size'
                stream.write(block)
        headers = {k: response.headers.get(k) for k in ['Content-Type', 'ETag', 'Last-Modified']}
    assert count > 0 and (expected is None or count == expected)
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive member names'
        assert sum(x.file_size for x in archive.infolist()) < 1024**3
        assert archive.testzip() is None
        members = [{'name': x.filename, 'bytes': x.file_size, 'CRC32': x.CRC} for x in archive.infolist()]
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'reference': reference,
        'resolved_public_url': final_url, 'response_metadata': headers,
        'archive': {'path': destination.relative_to(ROOT).as_posix(), 'sha256': sha(destination), 'bytes': count},
        'members': members,
        'scope': 'Public original data bytes only. No remote loader, model, package, or executable was downloaded or run. '
                 'Data version frozen by actual SHA; HTTP metadata not treated as scientific truth.'}
    immutable(current, result)
    return destination, result


def inspect(cohort):
    directory = gate()
    finished = OUT/'sources'/(cohort+'_inventory.json')
    if finished.exists():
        value = read(finished)
        assert value['all_passed']
        assert sha(ROOT/value['rows']['path']) == value['rows']['sha256']
        return value
    start = time.monotonic()
    archive_path, downloaded = acquire(cohort, directory)
    audit_path = OUT/'sources/schema_audit.json'
    audit = read(audit_path)
    assert audit['all_original_questions_scanned']
    identity_path = OUT/'sources/identity_audit.json'
    identity_audit = read(identity_path)
    assert identity_audit['all_original_question_occurrences_scanned']
    assert identity_audit['prior_schema_audit_sha256'] == sha(audit_path)
    source_audit = next(r for r in audit['sources'] if r['cohort'] == cohort)
    assert source_audit['archive']['sha256'] == downloaded['archive']['sha256']
    parser = quoref_rows if cohort == 'quoref' else drop_rows
    paths = ({'train': 'quoref-train-dev-v0.1/quoref-train-v0.1.json',
              'dev': 'quoref-train-dev-v0.1/quoref-dev-v0.1.json'} if cohort == 'quoref' else
             {'train': 'drop_dataset/drop_dataset_train.json',
              'dev': 'drop_dataset/drop_dataset_dev.json'})
    rows = []
    per_split = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split, member in paths.items():
            # Read a declared member directly; never extract zip paths.
            with archive.open(member) as entry:
                payload = json.load(io.TextIOWrapper(entry, encoding='utf-8'))
            exclusion = next(r for r in identity_audit['primary_exclusions'] if r['cohort'] == cohort and r['split'] == split)
            excluded = set(exclusion['question_ids'])
            selected = list(parser(payload, split, excluded_question_ids=excluded))
            assert len(selected) + exclusion['excluded_original_occurrences'] == source_audit['raw_questions_by_split'][split]
            per_split[split] = inventory(selected)
            rows += selected
            del payload, selected
    summary = inventory(rows)
    normalized = directory/'sources'/(cohort+'_raw_text_rows.json.gz')
    assert not normalized.exists(), 'Unregistered parsed data retained; inspect before reusing'
    compressed(normalized, rows)
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'parser': snapshot(Path(__file__).with_name('rdc_question_material.py')),
        'all_passed': True, 'cohort': cohort, 'source_archive': downloaded['archive'],
        'rows': {'path': normalized.relative_to(ROOT).as_posix(), 'sha256': sha(normalized)},
        'summary': summary, 'per_original_split': per_split, 'seconds': time.monotonic()-start,
        'source_annotation_exclusions': {'audit_path': audit_path.relative_to(ROOT).as_posix(), 'sha256': sha(audit_path),
            'identity_audit_path': identity_path.relative_to(ROOT).as_posix(), 'identity_audit_sha256': sha(identity_path),
            'distinct_question_ids': sum(r['distinct_excluded_ids'] for r in identity_audit['primary_exclusions'] if r['cohort'] == cohort),
            'original_question_occurrences': sum(r['excluded_original_occurrences'] for r in identity_audit['primary_exclusions'] if r['cohort'] == cohort),
            'rule': 'Only explicitly audited invalid/empty/duplicate-ID source questions excluded. All retained rows must pass the original strict checks; no corrected offsets, arbitrary duplicate selection or inferred targets.'},
        'unexecuted': ['Historical article/context exclusion, tokenization and model-specific eligibility.',
                       'Frozen train/validation/test assignment, native model execution, predictor fitting and answer scoring.'],
        'scope': 'Verified original source identity and annotation schema, not a language mechanism result. '
                 'Quoref multiple spans remain joint requirements; DROP validated annotations remain alternatives. '
                 'Unknown article IDs are not fabricated from passage IDs.'}
    immutable(finished, value)
    print('NATURAL_QUESTION_SOURCE_INVENTORY', cohort,
          {k: summary[k] for k in ['questions', 'exact_contexts', 'known_articles', 'answer_types']}, flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['quoref', 'drop'])
    inspect(parser.parse_args().cohort)
