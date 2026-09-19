"""Authorized scope amendment: preserve partial evidence; never mark old campaign passed."""
import shutil
from pathlib import Path
from rdc_feature_common import *

def main():
    source = RESULT / 'phase2691_crossmodel_role_confirmation'
    audit_path = source / 'analysis/status_audit_20260908/audit.json'
    audit = read(audit_path)
    out = CAMPAIGN / 'reconciliation'
    if (out/'result.json').exists():
        result = read(out/'result.json')
        for path, digest in result['archive_hashes'].items():
            assert sha(out/path) == digest
        print('RECONCILIATION_ALREADY_VERIFIED', flush=True)
        return
    for path, digest in audit['source_fingerprints'].items():
        assert sha(path) == digest, f'Re-audit required; source changed: {path}'
    archive = out/'archive'
    archive.mkdir(parents=True, exist_ok=True)
    selected = [audit_path, source/'protocol/frozen.json', source/'qwen14/material/cases.json',
                source/'qwen14/maps/global_sums.npz', source/'qwen14/analysis/native_noops.json',
                source/'analysis/resource_runtime_amendment.json']
    for cell in audit['cells']:
        key = cell['cell']
        selected += [source/f'qwen14/analysis/cell_{key}.json', source/f'qwen14/analysis/records_{key}.json',
                     source/f'qwen14/maps/counts_{key}.npz']
    # The two incomplete records are also preserved under stable semantic IDs below.
    by_id = {}
    for cell in audit['cells']:
        for row in read(source/f'qwen14/analysis/records_{cell["cell"]}.json'):
            by_id[row['case_id']] = row
    for path in (source/'qwen14/behavior').glob('case_*.json'):
        row = read(path)
        by_id.setdefault(row['case_id'], row)
    assert len(by_id) == 2818
    paths = {}
    for path in selected:
        dest = archive / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assert sha(path) == sha(dest)
        paths[str(dest.relative_to(out))] = sha(dest)
    save(archive/'stable_records.json', sorted(by_id.values(), key=lambda r:r['case_id']))
    paths['archive/stable_records.json'] = sha(archive/'stable_records.json')
    result = {'timestamp':stamp(), 'phase':2691, 'status':'partial_experiments_archived_scope_amended',
              'original_campaign_complete':False, 'original_campaign_all_checks_passed':False,
              'authorized_by':'User 2026-09-09: execute reviewed new scheme; old long queue deferred.',
              'stable_unique_records':2818, 'current2048_coverage':1922, 'cells':15,
              'deferred':['Q14 remaining generation and balanced map reconstruction','GLM4 old sign confirmation',
                          'DS native old sign confirmation','DS explicit-answer old sign confirmation'],
              'old_outputs_modified':False, 'archive_hashes':paths,
              'checks':{'source_hashes_valid':True,'stable_identity_deduplicated':True,'archive_verified':True},
              'next':'phase2692_rdc_feature_calibration and phase2693_rdc_language_atlas; old similarly numbered preparation directories remain historical unfinished artifacts.'}
    save(out/'result.json', result)
    save(CAMPAIGN/'plan.json', {'run_id':'rdc_feature_campaign_20260909','authorized':True,
        'phases':{'2691':'Partial evidence reconciliation and explicit scope amendment',
                  '2692':'S0 six known structures and numerical extraction calibration',
                  '2693':'S1 512 bilingual language inputs, full-coordinate extractors, live 3D and audit'},
        'budget':{'S0':1536,'S1':512,'disk_floor_bytes':8*1024**3,'nonquantized_model':'qwen3-4b',
                  'max_parallel_models':1,'expansion':'No unconditional S2-S4 batch; inspect frozen per-domain gains and actual resource cost.'},
        'deprecated_entrypoints':['phase2691_serial_tail.py','phase2693_campaign_terminal.py'],
        'scope':'New plan does not claim old four-protocol scientific completion; no old arrays deleted.'})
    print(json.dumps({k:v for k,v in result.items() if k!='archive_hashes'},ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
