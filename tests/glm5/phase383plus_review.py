"""Retrospective audit of exposed Phase 385/387-390 artifacts; no model run.

Reads discovery/confirm results and frozen metadata only. Does not read lockbox
materials, modify original artifacts, refit predictors, or change thresholds.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write
from tools import provenance

SRC = ROOT / 'tests/glm5/result/common_bench_b1_v3'
OUT = ROOT / 'tests/glm5/result/phase383plus_review_20261110'
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    p = SRC / name
    INPUTS[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    started = datetime.now().astimezone().isoformat()
    plan = read('plan.json')
    pre = read('stagec-prereg.json')
    frozen = read('stagec-frozen.json')
    report = read('stagec-confirm-report.json')
    dp = read('staged-prereg.json')
    df = read('staged-frozen.json')
    pilot = read('staged-pilot.json')
    items = read('staged-pilot-items.json')
    custody = []

    def check(target, expected, owner):
        path = ROOT / target
        actual = sha(path)
        INPUTS[str(path.relative_to(ROOT))] = {'sha256': actual, 'bytes': path.stat().st_size}
        custody.append({'owner': owner, 'target': target, 'expected': expected,
                        'actual': actual, 'matches': actual == expected})

    folder = 'tests/glm5/result/common_bench_b1_v3/'
    for code, expected, owner in [
        ('common_bench_b1.py', plan['code_sha256'], 'B plan'),
        ('common_bench_b1_stage_c.py', pre['code_sha256'], 'C prereg'),
        ('common_bench_b1_stage_c_eval.py', report['eval_code_sha256'], 'C report'),
        ('common_bench_b1_stage_d.py', dp['code_sha256'], 'D prereg'),
    ]:
        check('tests/glm5/' + code, expected, owner)
    for name, expected, owner in [
        ('plan.json', pre['benchmark_plan_sha256'], 'C prereg'),
        ('stagec-prereg.json', frozen['prereg_sha256'], 'C freeze'),
        ('stagec-estimators.npz', frozen['estimators_sha256'], 'C freeze'),
        ('stagec-frozen.json', report['frozen_sha256'], 'C confirm'),
        ('stagec-prereg.json', report['prereg_sha256'], 'C confirm'),
        ('stagec-native-confirm.json', report['native_manifest_sha256'], 'C confirm'),
        ('stagec-native-discovery.json', frozen['native_manifest_sha256'], 'C freeze'),
        ('stagec-frozen.json', dp['stagec_frozen_sha256'], 'D prereg'),
        ('plan.json', dp['benchmark_plan_sha256'], 'D prereg'),
        ('staged-prereg.json', df['prereg_sha256'], 'D freeze'),
        ('staged-centres.npz', df['centres_sha256'], 'D freeze'),
        ('staged-pilot.json', df['pilot_sha256'], 'D freeze'),
        ('staged-pilot-items.json', pilot['items_sha256'], 'D pilot'),
    ]:
        check(folder + name, expected, owner)

    # Preserve the precise ID-to-seed expression in the frozen stage D code.
    seeds = Counter(int.from_bytes(x['id'].encode()[-12:], 'little') % (2 ** 32) for x in items)
    assert len({x['id'] for x in items}) == len(items)
    seed_audit = {'items': len(items), 'unique_ids': len(items), 'unique_seed_suffixes': len(seeds),
                  'counts': dict(seeds), 'largest_two_fraction': sum(n for _, n in seeds.most_common(2)) / len(items),
                  'directed_colour_pairs_present': len({(x['gold'] % 5, x['target_color']) for x in items}),
                  'note': 'Modulo 2**32 of a little-endian 12-byte suffix retains only its first four bytes. '
                          'Several family/condition IDs therefore reuse the RNG stream and Gaussian direction; '
                          'this is not 512 independent random-direction draws. All 20 colour pairs are present.'}

    ids = [plan['choice_ids'][k] for k in plan['classes']]
    behavior = {}
    for split in ('discovery', 'confirm'):
        rows = read(split + '-materials.json')
        check(folder + split + '-materials.json', plan['materials'][split]['sha256'], 'B material freeze')
        bs = read('stagec-behaviour-' + split + '.json')
        assert [r['id'] for r in rows] == [b['id'] for b in bs]
        after_screen = [b for r, b in zip(rows, bs)
                        if not set(r['entities']) & set(report['behaviour']['dropped_entities'])] if split == 'confirm' else bs
        behavior[split] = {'items': len(bs), 'families': len({r['family'] for r in rows}),
                           'entities': len({e for r in rows for e in r['entities']}),
                           'forced_accuracy_all': sum(b['pred'] == b['gold'] for b in bs) / len(bs),
                           'top1_accuracy_all': sum(b['top1'] in ids[b['gold']] for b in bs) / len(bs),
                           'top1_accuracy_after_screen': sum(b['top1'] in ids[b['gold']] for b in after_screen) / len(after_screen)}

    # Version exposure check uses only discovery pilot records and confirm rows.
    seen_before_v3 = set()
    for parent in ('common_bench_b1', 'common_bench_b1_v2'):
        p = SRC.parent / parent / 'pilot-items.json'
        if p.exists():
            INPUTS[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
            olditems = json.loads(p.read_text(encoding='utf-8'))
            for x in olditems:
                seen_before_v3.update(x.get('entities', []))
                if x.get('entity'):
                    seen_before_v3.add(x['entity'])
    confirm = read('confirm-materials.json')
    cnames = {e for r in confirm for e in r['entities']}
    exposure = {'prior_pilot_entities': len(seen_before_v3), 'confirm_entities': len(cnames),
                'confirm_entities_seen_in_prior_pilots': len(cnames & seen_before_v3),
                'confirm_items_with_any_prior_pilot_entity': sum(bool(set(r['entities']) & seen_before_v3) for r in confirm),
                'scope': 'Earlier prompt-version behaviour exposure; does not imply current confirm activations were used to fit.'}

    arms = {k: {metric: v[metric] for metric in ('success', 'top1_success', 'unchanged', 'language_kept')}
            for k, v in pilot['summary']['arms'].items() if k.endswith('@1.0')}
    payload = {'schema': 'phase383plus_review.v1', 'scope': 'Retrospective audit of Phase 385 and 387-390; no model forward or new blind test',
               'started_at': started, 'model_snapshot_under_review': plan['model_snapshot'],
               'custody_checks': custody, 'all_checked_hashes_match': all(x['matches'] for x in custody),
               'native_arrays': 'Native manifest hashes checked; multi-GB arrays not reread in this audit',
               'random_seed_audit': seed_audit, 'stagec_behavior_recount': behavior, 'exposure': exposure,
               'stagec_verdicts_as_recorded': report['verdicts'], 'stagec_P3_as_recorded': {k: report['P3'][k] for k in ('A1', 'A2')},
               'staged_pilot_arms_as_recorded': arms, 'staged_lockbox_report_present': (SRC / 'staged-lockbox-report.json').exists(),
               'staged_lockbox_frame_report_present': (SRC / 'staged-lockbox_frame-report.json').exists(),
               'source_review_findings': [
                   'P3 tests nearest-centroid classification, not a state reconstruction error or output-distribution fidelity.',
                   'P4 resets prev=nxt to the observed state at every layer; this is teacher-forced one-step prediction.',
                   'C fit/evaluation behavior gate omits the registered top1>=0.80 term; retrospective top1 rates recorded here.',
                   'D summarise screens entities and correct cases but run_split does not apply the full B gate before supported verdicts.',
                   'D run_split still emits supported D1-D3 fields when controls_valid is false, with only a descriptive-only note.',
                   'D final-boundary injection is a useful residual-skip/direct-readout control; binary success rates do not estimate a mediated-effect fraction.',
                   'D random arm matches planned vector norm but does not record the realized BF16 norm mismatch.',
                   'D wrong_donor excludes target colour by construction; zero target success is not a test of harmlessness.',
                   'No D no-op/restore metric, unrelated-task retention, or post-intervention free generation is recorded.',
               ],
               'input_manifest': INPUTS, 'audit_source_sha256': sha(Path(__file__)), 'formal_evidence_promotion': 0}
    for p in (OUT, ROOT / 'results' / datetime.now().strftime('%Y-%m-%d')):
        check_write(p, 1 << 20, operation='Phase383之后结果审计')
    # This checkout has a different Windows owner. Trust only this explicit
    # workspace for these read-only Git calls; do not alter global Git config.
    original_git = provenance._git
    def audit_git(*args):
        proc = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=15, check=True)
        return proc.stdout.strip()
    provenance._git = audit_git
    try:
        path = provenance.write_result('phase383plus_review', payload, model_path=ROOT / 'models/hf/qwen3-4b',
                                       seed=202611100391, tool='phase383plus_review')
    finally:
        provenance._git = original_git
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'audit.json'
    target.write_bytes(path.read_bytes())
    saved = json.loads(target.read_text(encoding='utf-8'))
    assert not provenance.check_provenance(saved), provenance.check_provenance(saved)
    assert saved['payload']['audit_source_sha256'] == sha(Path(__file__))
    print(json.dumps({'audit': str(target), 'provenance_result': str(path), 'hash_checks': len(custody),
                      'hash_checks_passed': sum(x['matches'] for x in custody), 'seed_audit': seed_audit,
                      'behavior': behavior, 'exposure': exposure}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
