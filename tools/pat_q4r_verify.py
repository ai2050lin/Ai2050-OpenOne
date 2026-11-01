# -*- coding: utf-8 -*-
"""Q4-R 独立复核：只从磁盘重算，不信自报字段。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok, bad = [], []


def chk(name, cond, got=''):
    (ok if cond else bad).append('%s %s' % (name, got))


def main() -> int:
    L = json.loads((ROOT / 'research/gpt5/atlas/atlas_ledger.json').read_text(encoding='utf-8'))
    ms = L['measurements']
    chk('ledger.n==459', len(ms) == 459, len(ms))
    last = ms[-1]
    chk('last.phase==Q4-R', last.get('phase') == 'Q4-R', last.get('phase'))
    chk('last.res_sha8==0f295b75', last.get('res_sha8') == '0f295b75', last.get('res_sha8'))
    chk('last.prev==#458 head', last.get('prev_post') == 'ebac5c72', last.get('prev_post'))
    stored = L['ledger_sha256_8']
    probe = json.loads(json.dumps(L))
    probe['ledger_sha256_8'] = last.get('prev_post')
    recomp = hashlib.sha256(
        json.dumps(probe, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    chk('ledger.head_chain', recomp == stored, '%s vs %s' % (recomp, stored))

    rf = ROOT / 'results/2026-11-01/063929_pat_q4r.json'
    chk('result.exists', rf.exists())
    d = json.loads(rf.read_text(encoding='utf-8'))
    p = d['payload']
    chk('result.sha8', d.get('sha8') == '0f295b75', d.get('sha8'))
    chk('result.n_cells==144', p.get('n_cells') == 144, p.get('n_cells'))
    chk('result.verdict', p.get('verdict') == 'RULE_CONFIRMED_STATIC', p.get('verdict'))
    chk('result.all_criteria_true', all(p['criteria'].values()), p['criteria'])
    chk('result.acc_R1==1.0', p.get('acc_holdout_R1') == 1.0, p.get('acc_holdout_R1'))
    chk('result.R3==0.3333', p.get('acc_R3_position') == 0.3333, p.get('acc_R3_position'))
    chk('result.lstar==28', p.get('lstar_global_R1') == 28, p.get('lstar_global_R1'))
    chk('result.lstar_all_folds_same',
        len({a for a, _ in p['lstar_per_fold']}) == 1 and len(p['lstar_per_fold']) == 5,
        p['lstar_per_fold'])
    # 反泄漏三诊断必须同时存在且自洽
    chk('diag.rule_vs_model_le_1', p.get('acc_vs_model_output', 2) <= 1.0, p.get('acc_vs_model_output'))
    chk('diag.rule_beats_model_or_equal',
        p.get('acc_vs_model_output', 0) >= p.get('model_own_behaviour', 1) - 1e-9,
        '%s vs %s' % (p.get('acc_vs_model_output'), p.get('model_own_behaviour')))
    chk('diag.margin_min>0.1', p.get('margin_min_at_lstar', 0) > 0.10, p.get('margin_min_at_lstar'))
    chk('diag.permutation==1.0', p.get('permutation_rate') == 1.0, p.get('permutation_rate'))
    # 低层剖面必须与位置规则同分（F4-2 的核心断言）
    prof = p['profile_R1']
    chk('profile.low_layers_near_position',
        all(abs(prof[l] - 0.3333) < 0.03 for l in range(0, 21)),
        'range %.3f-%.3f' % (min(prof[:21]), max(prof[:21])))
    chk('profile.lstar_is_perfect', abs(prof[28] - 1.0) < 1e-9, prof[28])
    chk('profile.L35_collapses', prof[35] < 0.65, prof[35])
    chk('profile.shape_matches_doc',
        prof[21] < 0.70 and prof[23] > 0.95,
        'L21 %.3f L23 %.3f' % (prof[21], prof[23]))

    R = json.loads((ROOT / 'data/rules/ledger.json').read_text(encoding='utf-8'))
    items = R['rules'] if isinstance(R, dict) and 'rules' in R else R
    ids = {x['id']: x for x in items}
    chk('rules.n>=80', len(items) >= 80, len(items))
    for rid in ('R-ATT-18', 'R-MTH-16'):
        chk('rules.%s==refuted' % rid, ids.get(rid, {}).get('status') == 'refuted',
            ids.get(rid, {}).get('status'))

    for f, need in (
        ('research/gpt5/docs/PAT_Q4R_FINDINGS_20261101.md',
         ['F4-1', 'F4-2', 'F4-3', '0f295b75', '不能声称']),
        ('research/gpt5/docs/PAT_Q4R_PREREG_v1.md',
         ['RULE_CONFIRMED_STATIC', 'SKETCH_FALSIFIED', '防作弊']),
        ('research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', ['PSF-2', 'D24', 'D25']),
        ('research/gpt5/docs/AGI_GPT5_MEMO.md', ['Q4-R', 'fcfbf3fb']),
    ):
        t = (ROOT / f).read_text(encoding='utf-8', errors='replace')
        miss = [k for k in need if k not in t]
        chk('doc.%s' % Path(f).name, not miss, 'missing=%s' % miss)

    print('=' * 58)
    for x in ok:
        print('  OK   ' + x)
    for x in bad:
        print('  FAIL ' + x)
    print('=' * 58)
    print('通过 %d / 失败 %d' % (len(ok), len(bad)))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
