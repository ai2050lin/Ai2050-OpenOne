# -*- coding: utf-8 -*-
"""Q3-T3 独立复核：只从磁盘重算，不信任何自报字段。"""
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
    # ---- 1 atlas ledger ----
    L = json.loads((ROOT / 'research/gpt5/atlas/atlas_ledger.json').read_text(encoding='utf-8'))
    ms = L['measurements']
    chk('ledger.n==458', len(ms) == 458, len(ms))
    chk('ledger.negatives==22', len(L['negatives']) == 22, len(L['negatives']))
    last = ms[-1]
    chk('last.phase==Q3-T3', last.get('phase') == 'Q3-T3', last.get('phase'))
    chk('last.name==pat_q3t_t3', last.get('name') == 'pat_q3t_t3', last.get('name'))
    chk('last.res_sha8==b9fd3828', last.get('res_sha8') == 'b9fd3828', last.get('res_sha8'))
    chk('last.prev==#457 head', last.get('prev_post') == '3bd49fc6', last.get('prev_post'))

    # 末链复算：把 head 换回 prev_post 再算，应还原出存储的 head
    stored = L['ledger_sha256_8']
    probe = json.loads(json.dumps(L))
    probe['ledger_sha256_8'] = last.get('prev_post')
    recomp = hashlib.sha256(
        json.dumps(probe, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    chk('ledger.head_chain', recomp == stored, '%s vs %s' % (recomp, stored))

    # prev_post 覆盖率：已知遗留为 #1–#423 与 #431 缺该字段（链式协议是 #424 之后才启用的）
    # 复核口径：缺失集合必须**恰好等于**该遗留集合，不能多也不能少。
    legacy = set(range(1, 424)) | {431}
    missing = {i + 1 for i, m in enumerate(ms) if not m.get('prev_post')}
    chk('ledger.prev_legacy_exact', missing == legacy,
        'extra=%s short=%s' % (sorted(missing - legacy)[:5], sorted(legacy - missing)[:5]))
    chained = [i + 1 for i, m in enumerate(ms) if m.get('prev_post')]
    chk('ledger.chain_contiguous_from424',
        chained == [i for i in range(424, len(ms) + 1) if i != 431], chained[:3])

    neg = [n for n in L['negatives'] if n.get('id') == 'pat_q3t_neg_role_mode_reuse']
    chk('negative.role_mode_reuse', len(neg) == 1)
    if neg:
        chk('negative.status==refuted', neg[0].get('status') == 'refuted', neg[0].get('status'))

    # ---- 2 result file ----
    rf = ROOT / 'results/2026-11-01/061142_pat_q3t_t3.json'
    chk('result.exists', rf.exists())
    d = json.loads(rf.read_text(encoding='utf-8'))
    chk('result.sha8==b9fd3828', d.get('sha8') == 'b9fd3828', d.get('sha8'))
    p = d['payload']
    chk('result.n_cells==144', p.get('n_cells') == 144, p.get('n_cells'))
    chk('result.verdict', p.get('verdict') == 'NULL_decomposition', p.get('verdict'))
    chk('result.r2_ho<0.90', p.get('r2_holdout_median', 1) < 0.90, p.get('r2_holdout_median'))
    chk('result.reuse==1.609', p.get('reuse_raw_median') == 1.609, p.get('reuse_raw_median'))
    chk('result.K==32', p.get('K_dict') == 32, p.get('K_dict'))
    chk('result.nnz<=3', p.get('nnz_max', 9) <= 3, p.get('nnz_max'))
    # 复用驱动源独立复算
    se = p['reuse_raw']['same_entity']['reuse']
    de = p['reuse_raw']['diff_entity']['reuse']
    chk('result.entity_ratio>5', se / de > 5.0, round(se / de, 2))
    # bootstrap 自配对修复后的健全性：CI 必须包住点估计
    viol = []
    for tag in ('reuse_raw', 'reuse_centered'):
        for r in ('ag', 're', 'th'):
            v = p[tag][r]
            if not (v['ci'][0] <= v['reuse'] <= v['ci'][1]):
                viol.append('%s/%s' % (tag, r))
    chk('result.ci_brackets_point', not viol, viol)

    # ---- 3 rules ledger ----
    R = json.loads((ROOT / 'data/rules/ledger.json').read_text(encoding='utf-8'))
    items = R['rules'] if isinstance(R, dict) and 'rules' in R else R
    ids = {x['id']: x for x in items}
    chk('rules.n>=77', len(items) >= 77, len(items))
    for rid, st in (('R-MTH-11', 'refuted'), ('R-ATT-17', 'refuted'),
                    ('R-MET-22', 'refuted'), ('R-MET-23', 'refuted'),
                    ('R-MTH-12', 'refuted')):
        got = ids.get(rid, {}).get('status')
        chk('rules.%s==%s' % (rid, st), got == st, got)

    # ---- 4 docs ----
    for f, need in (
        ('research/gpt5/docs/PAT_Q3T_FINDINGS_20261101.md', ['F3-1', 'F3-2', 'F3-3', 'D17', 'D18', 'D19']),
        ('research/gpt5/docs/PAT_Q3T_PREREG_v1.md', ['b9fd3828', '预算（3 轮）耗尽']),
        ('research/gpt5/docs/AGI_GPT5_MEMO.md', ['Phase Q3-T3', 'ebac5c72']),
    ):
        t = (ROOT / f).read_text(encoding='utf-8', errors='replace')
        miss = [k for k in need if k not in t]
        chk('doc.%s' % Path(f).name, not miss, 'missing=%s' % miss)

    # ---- 5 版本文档未被动判据区 ----
    pre = (ROOT / 'research/gpt5/docs/PAT_Q3T_PREREG_v1.md').read_text(encoding='utf-8')
    chk('prereg.criteria_intact', 'reuse(role) = 该角色在 144 格中命中的字典列集合的 Jaccard 均值' in pre)

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
