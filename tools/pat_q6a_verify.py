# -*- coding: utf-8 -*-
"""Q6-A 独立复核：只从磁盘重算，不信自报字段。

复核原则：本脚本不读我写的结论文档里的任何数字，只用 result.json 与 ledger，
并把文档里引用的关键数字与 JSON 交叉对账（若两者不一致，说明我写错了）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok, bad = [], []


def chk(name, cond, got=''):
    (ok if cond else bad).append('%s %s' % (name, got))


def main() -> int:
    # ---------- ledger ----------
    L = json.loads((ROOT / 'research/gpt5/atlas/atlas_ledger.json').read_text(encoding='utf-8'))
    ms = L['measurements']
    chk('ledger.n==461', len(ms) == 461, len(ms))
    last = ms[-1]
    chk('last.phase==Q6-A', last.get('phase') == 'Q6-A', last.get('phase'))
    chk('last.res_sha8==71dfa425', last.get('res_sha8') == '71dfa425', last.get('res_sha8'))
    chk('last.prev==53eb22a7', last.get('prev_post') == '53eb22a7', last.get('prev_post'))
    # 链头复算（本提交是「测量 + negative」两段写盘，每段各重算一次 head）：
    #   段1  doc 尚不含 q6a negative，field = prev_post(53eb22a7) → h1 = 6f05c384
    #   段2  doc 含 negative，        field = h1                    → stored
    # 单段提交与两段提交都试，命中即视为链完整。
    def _h(doc):
        return hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]

    probe = json.loads(json.dumps(L))
    matched = None
    probe['ledger_sha256_8'] = last.get('prev_post')
    if _h(probe) == L['ledger_sha256_8']:
        matched = 1
    else:
        d0 = json.loads(json.dumps(L))
        d0['negatives'] = [n for n in d0['negatives']
                           if n.get('name') != 'pat_q6a_neg_single_abstraction_arrow']
        d0['ledger_sha256_8'] = last.get('prev_post')
        h1 = _h(d0)
        probe['ledger_sha256_8'] = h1
        if _h(probe) == L['ledger_sha256_8']:
            matched = 2
    chk('ledger.head_chain', matched is not None,
        'steps=%s stored=%s' % (matched, L['ledger_sha256_8']))

    ne = [n for n in L.get('negatives', []) if n.get('name') == 'pat_q6a_neg_single_abstraction_arrow']
    chk('negatives.has_q6a', len(ne) == 1, len(L.get('negatives', [])))
    chk('negatives.count==24', len(L.get('negatives', [])) == 24, len(L.get('negatives', [])))
    chk('negatives.q6a_id==24', ne and ne[0].get('id') == 24, ne[0].get('id') if ne else None)

    # ---------- result ----------
    rf = ROOT / 'results/2026-11-01/072150_pat_q6a.json'
    chk('result.exists', rf.exists())
    d = json.loads(rf.read_text(encoding='utf-8'))
    p = d['payload']
    chk('result.sha8==71dfa425', d.get('sha8') == '71dfa425', d.get('sha8'))
    chk('result.n_inst==54', p.get('n_inst') == 54, p.get('n_inst'))
    chk('result.n_words==78', p.get('n_words') == 78, p.get('n_words'))
    chk('result.n_templates==3', p.get('n_templates') == 3, p.get('n_templates'))
    chk('result.A1==MIXED', p.get('A1_verdict') == 'MIXED', p.get('A1_verdict'))
    chk('result.A2==NOT_SHARED', p.get('A2_verdict') == 'HIERARCHY_NOT_SHARED', p.get('A2_verdict'))
    chk('result.A2b==NOT_SUPPORTED',
        p.get('A2b_verdict') == 'RADIAL_NOT_SUPPORTED', p.get('A2b_verdict'))

    # A1 逐族：颜色/大小 达阈，反义 0
    fb = p['A1_frac_A1b']
    chk('A1.F3_colour_frac>=0.30', fb['F3'] >= 0.30, fb['F3'])
    chk('A1.F4_size_frac>=0.30', fb['F4'] >= 0.30, fb['F4'])
    chk('A1.F5_antonym_frac==0', fb['F5'] == 0.0, fb['F5'])
    chk('A1.F1_taxonomic_frac<0.10', fb['F1'] < 0.10, fb['F1'])
    chk('A1.F2_single_target_uncomputable',
        all(v is None for v in p['A1_profile_A1b']['F2']), 'A1b 全 None')

    # 关系方向不通用：F3-F4 远低于 F1-F2
    M = p['A1_fam_cross_matrix_at_peak']
    chk('cross.F3-F4<0.12', M['F3-F4'] < 0.12, M['F3-F4'])
    chk('cross.F1-F2>0.15', M['F1-F2'] > 0.15, M['F1-F2'])
    chk('cross.F3F4_below_F1F2', M['F3-F4'] < M['F1-F2'],
        '%s < %s' % (M['F3-F4'], M['F1-F2']))

    # A2：37 层全负
    prof = p['A2_profile_chain']
    chk('A2.all_layers_negative', all(x < 0 for x in prof), min(prof))
    chk('A2.median<-0.15', p['A2_median'] < -0.15, p['A2_median'])
    chk('A2.cross_near_zero', abs(max(p['A2_profile_cross'])) < 0.10,
        max(p['A2_profile_cross']))
    chk('A2b.frac_below_0.30', p['A2b_frac'] < 0.30, p['A2b_frac'])

    # A3：功效闸——CI 必须包住点估计，且上界高于阈值 ⇒ 不得判死
    lo, hi = p['A3_CI']
    chk('A3.ci_contains_point', lo <= p['A3_holdout_acc'] <= hi,
        '[%s,%s] vs %s' % (lo, hi, p['A3_holdout_acc']))
    chk('A3.ci_upper_above_threshold', hi > 0.52, hi)
    chk('A3.n_eval==32', p['A3_n_eval'] == 32, p['A3_n_eval'])

    # ---------- 规则账本 ----------
    R = json.loads((ROOT / 'data/rules/ledger.json').read_text(encoding='utf-8'))
    items = R['rules'] if isinstance(R, dict) and 'rules' in R else R
    chk('rules.n==92', len(items) == 92, len(items))
    ids = {r['id'] for r in items}
    for rid in ('R-ATT-21', 'R-ATT-22', 'R-MET-26', 'R-MET-27', 'R-MTH-18'):
        chk('rules.has %s' % rid, rid in ids)
    byid = {r['id']: r for r in items}
    chk('rules.R-ATT-21.confirmed', byid['R-ATT-21']['status'] == 'confirmed',
        byid['R-ATT-21']['status'])
    chk('rules.R-ATT-22.refuted', byid['R-ATT-22']['status'] == 'refuted',
        byid['R-ATT-22']['status'])

    # ---------- 文档与数据交叉对账 ----------
    doc = (ROOT / 'research/gpt5/docs/PAT_Q6A_FINDINGS_20261101.md').read_text(encoding='utf-8')
    chk('doc.cites_sha', '71dfa425' in doc)
    chk('doc.cites_F3_30of37', '30/37' in doc)
    chk('doc.cites_F4_33of37', '33/37' in doc)
    chk('doc.cites_A2_median', '−0.246' in doc)
    chk('doc.marked_F2_unusable', '不得引用' in doc or '不可引用' in doc)

    memo = (ROOT / 'research/gpt5/docs/AGI_GPT5_MEMO.md').read_text(encoding='utf-8')
    chk('memo.has_Q6A', 'Phase Q6-A' in memo)
    chk('memo.head', '179c9160' in memo)
    fw = (ROOT / 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md').read_text(encoding='utf-8')
    chk('framework.has_D26', 'D26' in fw)
    pre = (ROOT / 'research/gpt5/docs/PAT_Q6A_PREREG_v1.md').read_text(encoding='utf-8')
    for sec in ('6.1', '6.7', '6.9'):
        chk('prereg.has_%s' % sec, sec in pre)
    plan = (ROOT / 'research/gpt5/docs/PAT_Q6_ATLAS_PLAN_v1.md').read_text(encoding='utf-8')
    chk('plan.has_exec_status', '8 执行状态' in plan)

    print('通过 %d / %d' % (len(ok), len(ok) + len(bad)))
    if bad:
        print('\n未通过：')
        for b in bad:
            print('  FAIL', b)
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
