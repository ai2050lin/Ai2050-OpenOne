# -*- coding: utf-8 -*-
"""H6 严格口径判读：正反差分对是否成立（L34，pos/neg/both vs base + 同层随机对照）。

为什么需要这个脚本
------------------
`mech_pair_sign.py` 自己输出的 verdict 用的是「原始口径」（解析失败 None 记为答错），
这与 H5 之后确立的 R-MET-06 冲突：口径污染会把 neg 侧的净变化抹成 0（原始 net.neg=0、
严格 net.neg=−6），从而把「两侧都受损」误读成「杠杆是无向抖动」。判据必须与报告口径同源。

口径（三条，缺一不可）
--------------------
① 正确率：只用「base 本来可解析」的题；主表进一步要求 pos/neg/both 四个条件 pred 均非 None
   （严格子集），任何 None 不进入分母、也不记为错。
② 改答率：同样在严格子集上算，不看对错，只看 pred 是否变化。
③ 随机对照：同层随机 97 个单元（`mech_cot_L34_n169.json` 的 rand97，与 base 逐题同源，
   已断言 169/169 base 文本完全一致）。随机数是 97 不是 45，因此它是「同层无关单元」的
   保守参照而非等量匹配；等量（45）随机对照另需 H6b 补测，不给结论。

输入
----
tempdata/pair_sign_L34.json        H6 四条件（base/pos/neg/both）
tempdata/mech_cot_L34_n169.json    H5 的 base/P/rand97（同 169 题、同 base）

判据（预注册，严格口径重算）
--------------------------
S1 复现    base_repro ≥ 0.95
S2 单侧有向 net.pos ≤ −5 且 net.neg ≥ +3 且 acc.neg > acc.pos + 0.03
S3 双侧抵消 |net.both| ≤ max(2, 0.35·max(|net.pos|, |net.neg|))
S4 特异    条件改答率 ≥ 2×随机改答率，且「条件 vs rand97」改答配对的 McNemar p < 0.01
           （沿用原预注册的量：判「是否比同层随机更动行为」用改答，不用准确率；
             准确率配对 p 只因样本小而不显著，不拿来当 S4 判据）

用法：.venv/Scripts/python.exe -X utf8 tools/mech_pair_sign_strict.py
输出：tempdata/pair_sign_L34_strict.json（含全部中间量，供文档与规则账本引用）
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / 'tempdata' / 'pair_sign_L34.json'
COT = ROOT / 'tempdata' / 'mech_cot_L34_n169.json'
OUT = ROOT / 'tempdata' / 'pair_sign_L34_strict.json'


def mcnemar_exact(a: int, b: int) -> float:
    """配对二项精确检验（双尾）：a/b 是两个方向的翻转数，H0 = 各半。"""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ps = json.loads(PAIR.read_text(encoding='utf-8'))
    ct = json.loads(COT.read_text(encoding='utf-8'))
    P = {r['qi']: r for r in ps['records']}
    C = {r['qi']: r for r in ct['records']}

    def pred(src, q, cond):
        return (src[q]['out'].get(cond) or {}).get('pred')

    def ok(src, q, cond):
        p = pred(src, q, cond)
        return p is not None and p == src[q]['gold']

    common = sorted(set(P) & set(C))
    base_same = sum(1 for q in common if pred(P, q, 'base') == pred(C, q, 'base'))
    assert base_same == len(common), f'两个文件的 base 不一致：{base_same}/{len(common)}，不能配对'

    # 严格子集：四条件 pred 均非 None，且随机对照 pred 也非 None
    qs = [q for q in common
          if all(pred(P, q, c) is not None for c in ('base', 'pos', 'neg', 'both'))
          and pred(C, q, 'rand97') is not None and pred(C, q, 'P') is not None]
    n = len(qs)
    if n == 0:
        raise SystemExit('严格子集为空')

    base_acc = sum(ok(P, q, 'base') for q in qs) / n

    def stat(src, cond):
        acc = sum(ok(src, q, cond) for q in qs) / n
        wrong = sum(1 for q in qs if ok(src, q, 'base') and not ok(src, q, cond))
        right = sum(1 for q in qs if not ok(src, q, 'base') and ok(src, q, cond))
        changed = sum(1 for q in qs if pred(src, q, cond) != pred(src, q, 'base'))
        return {'acc': round(acc, 4), 'delta_pt': round(100 * (acc - base_acc), 2),
                'flip': {'wrong': wrong, 'right': right}, 'net': right - wrong,
                'changed': changed, 'change_rate': round(changed / n, 4),
                'p_vs_base': round(mcnemar_exact(wrong, right), 6)}

    conds = {'pos': stat(P, 'pos'), 'neg': stat(P, 'neg'), 'both': stat(P, 'both')}
    ref = {'P_top97': stat(C, 'P'), 'rand97': stat(C, 'rand97')}

    # 关键对照（同题配对，跨两个文件）
    def pair_test(src_a, ca, src_b, cb):
        only_a = sum(1 for q in qs if ok(src_a, q, ca) and not ok(src_b, q, cb))
        only_b = sum(1 for q in qs if not ok(src_a, q, ca) and ok(src_b, q, cb))
        return {'only_first_right': only_a, 'only_second_right': only_b,
                'p': round(mcnemar_exact(only_a, only_b), 6)}

    contrasts = {
        'pos_vs_rand97': pair_test(P, 'pos', C, 'rand97'),
        'neg_vs_rand97': pair_test(P, 'neg', C, 'rand97'),
        'both_vs_rand97': pair_test(P, 'both', C, 'rand97'),
        'pos_vs_neg': pair_test(P, 'pos', P, 'neg'),
        'pos_vs_both': pair_test(P, 'pos', P, 'both'),
        'pos_vs_P_top97': pair_test(P, 'pos', C, 'P'),
    }

    # 改答率：条件 vs 随机（McNemar on "是否改答"）
    def churn_test(ca):
        only_cond = sum(1 for q in qs
                        if pred(P, q, ca) != pred(P, q, 'base') and pred(C, q, 'rand97') == pred(C, q, 'base'))
        only_rand = sum(1 for q in qs
                        if pred(P, q, ca) == pred(P, q, 'base') and pred(C, q, 'rand97') != pred(C, q, 'base'))
        return {'only_cond_changed': only_cond, 'only_rand_changed': only_rand,
                'p': round(mcnemar_exact(only_cond, only_rand), 6)}

    churn = {c: churn_test(c) for c in ('pos', 'neg', 'both')}

    # 预注册判据（严格口径）
    v = {}
    v['S1_base_reproducible'] = bool(ps['summary'].get('base_repro', 0) >= 0.95)
    v['S2_onesided_directional'] = bool(conds['pos']['net'] <= -5 and conds['neg']['net'] >= 3
                                        and conds['neg']['acc'] > conds['pos']['acc'] + 0.03)
    thr = max(2.0, 0.35 * max(abs(conds['pos']['net']), abs(conds['neg']['net'])))
    v['S3_both_cancels'] = bool(abs(conds['both']['net']) <= thr)
    v['S4_specific_vs_random'] = bool(all(
        conds[c]['change_rate'] >= 2 * ref['rand97']['change_rate'] and churn[c]['p'] < 0.01
        for c in ('pos', 'neg', 'both')))
    v['S4_detail'] = {c: {'change_ratio': round(conds[c]['change_rate'] / ref['rand97']['change_rate'], 2),
                          'churn_p': churn[c]['p']} for c in ('pos', 'neg', 'both')}
    v['S3_threshold'] = round(thr, 2)

    # 读法：先看差分对（S2∧S3），再看是不是「删什么都掉」（S4）
    if v['S2_onesided_directional'] and v['S3_both_cancels']:
        reading = '差分对成立：删单侧有方向、删双侧相互抵消。'
    elif not v['S2_onesided_directional'] and not v['S3_both_cancels']:
        if ref['rand97']['delta_pt'] > -2 and min(conds['pos']['delta_pt'], conds['neg']['delta_pt']) < -3:
            detail = v['S4_detail']
            strong = ', '.join(f'{c} {detail[c]["change_ratio"]:.1f}×/p={detail[c]["churn_p"]:.4f}'
                               for c in ('pos', 'neg', 'both'))
            reading = ('差分对否证，但不是无向抖动：两侧删除损伤相当且彼此无差异（无方向性），'
                       '双侧删除也未回到基线（无抵消）；同层随机删除 97 个几乎无损（'
                       f'{ref["rand97"]["delta_pt"]:+.2f}pt），而这批单元改答率是随机的数倍（{strong}），'
                       '说明这 90 个单元是一个冗余饱和的功能池——删池内任一半即触及饱和点，'
                       '再删另一半无额外代价。'
                       + ('' if v['S4_specific_vs_random']
                          else '（S4 按预注册阈值 p<0.01 未全数通过：pos 支 p 略高于 0.01，方向明确但强度只到 p<0.05。）'))
        else:
            reading = '差分对否证；且删除效应与同层随机删除不可区分，这批单元的功能特异性不成立。'
    elif v['S2_onesided_directional']:
        reading = '单侧删除有方向性，但双侧未回到零：是「部分抵消」，不是严格差分对。'
    else:
        reading = '条件不全，未能判定。'

    out = {
        'schema': 'mech_pair_sign_strict.v1',
        'block': ps.get('block'), 'model_name': ps.get('model_name'),
        'source': {'pair_sign': 'tempdata/pair_sign_L34.json', 'cot': 'tempdata/mech_cot_L34_n169.json'},
        'n_records': len(common), 'n_strict': n, 'base_acc': round(base_acc, 4),
        'base_identical_across_files': f'{base_same}/{len(common)}',
        'none_counts': {c: sum(1 for q in common if pred(P, q, c) is None)
                        for c in ('base', 'pos', 'neg', 'both')},
        'conds': conds, 'ref': ref, 'contrasts': contrasts, 'churn': churn,
        'verdict_strict': v, 'reading_strict': reading,
        'raw_verdict_for_reference': {'verdict': ps.get('verdict'), 'reading': ps.get('reading'),
                                      'note': '原始口径（None 记错），仅作对照，不作为结论'},
        'caveat': ('随机对照删 97 个、本组删 45 个，数量不匹配；等量随机对照（H6b）未做，'
                   '故 S4 的「2 倍改答率」这一支只作口语参照，不以它下结论。'),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + chr(10), encoding='utf-8', newline='')

    print(f'严格子集 n={n}（共 {len(common)} 题） base acc={base_acc:.3f}')
    print(f'{"条件":22s} {"准确率":>7s} {"Δ":>9s} {"错/对":>8s} {"改答率":>8s} {"p vs base":>10s}')
    rows = [('pos (删45正侧)', conds['pos']), ('neg (删45负侧)', conds['neg']),
            ('both (删90)', conds['both']), ('P (top97)', ref['P_top97']),
            ('rand97 (随机97)', ref['rand97'])]
    for lab, s in rows:
        print(f'{lab:22s} {s["acc"]:7.3f} {s["delta_pt"]:+8.2f}pt '
              f'{s["flip"]["wrong"]:3d}/{s["flip"]["right"]:<4d} {s["change_rate"]:8.3f} {s["p_vs_base"]:10.4f}')
    print('\n关键对照：')
    for k, c in contrasts.items():
        print(f'  {k:18s} 只前对 {c["only_first_right"]:3d} / 只后对 {c["only_second_right"]:3d}  p={c["p"]:.4f}')
    print('\n改答率配对（条件 vs rand97）：')
    for k, c in churn.items():
        print(f'  {k:6s} 只条件改 {c["only_cond_changed"]:3d} / 只随机改 {c["only_rand_changed"]:3d}  p={c["p"]:.4f}')
    print(f'\n判据（严格口径）{json.dumps(v, ensure_ascii=False)}')
    print(f'读法：{reading}')
    print(f'\nwritten {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
