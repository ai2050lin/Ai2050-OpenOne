# -*- coding: utf-8 -*-
"""AM-v1：G5 第一次拼装——只用 G1–G4 的图谱部件预测留出实体上模型的答案分布（谱图预测保真度，AGENTS.md 5.4）。

任务族（确认集 197 个新实体 × 中 / 英；模型真实分布来自已冻结的采集）：
  color   颜色问答 11 色（EA-v2 目标，softmax）；
  size / weight / water / edible   二值属性问答（EA-v2 目标间隔，sigmoid）；
  arbitration   冲突上下文下 K 对 P 的对数几率（CF-v2，r ≥ 1 的 12 个条件）；
  truth   裸陈述句「这句话是对的 / 错的」间隔（MV-v2 行为门，4 格）。
预测方法（参数只在发现集 210 个实体上拟合；发现集内需要实体级预测时用 5 折交叉拟合）：
  atlas      图谱：G2 属性算子作用于实体中性编码（EA-v2 选定层与 λ）；仲裁 = G4 规律（线索 + 由颜色算子预测的知识强度）；
             真假 = G1 类别中心判定所述类别是否属实，再与否定组合（G3）后按发现集拟合的间隔规则给出；
  hs0        同样的拟合，但实体编码换成 hs[0]（词汇身份基线）；
  category   只用实体的标注类别：属性 = 发现集同类均值；仲裁的知识强度 = 同类均值颜色得分之差；真假 = 按标注类别完全已知（相当于知识上限）；
  uniform    候选上的均匀分布。仲裁另报「只用线索」，真假另报「只用否定」（不含实体知识）。
预测（assembly.CRITERIA，随预注册冻结）：
  H1  14 个「任务族 × 语言」中，atlas 的 KL 同时低于 hs0 与 category 的至少 11 个。
  H2  每种语言：7 个任务族 KL 平均后，atlas 相对 uniform 的恢复比例 ≥ 0.5。
用法：python scripts/g5_assembly_v1.py prereg|run --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import arbitration as A  # noqa: E402
from shared.dist_protocol import assembly as S  # noqa: E402
from shared.dist_protocol import entity_atlas as E  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402
import arbitration_v1 as CF  # noqa: E402
import mode_atlas_v1 as MV1  # noqa: E402
import mode_atlas_v2 as MV2  # noqa: E402

PREREG = ROOT / 'research/assembly/AM-v1.prereg.json'
ATTRS = ['color', 'size', 'weight', 'water', 'edible']
FAMILIES = ATTRS + ['arbitration', 'truth']
G1_LAYER = {'zh': 27, 'en': 10}


def prereg_doc():
    return {'id': 'AM-v1', 'schema': 'assembly_prereg.v1', 'created': '2026-11-01',
            'basis': 'G1–G4（Phase 264–271）的部件；AGENTS.md 5.4 谱图预测保真度',
            'question': '只用图谱部件（共享属性算子、类别中心、句型 / 真值组合、仲裁规律）与实体的中性编码，'
                        '能在多大程度上预测留出实体上模型的答案分布？是否胜过词汇身份（hs[0]）与标注类别基线？',
            'families': FAMILIES, 'doc': __doc__,
            'inputs_allowed': '提示的设计因子；实体在中性句式中的编码（EA-v1 / v2 采集）；发现集拟合的参数',
            'inputs_forbidden': '被预测提示本身的状态、logits、模型答案',
            'sources': {'EA-v2': 'research/entity_atlas/EA-v2.prereg.json', 'CF-v2': 'research/arbitration/CF-v2.prereg.json',
                        'MV-v2': 'research/mode_atlas/MV-v2.prereg.json'},
            'g1_layers': G1_LAYER, 'criteria': S.CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（AM-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def folds(n, k=5, seed=0):
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def ridge_predictions(Xd, Yd, Xc, lam):
    """发现集交叉拟合预测与确认集全量拟合预测。"""
    Pd = np.zeros_like(Yd, dtype=float)
    for p in folds(len(Yd)):
        tr = np.setdiff1d(np.arange(len(Yd)), p)
        Pd[p] = E.ridge_predict(E.ridge_fit(Xd[tr], Yd[tr], lam), Xd[p])
    return Pd, E.ridge_predict(E.ridge_fit(Xd, Yd, lam), Xc)


def centroid_predict(Xd, cd, Xc):
    cats = sorted(set(cd))
    C = np.stack([Xd[cd == c].mean(0) for c in cats])
    d = ((Xc[:, None] - C[None]) ** 2).sum(-1)
    return np.array(cats)[d.argmin(1)]


def centroid_cv(Xd, cd, Xc):
    Pd = np.empty(len(Xd), dtype=object)
    for p in folds(len(Xd)):
        tr = np.setdiff1d(np.arange(len(Xd)), p)
        Pd[p] = centroid_predict(Xd[tr], cd[tr], Xd[p])
    return Pd, centroid_predict(Xd, cd, Xc)


def do_run(model_path):
    from provenance import write_result
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = CF.key_of(model_path)
    v2rep = json.loads((ROOT / 'research/entity_atlas/EA-v2-qwen3-4b/report.json').read_text(encoding='utf-8'))['G2_confirm']
    z2 = np.load(V2.out_dir(key) / 'data.npz')
    v1 = np.load(V1.native_dir(key) / 'codes.npz')
    zc = np.load(CF.out_dir(key) / 'data.npz')
    zm = np.load(MV2.out_dir(key) / 'data.npz')
    conds = [tuple(c) for c in json.loads(CF.PREREG.read_text(encoding='utf-8'))['conditions']]
    drow, crow = V1.lexicon(), V2.new_entities()
    dcat = np.array([r['cat'] for r in drow]); ccat = np.array([r['cat'] for r in crow])
    res, cells = {}, {}
    for lang in ('zh', 'en'):
        Xd_all = v1[lang].astype(np.float64)
        Xc_all = z2[f'conf_{lang}_codes'].astype(np.float64)
        res[lang] = {}
        colour_pred = {}
        for attr in ATTRS:
            Yd, Yc = z2[f'disc_{lang}_{attr}'].astype(float), z2[f'conf_{lang}_{attr}'].astype(float)
            l, lam = v2rep[attr][lang]['layer'], v2rep[attr][lang]['lambda_rel']
            _, lam0, _ = E.select_layer_lambda([Xd_all[:, 0]], Yd if Yd.ndim == 2 else Yd[:, None], E.CRITERIA_V2['lambdas_rel'], 5, 0)
            atl_d, atl_c = ridge_predictions(Xd_all[:, l], Yd, Xc_all[:, l], lam)
            hs0_d, hs0_c = ridge_predictions(Xd_all[:, 0], Yd, Xc_all[:, 0], lam0)
            cm = {c: Yd[dcat == c].mean(0) for c in set(dcat)}
            cat_d = np.stack([np.mean([Yd[j] for j in range(len(Yd)) if dcat[j] == dcat[i] and j != i], 0) for i in range(len(Yd))])
            cat_c = np.stack([cm[c] for c in ccat])
            kind = 'cat' if attr == 'color' else 'bin'
            res[lang][attr] = S.family_scores(kind, Yc, {'atlas': atl_c, 'hs0': hs0_c, 'category': cat_c})
            if attr == 'color':
                colour_pred = {'atlas': (atl_d, atl_c), 'hs0': (hs0_d, hs0_c), 'category': (cat_d, cat_c)}
        # 仲裁：G4 规律，知识强度 = 各方法预测的颜色得分之差（P − K）
        data = {}
        for s, n in (('disc', len(drow)), ('conf', len(crow))):
            LO, PK = zc[f'{s}_{lang}_LO'], zc[f'{s}_{lang}_PK']
            P, K = PK[:, 0].astype(int), PK[:, 1].astype(int)
            si = 0 if s == 'disc' else 1
            strength = {m: colour_pred[m][si][np.arange(n), P] - colour_pred[m][si][np.arange(n), K] for m in colour_pred}
            rows = {k: [] for k in ('S', 'src', 'r', 'LO', 'ent', *strength)}
            for j, (src, r, Sx) in enumerate(conds):
                if r == 0:
                    continue
                for i in range(n):
                    rows['S'].append(Sx); rows['src'].append(int(src == 'ency')); rows['r'].append(r)
                    rows['LO'].append(LO[i, j]); rows['ent'].append(i)
                    for m in strength:
                        rows[m].append(strength[m][i])
            data[s] = {k: np.array(v, float) for k, v in rows.items()}
        preds = {}
        for m in ('atlas', 'hs0', 'category'):
            b = A.fit(A.design(data['disc']['S'], data['disc']['src'], data['disc']['r'], data['disc'][m]), data['disc']['LO'])
            preds[m] = A.design(data['conf']['S'], data['conf']['src'], data['conf']['r'], data['conf'][m]) @ b
        cue = [A.TERMS.index(t) for t in ('1', 'S', 'src', 'logr', 'S_logr')]
        Xd_ = A.design(data['disc']['S'], data['disc']['src'], data['disc']['r'], data['disc']['atlas'])[:, cue]
        preds['cues_only'] = A.design(data['conf']['S'], data['conf']['src'], data['conf']['r'], data['conf']['atlas'])[:, cue] @ A.fit(Xd_, data['disc']['LO'])
        res[lang]['arbitration'] = S.family_scores('bin', data['conf']['LO'], preds)
        # 真假判断：所述类别是否属实（类别中心判定）与否定组合；间隔 = a_n + b_n · 判定为真
        gd, gc = zm[f'disc_{lang}_gate'], zm[f'conf_{lang}_gate']        # [n, 4]，格序 2n + t
        stated = lambda rows: [[(r['cat'] if (c & 1) else MV1.false_cat(r)) for c in range(4)] for r in rows]
        sd, sc = stated(drow), stated(crow)
        l1 = G1_LAYER[lang]
        cls = {'atlas': centroid_cv(Xd_all[:, l1], dcat, Xc_all[:, l1]), 'hs0': centroid_cv(Xd_all[:, 0], dcat, Xc_all[:, 0]),
               'category': (dcat, ccat)}

        def truth_hat(pc, st):
            return np.array([[float((pc[i] == st[i][c]) != bool(c >> 1)) for c in range(4)] for i in range(len(st))])
        preds = {}
        for m, (pd_, pc_) in cls.items():
            Td, Tc = truth_hat(pd_, sd), truth_hat(pc_, sc)
            P = np.zeros_like(gc)
            for nn in (0, 1):
                cols = [2 * nn, 2 * nn + 1]
                X = np.c_[np.ones(Td[:, cols].size), Td[:, cols].ravel()]
                b = np.linalg.lstsq(X, gd[:, cols].ravel(), rcond=None)[0]
                P[:, cols] = b[0] + b[1] * Tc[:, cols]
            preds[m] = P.ravel()
        P = np.zeros_like(gc)
        for nn in (0, 1):
            P[:, [2 * nn, 2 * nn + 1]] = gd[:, [2 * nn, 2 * nn + 1]].mean()
        preds['negation_only'] = P.ravel()
        res[lang]['truth'] = S.family_scores('bin', gc.ravel(), preds)
        for f in FAMILIES:
            r_ = res[lang][f]
            r_['recovered_atlas'] = S.recovered(r_['atlas'], r_['uniform'])
            cells[f'{f}/{lang}'] = bool(r_['atlas'] < r_['hs0'] and r_['atlas'] < r_['category'])
        print(f'@@progress {lang} ' + ' '.join(f'{f}:{res[lang][f]["atlas"]:.3f}/{res[lang][f]["hs0"]:.3f}/{res[lang][f]["category"]:.3f}/{res[lang][f]["uniform"]:.3f}'
                                                for f in FAMILIES), flush=True)
    pooled = {lang: {m: float(np.mean([res[lang][f][m] for f in FAMILIES])) for m in ('atlas', 'hs0', 'category', 'uniform')} for lang in res}
    for lang in pooled:
        pooled[lang]['recovered_atlas'] = S.recovered(pooled[lang]['atlas'], pooled[lang]['uniform'])
        pooled[lang]['recovered_hs0'] = S.recovered(pooled[lang]['hs0'], pooled[lang]['uniform'])
        pooled[lang]['recovered_category'] = S.recovered(pooled[lang]['category'], pooled[lang]['uniform'])
    verdict = {'H1_cells': cells, 'H1': sum(cells.values()) >= C['H1_min_cells'],
               'H2': {lang: bool(pooled[lang]['recovered_atlas'] >= C['H2_recovered_min']) for lang in pooled}}
    rep = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
           'kl_by_family': res, 'pooled': pooled, 'verdict': verdict}
    dst = ROOT / 'research/assembly' / f'AM-v1-{Path(model_path).name}'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    p = write_result('g5_am_v1', rep, model_path=model_path, seed=C['seed'], tool='scripts/g5_assembly_v1.py')
    print(json.dumps({'pooled': pooled, 'verdict': verdict}, ensure_ascii=False, indent=1), '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
