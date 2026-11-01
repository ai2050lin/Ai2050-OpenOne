# -*- coding: utf-8 -*-
"""EA-v5：属性算子图谱——各属性是否各占独立维度、共享算子是否胜过相似查表、沿一个属性写入会不会串到别的属性（G2 补完）。

材料与 EA-v2 / v3 相同：发现集 210 个实体（拟合），确认集 197 个新实体（评估），属性目标是模型自己的回答（颜色 11 维、大小、重量、含水、可食）。
  A 独有部分（不用 GPU）  属性 k 的回答对其他全部属性的回答做线性回归（系数只在发现集拟合），残差 = 独有部分；
                          独有部分能否由实体编码读出（发现集选层与 λ，确认集评估一次；对照 hs[0] 与发现集目标打乱）。
     H1  每种语言 5 个属性中至少 3 个的独有部分可读出。
  B 相似查表（不用 GPU）  在 EA-v2 选定的层上，用发现集里编码最相似的 k 个实体的回答平均来预测确认集（k 由交叉验证选定），与共享算子的 R² 比较。
     H2  10 个属性 × 语言中至少 7 个：共享算子 R² > 相似查表 R²。
  C 写入串扰（GPU）       在共同写入层（中文 12、英文 13；规则见 common_layer_rule）上拟合 6 个方向：大小、重量、含水、可食、红、绿，
                          另加 4 个二值属性的独有方向（在残差目标上拟合）。沿每个方向在实体名 token 上加 ±6σ·v，读出全部 6 个属性的变化。
                          预测 = 线性读出算子自己的推算 w_j · (12σ_i v_i)。
     H3  每种语言 6 个读出中至少 5 个：最大 |摆幅| 来自对应的方向（对角占优）。
     H4  36 格实测摆幅与线性预测的 Spearman ≥ 0.6，且 |预测| ≥ 0.5 的非对角格中同号比例 ≥ 0.7。
     H5  每种语言 4 个二值属性中至少 2 个：独有方向的本属性摆幅 ≥ 1.0 且 ≥ 其他二值属性最大 |摆幅| 的 2 倍（可分离写入）。
判据在 entity_atlas.CRITERIA_V5，随预注册冻结。
用法：python scripts/entity_atlas_v5.py prereg|analyze|steer --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import entity_atlas as E  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402
import entity_atlas_v3 as V3  # noqa: E402

PREREG = ROOT / 'research/entity_atlas/EA-v5.prereg.json'
ATTRS = ['color', 'size', 'weight', 'water', 'edible']
BIN = ['size', 'weight', 'water', 'edible']
DIRS = ['size', 'weight', 'water', 'edible', 'red', 'green']
COMMON_LAYER = {'zh': 12, 'en': 13}


def common_layer_rule():
    """共同写入层：EA-v4 中该语言全部 5 个属性写入窗口的交集里，使 min_k（摆幅 / 该属性最大摆幅）最大的层。只用 EA-v4 的目标属性摆幅。"""
    r = json.loads((ROOT / 'research/entity_atlas/EA-v4-qwen3-4b/report.json').read_text(encoding='utf-8'))['items']
    out = {}
    for lang in ('zh', 'en'):
        its = [k for k in r if k.endswith(lang)]
        inter = set(range(36))
        for k in its:
            inter &= set(r[k]['write_window'])
        sc = {l: min(r[k]['rows'][l]['swing'] / max(x['swing'] for x in r[k]['rows']) for k in its) for l in sorted(inter)}
        out[lang] = {'layer': max(sc, key=sc.get), 'score': round(sc[max(sc, key=sc.get)], 3), 'intersection': sorted(inter)}
    return out


def prereg_doc():
    rule = common_layer_rule()
    assert {k: v['layer'] for k, v in rule.items()} == COMMON_LAYER
    return {'id': 'EA-v5', 'schema': 'entity_atlas_prereg.v5', 'created': '2026-11-01',
            'basis': 'EA-v2 / v3 / v4（Phase 265–267）：5 个属性 × 2 种语言都有共享读出算子与截止层之前的有效写入方向',
            'question': '这些属性在实体编码里是各占独立维度，还是一条共同轴的不同读法？共享线性算子是否胜过「找相似实体」？'
                        '沿一个属性写入会不会串到别的属性，串扰能否由线性读出算子预测？',
            'targets': 'EA-v2 预注册的属性目标（颜色减 11 色均值）', 'attrs': ATTRS,
            'A': '残差 = 对其他全部属性回答（颜色 11 列、二值 4 个）的最小二乘残差，系数只在发现集拟合；EA-v2 的全部层参与选层',
            'B': 'EA-v2 报告的选定层；相似 = 减去发现集均值后的余弦',
            'C': {'common_layer': COMMON_LAYER, 'common_layer_rule': common_layer_rule.__doc__, 'derivation': rule,
                  'directions': DIRS + [f'{a}_unique' for a in BIN], 'readouts': DIRS,
                  'entities': [r['id'] for r in V3.pick_entities()],
                  'intervention': 'block ℓ 的输入残差（hs[ℓ]）上，实体名全部 token 位置加 ±k·σ·v；σ = 发现集中性编码在 v 上投影的标准差',
                  'readout_definition': '颜色：目标色首 token 对数概率 − 11 色均值；二值：正向 − 反向完整答案对数概率（各取组内最大）',
                  'prediction': 'P[i, j] = w_j · (2kσ_i v_i)；w_j 为该层拟合的读出算子（颜色 c：W[:, c] − W 列均值）'},
            'criteria': E.CRITERIA_V5}


def do_prereg():
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（EA-v6）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def load(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    snap = model_snapshot(model_path)['sha256']
    key = snap[:16] + '-bf16'
    z = np.load(V2.out_dir(key) / 'data.npz')
    v1 = np.load(V1.native_dir(key) / 'codes.npz')
    T = {}
    for s in ('disc', 'conf'):
        for lang in ('zh', 'en'):
            c = z[f'{s}_{lang}_color']
            T[s, lang, 'color'] = c - c.mean(1, keepdims=True)
            for a in BIN:
                T[s, lang, a] = z[f'{s}_{lang}_{a}'].astype(np.float64)
    codes = {('disc', lang): v1[lang].astype(np.float64) for lang in ('zh', 'en')}
    codes.update({('conf', lang): z[f'conf_{lang}_codes'].astype(np.float64) for lang in ('zh', 'en')})
    return snap, T, codes


def others(T, s, lang, attr):
    return np.column_stack([T[s, lang, a] for a in ATTRS if a != attr])


def residual_targets(T, lang, attr):
    return E.residualize(T['disc', lang, attr], others(T, 'disc', lang, attr), T['conf', lang, attr], others(T, 'conf', lang, attr))


def report_path(model_path):
    d = ROOT / 'research/entity_atlas' / f'EA-v5-{Path(model_path).name}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def do_analyze(model_path):
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, T, codes = load(model_path)
    v2 = json.loads((ROOT / 'research/entity_atlas/EA-v2-qwen3-4b/report.json').read_text(encoding='utf-8'))['G2_confirm']
    L = codes['disc', 'zh'].shape[1]
    A, B = {}, {}
    for lang in ('zh', 'en'):
        Xd = [codes['disc', lang][:, l] for l in range(L)]
        Xc = [codes['conf', lang][:, l] for l in range(L)]
        for attr in ATTRS:
            Rd, Rc = residual_targets(T, lang, attr)
            Yd = T['disc', lang, attr]
            share = 1 - float(((Rd - Rd.mean(0)) ** 2).sum() / ((Yd - Yd.mean(0)) ** 2).sum())
            A[f'{attr}/{lang}'] = {'R2_by_other_attrs_disc': share, **E.unique_part_eval(Xd, Rd, Xc, Rc, C)}
            l = v2[attr][lang]['layer']
            kn = E.knn_eval(Xd[l], Yd, Xc[l], T['conf', lang, attr], C)
            B[f'{attr}/{lang}'] = {'layer': l, 'R2_operator': v2[attr][lang]['R2_operator'],
                                   'R2_category_mean': v2[attr][lang]['R2_category_mean'], **kn,
                                   'operator_beats_knn': bool(v2[attr][lang]['R2_operator'] > kn['R2_knn'])}
            print(f'@@progress {attr}/{lang} unique={A[f"{attr}/{lang}"]["R2_unique"]:.3f} '
                  f'op={B[f"{attr}/{lang}"]["R2_operator"]:.3f} knn={kn["R2_knn"]:.3f}', flush=True)
    verdict = {'H1': {lang: sum(A[f'{a}/{lang}']['supported'] for a in ATTRS) >= C['H1_min_attrs'] for lang in ('zh', 'en')},
               'H2': sum(b['operator_beats_knn'] for b in B.values()) >= C['H2_min_items']}
    p = report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
                'A_unique': A, 'B_knn': B})
    rep.setdefault('verdict', {}).update(verdict)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps(verdict, ensure_ascii=False))


def do_steer(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    p2 = json.loads(V2.PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, T, codes = load(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    blocks = model.model.layers
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    ents = [r for r in V2.new_entities() if r['id'] in set(pre['C']['entities'])]
    TP = p2['targets']
    palette = TP['color']['palette']
    rng = np.random.default_rng(C['seed'])
    rand_dirs = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], model.config.hidden_size))]
    ctl = {'pos': None, 'vecs': None}

    def pre_hook(mod, args, kwargs):
        if ctl['vecs'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        V = ctl['vecs'].to(h.dtype)
        for b in range(h.shape[0]):
            h[b, ctl['pos']] += V[b]
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def positions(text, name):
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        spans, i = [], text.find(name)
        while i >= 0:
            spans.append((i, i + len(name)))
            i = text.find(name, i + 1)
        return [j for j, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]

    def run(ids, vecs, layer, pos):
        hk = blocks[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        ctl.update(pos=pos, vecs=torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda'))
        try:
            return model(input_ids=torch.as_tensor([ids] * len(vecs), device='cuda')).logits.float()
        finally:
            ctl['vecs'] = None
            hk.remove()

    t0 = time.time()
    out = {}
    for lang in ('zh', 'en'):
        l = pre['C']['common_layer'][lang]
        Xd = codes['disc', lang][:, l]
        W, dirs = {}, {}
        for attr in ATTRS:
            Yd = T['disc', lang, attr]
            Y2 = Yd if Yd.ndim == 2 else Yd[:, None]
            _, lam, _ = E.select_layer_lambda([Xd], Y2, E.CRITERIA_V2['lambdas_rel'], 5, 0)
            Wm = E.ridge_fit(Xd, Y2, lam)['W']
            if attr == 'color':
                for c in ('red', 'green'):
                    W[c] = Wm[:, palette.index(c)] - Wm.mean(1)
            else:
                W[attr] = Wm[:, 0]
        for a in DIRS:
            dirs[a] = W[a]
        for a in BIN:
            Rd, _ = residual_targets(T, lang, a)
            _, lam, _ = E.select_layer_lambda([Xd], Rd[:, None], E.CRITERIA_V2['lambdas_rel'], 5, 0)
            dirs[f'{a}_unique'] = E.ridge_fit(Xd, Rd[:, None], lam)['W'][:, 0]
        names = list(dirs)
        units, sig = {}, {}
        for n in names:
            v = dirs[n] / np.linalg.norm(dirs[n])
            units[n], sig[n] = v, float((Xd @ v).std())
        P = {n: {j: float(W[j] @ (2 * C['k'] * sig[n] * units[n])) for j in DIRS} for n in names}
        vecs = [np.zeros(model.config.hidden_size)]
        for n in names:
            vecs += [C['k'] * sig[n] * units[n], -C['k'] * sig[n] * units[n]]
        rsig = float(np.mean(list(sig.values())))
        for rd in rand_dirs:
            vecs += [C['k'] * rsig * rd, -C['k'] * rsig * rd]
        vals = np.zeros((len(ents), len(DIRS), len(vecs)))
        for i, r in enumerate(ents):
            name = r['zh'] if lang == 'zh' else r['en']
            text = TP['color']['prompt'][lang].replace('{X}', r['zh']).replace('{E}', r['en'])
            ids = enc(text)
            lp = torch.log_softmax(run(ids, vecs, l, positions(text, name))[:, len(ids) - 1], -1)
            sc = lp[:, [enc(TP['color']['forms'][lang][col][0])[0] for col in palette]]
            for jj, c in enumerate(('red', 'green')):
                vals[i, 4 + jj] = (sc[:, palette.index(c)] - sc.mean(1)).cpu().numpy()
            for jj, a in enumerate(BIN):
                sp = TP[a][lang]
                text = sp['prompt'].replace('{X}', r['zh']).replace('{E}', r['en'])
                p = enc(text); pos = positions(text, name)

                def best(forms):
                    o = np.full(len(vecs), -1e9)
                    for f in forms:
                        aa = enc(f)
                        lpa = torch.log_softmax(run(p + aa, vecs, l, pos)[:, len(p) - 1:len(p) - 1 + len(aa)], -1)
                        o = np.maximum(o, lpa.gather(-1, torch.as_tensor(aa, device='cuda')[None, :, None]
                                                     .expand(len(vecs), -1, 1)).sum((1, 2)).cpu().numpy())
                    return o
                vals[i, jj] = best(sp['pos']) - best(sp['neg'])
            if i % 14 == 13:
                print(f'@@progress {lang} {i + 1}/{len(ents)} {round(time.time() - t0)}s', flush=True)
        m = vals.mean(0)                                   # [读出, 向量]
        sw = {n: {j: float(m[jj, 1 + 2 * k] - m[jj, 2 + 2 * k]) for jj, j in enumerate(DIRS)} for k, n in enumerate(names)}
        base = 1 + 2 * len(names)
        rsw = [{j: float(m[jj, base + 2 * q] - m[jj, base + 2 * q + 1]) for jj, j in enumerate(DIRS)} for q in range(len(rand_dirs))]
        S = np.array([[sw[i][j] for j in DIRS] for i in DIRS])
        Pm = np.array([[P[i][j] for j in DIRS] for i in DIRS])
        U = np.array([[sw[f'{a}_unique'][j] for j in DIRS] for a in BIN])
        v = E.crosstalk_verdict(S, Pm, U, [DIRS.index(a) for a in BIN], C)
        cosm = {i: {j: float(units[i] @ units[j]) for j in names} for i in names}
        out[lang] = {'layer': l, 'sigma': sig, 'swing': sw, 'predicted': P, 'random_swings': rsw, 'cos': cosm,
                     'item_swing_own_positive_frac': {n: float(np.mean(vals[:, DIRS.index(n.replace('_unique', '')),
                                                                            1 + 2 * k] - vals[:, DIRS.index(n.replace('_unique', '')),
                                                                                              2 + 2 * k] > 0))
                                                      for k, n in enumerate(names)},
                     'verdict': v}
        print(json.dumps({lang: {kk: v[kk] for kk in ('diag_top', 'H3', 'spearman', 'sign_agree_offdiag', 'H4', 'H5')}},
                         ensure_ascii=False), flush=True)
    p = report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
                'C_crosstalk': out, 'seconds_steer': round(time.time() - t0, 1), 'torch': torch.__version__})
    rep.setdefault('verdict', {}).update({f'{h}_{lang}': out[lang]['verdict'][h] for h in ('H3', 'H4', 'H5') for lang in out})
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'analyze', 'steer'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'analyze': lambda: do_analyze(a.model), 'steer': lambda: do_steer(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
