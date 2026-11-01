# -*- coding: utf-8 -*-
"""EA-v4：读出窗口与写入窗口的逐层扫描（Phase 266 的后续）。

对每个属性 × 语言、每一层 ℓ（hs[0..35]，即第 ℓ 块的输入；ℓ = 36 是最终归一化后的输出，不能写入，不扫）：
  读出  该层单独拟合的共享算子（发现集 210 个实体，λ 由发现集 5 折交叉验证选定），在确认集 197 个新实体上的 R²，减去类别均值 R²；
  写入  用该层的算子方向（颜色取 3 个目标色 红 / 绿 / 紫，方向 = W[:, c] − 列均值；二值属性取 W 的唯一一列），在属性问题提示里
        于实体名全部 token 位置的 hs[ℓ] 上加 ±6σ·v（σ = 发现集中性编码在 v 上投影的标准差），56 个确认集实体（同 EA-v3）；
        对照：2 个随机单位方向，同样的绝对大小。摆幅 = 平均 Δ(+6) − 平均 Δ(−6)。
窗口与预注册预测（判据在 CRITERIA_V4，随结果写出）：
  读出窗口 R = {ℓ : 读出 R² − 类别均值 R² ≥ 0.10}；写入窗口 W = {ℓ : 摆幅 ≥ 1.0 且大于该层每个随机方向的摆幅}。
  H1  EA-v3 写方向成立的 6 项，W 非空。
  H2  读出比写入延续得更晚：max(R) > max(W)。10 项中至少 7 项成立记为支持。
  H3  EA-v3 未成立的 4 项（颜色 / 中、大小 / 中、可食 / 中、可食 / 英），W 非空——即存在更早的有效写入层。逐项报告。
用法：python scripts/entity_atlas_v4.py prereg|run --model models/hf/qwen3-4b
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

PREREG = ROOT / 'research/entity_atlas/EA-v4.prereg.json'
CRITERIA_V4 = {
    'schema': 'entity_atlas.v4', 'layers': list(range(36)), 'k': 6, 'n_random': 2, 'color_targets': ['red', 'green', 'purple'],
    'read_margin_min': 0.10, 'write_swing_min': 1.0, 'H2_min_items': 7, 'seed': 0,
    'v3_passed': ['color/en', 'size/en', 'weight/zh', 'weight/en', 'water/zh', 'water/en'],
    'v3_failed': ['color/zh', 'size/zh', 'edible/zh', 'edible/en'],
}


def prereg_doc():
    return {'id': 'EA-v4', 'schema': 'entity_atlas_prereg.v4', 'created': '2026-11-01',
            'basis': 'EA-v3（Phase 266）：写入成立的 6 项选定层在 11–27，未成立的多在后层（31、33）',
            'question': '每个属性的读出窗口与写入窗口在哪些层？读出是否比写入延续得更晚？EA-v3 未成立的 4 项是否在更早的层可以写入？',
            'entities_write': [r['id'] for r in V3.pick_entities()],
            'confirmation_read': 'EA-v2 确认集全部 197 个实体', 'operators': '逐层单独拟合（发现集，λ 由 5 折交叉验证）',
            'criteria': CRITERIA_V4}


def do_prereg():
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（EA-v5）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def do_run(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.dist_protocol.census_runner import model_snapshot
    torch.set_grad_enabled(False)
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    p2 = json.loads(V2.PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap = model_snapshot(model_path)['sha256']
    key = snap[:16] + '-bf16'
    z = np.load(V2.out_dir(key) / 'data.npz')
    v1 = np.load(V1.native_dir(key) / 'codes.npz')
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    blocks = model.model.layers
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    ents = [r for r in V2.new_entities() if r['id'] in set(pre['entities_write'])]
    crows = V2.new_entities()
    dcat = np.array([r['cat'] for r in V1.lexicon()]); ccat = np.array([r['cat'] for r in crows])
    rng = np.random.default_rng(C['seed'])
    rand_dirs = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], model.config.hidden_size))]
    T = p2['targets']
    palette = T['color']['palette']
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

    def batch(ids, vecs, layer, pos):
        hk = blocks[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        ctl.update(pos=pos, vecs=torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda'))
        try:
            return model(input_ids=torch.as_tensor([ids] * len(vecs), device='cuda')).logits.float()
        finally:
            ctl['vecs'] = None
            hk.remove()

    t0 = time.time()
    part = V2.out_dir(key).parent / 'EA-v4' / 'partial.json'     # 逐项断点：进程被结束后重跑只补缺的项
    part.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(part.read_text(encoding='utf-8')) if part.exists() else {}
    for attr in ['color'] + list(V2.BINARY):
        for lang in ('zh', 'en'):
            if f'{attr}/{lang}' in out:
                continue
            Yd = z[f'disc_{lang}_{attr}']; Yc = z[f'conf_{lang}_{attr}']
            if attr == 'color':
                Yd = Yd - Yd.mean(1, keepdims=True); Yc = Yc - Yc.mean(1, keepdims=True)
            Yd2 = Yd if Yd.ndim == 2 else Yd[:, None]; Yc2 = Yc if Yc.ndim == 2 else Yc[:, None]
            cm = {c: Yd2[dcat == c].mean(0) for c in set(dcat)}
            r2_cat = E.r2(Yc2, np.stack([cm[c] for c in ccat]))
            Cc = z[f'conf_{lang}_codes'].astype(np.float64)
            rows = []
            for l in C['layers']:
                Xd = v1[lang][:, l].astype(np.float64)
                _, lam, _ = E.select_layer_lambda([Xd], Yd2, E.CRITERIA_V2['lambdas_rel'], 5, 0)
                W = E.ridge_fit(Xd, Yd2, lam)['W']
                read = E.r2(Yc2, E.ridge_predict(E.ridge_fit(Xd, Yd2, lam), Cc[:, l]))
                if attr == 'color':
                    dirs = {c: W[:, palette.index(c)] - W.mean(1) for c in C['color_targets']}
                else:
                    dirs = {'+': W[:, 0]}
                swings, rand_sw = [], []
                for c, v in dirs.items():
                    v = v / np.linalg.norm(v)
                    s = float((Xd @ v).std())
                    vecs = [np.zeros_like(v), C['k'] * s * v, -C['k'] * s * v] + \
                           [sg * C['k'] * s * rd for rd in rand_dirs for sg in (1, -1)]
                    vals = np.zeros((len(ents), len(vecs)))
                    for i, r in enumerate(ents):
                        name = r['zh'] if lang == 'zh' else r['en']
                        if attr == 'color':
                            text = T['color']['prompt'][lang].replace('{X}', r['zh']).replace('{E}', r['en'])
                            ids = enc(text)
                            lp = torch.log_softmax(batch(ids, vecs, l, positions(text, name))[:, len(ids) - 1], -1)
                            sc = lp[:, [enc(T['color']['forms'][lang][col][0])[0] for col in palette]]
                            vals[i] = (sc[:, palette.index(c)] - sc.mean(1)).cpu().numpy()
                        else:
                            sp = T[attr][lang]
                            text = sp['prompt'].replace('{X}', r['zh']).replace('{E}', r['en'])
                            p = enc(text); pos = positions(text, name)
                            def best(forms):
                                o = np.full(len(vecs), -1e9)
                                for f in forms:
                                    a = enc(f)
                                    lg = batch(p + a, vecs, l, pos)
                                    lpa = torch.log_softmax(lg[:, len(p) - 1:len(p) - 1 + len(a)], -1)
                                    o = np.maximum(o, lpa.gather(-1, torch.as_tensor(a, device='cuda')[None, :, None]
                                                                 .expand(len(vecs), -1, 1)).sum((1, 2)).cpu().numpy())
                                return o
                            vals[i] = best(sp['pos']) - best(sp['neg'])
                    m = vals.mean(0)
                    swings.append(float(m[1] - m[2]))
                    rand_sw.append([float(m[3 + 2 * j] - m[4 + 2 * j]) for j in range(len(rand_dirs))])
                sw = float(np.mean(swings))
                rs = [float(np.mean([r[j] for r in rand_sw])) for j in range(len(rand_dirs))]
                rows.append({'layer': l, 'read_r2': read, 'read_margin': read - r2_cat, 'swing': sw, 'swing_by_target': swings,
                             'random_swings': rs,
                             'in_read': bool(read - r2_cat >= C['read_margin_min']),
                             'in_write': bool(sw >= C['write_swing_min'] and sw > max(rs))})
            R = [r['layer'] for r in rows if r['in_read']]; Wn = [r['layer'] for r in rows if r['in_write']]
            item = f'{attr}/{lang}'
            out[item] = {'r2_category_mean': r2_cat, 'rows': rows, 'read_window': R, 'write_window': Wn,
                         'read_last': max(R) if R else None, 'write_last': max(Wn) if Wn else None,
                         'write_first': min(Wn) if Wn else None,
                         'H2_read_outlasts_write': bool(R and (not Wn or max(R) > max(Wn)))}
            part.write_text(json.dumps(out, ensure_ascii=False, default=float), encoding='utf-8')
            print(f'@@progress {item} R={R[:1] + R[-1:] if R else []} W={Wn[:1] + Wn[-1:] if Wn else []} {round(time.time() - t0)}s', flush=True)
    verdict = {'H1': {i: bool(out[i]['write_window']) for i in C['v3_passed']},
               'H2_items': {i: out[i]['H2_read_outlasts_write'] for i in out},
               'H3': {i: bool(out[i]['write_window']) for i in C['v3_failed']}}
    verdict['H1_all'] = all(verdict['H1'].values())
    verdict['H2_supported'] = sum(verdict['H2_items'].values()) >= C['H2_min_items']
    rep = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
           'verdict': verdict, 'items': out, 'seconds': round(time.time() - t0, 1), 'torch': torch.__version__}
    dst = ROOT / 'research/entity_atlas' / f'EA-v4-{Path(model_path).name}'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps(verdict, ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
