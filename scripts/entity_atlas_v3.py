# -*- coding: utf-8 -*-
"""EA-v3：沿共享属性算子方向的双向干预——读出方向能否写入（Phase 265 的「读写之分」）。

  算子    与 EA-v2 相同：发现集（EA-v1 的 210 个实体）中性句式编码 → 模型自己的回答，层与 λ 取 EA-v2 报告的选定值，重新拟合。
          方向：颜色 c 的方向 = W[:, c] − 11 色列均值；二值属性方向 = W 的唯一一列；单位化为 v。
          尺度 σ = 发现集中性编码在 v 上投影的标准差。
  干预    在属性问题提示（与 EA-v2 相同）里，于实体名的全部 token 位置，在选定层 ℓ 的残差（第 ℓ 块的输入，即 hs[ℓ]）上加 k·σ·v，
          k ∈ CRITERIA_V3.k_grid；对照：5 个随机单位方向，同样的绝对大小 k·σ。
  实体    确认集（EA-v2 新实体）中每个类别按 id 的 sha256 取前 8 个，共 56 个；它们从未参与算子拟合。
  量      颜色：目标色 c ∈ {红, 黄, 绿, 白, 黑, 紫}，目标量 = 目标色首 token 对数概率 − 11 色首 token 对数概率均值；
          二值属性：目标量 = 正向答案 − 反向答案的完整答案对数概率差（各取组内最大）。Δ(k) = 目标量(k) − 目标量(0)。
  判据    entity_atlas.CRITERIA_V3 / steer_verdict：每个属性 × 语言单独判定「写方向成立」。
用法：python scripts/entity_atlas_v3.py prereg|run --model models/hf/qwen3-4b
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

PREREG = ROOT / 'research/entity_atlas/EA-v3.prereg.json'
TARGET_COLORS = ['red', 'yellow', 'green', 'white', 'black', 'purple']
PER_CAT = 8


def pick_entities():
    rows = V2.new_entities()
    out = []
    for c in sorted({r['cat'] for r in rows}):
        rs = sorted([r for r in rows if r['cat'] == c], key=lambda r: hashlib.sha256(f"EA-v3|{r['id']}".encode()).hexdigest())
        out += rs[:PER_CAT]
    return out


def prereg_doc():
    v2 = json.loads((ROOT / 'research/entity_atlas/EA-v2-qwen3-4b/report.json').read_text(encoding='utf-8'))
    layers = {t: {lang: {'layer': v2['G2_confirm'][t][lang]['layer'], 'lambda_rel': v2['G2_confirm'][t][lang]['lambda_rel']}
                  for lang in ('zh', 'en')} for t in v2['G2_confirm']}
    return {
        'id': 'EA-v3', 'schema': 'entity_atlas_prereg.v3', 'created': '2026-11-01',
        'basis': 'EA-v2（Phase 265）确认的共享属性算子；本检验问「读出方向能否写入」',
        'question': '在实体名位置沿属性算子方向加正 / 负扰动，模型对该属性的回答是否随之单调变化，且大于同样大小的随机方向？',
        'operators': layers, 'operator_source': 'EA-v2 发现集编码与目标（重新拟合，层与 λ 取上表）',
        'entities': [r['id'] for r in pick_entities()], 'target_colors': TARGET_COLORS,
        'intervention': 'block ℓ 的输入残差（hs[ℓ]）上，实体名全部 token 位置加 k·σ·v；σ = 发现集中性编码在 v 上投影的标准差',
        'prompts': 'EA-v2 预注册中的属性问题提示（颜色用首 token 对数概率，二值属性用完整答案对数概率）',
        'criteria': E.CRITERIA_V3,
        'decision_rules': {'binary': '每个属性 × 语言：steer_verdict 的 supported_write',
                           'color': '每种语言：6 个目标色中至少 4 个 steer_verdict 成立，记为该语言颜色写方向成立'},
    }


def do_prereg():
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（EA-v4）')
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
    layers = model.model.layers
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    ents = [r for r in V2.new_entities() if r['id'] in set(pre['entities'])]
    rng = np.random.default_rng(C['seed'])
    d = model.config.hidden_size
    rand_dirs = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], d))]
    ks = C['k_grid']
    T = p2['targets']
    palette = T['color']['palette']
    ctl = {'layer': None, 'pos': None, 'vecs': None}

    def pre_hook(mod, args, kwargs):
        if ctl['vecs'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        V = ctl['vecs'].to(h.dtype)                                    # [batch, d]
        for b in range(h.shape[0]):
            h[b, ctl['pos']] += V[b]
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def entity_positions(text, name):
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        spans, i = [], text.find(name)
        while i >= 0:
            spans.append((i, i + len(name)))
            i = text.find(name, i + 1)
        return [j for j, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]

    def run_batch(ids_list, vecs, layer, pos):
        hk = layers[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        ctl.update(layer=layer, pos=pos, vecs=torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda'))
        try:
            T_ = max(map(len, ids_list))
            x = torch.full((len(ids_list), T_), tok.pad_token_id or 0, dtype=torch.long)
            m = torch.zeros_like(x)
            for i, s in enumerate(ids_list):
                x[i, :len(s)] = torch.as_tensor(s); m[i, :len(s)] = 1
            return model(input_ids=x.cuda(), attention_mask=m.cuda()).logits.float()
        finally:
            ctl['vecs'] = None
            hk.remove()

    t0 = time.time()
    results = {}
    for attr in ['color'] + list(V2.BINARY):
        for lang in ('zh', 'en'):
            spec = pre['operators'][attr][lang]
            l, lam = spec['layer'], spec['lambda_rel']
            Xd = v1[lang][:, l].astype(np.float64)
            Yd = z[f'disc_{lang}_{attr}']
            if attr == 'color':
                Yd = Yd - Yd.mean(1, keepdims=True)
            W = E.ridge_fit(Xd, Yd if Yd.ndim == 2 else Yd[:, None], lam)['W']   # [d, n_out]
            if attr == 'color':
                dirs = {c: W[:, palette.index(c)] - W.mean(1) for c in TARGET_COLORS}
            else:
                dirs = {'+': W[:, 0]}
            dirs = {c: v / np.linalg.norm(v) for c, v in dirs.items()}
            sig = {c: float((Xd @ v).std()) for c, v in dirs.items()}
            per_target = {}
            for c, v in dirs.items():
                conds = [('op', k, k * sig[c] * v) for k in ks] + [(f'r{j}', k, k * sig[c] * rd) for j, rd in enumerate(rand_dirs) for k in ks]
                vals = np.zeros((len(ents), len(conds)))
                for i, r in enumerate(ents):
                    if attr == 'color':
                        text = T['color']['prompt'][lang].replace('{X}', r['zh']).replace('{E}', r['en'])
                        ids = enc(text)
                        pos = entity_positions(text, r['zh'] if lang == 'zh' else r['en'])
                        lg = run_batch([ids] * len(conds), [cd[2] for cd in conds], l, pos)[:, len(ids) - 1]
                        lp = torch.log_softmax(lg, -1)
                        first = [enc(T['color']['forms'][lang][col][0])[0] for col in palette]
                        sc = lp[:, first]
                        vals[i] = (sc[:, palette.index(c)] - sc.mean(1)).cpu().numpy()
                    else:
                        sp = T[attr][lang]
                        text = sp['prompt'].replace('{X}', r['zh']).replace('{E}', r['en'])
                        p_ids = enc(text)
                        pos = entity_positions(text, r['zh'] if lang == 'zh' else r['en'])
                        def best(forms):
                            out = np.full(len(conds), -1e9)
                            for f in forms:
                                a = enc(f)
                                lg = run_batch([p_ids + a] * len(conds), [cd[2] for cd in conds], l, pos)
                                lpa = torch.log_softmax(lg[:, len(p_ids) - 1:len(p_ids) - 1 + len(a)], -1)
                                s = lpa.gather(-1, torch.as_tensor(a, device='cuda')[None, :, None].expand(len(conds), -1, 1)).sum((1, 2))
                                out = np.maximum(out, s.cpu().numpy())
                            return out
                        vals[i] = best(sp['pos']) - best(sp['neg'])
                k0 = ks.index(0)
                op = vals[:, :len(ks)]
                d_op = op - op[:, [k0]]
                d_rand = np.stack([vals[:, len(ks) * (j + 1):len(ks) * (j + 2)] - vals[:, len(ks) * (j + 1) + k0][:, None]
                                   for j in range(len(rand_dirs))])
                per_target[c] = {'verdict': E.steer_verdict(d_op, d_rand, ks, crit=C), 'sigma': sig[c],
                                 'base_mean': float(op[:, k0].mean())}
            if attr == 'color':
                d_all = np.concatenate([np.array([per_target[c]['verdict']['mean_delta_by_k']]) for c in TARGET_COLORS])
                summary = {'targets_supported': [c for c in TARGET_COLORS if per_target[c]['verdict']['supported_write']],
                           'mean_swing': float(np.mean([per_target[c]['verdict']['swing'] for c in TARGET_COLORS])),
                           'mean_random_swing_max': float(np.mean([max(per_target[c]['verdict']['random_swings']) for c in TARGET_COLORS])),
                           'mean_delta_by_k_over_targets': d_all.mean(0).tolist()}
                summary['supported_write'] = len(summary['targets_supported']) >= 4      # 6 个目标色中至少 4 个
            else:
                summary = per_target['+']['verdict']
            results[f'{attr}/{lang}'] = {'layer': l, 'per_target': per_target, 'summary': summary}
            print(f'@@progress {attr}/{lang} write={summary["supported_write"]} {round(time.time() - t0)}s', flush=True)
    rep = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
           'n_entities': len(ents), 'color_rule': '颜色：6 个目标色中至少 4 个写方向成立，记为该语言颜色写方向成立（预注册）',
           'results': results, 'seconds': round(time.time() - t0, 1), 'torch': torch.__version__}
    dst = ROOT / 'research/entity_atlas' / f'EA-v3-{Path(model_path).name}'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: {'layer': v['layer'], **{kk: v['summary'][kk] for kk in ('supported_write', 'swing', 'random_swings', 'mean_delta_by_k',
                                                                                      'targets_supported', 'mean_swing', 'mean_random_swing_max')
                                                  if kk in v['summary']}} for k, v in results.items()}, ensure_ascii=False, indent=1, default=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
