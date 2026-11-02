# -*- coding: utf-8 -*-
"""CF-v2：上下文与知识冲突时的仲裁规律（G4 第一步）。

材料  实体同 G1–G3（发现集 210、确认集 197）。先验颜色 P = 模型在无上下文颜色问答中的首选（EA-v2 目标，分语言）；
      冲突颜色 K = 该实体先验最低的 5 种颜色中按 sha256 选 1 种。上下文 = 来源 src（童话书 / 百科全书）的 r 句冲突陈述（r = 1, 2, 3；
      另有 r = 0 无上下文基线），再提问：范围 S = 「在这本书里」（上下文）或「在现实世界里」（知识）。
读出  答案位置 LO = logit(K 首 token) − logit(P 首 token)（与对数概率差相同）；逐层直接作用（DLA）：每层注意力与 MLP 输出
      经最终归一化（冻结 RMS）后在 W_U[K] − W_U[P] 上的投影；各项之和与 LO 对账。
预测（arbitration.CRITERIA）：
  H1 条件化仲裁  r = 1、童话来源：上下文范围 LO > 0 的比例 ≥ 0.7，现实范围 LO < 0 的比例 ≥ 0.7（每种语言，全部实体）。
  H2 规律        LO ≈ β0 + β_S·S + β_src·src + β_r·log r + β_rS·S·log r + γ·s + γ_S·S·s（发现集拟合），确认集 R² ≥ 0.5。
  H3 知识抵抗    现实范围的 γ 的 95% 区间（按实体 bootstrap）上界 < 0（每种语言）。
  H4 编码预测    用 EA-v2 颜色算子从中性实体编码预测的强度 ŝ 代替 s：确认集 R² 下降 ≤ 0.10，且现实范围 γ 的区间上界 < 0。
  H5 写入        确认集 56 个实体（EA-v3 选取）、r = 1、童话来源：在共同写入层（中 12 / 英 13）、全部实体名 token 上沿「削弱先验色」
                 方向 −v_P 加 k·σ（v_P = EA 颜色算子在该层的 P 列减列均值），现实范围的 LO 按 entity_atlas.steer_verdict 成立（每种语言）。
用法：python scripts/arbitration_v1.py prereg|collect|analyze|steer --model models/hf/qwen3-4b
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

from shared.dist_protocol import arbitration as A  # noqa: E402
from shared.dist_protocol import entity_atlas as E  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402
import entity_atlas_v3 as V3  # noqa: E402

PREREG = ROOT / 'research/arbitration/CF-v2.prereg.json'
CONTEXT = {
    'zh': {'story': ['在一本童话书里，{X}是{K}的。', '这本童话书里的{X}都是{K}的。', '童话书里又一次写到，{X}是{K}的。'],
           'ency': ['一本百科全书写道，{X}是{K}的。', '这本百科全书里说{X}都是{K}的。', '百科全书又一次写到，{X}是{K}的。']},
    'en': {'story': ['In a fairy-tale book, the {E} is {K}.', 'In this fairy-tale book, every {E} is {K}.',
                     'The fairy-tale book says again that the {E} is {K}.'],
           'ency': ['An encyclopedia states that the {E} is {K}.', 'This encyclopedia says every {E} is {K}.',
                    'The encyclopedia says again that the {E} is {K}.']},
}
QUESTION = {
    'zh': {1: '问：在这本书里，{X}是什么颜色的？\n答：在这本书里，{X}是', 0: '问：在现实世界里，{X}通常是什么颜色的？\n答：在现实世界里，{X}通常是'},
    'en': {1: 'Q: In this book, what color is the {E}?\nA: In this book, the {E} is',
           0: 'Q: In the real world, what color is the {E} usually?\nA: In the real world, the {E} is usually'},
}
COMMON_LAYER = {'zh': 12, 'en': 13}        # EA-v5 的共同写入层


def palette_forms():
    p = json.loads(V2.PREREG.read_text(encoding='utf-8'))['targets']['color']
    return p['palette'], p['forms']


def colour_targets(key):
    z = np.load(V2.out_dir(key) / 'data.npz')
    return {(s, lang): z[f'{s}_{lang}_color'] for s in ('disc', 'conf') for lang in ('zh', 'en')}


def pick_pk(scores, ent_id, lang):
    """P = 首选色；K = 先验最低的 5 种颜色中按 sha256 选 1 种。返回 (P, K, s = score[P] − score[K])。"""
    P = int(np.argmax(scores))
    low = list(np.argsort(scores)[:5])
    K = int(low[hashlib.sha256(f'CF-v2|{ent_id}|{lang}'.encode()).digest()[0] % 5])
    return P, K, float(scores[P] - scores[K])


def prompt(r_, lang, src, r, S, kword):
    ctx = CONTEXT[lang][src][:r] if r else []
    sep = '' if lang == 'zh' else ' '
    fill = lambda s: s.replace('{X}', r_['zh']).replace('{E}', r_['en']).replace('{K}', kword)
    text = sep.join(fill(c) for c in ctx)
    return (text + '\n' if text else '') + fill(QUESTION[lang][S])


def conditions():
    out = [(src, r, S) for src in ('story', 'ency') for r in (1, 2, 3) for S in (0, 1)]
    return out + [('none', 0, S) for S in (0, 1)]


def prereg_doc():
    return {'id': 'CF-v2', 'schema': 'arbitration_prereg.v1', 'created': '2026-11-01',
            'basis': 'G1–G3（Phase 264–270）：实体属性的共享算子与截止层、句型与真值方向；CF-v1（Phase 263）已随 2026-11-01 的回退撤销，本版重新设计',
            'question': '上下文给出与模型知识冲突的颜色时，模型按提问范围（书里 / 现实）读哪一个？对数几率是否服从「线索强度 + 知识强度」的简单规律？'
                        '知识强度能否由实体编码（G2 算子）预测？削弱实体位置的先验色方向，仲裁是否按方向改变？',
            'context': CONTEXT, 'question_templates': {lang: {('book' if k else 'world'): v for k, v in q.items()} for lang, q in QUESTION.items()},
            'conditions': conditions(), 'pk_rule': pick_pk.__doc__, 'colour_source': 'EA-v2 颜色目标（完整答案对数概率，分语言）',
            'context_colour_word': '中文用表面形式首项（如「蓝色」），英文去掉前导空格；LO 用首 token',
            's_hat': 'EA-v2 报告的颜色算子层与 λ；发现集用 5 折交叉拟合预测，确认集用发现集全量拟合',
            'steer': {'layer': COMMON_LAYER, 'entities': [r['id'] for r in V3.pick_entities()], 'condition': 'src = story, r = 1, S ∈ {0, 1}',
                      'positions': '提示中实体名的全部 token', 'direction': '−v_P（单位化），σ = 发现集中性编码在 v_P 上投影的标准差'},
            'dla': '每层 self_attn 与 mlp 在答案位置的输出 / 最终归一化输入的 RMS × norm.weight · (W_U[K] − W_U[P])；嵌入项同理',
            'criteria': A.CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（CF-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def key_of(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    snap = model_snapshot(model_path)['sha256']
    return snap, snap[:16] + '-bf16'


def out_dir(key):
    return V1.native_dir(key).parents[2] / '_arbitration' / key / 'CF-v2'


def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    return tok, model


def do_collect(model_path):
    import torch
    snap, key = key_of(model_path)
    out = out_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(model_path)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    palette, forms = palette_forms()
    T = colour_targets(key)
    blocks = model.model.layers
    nL = len(blocks)
    cap = {}
    hooks = []
    for i, b in enumerate(blocks):
        hooks.append(b.self_attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(('a', i), o[0][0, -1].float())))
        hooks.append(b.mlp.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(('m', i), o[0, -1].float())))
    hooks.append(model.model.norm.register_forward_hook(lambda m, a, o: cap.__setitem__('pre', a[0][0, -1].float())))
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: cap.__setitem__('emb', o[0, -1].float())))
    Wu = model.lm_head.weight
    nw = model.model.norm.weight.float()
    eps = model.model.norm.variance_epsilon
    conds = conditions()
    t0 = time.time()
    arrays, meta = {}, {}
    for s, rows in (('disc', V1.lexicon()), ('conf', V2.new_entities())):
        for lang in ('zh', 'en'):
            n = len(rows)
            LO = np.zeros((n, len(conds)))
            DLA = np.zeros((n, len(conds), 2 * nL + 1), dtype=np.float32)       # [emb, attn_0, mlp_0, attn_1, ...]
            PK = np.zeros((n, 3))
            for i, r in enumerate(rows):
                P, K, sv = pick_pk(T[s, lang][i], r['id'], lang)
                PK[i] = (P, K, sv)
                kw = forms[lang][palette[K]][0].strip()
                tk, tp = enc(forms[lang][palette[K]][0])[0], enc(forms[lang][palette[P]][0])[0]
                du = (Wu[tk] - Wu[tp]).float()
                for j, (src, rr, S) in enumerate(conds):
                    ids = enc(prompt(r, lang, src, rr, S, kw))
                    lg = model(input_ids=torch.as_tensor([ids], device='cuda')).logits[0, -1].float()
                    LO[i, j] = float(lg[tk] - lg[tp])
                    pre = cap['pre']
                    scale = nw / torch.sqrt((pre ** 2).mean() + eps)
                    proj = lambda v: float((v * scale) @ du)
                    DLA[i, j, 0] = proj(cap['emb'])
                    for l in range(nL):
                        DLA[i, j, 1 + 2 * l] = proj(cap['a', l])
                        DLA[i, j, 2 + 2 * l] = proj(cap['m', l])
            arrays[f'{s}_{lang}_LO'], arrays[f'{s}_{lang}_DLA'], arrays[f'{s}_{lang}_PK'] = LO, DLA, PK
            print(f'@@progress {s}/{lang} recon_err={float(np.abs(DLA.sum(-1) - LO).max()):.3f} {round(time.time() - t0)}s', flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(out / 'data.npz', **arrays)
    man = {'model': model_path, 'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(),
           'conditions': conds, 'seconds': round(time.time() - t0, 1), 'torch': torch.__version__,
           'sha256_data': hashlib.sha256((out / 'data.npz').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(man, ensure_ascii=False))


def s_hat(key, lang):
    """EA-v2 颜色算子（报告中的层与 λ）预测的 11 色得分：发现集 5 折交叉拟合，确认集全量拟合。"""
    v2 = json.loads((ROOT / 'research/entity_atlas/EA-v2-qwen3-4b/report.json').read_text(encoding='utf-8'))['G2_confirm']['color'][lang]
    l, lam = v2['layer'], v2['lambda_rel']
    T = colour_targets(key)
    Yd = T['disc', lang] - T['disc', lang].mean(1, keepdims=True)
    Xd = np.load(V1.native_dir(key) / 'codes.npz')[lang][:, l].astype(np.float64)
    Xc = np.load(V2.out_dir(key) / 'data.npz')[f'conf_{lang}_codes'][:, l].astype(np.float64)
    Pd = np.zeros_like(Yd)
    for p in np.array_split(np.random.default_rng(0).permutation(len(Yd)), 5):
        tr = np.setdiff1d(np.arange(len(Yd)), p)
        Pd[p] = E.ridge_predict(E.ridge_fit(Xd[tr], Yd[tr], lam), Xd[p])
    return Pd, E.ridge_predict(E.ridge_fit(Xd, Yd, lam), Xc)


def report_path(model_path):
    d = ROOT / 'research/arbitration' / f'CF-v2-{Path(model_path).name}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, upd, verdict):
    p = report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    rep.setdefault('verdict', {}).update(verdict)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


def do_analyze(model_path):
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = key_of(model_path)
    z = np.load(out_dir(key) / 'data.npz')
    conds = [tuple(c) for c in pre['conditions']]
    upd = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C}
    verdict = {'H1': {}, 'H2': {}, 'H3': {}, 'H4': {}}
    res = {}
    for lang in ('zh', 'en'):
        Sh = dict(zip(('disc', 'conf'), s_hat(key, lang)))
        data = {}
        for s in ('disc', 'conf'):
            LO, PK = z[f'{s}_{lang}_LO'], z[f'{s}_{lang}_PK']
            P, K = PK[:, 0].astype(int), PK[:, 1].astype(int)
            sh = Sh[s][np.arange(len(P)), P] - Sh[s][np.arange(len(P)), K]
            rows = {k: [] for k in ('S', 'src', 'r', 's', 's_hat', 'LO', 'ent')}
            for j, (src, r, S) in enumerate(conds):
                if r == 0:
                    continue
                for i in range(len(LO)):
                    for k_, v in zip(rows, (S, int(src == 'ency'), r, PK[i, 2], sh[i], LO[i, j], i)):
                        rows[k_].append(v)
            data[s] = {k: np.array(v) for k, v in rows.items()}
            data[s]['_LO'], data[s]['_PK'], data[s]['_sh'] = LO, PK, sh
        allLO = np.concatenate([data['disc']['_LO'], data['conf']['_LO']])
        ci = {c: j for j, c in enumerate(conds)}
        h1 = {'context_win': float(np.mean(allLO[:, ci['story', 1, 1]] > 0)), 'world_prior_win': float(np.mean(allLO[:, ci['story', 1, 0]] < 0))}
        h1['pass'] = bool(h1['context_win'] >= C['H1_win_min'] and h1['world_prior_win'] >= C['H1_win_min'])
        cell_means = {f'{src}/r{r}/{"book" if S else "world"}': {'mean_LO': float(allLO[:, j].mean()), 'frac_K': float(np.mean(allLO[:, j] > 0))}
                      for j, (src, r, S) in enumerate(conds)}
        law = A.law_eval(data['disc'], data['conf'], 's', C)
        law_hat = A.law_eval(data['disc'], data['conf'], 's_hat', C)
        corr = float(np.corrcoef(np.r_[data['disc']['_PK'][:, 2], data['conf']['_PK'][:, 2]],
                                 np.r_[data['disc']['_sh'], data['conf']['_sh']])[0, 1])
        res[lang] = {'H1': h1, 'cells': cell_means, 'law': law, 'law_s_hat': law_hat, 'corr_s_s_hat': corr}
        verdict['H1'][lang] = h1['pass']
        verdict['H2'][lang] = bool(law['R2_conf'] >= C['H2_r2_min'])
        verdict['H3'][lang] = bool(law['ci95']['s'][1] < 0)
        verdict['H4'][lang] = bool(law['R2_conf'] - law_hat['R2_conf'] <= C['H4_r2_drop_max'] and law_hat['ci95']['s'][1] < 0)
        # DLA 描述：r = 1 童话来源，两种范围；逐层注意力 / MLP 直接作用的平均，及其差（书里 − 现实）
        D = np.concatenate([z[f'disc_{lang}_DLA'], z[f'conf_{lang}_DLA']])
        jb, jw = ci['story', 1, 1], ci['story', 1, 0]
        res[lang]['dla'] = {'recon_max_abs_err': float(np.abs(D.sum(-1) - allLO).max()),
                            'book': D[:, jb].mean(0).tolist(), 'world': D[:, jw].mean(0).tolist(),
                            'order': 'emb, attn_0, mlp_0, attn_1, mlp_1, ...'}
        print(f'@@progress {lang} H1={h1} R2={law["R2_conf"]:.3f} cues={law["R2_conf_cues_only"]:.3f} '
              f'gamma={law["coef"]["s"]:.3f} ci={law["ci95"]["s"]} R2hat={law_hat["R2_conf"]:.3f}', flush=True)
    upd['results'] = res
    save(model_path, upd, verdict)
    print(json.dumps(verdict, ensure_ascii=False))


def do_steer(model_path):
    import torch
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = key_of(model_path)
    tok, model = load_model(model_path)
    blocks = model.model.layers
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    palette, forms = palette_forms()
    T = colour_targets(key)
    codes = np.load(V1.native_dir(key) / 'codes.npz')
    crow = V2.new_entities()
    idx = {r['id']: i for i, r in enumerate(crow)}
    ents = [r for r in crow if r['id'] in set(pre['steer']['entities'])]
    rng = np.random.default_rng(C['seed'])
    rand_dirs = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], model.config.hidden_size))]
    k = C['k_grid']
    ctl = {'pos': None, 'vecs': None}

    def pre_hook(mod, args, kwargs):
        if ctl['vecs'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        h[:, ctl['pos']] += ctl['vecs'].to(h.dtype)[:, None, :]
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
    t0 = time.time()
    out = {}
    for lang in ('zh', 'en'):
        l = COMMON_LAYER[lang]
        Xd = codes[lang][:, l].astype(np.float64)
        Yd = T['disc', lang] - T['disc', lang].mean(1, keepdims=True)
        _, lam, _ = E.select_layer_lambda([Xd], Yd, E.CRITERIA_V2['lambdas_rel'], 5, 0)
        W = E.ridge_fit(Xd, Yd, lam)['W']
        res = {S: {'op': [], 'rnd': [[] for _ in rand_dirs]} for S in (0, 1)}
        hk = blocks[l].register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            for r in ents:
                P, K, _ = pick_pk(T['conf', lang][idx[r['id']]], r['id'], lang)
                v = W[:, P] - W.mean(1)
                v = v / np.linalg.norm(v)
                sig = float((Xd @ v).std())
                kw = forms[lang][palette[K]][0].strip()
                tk, tp = enc(forms[lang][palette[K]][0])[0], enc(forms[lang][palette[P]][0])[0]
                vecs = [-kk * sig * v for kk in k] + [kk * sig * rd for rd in rand_dirs for kk in k]
                for S in (0, 1):
                    text = prompt(r, lang, 'story', 1, S, kw)
                    ids = enc(text)
                    ctl.update(pos=positions(text, r['zh'] if lang == 'zh' else r['en']),
                               vecs=torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda'))
                    lg = model(input_ids=torch.as_tensor([ids] * len(vecs), device='cuda')).logits[:, -1].float()
                    lo = (lg[:, tk] - lg[:, tp]).cpu().numpy()
                    nk = len(k)
                    d = lo - lo[k.index(0)]
                    res[S]['op'].append(d[:nk])
                    for j in range(len(rand_dirs)):
                        res[S]['rnd'][j].append(d[(1 + j) * nk:(2 + j) * nk])
        finally:
            ctl['vecs'] = None
            hk.remove()
        out[lang] = {'layer': l}
        for S in (0, 1):
            vv = E.steer_verdict(np.array(res[S]['op']), np.array([np.array(x) for x in res[S]['rnd']]), k, {'min_swing': C['min_swing']})
            out[lang]['book' if S else 'world'] = vv
            print(f'@@progress {lang} {"book" if S else "world"} swing={vv["swing"]:.2f} rand={[round(x, 2) for x in vv["random_swings"]]} '
                  f'ok={vv["supported_write"]} {round(time.time() - t0)}s', flush=True)
    save(model_path, {'H5_steer': out, 'seconds_steer': round(time.time() - t0, 1)},
         {'H5': {lang: out[lang]['world']['supported_write'] for lang in out}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze', 'steer'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model), 'analyze': lambda: do_analyze(a.model),
     'steer': lambda: do_steer(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
