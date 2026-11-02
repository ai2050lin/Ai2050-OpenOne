# -*- coding: utf-8 -*-
"""PS-v1：真值写入的「层 × 位置」扫描（TS）与仲裁上下文搬运的逐头路径（AH），在 Qwen3-4B（bf16）与 GLM4-9B（int8）上。

TS  问题：XM-v1 中 GLM 的真值写入（C6，句末、3 个相对深度）不成立，是层不对、位置不对，还是模型本身不区分真假？
    材料：MV-v2 裸陈述句（4 格 = 否定 × 所述类别对错）+ 行为门后缀；干预实体 = XM-v1 C6 的 56 个确认集实体，按类别确定性分 A / B。
    方向：层 l、位置组 p 上「真 − 假」的发现集均值差（end / span 用句末状态，final 用答案位置状态），σ 为发现集投影标准差，k·σ·v 加在第 l 块输入。
    读出：后缀之后「对 − 错」完整答案对数概率间隔，相对 k = 0 的变化。
    H1  行为门 ≥ 0.7 的语言上，GLM 在 A 选出的格在 B 上确认（steer_verdict）。
    H2  GLM（过门语言）：答案位置（final）的最大 A 摆幅大于句末（end）的最大 A 摆幅。
    描述：两个模型全网格摆幅图；句末与答案位置的真值可读出性（发现集拟合、确认集检验）与两种方向的 cos；Qwen3-4B 作参照。
AH  问题：「书里」范围下把上下文颜色写到答案的注意力，是否集中在少数几个头？这些头是否看上下文颜色词？消融后上下文是否失去胜出？
    「现实」范围下是不是同一批头被关小？
    材料：CF-v2 条件 (童话, r = 1, S ∈ {书里, 现实})，发现集 210 / 确认集 197 实体，两种语言；P / K 用本模型颜色目标（XM-v1 data.npz）。
    逐头直接作用：o_proj 输入的头切片 z_h 乘 W_o 对应列（W_o 由 o_proj(单位阵) 取得，int8 也适用），在冻结的最终 RMS 下投到 Wu[K] − Wu[P]。
    H1  确认集前 10 头（只按发现集选）占全部正向头直接作用之和 ≥ 0.5（每个模型 × 语言）。
    H2  前 10 头从答案位置注视上下文颜色词的平均注意力 ≥ 0.2 且 ≥ 全部头平均的 3 倍。
    H3  确认集上把前 10 头在答案位置的输出置零：书里 LO 平均下降 ≥ 每个同层随机集合的 3 倍，且上下文胜出率下降 ≥ 0.2（总效应，含下游补偿）。
    H4  现实范围下前 10 头的直接作用之和 ≤ 书里的 0.5 倍。
用法：python scripts/path_scan_v1.py prereg | truth --model M [--int8] | heads --model M [--int8] | analyze
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import mode_atlas as M  # noqa: E402
from shared.dist_protocol import path_scan as P  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402
import entity_atlas_v3 as V3  # noqa: E402
import mode_atlas_v1 as MV1  # noqa: E402
import mode_atlas_v2 as MV2  # noqa: E402
import arbitration_v1 as CF  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/path_scan/PS-v1.prereg.json'
RUNS = {'qwen3-4b': False, 'glm4-9b-chat-hf': True}      # 模型 → 是否 int8


def prereg_doc():
    return {'id': 'PS-v1', 'schema': 'path_scan_prereg.v1', 'created': '2026-11-02',
            'basis': 'XM-v1（Phase 273）：GLM4-9B 真值写入 C6 未复现；CF-v2 / XM-v1 C9：书里范围上下文答案主要由后段注意力直接写出',
            'models': {k: {'path': f'models/hf/{k}', 'precision': 'int8 (bitsandbytes LLM.int8)' if v else 'bf16'} for k, v in RUNS.items()},
            'data': 'XM-v1 data.npz（句末状态、颜色目标、行为门）；答案位置状态与逐头量在本版新采',
            'steer_entities': [r['id'] for r in V3.pick_entities()],
            'gate_from_xm_v1': 'XM-v1 C4 成对行为门：GLM zh 0.545（未过）、en 0.753；Qwen3-4B zh 0.909、en 0.990。未过门的语言只描述',
            'attention_impl': 'AH 用 eager 注意力以取得注意力权重（与 sdpa 有数值差异，LO 另行记录）',
            'power_note': 'H3 的胜出率为逐题配对二值：check_phase(0.2, 197) 所需 n = 20，检验力 1.0；TS 为连续摆幅判定，沿用 XM-v1 的 56 实体 × 4 句（A / B 各一半）',
            'doc': __doc__, 'criteria_ts': P.CRITERIA_TS, 'criteria_ah': P.CRITERIA_AH}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（PS-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def load(model_path, int8, eager=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)} if int8 else {}
    if eager:
        kw['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda', **kw).eval()
    return tok, model


def out_dir(key):
    return XM.out_dir(key).parent / 'PS-v1'


def report_path(model_path, int8):
    d = ROOT / 'research/path_scan' / f'PS-v1-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def answer_lp(R, seqs, starts, answers, layer=None, pos_rows=None, vecs=None):
    """只在答案位置取 logits（避免整段 × 词表的显存）；可选在第 layer 块输入的指定位置加向量。返回每行答案完整对数概率。"""
    torch = R.torch
    ids, att = R._batch(seqs)
    hk = None
    if layer is not None:
        V = torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda')

        def hook(mod, args, kwargs):
            h = (args[0] if args else kwargs['hidden_states']).clone()
            for b, ps in enumerate(pos_rows):
                h[b, ps] += V[b].to(h.dtype)
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs['hidden_states'] = h
            return args, kwargs
        hk = R.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        hs = R.model.model(input_ids=ids, attention_mask=att).last_hidden_state
    finally:
        if hk is not None:
            hk.remove()
    rows, pos, tgt = [], [], []
    for b, (st, a) in enumerate(zip(starts, answers)):
        for j, t in enumerate(a):
            rows.append(b); pos.append(st - 1 + j); tgt.append(t)
    rows_t = torch.as_tensor(rows, device='cuda')
    lp = torch.log_softmax(R.model.lm_head(hs[rows_t, torch.as_tensor(pos, device='cuda')]).float(), -1)
    lp = lp[torch.arange(len(rows), device='cuda'), torch.as_tensor(tgt, device='cuda')]
    out = torch.zeros(len(seqs), device='cuda').index_add_(0, rows_t, lp)
    return out.cpu().numpy()


# ---------------- TS：真值写入扫描 ----------------

def do_truth(model_path, int8):
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria_ts']
    snap, key = XM.key_of(model_path, int8)
    od = out_dir(key)
    od.mkdir(parents=True, exist_ok=True)
    with np.load(XM.out_dir(key) / 'data.npz') as f:
        bare = {k_: f[k_] for k_ in f.files if k_.endswith('_bare')}
    tok, model = load(model_path, int8)
    R = XM.Runner(tok, model)
    enc, L = R.enc, R.L
    rows = XM.rows()
    t0 = time.time()
    # 1 答案位置（后缀最后一个 token）的逐层状态，发现集与确认集
    fin_p = od / 'final_states.npz'
    if fin_p.exists():
        with np.load(fin_p) as f:
            fin = {k_: f[k_] for k_ in f.files}
    else:
        fin = {}
        for s, rs in rows.items():
            for lang in ('zh', 'en'):
                suf = MV1.GATE[lang][1]
                F = np.zeros((len(rs), 4, L + 1, model.config.hidden_size), dtype=np.float16)
                for i in range(0, len(rs), 8):
                    chunk = rs[i:i + 8]
                    seqs = [enc(t) + enc(suf) for r in chunk for t in MV2.bare(r, lang)]
                    F[i:i + len(chunk)] = R.last_states(seqs).reshape(len(chunk), 4, L + 1, -1)
                fin[f'{s}_{lang}_final'] = F
                print(f'@@progress final {s}/{lang} {round(time.time() - t0)}s', flush=True)
        np.savez(fin_p, **fin)
    # 2 描述：两种位置的真值可读出性与方向 cos（发现集拟合、确认集检验）
    tm = np.array([0, 1, 1, 0], bool)
    nd, nc = len(rows['disc']), len(rows['conf'])
    grid = list(range(0, L, C['layer_step']))
    read = {}
    for lang in ('zh', 'en'):
        read[lang] = []
        for l in grid:
            row = {'layer': l}
            vv = {}
            for pos, src in (('end', bare), ('final', fin)):
                key_ = lambda s: f'{s}_{lang}_bare' if pos == 'end' else f'{s}_{lang}_final'
                Xd = src[key_('disc')][:, :, l].astype(np.float64).reshape(-1, src[key_('disc')].shape[-1])
                Xc = src[key_('conf')][:, :, l].astype(np.float64).reshape(-1, src[key_('conf')].shape[-1])
                pr = M.probe(Xd, np.tile(tm, nd), np.repeat(np.arange(nd), 4), Xc, np.tile(tm, nc), M.CRITERIA_V2)
                row[f'{pos}_acc'] = pr['acc']
                Sd = src[key_('disc')][:, :, l].astype(np.float64)
                vv[pos] = Sd[:, tm].mean((0, 1)) - Sd[:, ~tm].mean((0, 1))
            row['cos_end_final'] = M.cos(vv['end'], vv['final'])
            read[lang].append(row)
        print(f'@@progress read {lang} {round(time.time() - t0)}s', flush=True)
    # 3 扫描
    ents = V3.pick_entities()
    A = P.split_ab([r['id'] for r in ents], [r['cat'] for r in ents])
    rng = np.random.default_rng(C['seed'])
    rand = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], model.config.hidden_size))]
    k = C['k_grid']
    nk, nv = len(k), len(k) * (1 + C['n_random'])
    raw = {}
    for lang in ('zh', 'en'):
        _, suf, tr, fa = MV1.GATE[lang]
        sid, tids, fids = enc(suf), enc(tr[0]), enc(fa[0])
        dirs = {}
        for l in grid:
            for pos in C['positions']:
                Sd = (fin[f'disc_{lang}_final'] if pos == 'final' else bare[f'disc_{lang}_bare'])[:, :, l].astype(np.float64)
                v = Sd[:, tm].mean((0, 1)) - Sd[:, ~tm].mean((0, 1))
                v /= np.linalg.norm(v)
                sg = float((Sd.reshape(-1, Sd.shape[-1]) @ v).std())
                dirs[l, pos] = [kk * sg * v for kk in k] + [kk * sg * rd for rd in rand for kk in k]
        D = np.zeros((len(grid), len(C['positions']), len(ents), 4, nv), dtype=np.float32)
        for ei, r in enumerate(ents):
            sents = [enc(t) for t in MV2.bare(r, lang)]
            for li, l in enumerate(grid):
                seqs, starts, ans, prow, vecs = [], [], [], [], []
                for pos in C['positions']:
                    for ids in sents:
                        p = ids + sid
                        ppos = {'end': [len(ids) - 1], 'span': list(range(len(ids))), 'final': [len(p) - 1]}[pos]
                        for a in (tids, fids):
                            for vec in dirs[l, pos]:
                                seqs.append(p + a); starts.append(len(p)); ans.append(a); prow.append(ppos); vecs.append(vec)
                lp = answer_lp(R, seqs, starts, ans, l, prow, vecs).reshape(len(C['positions']), 4, 2, nv)
                mg = lp[:, :, 0] - lp[:, :, 1]
                D[li, :, ei] = mg - np.concatenate([mg[..., j * nk + k.index(0)][..., None].repeat(nk, -1) for j in range(1 + C['n_random'])], -1)
            if ei % 8 == 7:
                print(f'@@progress scan {lang} ent {ei + 1}/{len(ents)} {round(time.time() - t0)}s', flush=True)
        raw[lang] = D
    np.savez(od / 'truth_scan.npz', **{f'{lang}_D': raw[lang] for lang in raw}, A=A, grid=np.array(grid))
    # 4 判定
    gate = json.loads(XM.report_path(model_path, int8).read_text(encoding='utf-8'))['results']['C4']
    res = {}
    for lang in ('zh', 'en'):
        D = raw[lang]
        itemA = np.repeat(A, 4)
        cells = {}
        swing = {}
        for li, l in enumerate(grid):
            for pi, pos in enumerate(C['positions']):
                d = D[li, pi].reshape(-1, nv).astype(np.float64)
                cells[f'L{l}/{pos}'] = {'op': d[:, :nk], 'rand': np.stack([d[:, (1 + j) * nk:(2 + j) * nk] for j in range(C['n_random'])])}
                v = P.steer_verdict(cells[f'L{l}/{pos}']['op'], cells[f'L{l}/{pos}']['rand'], k, C['min_swing'])
                swing[f'L{l}/{pos}'] = {'swing': v['swing'], 'supported_all': v['supported_write']}
        sc = P.select_and_confirm(cells, itemA, C)
        best = lambda pos: max(sc['swing_A'][f'L{l}/{pos}'] for l in grid)
        g = gate[lang]['gate']['acc']
        res[lang] = {'gate': g, 'gate_pass': bool(g >= C['gate_min']), 'select': sc, 'grid': swing,
                     'best_A_swing': {pos: best(pos) for pos in C['positions']},
                     'H2_final_over_end': bool(best('final') > best('end')), 'readability': read[lang]}
        print(f'@@progress TS {lang} gate={g:.2f} cell={sc["cell"]} confirmed={sc["confirmed"]} best={res[lang]["best_A_swing"]}', flush=True)
    save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                            'precision': 'int8' if int8 else 'bf16', 'n_blocks': L, 'TS': res, 'seconds_TS': round(time.time() - t0, 1)})


# ---------------- AH：仲裁上下文搬运头 ----------------

def do_heads(model_path, int8):
    import torch
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria_ah']
    snap, key = XM.key_of(model_path, int8)
    od = out_dir(key)
    od.mkdir(parents=True, exist_ok=True)
    with np.load(XM.out_dir(key) / 'data.npz') as f:
        colour = {k_: f[k_] for k_ in f.files if k_.endswith('_color')}
    tok, model = load(model_path, int8, eager=True)
    R = XM.Runner(tok, model)
    enc, L = R.enc, R.L
    cfg = model.config
    H = cfg.num_attention_heads
    dh = getattr(cfg, 'head_dim', None) or cfg.hidden_size // H
    palette, forms = CF.palette_forms()
    rows = XM.rows()
    t0 = time.time()
    # W_o（转置）：o_proj(I) − o_proj(0)，对 int8 也给出模型实际使用的权重
    Wt, bias = [], []
    for b in R.blocks:
        op = b.self_attn.o_proj
        I = torch.eye(H * dh, device='cuda', dtype=torch.bfloat16)
        z0 = op(torch.zeros(1, H * dh, device='cuda', dtype=torch.bfloat16)).float()
        Wt.append((op(I).float() - z0).to(torch.bfloat16))
        bias.append(z0[0])
    cap = {}
    hooks = [b.self_attn.o_proj.register_forward_pre_hook(lambda m, a, li=li: cap.__setitem__(('z', li), a[0])) for li, b in enumerate(R.blocks)]
    hooks += [b.self_attn.register_forward_hook(lambda m, a, o, li=li: cap.__setitem__(('a', li), o[0])) for li, b in enumerate(R.blocks)]
    hooks.append(model.model.norm.register_forward_hook(lambda m, a, o: cap.__setitem__('pre', a[0])))
    nw = model.model.norm.weight.float()
    eps = getattr(model.model.norm, 'variance_epsilon', getattr(model.model.norm, 'eps', 1e-6))
    Wu = model.lm_head.weight
    ablate = {'sets': None}

    def abl_hook(li):
        def f(m, a):
            if ablate['sets'] is None:
                return None
            z = a[0].clone()
            for b, (heads, last) in enumerate(ablate['sets']):
                for h in heads.get(li, ()):
                    z[b, last, h * dh:(h + 1) * dh] = 0
            return (z,) + tuple(a[1:])
        return f
    hooks += [b.self_attn.o_proj.register_forward_pre_hook(abl_hook(li)) for li, b in enumerate(R.blocks)]

    def items(s, lang):
        out = []
        for i, r in enumerate(rows[s]):
            Pc, Kc, _ = CF.pick_pk(colour[f'{s}_{lang}_color'][i], r['id'], lang)
            kw = forms[lang][palette[Kc]][0].strip()
            tk, tp = enc(forms[lang][palette[Kc]][0])[0], enc(forms[lang][palette[Pc]][0])[0]
            for Sx in (1, 0):
                text = CF.prompt(r, lang, 'story', 1, Sx, kw)
                ids = enc(text)
                out.append({'i': i, 'S': Sx, 'ids': ids, 'tk': tk, 'tp': tp, 'kpos': XM.positions(tok, text, kw),
                            'epos': XM.positions(tok, text, r['zh'] if lang == 'zh' else r['en'])})
        return out

    def run(batch, sets=None, full=False):
        ablate['sets'] = sets
        try:
            ids, att = R._batch([x['ids'] for x in batch])
            o = model.model(input_ids=ids, attention_mask=att, output_attentions=full)
        finally:
            ablate['sets'] = None
        ar = torch.arange(len(batch), device='cuda')
        last = torch.as_tensor([len(x['ids']) - 1 for x in batch], device='cuda')
        hl = o.last_hidden_state[ar, last]
        lg = model.lm_head(hl).float()
        tk = torch.as_tensor([x['tk'] for x in batch], device='cuda'); tp = torch.as_tensor([x['tp'] for x in batch], device='cuda')
        LO = (lg[ar, tk] - lg[ar, tp]).cpu().numpy()
        if not full:
            return LO
        v = nw[None] * (Wu[tk] - Wu[tp]).float()                                # [B, d]
        pre_ = cap['pre'][ar, last].float()
        rms = torch.sqrt((pre_ ** 2).mean(-1) + eps)                              # [B]
        dla = torch.zeros(len(batch), L, H, device='cuda')
        recon = torch.zeros(len(batch), device='cuda')
        for li in range(L):
            u = (v.to(torch.bfloat16) @ Wt[li].T).float()                         # [B, H·dh]
            z = cap['z', li][ar, last].float()
            dla[:, li] = (z * u).view(len(batch), H, dh).sum(-1) / rms[:, None]
            tot = (cap['a', li][ar, last].float() * v).sum(-1) / rms
            recon = torch.maximum(recon, (dla[:, li].sum(-1) + (bias[li][None] * v).sum(-1) / rms - tot).abs())
        AK = torch.zeros(len(batch), L, H, device='cuda')
        AE = torch.zeros(len(batch), L, H, device='cuda')
        for li, a in enumerate(o.attentions):
            for b, x in enumerate(batch):
                AK[b, li] = a[b, :, last[b], x['kpos']].float().sum(-1)
                AE[b, li] = a[b, :, last[b], x['epos']].float().sum(-1)
        return LO, dla.cpu().numpy(), AK.cpu().numpy(), AE.cpu().numpy(), recon.cpu().numpy()

    data, res = {}, {}
    for lang in ('zh', 'en'):
        for s in ('disc', 'conf'):
            it = items(s, lang)
            n = len(rows[s])
            LO = np.zeros((n, 2)); DLA = np.zeros((n, 2, L, H), np.float32); AK = np.zeros_like(DLA); AE = np.zeros_like(DLA)
            rec = 0.0
            for j in range(0, len(it), 16):
                bt = it[j:j + 16]
                lo, d, ak, ae, rc = run(bt, full=True)
                rec = max(rec, float(rc.max()))
                for b, x in enumerate(bt):
                    c = 1 - x['S']                                           # 0 = 书里，1 = 现实
                    LO[x['i'], c], DLA[x['i'], c], AK[x['i'], c], AE[x['i'], c] = lo[b], d[b], ak[b], ae[b]
            data.update({f'{s}_{lang}_LO': LO, f'{s}_{lang}_DLA': DLA, f'{s}_{lang}_AK': AK, f'{s}_{lang}_AE': AE})
            print(f'@@progress AH collect {s}/{lang} recon={rec:.4f} {round(time.time() - t0)}s', flush=True)
            data[f'{s}_{lang}_recon'] = np.array(rec)
        # 选头（只用发现集书里）、确认集判定
        top = P.top_heads(data[f'disc_{lang}_DLA'][:, 0], C['top_k'])
        rsets = P.random_matched(top, H, C['n_random_sets'], C['seed'])
        Dc = data[f'conf_{lang}_DLA']
        h1 = P.concentration(Dc[:, 0], top)
        h2 = P.attn_focus(data[f'conf_{lang}_AK'][:, 0], top, C)
        h2['entity_top_mean'] = float(data[f'conf_{lang}_AE'][:, 0].reshape(len(Dc), -1)[:, top].mean())
        h2['world'] = P.attn_focus(data[f'conf_{lang}_AK'][:, 1], top, C)
        # 消融：确认集，书里与现实，基线 / 前 k / 随机集合
        it = items('conf', lang)
        as_sets = lambda flat: {li: [i % H for i in flat if i // H == li] for li in {i // H for i in flat}}
        variants = [{}] + [as_sets(top)] + [as_sets(s_) for s_ in rsets]
        AB = np.zeros((len(rows['conf']), 2, len(variants)))
        for x in it:
            batch = [x] * len(variants)
            lo = run(batch, sets=[(v, len(x['ids']) - 1) for v in variants])
            AB[x['i'], 1 - x['S']] = lo
        data[f'conf_{lang}_ABL'] = AB
        h3 = P.ablation_verdict(AB[:, 0, 0], AB[:, 0, 1], AB[:, 0, 2:].T, C)
        h3['world'] = {'lo_change': float((AB[:, 1, 1] - AB[:, 1, 0]).mean()),
                       'random_changes': [float((AB[:, 1, 2 + j] - AB[:, 1, 0]).mean()) for j in range(len(rsets))],
                       'prior_win_base': float((AB[:, 1, 0] < 0).mean()), 'prior_win_ablated': float((AB[:, 1, 1] < 0).mean())}
        h3['direct_top_sum'] = h1['top_sum']
        h3['eager_vs_xm_note'] = '基线 LO 为 eager 注意力；XM-v1 为 sdpa'
        mw = Dc[:, 1].reshape(len(Dc), -1).mean(0)
        h4 = {'world_top_sum': float(mw[top].sum()), 'book_top_sum': h1['top_sum'],
              'ratio': float(mw[top].sum() / max(h1['top_sum'], 1e-30)),
              'pass': bool(mw[top].sum() <= C['H4_ratio_max'] * h1['top_sum'])}
        res[lang] = {'top_heads': [{'layer': i // H, 'head': i % H, 'disc_mean': float(data[f'disc_{lang}_DLA'][:, 0].reshape(-1, L * H)[:, i].mean()),
                                    'conf_mean': float(Dc[:, 0].reshape(-1, L * H)[:, i].mean()),
                                    'conf_world_mean': float(mw[i]),
                                    'attn_kw': float(data[f'conf_{lang}_AK'][:, 0].reshape(-1, L * H)[:, i].mean()),
                                    'attn_kw_world': float(data[f'conf_{lang}_AK'][:, 1].reshape(-1, L * H)[:, i].mean())} for i in top],
                     'random_sets': rsets, 'H1': {**h1, 'pass': bool(h1['share'] >= C['H1_share_min'])}, 'H2': h2, 'H3': h3, 'H4': h4,
                     'recon_max_abs_err': float(max(data[f'disc_{lang}_recon'], data[f'conf_{lang}_recon'])),
                     'win_book_eager': float((data[f'conf_{lang}_LO'][:, 0] > 0).mean()),
                     'win_world_eager': float((data[f'conf_{lang}_LO'][:, 1] < 0).mean())}
        print(f'@@progress AH {lang} H1={h1["share"]:.2f} H2={h2["top_mean"]:.2f}/{h2["ratio"]:.1f} H3 drop={h3["lo_drop"]:.2f} '
              f'rand={max(h3["random_drops"]):.2f} win {h3["win_base"]:.2f}->{h3["win_ablated"]:.2f} H4={h4["ratio"]:.2f} {round(time.time() - t0)}s', flush=True)
    for h in hooks:
        h.remove()
    np.savez(od / 'heads.npz', **data)
    save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                            'precision': 'int8' if int8 else 'bf16', 'n_blocks': L, 'n_heads': H, 'AH': res, 'seconds_AH': round(time.time() - t0, 1)})


# ---------------- 汇总 ----------------

def posthoc_positions(model_path, int8, C):
    """事后（不进判定）：hs[0] 上真假句末 token 相同，方向为零、摆幅为 NaN；H2 的最大摆幅改用 nanmax 重算。
    另对每种位置分别在 A 上选格（成立格中摆幅最大）、在 B 上确认，看句末 / 全句位置是否也有可确认的写入层。"""
    _, key = XM.key_of(model_path, int8)
    with np.load(out_dir(key) / 'truth_scan.npz') as f:
        A, grid = f['A'], [int(x) for x in f['grid']]
        raw = {lang: f[f'{lang}_D'] for lang in ('zh', 'en')}
    k, nk = C['k_grid'], len(C['k_grid'])
    out = {}
    for lang, D in raw.items():
        itemA = np.repeat(A, 4)
        out[lang] = {}
        for pi, pos in enumerate(C['positions']):
            cells = {}
            for li, l in enumerate(grid):
                d = D[li, pi].reshape(-1, D.shape[-1]).astype(np.float64)
                if not np.isfinite(d).all():
                    continue
                cells[f'L{l}/{pos}'] = {'op': d[:, :nk], 'rand': np.stack([d[:, (1 + j) * nk:(2 + j) * nk] for j in range(C['n_random'])])}
            sc = P.select_and_confirm(cells, itemA, C)
            out[lang][pos] = {'best_A_swing': float(np.nanmax(list(sc['swing_A'].values()))), 'cell': sc['cell'], 'confirmed': sc['confirmed'],
                              'B_swing': sc['B']['swing'] if sc.get('B') else None,
                              'supported_A_layers': [c.split('/')[0] for c in cells
                                                     if P.steer_verdict(cells[c]['op'][itemA], cells[c]['rand'][:, itemA], k, C['min_swing'])['supported_write']]}
        out[lang]['H2_final_over_end_nanmax'] = bool(out[lang]['final']['best_A_swing'] > out[lang]['end']['best_A_swing'])
    return out


def do_analyze():
    from provenance import write_result
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    out = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'models': {}, 'verdict': {}}
    for name, int8 in RUNS.items():
        mp = f'models/hf/{name}'
        p = report_path(mp, int8)
        if not p.exists():
            continue
        rep = json.loads(p.read_text(encoding='utf-8'))
        v = {}
        if 'TS' in rep:
            ph = posthoc_positions(mp, int8, pre['criteria_ts'])
            rep = save(mp, int8, {'TS_posthoc': ph})
            gl = [lang for lang in ('zh', 'en') if rep['TS'][lang]['gate_pass']]
            v['TS_H1'] = {lang: rep['TS'][lang]['select']['confirmed'] for lang in gl}
            v['TS_H2'] = {lang: ph[lang]['H2_final_over_end_nanmax'] for lang in gl}
        out['models'][name] = rep
        if 'AH' in rep:
            for h in ('H1', 'H2', 'H3', 'H4'):
                v[f'AH_{h}'] = {lang: rep['AH'][lang][h]['pass'] for lang in ('zh', 'en')}
        out['verdict'][name] = v
    p = ROOT / 'research/path_scan/PS-v1-summary.json'
    p.write_text(json.dumps({k: out[k] for k in ('prereg_sha256', 'verdict')}, ensure_ascii=False, indent=1), encoding='utf-8')
    r = write_result('ps_v1', out, model_path=';'.join(f'models/hf/{k}' for k in RUNS), seed=pre['criteria_ts']['seed'], tool='scripts/path_scan_v1.py')
    print(json.dumps(out['verdict'], ensure_ascii=False, indent=1), '\nprovenance →', r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'truth', 'heads', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--int8', action='store_true')
    a = ap.parse_args()
    {'prereg': do_prereg, 'truth': lambda: do_truth(a.model, a.int8), 'heads': lambda: do_heads(a.model, a.int8),
     'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
