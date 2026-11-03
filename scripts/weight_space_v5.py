# -*- coding: utf-8 -*-
"""WS-v5：模式状态的自持——同样内容、不同顺序，中性段之后模型还记得「最后是哪种模式」吗？

接 Phase 287（WS-v4）：上游 8 个头按「同模式」读取（查询的代码分量决定看不看代码 key）。候选不动点结构预测：
上下文同时有散文与代码时，最近的模式决定状态，状态再决定读哪一类 token，从而自持。
  材料：MF-v1 发现集（16 任务 × 5 填充 × 中 / 英）。同一任务同一填充的散文 P 与代码 C 按两种顺序拼接，
      PC = P + 空行 + C + 空行 + N，CP = C + 空行 + P + 空行 + N；N 为确定性数字列表（2 / 8 / 32 个数字，约 4 / 16 / 64 个 token）。
  模式投影 m_t = hs[33] · v_33 / Δμ_33（v_33：发现集 token 配平代码方向；Δμ：发现集共用 token 位置上代码与散文的平均投影差）。
  H1（自持）：N 末位 gap(L) = mean(m_PC − m_CP) ≥ 0.2（L ≈ 4），且 gap(64) / gap(4) ≥ 0.5。
  H2（状态决定选择）：L ≈ 16 的 N 末位，8 个头对代码 token 的注意力质量比 PC / CP ≥ 2（两种顺序的上下文 token 相同），
      且高于 WS-v2 的 5 组同块随机头的同一比值平均。
  H3（因果）：CP 顺序、L ≈ 16，只在 N 的位置上给 8 个头的查询加 +4σ 代码分量（σ 与随机方向同 WS-v4），
      N 末位代码质量比 ≥ 1.5 × 原始，且 m 向代码移动 ≥ 0.1 × gap(16)；随机方向同样移动的 m 变化 ≤ 真实的 1/3。
  描述：各长度下的逐位置 gap 曲线；下一 token 的代码专有词质量差。
注意力用 WS-v3 的自定义 eager。被测模型：Qwen3-4B（bf16，不使用量化）。
用法：python scripts/weight_space_v5.py prereg | run | analyze
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import weight_read as W  # noqa: E402
from shared.dist_protocol import mode_field as F  # noqa: E402
import mf_materials_v1 as MAT  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402
import weight_space_v1 as WS1  # noqa: E402
import weight_space_v2 as WS2  # noqa: E402
import weight_space_v4 as WS4  # noqa: E402

PREREG = ROOT / 'research/weight_space/WS-v5.prereg.json'
MODEL = WS1.MODEL
STY = F.STYLES
CODE = STY.index('code')
ML = 33
NNUM = {4: 2, 16: 8, 64: 32}


def prereg_doc():
    return {'id': 'WS-v5', 'schema': 'weight_space_prereg.v5', 'created': '2026-11-02',
            'basis': 'Phase 287（WS-v4）：同模式读取由 QK 双线性实现；本版检验它是否带来模式状态的自持（候选不动点）',
            'models': {'qwen3-4b': 'bf16'}, 'precision_rule': '不使用任何量化',
            'heads': 'research/weight_space/WS-v2-qwen3-4b/report.json 的 selected 与 random_sets',
            'sigma': 'research/weight_space/WS-v4-qwen3-4b/report.json 的 sigma（同一 σ 与随机方向种子）',
            'materials': 'MF-v1 发现集任务的散文与代码文本，两种顺序 + 数字列表中性段',
            'power_note': '每种语言 80 对 × 2 种顺序 × 3 种长度；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_V5}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（WS-v6）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/WS-v5-qwen3-4b'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(upd):
    p = report_path()
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def items():
    out = []
    by = {}
    for c in MAT.corpus():
        if c['split'] == 'disc':
            by.setdefault(c['item'], {})[c['style']] = c['text']
    for it, d in sorted(by.items()):
        for lang in ('zh', 'en'):
            out.append({'item': it, 'lang': lang, 'P': d[lang], 'C': d['code']})
    return out


def do_run():
    import torch
    import transformers.models.qwen3.modeling_qwen3 as MQ
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(MODEL, False)
    ws2 = json.loads(WS2.report_path().read_text(encoding='utf-8'))
    ws4 = json.loads(WS4.report_path().read_text(encoding='utf-8'))
    sel = [tuple(x) for x in ws2['selected']]
    rsets = [[tuple(x) for x in s_] for s_ in ws2['random_sets']]
    allh = sel + [x for s_ in rsets for x in s_]
    layers = sorted({l for l, _ in allh})
    sel_layers = sorted({l for l, _ in sel})
    z = MF1.load_fields(key)
    y, tk, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    vdir = {}
    for l in sel_layers + [ML]:
        X = z['Hs'][D[sh], l].astype(np.float32)
        vdir[l] = W.unit(F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], CODE))
        if l == ML:
            pr = X @ vdir[l]
            ys = y[sh][D[sh]]
            dmu = float(pr[ys == CODE].mean() - pr[ys != CODE].mean())
        del X
    del z
    gc.collect()
    tok, model = XM.load(MODEL, False)
    R = XM.Runner(tok, model)
    blocks = R.blocks
    cfg = model.config
    nh = cfg.num_attention_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nh)
    rng = np.random.default_rng(json.loads(WS4.PREREG.read_text(encoding='utf-8'))['criteria']['seed'])
    rdir = W.unit(rng.normal(0, 1, cfg.hidden_size))                 # 与 WS-v4 的第一个随机方向相同
    t0 = time.time()
    ctl = {'collect': {l: sorted({h for l2, h in allh if l2 == l}) for l in layers}, 'store': {}, 'q': {}, 'qmask': None}
    col = {l: {h: i for i, h in enumerate(v)} for l, v in ctl['collect'].items()}

    def ws_attn(module, query, key_, value, attention_mask, scaling, dropout=0.0, **kw):
        k_ = MQ.repeat_kv(key_, module.num_key_value_groups)
        v_ = MQ.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k_.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        lam = module.layer_idx
        if ctl['collect'].get(lam):
            ctl['store'][lam] = w[0, ctl['collect'][lam], -1].float()          # 只要末位查询 [heads, T]
        return torch.matmul(w, v_).transpose(1, 2).contiguous(), None

    MQ.eager_attention_forward = ws_attn
    model.set_attn_implementation('eager')

    def q_hook(l):
        def f(m, a, o):
            for h, dv in ctl['q'].get(l, []):
                o[..., h * hd:(h + 1) * hd] += dv * ctl['qmask'][..., None]
            return o
        return f

    hooks = [blocks[l].self_attn.q_proj.register_forward_hook(q_hook(l)) for l in sel_layers]
    gin = {l: blocks[l].input_layernorm.weight.float() for l in sel_layers}

    def shift(name, kk):
        ctl['q'] = {}
        if name is None:
            return
        for l, h in sel:
            v = vdir[l] if name == 'code' else rdir
            u = gin[l] * torch.as_tensor(v, device='cuda', dtype=torch.float32)
            u = u / u.norm()
            sig = ws4['sigma'][f'{l}.{name}']
            dv = kk * sig * (blocks[l].self_attn.q_proj.weight.float() @ u)[h * hd:(h + 1) * hd]
            ctl['q'].setdefault(l, []).append((h, dv.to(torch.bfloat16)))

    excl = json.loads((MF1.out_dir(key) / 'vocab.json').read_text(encoding='utf-8'))['excl']
    EXc = torch.as_tensor(excl['code'], device='cuda')
    v33 = torch.as_tensor(vdir[ML], device='cuda', dtype=torch.float32)

    def one(P, C, order, L, it):
        a, b = (P, C) if order == 'PC' else (C, P)
        head = a + '\n\n' + b + '\n\n'
        text = head + W.neutral_segment(it, NNUM[L])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        seq = enc.input_ids
        n0 = len(head) - 2
        ca, cb = (len(P) + 2, len(P) + 2 + len(C)) if order == 'PC' else (0, len(C))
        codek = np.array([ca <= s_ < cb for s_, _ in enc.offset_mapping])
        neut = np.array([s_ >= len(head) for s_, _ in enc.offset_mapping])
        ctl['qmask'] = torch.as_tensor(neut[None], device='cuda', dtype=torch.bfloat16)
        i_, att = R._batch([seq])
        o = R.model(input_ids=i_, attention_mask=att, output_hidden_states=True)
        mvec = (o.hidden_states[ML][0].float() @ v33).cpu().numpy() / dmu
        lp = torch.log_softmax(o.logits[0, -1].float(), -1)
        t = len(seq) - 1
        ratios = {hh: W.mass_ratio(ctl['store'][hh[0]][col[hh[0]][hh[1]]].cpu().numpy(), t, codek) for hh in allh}
        return {'m_end': float(mvec[-1]), 'm_neutral': mvec[neut].tolist(), 'n_neutral': int(neut.sum()),
                'code_mass': float(torch.logsumexp(lp[EXc], -1)), 'ratios': ratios, 'n_tokens': len(seq), 'neutral_start_char': n0}

    its = items()
    res = {}
    shift(None, 0)
    for L in crit['neutral_lengths']:
        for order in ('PC', 'CP'):
            res[(order, L, 'orig')] = [one(x['P'], x['C'], order, L, x['item']) for x in its]
        print(f'@@L {L} {round(time.time() - t0)}s', flush=True)
    for name in ('code', 'rand'):
        shift(name, 4)
        res[('CP', 16, f'q_{name}_4')] = [one(x['P'], x['C'], 'CP', 16, x['item']) for x in its]
        shift(None, 0)
        print(f'@@shift {name} {round(time.time() - t0)}s', flush=True)
    for h_ in hooks:
        h_.remove()
    rk = lambda hh: f'{hh[0]}.{hh[1]}'
    summ = {}
    for (order, L, cond), rows in res.items():
        summ[f'{order}|{L}|{cond}'] = {
            'm_end': float(np.mean([r['m_end'] for r in rows])), 'code_mass': float(np.mean([r['code_mass'] for r in rows])),
            'ratio': {rk(hh): float(np.nanmean([r['ratios'][hh] for r in rows])) for hh in allh},
            'n_neutral_mean': float(np.mean([r['n_neutral'] for r in rows])),
            'm_neutral_curve': np.nanmean(np.array([r['m_neutral'][:min(len(x['m_neutral']) for x in rows)] for r in rows]), 0).tolist()}
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'selected': [list(x) for x in sel], 'random_sets': [[list(x) for x in s_] for s_ in rsets], 'dmu_33': dmu,
          'n_pairs': len(its), 'summary': summ, 'seconds': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    S = rep['summary']
    g = lambda L: S[f'PC|{L}|orig']['m_end'] - S[f'CP|{L}|orig']['m_end']
    gaps = {L: g(L) for L in crit['neutral_lengths']}
    pers = W.persistence(gaps, 4, 64)
    h1 = {'gap': {str(k_): v for k_, v in gaps.items()}, 'persistence_64_over_4': pers,
          'pass': bool(gaps[4] >= crit['H1_gap_min'] and pers >= crit['H1_persist_min'])}
    sel = [f'{l}.{h}' for l, h in rep['selected']]
    rsets = [[f'{l}.{h}' for l, h in s_] for s_ in rep['random_sets']]
    mr = lambda key_, hs: float(np.mean([S[key_]['ratio'][h] for h in hs]))
    sel_pc, sel_cp = mr('PC|16|orig', sel), mr('CP|16|orig', sel)
    rr = [mr('PC|16|orig', s_) / mr('CP|16|orig', s_) for s_ in rsets]
    h2 = {'sel_PC': sel_pc, 'sel_CP': sel_cp, 'sel_ratio': sel_pc / sel_cp, 'random_ratios': rr, 'random_mean': float(np.mean(rr)),
          'pass': bool(sel_pc / sel_cp >= crit['H2_select_ratio_min'] and sel_pc / sel_cp > float(np.mean(rr)))}
    base = S['CP|16|orig']
    sc, sr = S['CP|16|q_code_4'], S['CP|16|q_rand_4']
    gain = mr('CP|16|q_code_4', sel) / sel_cp
    move = (sc['m_end'] - base['m_end']) / gaps[16]
    move_r = (sr['m_end'] - base['m_end']) / gaps[16]
    h3 = {'select_gain': gain, 'select_gain_random': mr('CP|16|q_rand_4', sel) / sel_cp, 'state_move_frac': move,
          'state_move_frac_random': move_r,
          'pass': bool(gain >= crit['H3_select_gain_min'] and move >= crit['H3_state_frac_min']
                       and abs(move_r) <= crit['H3_control_max_frac'] * move)}
    desc = {'code_mass_PC_minus_CP': {str(L): S[f'PC|{L}|orig']['code_mass'] - S[f'CP|{L}|orig']['code_mass'] for L in crit['neutral_lengths']},
            'n_neutral_tokens': {str(L): S[f'PC|{L}|orig']['n_neutral_mean'] for L in crit['neutral_lengths']},
            'gap_curve_64': (np.array(S['PC|64|orig']['m_neutral_curve'][:64]) - np.array(S['CP|64|orig']['m_neutral_curve'][:64])).tolist(),
            'm_end': {k_: v['m_end'] for k_, v in S.items()}}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'describe': desc, 'verdict': verdict})
    p = write_result('ws_v5_qwen3-4b', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/weight_space_v5.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\ncode_mass', desc['code_mass_PC_minus_CP'], '\nn_neutral', desc['n_neutral_tokens'],
          '\ncurve64', [round(x, 3) for x in desc['gap_curve_64'][::4]], '\nm_end', {k_: round(v, 3) for k_, v in desc['m_end'].items()}, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
