# -*- coding: utf-8 -*-
"""WS-v4：门控上游头的「同模式读取」——查询的代码分量是否决定它们看不看代码 key。

接 Phase 286（WS-v3）：8 个上游头在代码位置偏好语法 token，在散文位置回避代码 token；候选解释为模式匹配的 QK。
  代码方向 v_λ：MF-v1 发现集 hs[λ] 的 token 配平方向（λ = 头所在块，注意力读的是块输入）。
  H1（只读权重）：头 h 的「代码—代码 QK 对齐」a_h = cos(γ_q⊙W_Q^h(γ_in⊙v), B·γ_k⊙W_K^g(γ_in⊙v))，
      B = 相对距离 1..32 上平均的 RoPE 双线性矩阵（q_norm / k_norm 取其权重的线性近似）。
      对 200 个随机残差方向求 z；8 个头的平均 z ≥ 2，且高于 WS-v2 的 5 组同块随机头（40 个头）的平均 z。
  H2（查询端干预）：只在这 8 个头的查询上（q_proj 输出的该头切片，全部位置）加 k·σ·W_Q^h(γ_in⊙v̂)，k ∈ {−4, −2, 0, 2, 4}，
      σ = 原始前向中归一化输入在 γ_in⊙v 单位方向上投影的标准差；key 与其余头不变。
      md 混合文本（发现集任务，同 WS-v3）上：结语散文位置对代码段 key 的注意力质量比随 k 单调不减，且 +4 / 0 ≥ 1.5；
      代码位置对代码 key 的质量比也随 k 单调不减；且 |log(+4 / 0)| ≥ 3 × 随机方向同样移动的 |log 效应|。
  H3（key 端干预）：把代码段 token 的 key（这些头所在 KV 组的 k_proj 输出切片）沿代码方向移动 −4σ（推向非代码），
      代码位置对代码 key 的质量比 −4 / 0 ≤ 0.8，且 |log 效应| ≥ 3 × 随机方向。
  描述：cmt 注释位置；8 个头对其他三种风格方向（中文、英文、论文）的同类对齐 z。
注意力用 WS-v3 的自定义 eager 计算。被测模型：Qwen3-4B（bf16，不使用量化）。
用法：python scripts/weight_space_v4.py prereg | run | analyze
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
import mode_field_v1 as MF1  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402
import weight_space_v1 as WS1  # noqa: E402
import weight_space_v2 as WS2  # noqa: E402
import weight_space_v3 as WS3  # noqa: E402

PREREG = ROOT / 'research/weight_space/WS-v4.prereg.json'
MODEL = WS1.MODEL
STY = F.STYLES
CODE = STY.index('code')


def prereg_doc():
    return {'id': 'WS-v4', 'schema': 'weight_space_prereg.v4', 'created': '2026-11-02',
            'basis': 'Phase 286（WS-v3）：上游头在散文位置回避代码 token、在代码位置偏好语法 token；候选「同模式读取」',
            'models': {'qwen3-4b': 'bf16'}, 'precision_rule': '不使用任何量化',
            'heads': 'research/weight_space/WS-v2-qwen3-4b/report.json 的 selected 与 random_sets',
            'materials': 'WS-v3 的发现集混合文本（md 用于 H2 / H3，cmt 描述）；方向用 MF-v1 发现集',
            'rope_note': 'B 为 HF 约定 RoPE 在相对距离 1..32 上的平均；q_norm / k_norm 的 RMS 缩放不进入余弦',
            'power_note': 'H1：200 个随机方向；H2 / H3：md 160 段文本的全部结语与代码位置；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_V4}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（WS-v5）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/WS-v4-qwen3-4b'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(upd):
    p = report_path()
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def do_run():
    import torch
    import transformers.models.qwen3.modeling_qwen3 as MQ
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(MODEL, False)
    ws2 = json.loads(WS2.report_path().read_text(encoding='utf-8'))
    sel = [tuple(x) for x in ws2['selected']]
    rheads = [tuple(x) for s_ in ws2['random_sets'] for x in s_]
    layers = sorted({l for l, _ in sel + rheads})
    z = MF1.load_fields(key)
    y, tk, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    vdir = {}
    for l in layers:
        X = z['Hs'][D[sh], l].astype(np.float32)
        vdir[l] = {s: W.unit(F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], s)) for s in range(4)}
        del X
    del z
    gc.collect()
    tok, model = XM.load(MODEL, False)
    R = XM.Runner(tok, model)
    blocks = R.blocks
    cfg = model.config
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nh)
    grp = nh // nkv
    t0 = time.time()
    # ---------- H1：只读权重 ----------
    inv = model.model.rotary_emb.inv_freq.float().cpu().numpy()
    B = W.rope_avg(inv, crit['rope_deltas'])
    rng = np.random.default_rng(crit['seed'])
    rdirs = [W.unit(rng.normal(0, 1, cfg.hidden_size)) for _ in range(crit['n_random_dirs'])]
    wq_cache = {}

    def mats(l):
        if l not in wq_cache:
            at = blocks[l].self_attn
            wq_cache[l] = (at.q_proj.weight.float().cpu().numpy(), at.k_proj.weight.float().cpu().numpy(),
                           at.q_norm.weight.float().cpu().numpy(), at.k_norm.weight.float().cpu().numpy(),
                           blocks[l].input_layernorm.weight.float().cpu().numpy())
        return wq_cache[l]

    def align(l, h, v):
        Wq, Wk, gq, gk, gin = mats(l)
        g_ = h // grp
        return W.qk_alignment(Wq[h * hd:(h + 1) * hd], gq, Wk[g_ * hd:(g_ + 1) * hd], gk, gin, v, B)

    zs = {}
    for l, h in sel + rheads:
        null = [align(l, h, r) for r in rdirs]
        zs[(l, h)] = {STY[s]: W.z_vs(align(l, h, vdir[l][s]), null) for s in range(4)}
        zs[(l, h)]['a_code'] = align(l, h, vdir[l][CODE])
    print('@@H1 sel z', [round(zs[x]['code'], 2) for x in sel], f'{round(time.time() - t0)}s', flush=True)
    # ---------- 前向：自定义 eager 注意力 + 查询 / key 移动 ----------
    ctl = {'collect': {}, 'store': {}, 'q': {}, 'k': {}, 'kmask': None, 'cap': False, 'xn': {}}

    def ws_attn(module, query, key_, value, attention_mask, scaling, dropout=0.0, **kw):
        k_ = MQ.repeat_kv(key_, module.num_key_value_groups)
        v_ = MQ.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k_.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        lam = module.layer_idx
        if ctl['collect'].get(lam):
            ctl['store'][lam] = w[:, ctl['collect'][lam]].float()
        return torch.matmul(w, v_).transpose(1, 2).contiguous(), None

    MQ.eager_attention_forward = ws_attn
    model.set_attn_implementation('eager')
    sel_layers = sorted({l for l, _ in sel})

    def q_hook(l):
        def f(m, a, o):
            for h, dv in ctl['q'].get(l, []):
                o[..., h * hd:(h + 1) * hd] += dv
            return o
        return f

    def k_hook(l):
        def f(m, a, o):
            for g_, dv in ctl['k'].get(l, []):
                o[..., g_ * hd:(g_ + 1) * hd] += dv * ctl['kmask'][..., None]
            return o
        return f

    def ln_hook(l):
        def f(m, a, o):
            if ctl['cap']:
                ctl['xn'][l] = o.float()
        return f

    hooks = [blocks[l].self_attn.q_proj.register_forward_hook(q_hook(l)) for l in sel_layers]
    hooks += [blocks[l].self_attn.k_proj.register_forward_hook(k_hook(l)) for l in sel_layers]
    hooks += [blocks[l].input_layernorm.register_forward_hook(ln_hook(l)) for l in sel_layers]
    ctl['collect'] = {l: sorted({h for l2, h in sel if l2 == l}) for l in sel_layers}
    col = {l: {h: i for i, h in enumerate(v)} for l, v in ctl['collect'].items()}
    gin = {l: blocks[l].input_layernorm.weight.float() for l in sel_layers}
    unit_dir = {}
    for l in sel_layers:
        ud = gin[l] * torch.as_tensor(vdir[l][CODE], device='cuda', dtype=torch.float32)
        unit_dir[(l, 'code')] = ud / ud.norm()
        r = gin[l] * torch.as_tensor(rdirs[0], device='cuda', dtype=torch.float32)
        unit_dir[(l, 'rand')] = r / r.norm()
    items = []
    for it in WS3.mixed_disc():
        text, segs = '', []
        for s_, seg in it['pieces']:
            segs.append((len(text), len(text) + len(s_), seg))
            text += s_
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        seg_of = np.array([next(sg for a, b, sg in segs if a <= st_ < b) if st_ < len(text) else segs[-1][2] for st_, _ in enc.offset_mapping])
        items.append((it['kind'], enc.input_ids, seg_of))
    # σ：原始前向中归一化输入在单位方向上的投影标准差
    proj = {k_: [] for k_ in unit_dir}
    ctl['cap'] = True
    for kind, seq, seg_of in items:
        i_, att = R._batch([seq])
        R.model.model(input_ids=i_, attention_mask=att)
        for (l, nm), u in unit_dir.items():
            proj[(l, nm)].append((ctl['xn'][l][0] @ u).cpu().numpy())
    ctl['cap'] = False
    sig = {k_: float(np.concatenate(v).std()) for k_, v in proj.items()}

    def run(qshift=None, kshift=None):
        """qshift / kshift：(方向名, k)。返回各头在 md 结语、md 代码、cmt 注释位置上对代码 key 的平均质量比。"""
        out = {(l, h): {'md_closing': [], 'md_code': [], 'cmt_comment': []} for l, h in sel}
        for kind, seq, seg_of in items:
            codek = seg_of == 'code'
            ctl['q'], ctl['k'] = {}, {}
            for l, h in sel:
                at = blocks[l].self_attn
                if qshift:
                    nm, kk = qshift
                    dv = kk * sig[(l, nm)] * (at.q_proj.weight.float() @ unit_dir[(l, nm)])[h * hd:(h + 1) * hd]
                    ctl['q'].setdefault(l, []).append((h, dv.to(at.q_proj.weight.dtype)))
                if kshift:
                    nm, kk = kshift
                    g_ = h // grp
                    if all(g_ != g2 for g2, _ in ctl['k'].get(l, [])):
                        dv = kk * sig[(l, nm)] * (at.k_proj.weight.float() @ unit_dir[(l, nm)])[g_ * hd:(g_ + 1) * hd]
                        ctl['k'].setdefault(l, []).append((g_, dv.to(at.k_proj.weight.dtype)))
            ctl['kmask'] = torch.as_tensor(codek[None], device='cuda', dtype=torch.bfloat16)
            i_, att = R._batch([seq])
            R.model.model(input_ids=i_, attention_mask=att)
            A = {l: ctl['store'][l][0].cpu().numpy() for l in sel_layers}
            for t in range(8, len(seq)):
                pt = {'closing': 'md_closing', 'code': 'md_code', 'comment': 'cmt_comment'}.get(seg_of[t])
                if pt is None or (pt.startswith('md') != (kind == 'md')):
                    continue
                for l, h in sel:
                    out[(l, h)][pt].append(W.mass_ratio(A[l][col[l][h], t], t, codek))
        ctl['q'], ctl['k'] = {}, {}
        return {pt: float(np.mean([np.nanmean(out[x][pt]) for x in sel])) for pt in ('md_closing', 'md_code', 'cmt_comment')}

    res = {}
    for kk in crit['shift_k']:
        res[f'q_code_{kk}'] = run(qshift=('code', kk)) if kk else run()
        print(f'@@q {kk}', res[f'q_code_{kk}'], f'{round(time.time() - t0)}s', flush=True)
    for kk in (-4, 4):
        res[f'q_rand_{kk}'] = run(qshift=('rand', kk))
    res['k_code_-4'] = run(kshift=('code', -4))
    res['k_rand_-4'] = run(kshift=('rand', -4))
    print('@@k', res['k_code_-4'], res['k_rand_-4'], flush=True)
    for h_ in hooks:
        h_.remove()
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'selected': [list(x) for x in sel], 'z': {f'{l}.{h}': v for (l, h), v in zs.items()},
          'random_heads': [list(x) for x in rheads], 'sigma': {f'{l}.{nm}': v for (l, nm), v in sig.items()},
          'ratios': res, 'seconds': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    z = rep['z']
    sel = [f'{l}.{h}' for l, h in rep['selected']]
    rh = [f'{l}.{h}' for l, h in rep['random_heads']]
    zs, zr = float(np.mean([z[k_]['code'] for k_ in sel])), float(np.mean([z[k_]['code'] for k_ in rh]))
    h1 = {'z_sel_mean': zs, 'z_random_heads_mean': zr, 'z_sel': {k_: z[k_]['code'] for k_ in sel},
          'pass': bool(zs >= crit['H1_z_min'] and zs > zr)}
    r = rep['ratios']
    ks = crit['shift_k']
    cl = [r[f'q_code_{k}']['md_closing'] for k in ks]
    cd = [r[f'q_code_{k}']['md_code'] for k in ks]
    eff = np.log(r['q_code_4']['md_closing'] / r['q_code_0']['md_closing'])
    effr = np.log(r['q_rand_4']['md_closing'] / r['q_code_0']['md_closing'])
    h2 = {'closing_by_k': dict(zip(map(str, ks), cl)), 'code_by_k': dict(zip(map(str, ks), cd)),
          'ratio_plus4_over_0': float(np.exp(eff)), 'random_log_effect': float(effr),
          'pass': bool(W.monotone(cl) and W.monotone(cd) and np.exp(eff) >= crit['H2_ratio_min']
                       and abs(eff) >= crit['H2_over_random'] * abs(effr))}
    ek = np.log(r['k_code_-4']['md_code'] / r['q_code_0']['md_code'])
    ekr = np.log(r['k_rand_-4']['md_code'] / r['q_code_0']['md_code'])
    h3 = {'ratio_minus4_over_0': float(np.exp(ek)), 'random_ratio': float(np.exp(ekr)),
          'pass': bool(np.exp(ek) <= crit['H3_key_ratio_max'] and abs(ek) >= crit['H2_over_random'] * abs(ekr))}
    desc = {'other_style_z_sel_mean': {s: float(np.mean([z[k_][s] for k_ in sel])) for s in STY},
            'other_style_z_random_mean': {s: float(np.mean([z[k_][s] for k_ in rh])) for s in STY},
            'cmt_comment_by_k': {str(k): r[f'q_code_{k}']['cmt_comment'] for k in ks}}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'describe': desc, 'verdict': verdict})
    p = write_result('ws_v4_qwen3-4b', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/weight_space_v4.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\ndesc', desc, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
