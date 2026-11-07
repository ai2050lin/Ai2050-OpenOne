# -*- coding: utf-8 -*-
"""主线 H2（MA-H2）：discovery 上的全场采集与响应发现（ResearchPlan v6 第 3、8 节，切片 A，中文主线）。

材料  MA-H1 的 discovery 划分（84 个实体）× MA-H0-v5 的 32 格 = 2,688 条，构造与 MA-H1 材料检查完全相同（种子 MA-H1）。
      不读取 confirm / lockbox。MA-H1 排除的两格照常采集，分析中单列。
采集  Qwen3-4B bf16，attn_implementation = eager（为取注意力权重）。逐条前向，写入节点本地 memmap：
      · hs：6 个读出位置 × 37 层（hs[0] = 词嵌入，hs[l] = 第 l 块输出；hs[36] 是否已过最终归一化按 lm_head 与真实 logits 对账记录）
        读出位置：ans 答案前缀末位 / qname 问题中的目标名 / scope 问题中的范围词 / tname 目标记录中的目标名 / tcol 目标记录中的颜色 / lastcol 资料中最后一个颜色
      · mlp：答案位置各块 MLP 中间激活 A = silu(G)⊙V（down_proj 的输入），36 × 9,728
      · attn：答案位置各块各头的注意力行，36 × 32 × 256（右侧补零，按 token 数截断）
      · kv：tname、tcol、lastcol 三个位置各块 k_proj / v_proj 输出（RoPE 之前），36 × 8 × 128 × 2
      · lens：答案位置逐层读出（中间层先过最终归一化再接 lm_head），记录 K、P 与最高干扰色的首 token logit
分析  只做描述与候选观察，判据见 CRITERIA（随结果返回，看到结果后不改）。
用法  python scripts/mainline_h2.py prereg|collect|analyze --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

import arbitration_v1 as CF  # noqa: E402
import mainline_h0 as V3  # noqa: E402
import mainline_h0_v5 as V5  # noqa: E402
import mainline_h1 as H1  # noqa: E402

OUT = ROOT / 'research/mainline/H2'
PREREG = OUT / 'MA-H2.prereg.json'
SEED = 'MA-H1'                       # 与 MA-H1 材料检查相同的构造
POS = ('ans', 'qname', 'scope', 'tname', 'tcol', 'lastcol')
KV_POS = ('tname', 'tcol', 'lastcol')
TMAX = 256
FACTORS = ('scope', 'pos', 'k', 'draft', 'form')
LEVEL0 = {'scope': 'doc', 'pos': 'first', 'k': 2, 'draft': 0, 'form': 'single'}
PAIRS = (('scope', 'draft'), ('scope', 'form'), ('scope', 'pos'), ('form', 'pos'))
EXCLUDED = [('real', 'last', 3, 0, 'pair'), ('real', 'last', 3, 1, 'pair')]     # MA-H1 材料检查未过，单列
SCOPE_WORD = {'doc': '按这份资料', 'real': '在现实世界里'}

CRITERIA = {
    'readable': 'AUC ≥ 0.75，且比同一位置 hs[0] 的 AUC 高至少 0.10（按实体二分、5 次随机切分取平均；方向只在训练半拟合）',
    'readable_auc': 0.75, 'readable_margin': 0.10, 'splits': 5,
    'r1_r2': '现实范围中最终读出正确（第 36 层 K−P < 0）的条目里，第 8–32 层任一层 K−P > 0 的比例：≥ 0.5 记为支持 R2（晚仲裁），≤ 0.2 记为支持 R1（早门控），之间记为混合',
    'r2_share': 0.5, 'r1_share': 0.2, 'lens_layers': [8, 32],
    'caveat': '逐层读出是读出近似，早层不可靠；这里只作候选观察，R1 / R2 的确认要到 H4 的路径干预',
}


def local_dir(key):
    return Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native')) / '_mainline' / 'MA-H2' / key


def prereg_doc():
    return {'id': 'MA-H2', 'schema': 'mainline_prereg.v1', 'created': '2026-11-07', 'stage': 'H2',
            'basis': 'MA-H1（Phase 323）discovery 划分与材料', 'split': 'discovery（84 个实体），不读 confirm / lockbox',
            'positions': POS, 'kv_positions': KV_POS, 'layers': 'hs[0]–hs[36] 全部', 'factors': FACTORS, 'pairs': PAIRS,
            'excluded_cells': EXCLUDED, 'boundaries': __doc__.split('采集')[1].split('分析')[0].strip(),
            'outputs': ['逐层读出曲线与 R1 / R2 判定', '单因素效应（原始 L2 与标准化 RMS）与可读出性（实体留出 AUC，含 hs[0] 基线）',
                        '两因素交互范数比', '答案位置的注意力去向（按头）', '答案位置 MLP 单元的范围效应排序'],
            'criteria': CRITERIA}


def do_prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    d = hashlib.sha256(text.encode()).hexdigest()
    PREREG.with_suffix('.sha256').write_text(d + '\n', encoding='utf-8', newline='\n')
    print('prereg', d)


def build(model, tok, palette, forms):
    rows = H1.read_discovery()
    prior = {'_scores': {}}
    for r in rows:
        sc, _ = V3.score(model, tok, '问：{X}通常是什么颜色的？\n答：{X}通常是'.format(X=r['zh']), {c: forms['zh'][c] for c in palette})
        prior['_scores'][r['id']] = np.array([sc[c] for c in palette])
        prior[r['id']] = palette[int(np.argmax(prior['_scores'][r['id']]))]
    V5.SEED = SEED
    names = {r['id']: r['zh'] for r in rows}
    return V5.build_items('zh', rows, rows, prior, palette, forms), names


def char_positions(prompt, cell, X, labels, forms):
    word = lambda c: forms['zh'][c][0].strip()
    q0 = prompt.rindex('问：')
    body_end = prompt.index('\n（') if cell[3] else prompt.index('\n\n问')
    rec = prompt.index(f'{X}是{word(labels["K"])}的')
    sc = prompt.index(SCOPE_WORD[cell[0]], q0)
    qn = prompt.index(X, q0)
    lc = prompt.rindex(word(labels['last']), 0, body_end)
    return {'ans': len(prompt) - 1, 'qname': qn + len(X) - 1, 'scope': sc + len(SCOPE_WORD[cell[0]]) - 1,
            'tname': rec + len(X) - 1, 'tcol': rec + len(X) + 1 + len(word(labels['K'])) - 1, 'lastcol': lc + len(word(labels['last'])) - 1}


def to_token(offsets, ch):
    for i, (a, b) in enumerate(offsets):
        if a <= ch < b:
            return i
    raise ValueError(f'字符 {ch} 不在任何 token 中')


def do_collect(model_path):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    snap, key = CF.key_of(model_path)
    out = local_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    palette, forms = CF.palette_forms()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
    items, names = build(model, tok, palette, forms)
    N, L, D, F, H, KH, HD = len(items), 37, model.config.hidden_size, model.config.intermediate_size, 32, 8, 128
    mm = lambda name, shape: np.lib.format.open_memmap(out / f'{name}.npy', mode='w+', dtype=np.float16, shape=shape)
    HS = mm('hs', (N, len(POS), L, D))
    MLP = mm('mlp', (N, L - 1, F))
    ATT = mm('attn', (N, L - 1, H, TMAX))
    KV = mm('kv', (N, len(KV_POS), L - 1, 2, KH, HD))
    LENS = np.zeros((N, L, 3), dtype=np.float32)
    TOKPOS = np.zeros((N, len(POS)), dtype=np.int32)
    NTOK = np.zeros(N, dtype=np.int32)
    cap = {}
    hooks = []
    for i, b in enumerate(model.model.layers):
        hooks.append(b.mlp.down_proj.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(('mlp', i), a[0][0])))
        hooks.append(b.self_attn.k_proj.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(('k', i), o[0])))
        hooks.append(b.self_attn.v_proj.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(('v', i), o[0])))
    norm, head = model.model.norm, model.lm_head
    meta, recon = [], []
    t0 = time.time()
    for n, (cell, eid, prompt, cands, labels) in enumerate(items):
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        assert len(ids) <= TMAX
        ch = char_positions(prompt, cell, names[eid], labels, forms)
        tp = [to_token(offs, ch[p]) for p in POS]
        o = model(input_ids=torch.as_tensor([ids], device='cuda'), output_hidden_states=True, output_attentions=True)
        hs = torch.stack(o.hidden_states)[:, 0]                     # [37, T, D]
        HS[n] = hs[:, tp].transpose(0, 1).float().cpu().numpy().astype(np.float16)
        MLP[n] = torch.stack([cap['mlp', i][-1] for i in range(L - 1)]).float().cpu().numpy().astype(np.float16)
        att = torch.stack([a[0, :, -1] for a in o.attentions]).float()  # [36, H, T]
        ATT[n, :, :, :len(ids)] = att.cpu().numpy().astype(np.float16)
        for j, p in enumerate(KV_POS):
            t = tp[POS.index(p)]
            KV[n, j] = torch.stack([torch.stack([cap['k', i][t].view(KH, HD), cap['v', i][t].view(KH, HD)]) for i in range(L - 1)]).float().cpu().numpy().astype(np.float16)
        tk = lambda c: tok(forms['zh'][c][0], add_special_tokens=False).input_ids[0]
        idx = [tk(labels['K']), tk(labels['P'])] + [tk(c) for c in labels['D']]
        x = hs[:, -1]
        lens_in = torch.cat([norm(x[:-1]), x[-1:]])                  # hs[36] 已过最终归一化（下方对账）
        lg = head(lens_in)[:, idx].float()
        LENS[n, :, 0], LENS[n, :, 1] = lg[:, 0].cpu().numpy(), lg[:, 1].cpu().numpy()
        LENS[n, :, 2] = (lg[:, 2:].max(1).values if lg.shape[1] > 2 else torch.full_like(lg[:, 0], -1e4)).cpu().numpy()
        if n < 8:
            recon.append(float((head(x[-1:])[0] - o.logits[0, -1]).abs().max()))
        TOKPOS[n], NTOK[n] = tp, len(ids)
        meta.append({'cell': list(cell), 'entity': eid, 'K': labels['K'], 'P': labels['P'], 'D': labels['D'], 'last': labels['last'],
                     'partner': labels['partner'], 'draft': labels['draft'], 'correct': labels['correct']})
        if n % 300 == 0:
            print(f'@@progress {n}/{N} {round(time.time() - t0)}s', flush=True)
    for h in hooks:
        h.remove()
    for a in (HS, MLP, ATT, KV):
        a.flush()
    np.save(out / 'lens.npy', LENS)
    np.save(out / 'tokpos.npy', TOKPOS)
    np.save(out / 'ntok.npy', NTOK)
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    files = {}
    for f in sorted(out.glob('*')):
        if f.name == 'manifest.json':
            continue
        h = hashlib.sha256()
        with open(f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b''):
                h.update(chunk)
        files[f.name] = {'bytes': f.stat().st_size, 'sha256': h.hexdigest()}
    man = {'id': 'MA-H2', 'model': model_path, 'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(),
           'n_items': N, 'positions': POS, 'kv_positions': KV_POS, 'tmax': TMAX, 'seconds': round(time.time() - t0, 1),
           'torch': torch.__version__, 'transformers': transformers.__version__, 'attn_implementation': 'eager', 'dtype': 'bfloat16（存为 float16）',
           'hs_last_is_final_normed': {'lm_head(hs[36]) 与真实 logits 最大绝对差（前 8 条）': max(recon)},
           'kv_boundary': 'k_proj / v_proj 输出，RoPE 之前', 'mlp_boundary': 'down_proj 输入 A = silu(G)⊙V', 'files': files}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps({**man, 'local_dir': str(out)}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('collect done', man['seconds'], 's', 'recon', max(recon), 'GB', round(sum(v['bytes'] for v in files.values()) / 2 ** 30, 2))


def auc(pos, neg):
    """Mann–Whitney AUC，并列值取平均秩（2026-11-07 修正：最初用 argsort 秩，并列时 AUC 可为 0 或 1）。"""
    from scipy.stats import rankdata
    s = np.concatenate([pos, neg])
    r = rankdata(s)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def readability(X, y, ent, rng, n_split):
    """实体二分：训练半上用均值差定方向，测试半上投影求 AUC。X [N, D]。"""
    ents = np.unique(ent)
    vals = []
    for _ in range(n_split):
        tr = set(rng.choice(ents, len(ents) // 2, replace=False))
        m = np.array([e in tr for e in ent])
        w = X[m & y].mean(0) - X[m & ~y].mean(0)
        p = X[~m] @ w
        vals.append(auc(p[y[~m]], p[~y[~m]]))
    return float(np.mean(vals))


def do_analyze(model_path):
    _, key = CF.key_of(model_path)
    d = local_dir(key)
    meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    HS = np.load(d / 'hs.npy', mmap_mode='r')
    MLP = np.load(d / 'mlp.npy', mmap_mode='r')
    ATT = np.load(d / 'attn.npy', mmap_mode='r')
    LENS = np.load(d / 'lens.npy')
    TP = np.load(d / 'tokpos.npy')
    N, P, L, D = HS.shape
    cell = [tuple(m['cell']) for m in meta]
    ent = np.array([m['entity'] for m in meta])
    keep = np.array([c not in EXCLUDED for c in cell])
    lev = {f: np.array([c[i] != LEVEL0[f] for c in cell]) for i, f in enumerate(FACTORS)}
    rng = np.random.default_rng(0)
    rep = {'id': 'MA-H2', 'criteria': CRITERIA, 'n_items': N, 'n_analyzed': int(keep.sum()), 'excluded_cells': EXCLUDED}

    # A. 逐层读出与 R1 / R2
    lo = LENS[:, :, 0] - LENS[:, :, 1]                       # K − P
    real, doc = np.array([c[0] == 'real' for c in cell]), np.array([c[0] == 'doc' for c in cell])
    a, b = CRITERIA['lens_layers']
    rc = keep & real & (lo[:, -1] < 0)
    share = float((lo[rc, a:b + 1].max(1) > 0).mean())
    verdict = '支持 R2（晚仲裁）' if share >= CRITERIA['r2_share'] else '支持 R1（早门控）' if share <= CRITERIA['r1_share'] else '混合'
    dc = keep & doc & (lo[:, -1] > 0)
    rep['lens'] = {'mean_K_minus_P': {'doc': lo[keep & doc].mean(0).round(3).tolist(), 'real': lo[keep & real].mean(0).round(3).tolist()},
                   'final_correct_share': {'doc': float((lo[keep & doc, -1] > 0).mean()), 'real': float((lo[keep & real, -1] < 0).mean())},
                   'real_K_lead_share': share, 'n_real_correct': int(rc.sum()), 'verdict': verdict,
                   'doc_P_lead_share': float((lo[dc, a:b + 1].min(1) < 0).mean()),
                   'first_layer_correct_sign': {'doc': int(np.argmax((lo[keep & doc] > 0).mean(0) > 0.5)), 'real': int(np.argmax((lo[keep & real] < 0).mean(0) > 0.5))}}
    print('lens', rep['lens']['real_K_lead_share'], verdict, rep['lens']['final_correct_share'])

    # B. 单因素效应与可读出性（每个位置 × 层；载入一个位置的全部层）
    eff = {f: {} for f in FACTORS}
    for pi, pname in enumerate(POS):
        X = np.asarray(HS[:, pi], dtype=np.float32)[keep]       # [n, L, D]
        for f in FACTORS:
            y = lev[f][keep]
            raw, std, rd = [], [], []
            for l in range(L):
                x = X[:, l]
                m1, m0 = x[y].mean(0), x[~y].mean(0)
                sd = np.sqrt((x[y].var(0) + x[~y].var(0)) / 2) + 1e-6
                raw.append(float(np.linalg.norm(m1 - m0)))
                std.append(float(np.sqrt(np.mean(((m1 - m0) / sd) ** 2))))
                rd.append(readability(x, y, ent[keep], rng, CRITERIA['splits']) if l in (0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36) else None)
            base = rd[0]
            readable = [l for l, v in enumerate(rd) if v is not None and v >= CRITERIA['readable_auc'] and v - base >= CRITERIA['readable_margin']]
            eff[f][pname] = {'raw_l2': [round(v, 3) for v in raw], 'std_rms': [round(v, 4) for v in std],
                             'auc': [None if v is None else round(v, 3) for v in rd], 'readable_layers': readable,
                             'first_readable': readable[0] if readable else None}
        print('effects', pname, {f: eff[f][pname]['first_readable'] for f in FACTORS}, flush=True)
        if pname == 'ans':
            inter = {}
            for f, g in PAIRS:
                yf, yg = lev[f][keep], lev[g][keep]
                rows = []
                for l in range(L):
                    x = X[:, l]
                    mu = {(i, j): x[(yf == i) & (yg == j)].mean(0) for i in (0, 1) for j in (0, 1)}
                    I = mu[1, 1] - mu[1, 0] - mu[0, 1] + mu[0, 0]
                    mf = (mu[1, 0] + mu[1, 1] - mu[0, 0] - mu[0, 1]) / 2
                    mg = (mu[0, 1] + mu[1, 1] - mu[0, 0] - mu[1, 0]) / 2
                    rows.append(round(float(np.linalg.norm(I) / ((np.linalg.norm(mf) + np.linalg.norm(mg)) / 2 + 1e-6)), 3))
                inter[f'{f}×{g}'] = rows
            rep['interaction_ratio_ans'] = inter
        del X
    rep['effects'] = eff

    # C. 答案位置的注意力去向
    targets = ('tname', 'tcol', 'lastcol', 'qname')
    mass = {t: np.zeros((N, L - 1, 32), dtype=np.float32) for t in targets}
    for n in range(N):
        A = np.asarray(ATT[n], dtype=np.float32)
        for t in targets:
            mass[t][n] = A[:, :, TP[n, POS.index(t)]]
    att = {}
    for t in targets:
        att[t] = {'layer_mean': {'doc': mass[t][keep & doc].mean((0, 2)).round(4).tolist(), 'real': mass[t][keep & real].mean((0, 2)).round(4).tolist()}}
    md, mr = mass['tcol'][keep & doc].mean(0), mass['tcol'][keep & real].mean(0)
    top = np.dstack(np.unravel_index(np.argsort(md.ravel())[::-1][:15], md.shape))[0]
    att['top_heads_to_tcol'] = [{'layer': int(l), 'head': int(h), 'doc': round(float(md[l, h]), 3), 'real': round(float(mr[l, h]), 3)} for l, h in top]
    rep['attention'] = att
    print('top heads to K', att['top_heads_to_tcol'][:5])

    # D. 答案位置 MLP 单元的范围效应
    y = lev['scope'][keep]
    units = []
    for l in range(L - 1):
        x = np.asarray(MLP[:, l], dtype=np.float32)[keep]
        sd = np.sqrt((x[y].var(0) + x[~y].var(0)) / 2) + 1e-4
        z = (x[y].mean(0) - x[~y].mean(0)) / sd
        for u in np.argsort(-np.abs(z))[:5]:
            units.append({'layer': l, 'unit': int(u), 'std_diff_real_minus_doc': round(float(z[u]), 3)})
    rep['mlp_scope_units_top'] = sorted(units, key=lambda r: -abs(r['std_diff_real_minus_doc']))[:30]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    from provenance import write_result
    print('report', write_result('MA-H2', rep, model_path=model_path, seed=0, tool='scripts/mainline_h2.py'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
