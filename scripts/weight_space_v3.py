# -*- coding: utf-8 -*-
"""WS-v3：门控上游头的 QK 侧——它们看哪里，禁止它们看语法 token 会怎样。

接 Phase 285（WS-v2）：8 个上游头（32.24 / 34.8 / 32.26 / 32.25 / 32.19 / 22.18 / 32.21 / 27.3）写入代码方向；
描述性统计显示它们在 MF-v1 确认集代码文本中注意代码语法骨架 token。本版在新材料上检验，并做头级注意力敲除。
  新材料：用 MF-v1 发现集任务按 MF-v5 的方式生成混合文本（md：说明 + ```python 代码块``` + 结语；cmt：代码中插两行注释）。
  语法骨架 token（预注册，weight_read.is_syntax）：块头（含冒号且以换行结尾）、Python 关键字、运算符。
  H1（回看）：md 文本结语散文位置上，8 个头对代码段 token 的注意力质量 / 代码段在可见 token 中的份额 ≥ 1.5，且高于 WS-v2 的 5 组同块随机头的平均。
  H2（语法选择）：代码位置（md 代码块与 cmt 代码，t ≥ 8）上，8 个头对语法骨架 token 的注意力质量 / 其份额 ≥ 2，且高于随机头平均。
  H3（因果，QK）：只禁止 8 个头注意语法骨架 token（该头对这些 key 的注意力置为 0 后重新归一化，其余头不变），
      MF-v1 确认集门控得分「代码 − 散文」差下降 ≥ 10%，且 ≥ 2 × 「同样的头禁止等数随机非语法 token」的下降，
      且 ≥ 2 × 「随机头禁止语法 token」的下降（5 组平均）。
  描述：cmt 注释位置对代码 token 的回看；md 结语位置禁止 8 个头注意代码段时门控得分的变化。
注意力计算统一用 eager 形式（含原始与敲除运行），门控得分与 WS-v1 / v2 的定义与符号相同。
被测模型：Qwen3-4B（bf16，不使用量化）。
用法：python scripts/weight_space_v3.py prereg | run | analyze
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

from shared.dist_protocol import weight_read as W  # noqa: E402
from shared.dist_protocol import mode_field as F  # noqa: E402
import mf_materials_v1 as MAT  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402
import weight_space_v1 as WS1  # noqa: E402
import weight_space_v2 as WS2  # noqa: E402

PREREG = ROOT / 'research/weight_space/WS-v3.prereg.json'
MODEL = WS1.MODEL
STY = F.STYLES
CODE = STY.index('code')
TL = WS1.TL


def prereg_doc():
    return {'id': 'WS-v3', 'schema': 'weight_space_prereg.v3', 'created': '2026-11-02',
            'basis': 'Phase 285（WS-v2）：8 个上游头写入代码方向，确认集代码文本上注意语法骨架 token（描述）；本版用新材料与头级敲除检验',
            'models': {'qwen3-4b': 'bf16'}, 'precision_rule': '不使用任何量化',
            'heads': 'research/weight_space/WS-v2-qwen3-4b/report.json 的 selected 与 random_sets',
            'materials': '混合文本由 MF-v1 发现集任务生成（WS-v2 的注意力描述只用了确认集代码文本）；H3 用 MF-v1 确认集',
            'syntax_class': {'keywords': sorted(W.KEYWORDS), 'operators': sorted(W.OPERATORS), 'block_header': '含冒号且以换行结尾'},
            'power_note': 'H1 / H2：发现集 16 个任务 × 5 填充 × 中英 的混合文本；H3：确认集 160 段文本；5 组随机头；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_V3}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（WS-v4）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/WS-v3-qwen3-4b'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(upd):
    p = report_path()
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def mixed_disc():
    """与 MAT.mixed_corpus 相同的构造，但用发现集任务。"""
    out = []
    for t in MAT.split()['disc']:
        for j, f in enumerate(MAT.fillers(t)):
            code = MAT.render(t, 'code', f)
            for lang in ('zh', 'en'):
                prose = MAT.render(t, lang, f)
                pieces = [(prose + '\n\n', 'prose'), ('```python\n', 'fence'), (code, 'code'), ('```\n\n', 'fence'), (MAT.CLOSING[lang], 'closing')]
                out.append({'kind': 'md', 'pieces': pieces})
                lines = code.split('\n')
                ret = max(i for i, ln in enumerate(lines) if ln.strip().startswith('return'))
                pieces = [(lines[0] + '\n', 'code'), ('    # ' + MAT._first_sentence(prose, lang) + '\n', 'comment'),
                          ('\n'.join(lines[1:ret]) + '\n', 'code'), ('    # ' + MAT._last_sentence(prose, lang) + '\n', 'comment'),
                          ('\n'.join(lines[ret:]), 'code')]
                out.append({'kind': 'cmt', 'pieces': pieces})
    return out


def do_run():
    import torch
    import transformers.models.qwen3.modeling_qwen3 as MQ
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(MODEL, False)
    od = MF1.out_dir(key)
    ws1 = json.loads(WS1.report_path().read_text(encoding='utf-8'))
    ws2 = json.loads(WS2.report_path().read_text(encoding='utf-8'))
    sel = [tuple(x) for x in ws2['selected']]
    rsets = [[tuple(x) for x in s_] for s_ in ws2['random_sets']]
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of = [int(x) for x in U['top']], U['layer_of']
    off = {int(k_): v for k_, v in json.loads(WS1.MF4.read_text(encoding='utf-8'))['unit_offsets'].items()}
    tl_units = {l: [int(i - off[l]) for i in top if int(layer_of[i]) == l] for l in TL}
    sgn = {l: np.array(ws1['unit_signs'][str(l)]) for l in TL}
    shared_all = MF1.load_fields(key)['shared'].astype(bool)
    tok, model = XM.load(MODEL, False)
    R = XM.Runner(tok, model)
    C, ids = MF1.tokenize(tok)
    blocks = R.blocks
    t0 = time.time()
    # ---------- eager 注意力，可按头禁止 key、可收集指定头的注意力 ----------
    ctl = {'mask': {}, 'keymask': None, 'collect': {}, 'store': {}}

    def ws_attn(module, query, key_, value, attention_mask, scaling, dropout=0.0, **kw):
        k_ = MQ.repeat_kv(key_, module.num_key_value_groups)
        v_ = MQ.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k_.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        lam = module.layer_idx
        if ctl['keymask'] is not None and ctl['mask'].get(lam):
            hs_ = torch.as_tensor(ctl['mask'][lam], device=w.device)
            km = ctl['keymask'][:, None, None, :w.shape[-1]].expand(-1, len(hs_), w.shape[2], -1)
            sub = w[:, hs_].masked_fill(km, torch.finfo(w.dtype).min)
            w = w.clone()
            w[:, hs_] = sub
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        if ctl['collect'].get(lam):
            ctl['store'][lam] = w[:, ctl['collect'][lam]].float()
        out = torch.matmul(w, v_).transpose(1, 2).contiguous()
        return out, None

    MQ.eager_attention_forward = ws_attn
    model.set_attn_implementation('eager')
    uidx = {l: torch.as_tensor(tl_units[l], device='cuda') for l in TL}
    sg_t = {l: torch.as_tensor(sgn[l], device='cuda', dtype=torch.float32) for l in TL}
    gcap = {}
    hooks = [blocks[l].mlp.gate_proj.register_forward_hook(
        (lambda l: lambda m, a, o: gcap.__setitem__(l, (o[..., uidx[l]].float() * sg_t[l]).sum(-1)))(l)) for l in TL]
    syn_id = {}

    def syn_flags(seq):
        out = []
        for t in seq:
            if t not in syn_id:
                syn_id[t] = W.is_syntax(tok.decode([t]))
            out.append(syn_id[t])
        return np.array(out, bool)

    # ---------- H3：确认集门控得分差，各种敲除 ----------
    starts = np.concatenate([[0], np.cumsum([len(x) for x in ids])])
    conf = [i for i, cc in enumerate(C) if cc['split'] == 'conf']
    is_code = lambda i: STY.index(C[i]['style']) == CODE
    rng = np.random.default_rng(crit['seed'])
    km_syn, km_rand = {}, {}
    for i in conf:
        f = syn_flags(ids[i])
        f[0] = False
        km_syn[i] = f
        pool = np.where(~f)[0]
        pool = pool[pool > 0]
        r = np.zeros(len(f), bool)
        r[rng.choice(pool, min(int(f.sum()), len(pool)), replace=False)] = True
        km_rand[i] = r

    def gate_contrast(heads, kmaps):
        ctl['mask'] = {}
        for lam, h in heads or []:
            ctl['mask'].setdefault(lam, []).append(h)
        g, n = [0.0, 0.0], [0, 0]
        for j in range(0, len(conf), 16):
            ch = conf[j:j + 16]
            i_, att = R._batch([ids[i] for i in ch])
            if kmaps is not None:
                km = torch.zeros(i_.shape, dtype=torch.bool, device='cuda')
                for b, i in enumerate(ch):
                    km[b, :len(ids[i])] = torch.as_tensor(kmaps[i], device='cuda')
                ctl['keymask'] = km
            R.model.model(input_ids=i_, attention_mask=att)
            ctl['keymask'] = None
            for b, i in enumerate(ch):
                k = len(ids[i])
                m = torch.as_tensor(shared_all[starts[i]:starts[i] + k], device='cuda')
                c = int(is_code(i))
                g[c] += sum(gcap[l][b, :k][m].sum().item() for l in TL)
                n[c] += int(m.sum().item())
        ctl['mask'] = {}
        return g[1] / n[1] - g[0] / n[0]

    G = {'orig': gate_contrast(None, None), 'sel_syntax': gate_contrast(sel, km_syn), 'sel_random_tokens': gate_contrast(sel, km_rand)}
    for j, s_ in enumerate(rsets):
        G[f'rand{j}_syntax'] = gate_contrast(s_, km_syn)
    print('@@G', {k_: round(v, 2) for k_, v in G.items()}, f'{round(time.time() - t0)}s', flush=True)
    frac_syn = float(np.mean([km_syn[i][1:].mean() for i in conf if is_code(i)]))
    # ---------- H1 / H2：混合文本上的注意力 ----------
    mixed = mixed_disc()
    allh = {'sel': sel, **{f'rand{j}': s_ for j, s_ in enumerate(rsets)}}
    need = {}
    for hs_ in allh.values():
        for lam, h in hs_:
            need.setdefault(lam, set()).add(h)
    ctl['collect'] = {lam: sorted(v) for lam, v in need.items()}
    col = {lam: {h: k_ for k_, h in enumerate(v)} for lam, v in ctl['collect'].items()}
    stats = {g_: {'lookback_md': [], 'lookback_cmt': [], 'syntax': []} for g_ in allh}
    gate_md = {'orig': [], 'sel_mask_code': [], 'rand_mask_code': []}
    for item in mixed:
        text, segs = '', []
        for s_, seg in item['pieces']:
            segs.append((len(text), len(text) + len(s_), seg))
            text += s_
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        seq = enc.input_ids
        seg_of = [next(sg for a, b, sg in segs if a <= st_ < b) if st_ < len(text) else segs[-1][2] for st_, _ in enc.offset_mapping]
        seg_of = np.array(seg_of)
        codek = seg_of == 'code'
        syn = syn_flags(seq) & codek
        i_, att = R._batch([seq])
        R.model.model(input_ids=i_, attention_mask=att)
        A = {lam: ctl['store'][lam][0].cpu().numpy() for lam in ctl['collect']}
        gate_now = sum(gcap[l][0] for l in TL).cpu().numpy()
        for t in range(crit['min_pos'], len(seq)):
            for g_, hs_ in allh.items():
                vals = {}
                if item['kind'] == 'md' and seg_of[t] == 'closing':
                    vals['lookback_md'] = [W.mass_ratio(A[lam][col[lam][h], t], t, codek) for lam, h in hs_]
                if item['kind'] == 'cmt' and seg_of[t] == 'comment':
                    vals['lookback_cmt'] = [W.mass_ratio(A[lam][col[lam][h], t], t, codek) for lam, h in hs_]
                if seg_of[t] == 'code':
                    vals['syntax'] = [W.mass_ratio(A[lam][col[lam][h], t], t, syn) for lam, h in hs_]
                for k_, v in vals.items():
                    stats[g_][k_].append(float(np.nanmean(v)))
        if item['kind'] == 'md':
            cl = seg_of == 'closing'
            gate_md['orig'].append(float(gate_now[cl].mean()))
            for name, hs_ in (('sel_mask_code', sel), ('rand_mask_code', rsets[0])):
                ctl['mask'] = {}
                for lam, h in hs_:
                    ctl['mask'].setdefault(lam, []).append(h)
                ctl['keymask'] = torch.as_tensor(codek[None], device='cuda')
                R.model.model(input_ids=i_, attention_mask=att)
                ctl['keymask'], ctl['mask'] = None, {}
                gate_md[name].append(float(sum(gcap[l][0] for l in TL).cpu().numpy()[cl].mean()))
    ctl['collect'] = {}
    for h_ in hooks:
        h_.remove()
    summ = {g_: {k_: {'mean': float(np.nanmean(v)), 'n': int(np.sum(~np.isnan(v)))} for k_, v in d_.items()} for g_, d_ in stats.items()}
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'attn_impl': 'eager（自定义，与原始 eager 相同的计算加按头 key 屏蔽）', 'selected': [list(x) for x in sel],
          'G': G, 'syntax_share_code_texts': frac_syn, 'attention': summ,
          'gate_md_closing': {k_: float(np.mean(v)) for k_, v in gate_md.items()}, 'n_mixed': len(mixed),
          'seconds': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    a = rep['attention']
    rn = [k_ for k_ in a if k_.startswith('rand')]
    rmean = lambda k_: float(np.mean([a[r][k_]['mean'] for r in rn]))
    h1 = {'sel': a['sel']['lookback_md']['mean'], 'random_mean': rmean('lookback_md'),
          'pass': bool(a['sel']['lookback_md']['mean'] >= crit['H1_lookback_min'] and a['sel']['lookback_md']['mean'] > rmean('lookback_md'))}
    h2 = {'sel': a['sel']['syntax']['mean'], 'random_mean': rmean('syntax'),
          'pass': bool(a['sel']['syntax']['mean'] >= crit['H2_syntax_min'] and a['sel']['syntax']['mean'] > rmean('syntax'))}
    G = rep['G']
    drop = lambda k_: 1 - G[k_] / G['orig']
    rd = [drop(k_) for k_ in G if k_.startswith('rand')]
    h3 = {'drop_sel_syntax': drop('sel_syntax'), 'drop_sel_random_tokens': drop('sel_random_tokens'),
          'drop_random_heads_syntax': rd, 'drop_random_heads_mean': float(np.mean(rd)),
          'pass': bool(drop('sel_syntax') >= crit['H3_drop_min'] and
                       drop('sel_syntax') >= crit['H3_over_control'] * max(drop('sel_random_tokens'), 0) and
                       drop('sel_syntax') >= crit['H3_over_control'] * max(float(np.mean(rd)), 0))}
    gm = rep['gate_md_closing']
    desc = {'lookback_cmt': {'sel': a['sel']['lookback_cmt']['mean'], 'random_mean': rmean('lookback_cmt')},
            'gate_md_closing': gm}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'describe': desc, 'verdict': verdict})
    p = write_result('ws_v3_qwen3-4b', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/weight_space_v3.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\ndesc', desc, '\nper-set', json.dumps(a), '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
