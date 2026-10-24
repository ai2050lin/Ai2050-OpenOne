# -*- coding: utf-8 -*-
"""EXPL-v2 测量与判定（research/interventions/EXPL-v2.prereg.json）。

1. 文本测试：与 EXPL-v1 相同的前缀与判据（文本上限放宽到 200 字）；对照在各组内同层错位配对（种子 20261024）。
2. 检索验证：每个池位置用来源 token 流中该位置之前最多 256 个 token 单条前向，取该位置的 A，s = A / 普查最大值；
   s ≥ 0.3 为正例、s < 0.1 为负例；正例召回率 ≥ 0.5 且负例误报率 ≤ 0.2 为通过，正例 < 4 个为不判定。
   池中的普查正极位置另做一次只保留左侧 28 token 的前向，供「工具（上下文依赖）」规则使用。
3. 自动原因规则（训练、工具），其余未解释单元写出分类证据，交给分类子代理。

用法：python scripts/run_expl_v2_measure.py --work <scratchpad/expl2> --out research/interventions/EXPL-v2-results.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import neuron_census_service as S  # noqa: E402

PRE = ROOT / 'research/interventions/EXPL-v2.prereg.json'
UNITS = ROOT / 'research/interventions/EXPL-v2.units.json'
V1_DIR = ROOT / 'research/interventions/EXPL-v1'
V1_RES = ROOT / 'research/interventions/EXPL-v1-results.json'
MK, RUN = '6be1034f0b55a6df-bf16', '2ca5f0a756d36c8a-20000000'
PREFIX = '以下是一段文字。The following is a short text.\n'
KINDS = ('new', 'near_miss', 'distant')


def sha_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main(argv=None) -> int:
    import torch
    from scipy import stats  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args(argv)
    if sha_lf(PRE) != PRE.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:
        raise SystemExit('EXPL-v2 预注册文件哈希不符')
    pre = json.loads(PRE.read_text(encoding='utf-8'))
    ub = UNITS.read_bytes()
    if hashlib.sha256(ub).hexdigest() != pre['units']['sha256_raw'] and hashlib.sha256(ub.replace(b'\r\n', b'\n')).hexdigest() != pre['units']['sha256_raw']:
        raise SystemExit('单元清单哈希不符')
    U = json.loads(ub.decode('utf-8'))
    W = Path(args.work)
    units = {}
    for arm in ('A', 'B'):
        for u in U[f'arm_{arm}']['units']:
            units[u['code']] = {**u, 'arm': arm}
    expl = {it['code']: it for f in sorted(glob.glob(str(V1_DIR / 'explain_*.json'))) for it in json.loads(Path(f).read_text(encoding='utf-8'))['items'] if it['code'] in units}
    expl.update({it['code']: it for f in sorted(glob.glob(str(W / 'explain_B_*.json'))) for it in json.loads(Path(f).read_text(encoding='utf-8'))['items']})
    tests = {it['code']: it for f in sorted(glob.glob(str(W / 'tests_*.json'))) for it in json.loads(Path(f).read_text(encoding='utf-8'))['items']}
    labels = {it['code']: it['labels'] for f in sorted(glob.glob(str(W / 'labels_*.json'))) for it in json.loads(Path(f).read_text(encoding='utf-8'))['items']}
    pools = json.loads((W / 'pools.json').read_text(encoding='utf-8'))
    man = S.census_manifest(MK, RUN)
    cman, streams = S.corpus(man['corpus_fp'])
    t0 = time.time()

    # ---- 文本测试的待测文本与对照配对 ----
    rng = np.random.default_rng(20261024)
    partner = {}
    for arm in ('A', 'B'):
        by_layer = {}
        for c, u in sorted(units.items()):
            if u['arm'] == arm:
                by_layer.setdefault(u['block'], []).append(c)
        for b, codes in sorted(by_layer.items()):
            with_new = [c for c in codes if len((tests.get(c) or {}).get('new') or []) > 0]
            perm = list(rng.permutation(len(codes)))
            for i, c in enumerate(codes):
                cands = [codes[(perm[i] + k) % len(codes)] for k in range(1, len(codes) + 1)]
                partner[c] = next((p for p in cands if p != c and p in with_new), None)
    jobs = {}

    def add(text, code, kind, i):
        if isinstance(text, str) and text.strip():
            jobs.setdefault(text.strip(), []).append((code, kind, i))
    for c, t in tests.items():
        for kind in KINDS:
            for i, text in enumerate((t.get(kind) or [])[:5]):
                add(text, c, kind, i)
    for c, p in partner.items():
        if p:
            for i, text in enumerate(tests[p]['new'][:5]):
                add(text, c, 'control', i)

    tok = AutoTokenizer.from_pretrained(str(ROOT / pre['model']['path']))
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / pre['model']['path']), dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    layers = list(model.get_decoder().layers)
    need = {}
    for u in units.values():
        need.setdefault(u['block'], set()).add(u['unit'])
    need = {b: torch.tensor(sorted(s), device='cuda') for b, s in need.items()}
    col = {b: {int(j): k for k, j in enumerate(need[b].tolist())} for b in need}
    cap = {}
    for b in need:
        layers[b].mlp.gate_proj.register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(('g', b), o[..., need[b]].float()))
        layers[b].mlp.up_proj.register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(('v', b), o[..., need[b]].float()))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def run(seqs, start):
        """seqs：token id 列表；start[r]：从该位置起计入（最大值）；返回 {block: (B, n_units_b)} 的最大激活。"""
        W_ = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), W_), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), W_), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        with torch.inference_mode():
            model.model(input_ids=ids.cuda(), attention_mask=mask.cuda())
        valid = mask.cuda().bool().clone()
        for r, st in enumerate(start):
            valid[r, :st] = False
        out = {}
        for b in need:
            A = torch.nn.functional.silu(cap[('g', b)]) * cap[('v', b)]
            out[b] = A.masked_fill(~valid[..., None], float('-inf')).max(1).values.cpu().numpy()
        cap.clear()
        return out

    # 文本测试前向
    n_pre = len(tok(PREFIX, add_special_tokens=False)['input_ids'])
    res_text = {}
    texts = list(jobs)
    for s in range(0, len(texts), args.batch):
        chunk = texts[s:s + args.batch]
        out = run([tok(PREFIX + t, add_special_tokens=False)['input_ids'] for t in chunk], [n_pre] * len(chunk))
        for r, t in enumerate(chunk):
            for (c, kind, i) in jobs[t]:
                u = units[c]
                res_text[(c, kind, i)] = float(out[u['block']][r, col[u['block']][u['unit']]])

    # 检索池真值（256 token 上下文；正极位置另做 28 token）
    pos_jobs = []
    for c, p in pools.items():
        for row in p['rows']:
            g = row['gidx']
            src, off = g >> S.SRC_SHIFT, g & ((1 << S.SRC_SHIFT) - 1)
            pos_jobs.append((c, row['slot'], 256, [int(x) for x in streams[src][max(0, off - 255):off + 1]]))
            if row['kind'] == 'pos':
                pos_jobs.append((c, row['slot'], 28, [int(x) for x in streams[src][max(0, off - 27):off + 1]]))
    res_pool = {}
    for s in range(0, len(pos_jobs), args.batch):
        chunk = pos_jobs[s:s + args.batch]
        out = run([j[3] for j in chunk], [len(j[3]) - 1 for j in chunk])
        for r, (c, slot, ctx, _) in enumerate(chunk):
            u = units[c]
            res_pool[(c, slot, ctx)] = float(out[u['block']][r, col[u['block']][u['unit']]])

    # ---- 判定 ----
    v1 = {r['code']: r for r in json.loads(V1_RES.read_text(encoding='utf-8'))['units']}
    rows_out = []
    for c in sorted(units):
        u = units[c]
        e = expl.get(c) or {}
        cm = u['census_max']
        rec = {'code': c, 'arm': u['arm'], 'block': u['block'], 'unit': u['unit'], 'census_max': cm,
               'explanation': e.get('explanation'), 'confidence': e.get('confidence')}
        if not e or e.get('confidence') == '无法解释':
            rec.update(text_outcome='lose', retrieval='不判定', explained=False, explained_strict=False, reason='无法解释')
        else:
            s = {k: [res_text[(c, k, i)] / cm for i in range(5) if (c, k, i) in res_text] for k in KINDS + ('control',)}
            strong = {k: (float(np.mean(np.array(v) >= 0.3)) if v else None) for k, v in s.items()}
            med = {k: (float(np.median(v)) if v else None) for k, v in s.items()}
            c1 = int(np.sum(np.array(s['new']) >= 0.3)) >= 2
            c2 = med['new'] is not None and med['near_miss'] is not None and med['new'] - med['near_miss'] >= 0.10
            c3 = strong['control'] is not None and strong['new'] - strong['control'] >= 0.2
            text_out = 'win' if (c1 and c2 and c3) else ('lose' if med['new'] is None or med['near_miss'] is None or med['new'] <= med['near_miss'] else 'tie')
            pool = pools[c]
            lab = labels.get(c) or {}
            pos_rows = [r for r in pool['rows'] if res_pool[(c, r['slot'], 256)] / cm >= 0.3]
            neg_rows = [r for r in pool['rows'] if res_pool[(c, r['slot'], 256)] / cm < 0.1]
            yes = lambda r: lab.get(str(r['slot'])) == '是'  # noqa: E731
            if len(pos_rows) < 4 or not lab:
                retr, recall, fpr = '不判定', None, None
            else:
                recall = float(np.mean([yes(r) for r in pos_rows]))
                fpr = float(np.mean([yes(r) for r in neg_rows])) if neg_rows else 0.0
                retr = '通过' if recall >= 0.5 and fpr <= 0.2 else '未通过'
            rec.update(s=s, strong_rate=strong, median=med, criteria=[c1, c2, c3], text_outcome=text_out, partner=partner.get(c),
                       retrieval=retr, recall=recall, fpr=fpr, n_pos=len(pos_rows), n_neg=len(neg_rows),
                       explained=(text_out == 'win' or retr == '通过'), explained_strict=(text_out == 'win' and retr == '通过'))
        # 自动原因规则
        if (not rec['explained']) or rec['text_outcome'] in ('tie', 'lose'):
            L = S.census_layer(MK, RUN, u['block'])
            hist = L['hist'][u['unit']].sum(1)
            n = float(L['n'])
            rate10 = float(sum(hist[i] for i in range(S.N_SIDE + 1, 2 * S.N_SIDE + 1) if S._bin_lower(i) >= 0.1 * cm) / n) if cm > 0 else 0.0
            per_page = rate10 * S.TOKENS_PER_PAGE
            census_pos = [r for r in pools[c]['rows'] if r['kind'] == 'pos']
            ctx_dep = [res_pool[(c, r['slot'], 256)] / cm >= 0.3 and res_pool[(c, r['slot'], 28)] / cm < 0.3 for r in census_pos]
            rec['auto'] = {'per_page_gt10pct': per_page, 'context_dependent_share': float(np.mean(ctx_dep)) if ctx_dep else None}
            if cm < 0.5 or per_page > 20:
                rec['cause'], rec['cause_source'] = '训练', '自动规则（弱 / 无稀疏强激活尾部）'
            elif ctx_dep and np.mean(ctx_dep) >= 0.5:
                rec['cause'], rec['cause_source'] = '工具', '自动规则（上下文依赖）'
            else:
                rec['cause'], rec['cause_source'] = None, '待分类'
        if u['arm'] == 'A' and c in v1:
            rec['v1_outcome'] = v1[c]['outcome']
        rows_out.append(rec)

    def frac(rows, key, val=True):
        return float(np.mean([r.get(key) == val for r in rows])) if rows else None
    A = [r for r in rows_out if r['arm'] == 'A']
    B = [r for r in rows_out if r['arm'] == 'B']
    mc = {'v1_win_v2_win': sum(r.get('v1_outcome') == 'win' and r['text_outcome'] == 'win' for r in A),
          'v1_win_v2_not': sum(r.get('v1_outcome') == 'win' and r['text_outcome'] != 'win' for r in A),
          'v1_not_v2_win': sum(r.get('v1_outcome') != 'win' and r['text_outcome'] == 'win' for r in A),
          'v1_not_v2_not': sum(r.get('v1_outcome') != 'win' and r['text_outcome'] != 'win' for r in A)}
    judged = [r for r in rows_out if r['retrieval'] in ('通过', '未通过') and r.get('text_outcome') in ('win', 'tie', 'lose') and r.get('reason') != '无法解释']
    summary = {
        'B_explained_share': frac(B, 'explained'), 'A_explained_share': frac(A, 'explained'),
        'B_explained_strict_share': frac(B, 'explained_strict'), 'A_explained_strict_share': frac(A, 'explained_strict'),
        'B_text_win_share': frac(B, 'text_outcome', 'win'), 'A_text_win_share': frac(A, 'text_outcome', 'win'),
        'A_v1_win_share': float(np.mean([r.get('v1_outcome') == 'win' for r in A])),
        'A_mcnemar_counts': mc,
        'retrieval': {arm: {'通过': sum(r['retrieval'] == '通过' for r in rows), '未通过': sum(r['retrieval'] == '未通过' for r in rows),
                            '不判定': sum(r['retrieval'] == '不判定' for r in rows),
                            'recall_median': float(np.median([r['recall'] for r in rows if r.get('recall') is not None])) if any(r.get('recall') is not None for r in rows) else None,
                            'fpr_median': float(np.median([r['fpr'] for r in rows if r.get('fpr') is not None])) if any(r.get('fpr') is not None for r in rows) else None}
                      for arm, rows in (('A', A), ('B', B))},
        'text_retrieval_agreement': float(np.mean([(r['text_outcome'] == 'win') == (r['retrieval'] == '通过') for r in judged])) if judged else None,
        'B_by_confidence': {k: {'n': len([r for r in B if r.get('confidence') == k]), 'explained_share': frac([r for r in B if r.get('confidence') == k], 'explained')}
                            for k in ('高', '中', '低', '无法解释')},
        'distant_strong_rate_all': float(np.mean([x >= 0.3 for r in rows_out if 's' in r for x in r['s']['distant']])),
        'near_miss_strong_rate_all': float(np.mean([x >= 0.3 for r in rows_out if 's' in r for x in r['s']['near_miss']])),
        'new_strong_rate_all': float(np.mean([x >= 0.3 for r in rows_out if 's' in r for x in r['s']['new']])),
    }
    wB = summary['B_explained_share']
    summary['decision'] = ('中层也以原生神经元为原子，v1 的中层缺口主要来自测试格式' if wB >= 0.6 else
                           '中层以原生神经元为原子，但为 not_explained 单元补充子空间或字典节点' if wB >= 0.3 else
                           '中层的原子层改用残差子空间或字典特征')
    # 自检：随机 20 个普查正极位置，256 上下文重测与普查记录值
    sc = []
    rng2 = np.random.default_rng(11)
    for c in rng2.choice(sorted(pools), 20, replace=False):
        r = next(x for x in pools[c]['rows'] if x['kind'] == 'pos')
        L = S.census_layer(MK, RUN, units[c]['block'])
        g = r['gidx']
        k = int(np.where(L['top_idx'][units[c]['unit']] == g)[0][0])
        ref = float(L['top_val'][units[c]['unit'], k])
        sc.append(abs(res_pool[(c, r['slot'], 256)] - ref) / max(abs(ref), 1e-6))
    out = {'schema': 'explanation_test_result.v2', 'prereg_id': pre['id'], 'prereg_sha256': sha_lf(PRE), 'seconds': round(time.time() - t0, 1),
           'n_texts': len(texts), 'n_pool_forwards': len(pos_jobs),
           'self_check': {'n': len(sc), 'rel_err_median': float(np.median(sc)), 'rel_err_max': float(np.max(sc))},
           'summary': summary, 'units': rows_out}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    # 分类子代理的证据（只含待分类单元）
    todo = [r for r in rows_out if r.get('cause_source') == '待分类']
    lines = ['# 失败原因分类证据', '']
    for r in todo:
        u = units[r['code']]
        card = S.unit_card(model_key=MK, block=u['block'], unit=u['unit'], run=RUN, n_poles=20)['census']
        lines += [f"## {r['code']}（第 {u['block']} 层）", f"- 解释：{r['explanation']}（置信度 {r['confidence']}）"]
        if 's' in r:
            lines.append(f"- 文本测试（s = 激活 / 普查最大值；≥0.3 为强）：new {[round(x, 2) for x in r['s']['new']]}；near_miss {[round(x, 2) for x in r['s']['near_miss']]}；distant {[round(x, 2) for x in r['s']['distant']]}；结论 {r['text_outcome']}")
            lines.append(f"- 检索验证：{r['retrieval']}（正例 {r['n_pos']} 个，召回 {r['recall']}，负例误报 {r['fpr']}）")
        lines.append(f"- 上下文依赖（256 token 下强、28 token 下不强的普查正极比例）：{r['auto']['context_dependent_share']}")
        lines.append('- 普查正极片段（相对强度｜上下文）：')
        mx = card['max']
        for p in card['poles']['pos']:
            lines.append(f"  - {p['a'] / mx:.2f}｜{(p['left'] + '【' + p['hit'] + '】' + p['right']).replace(chr(10), '⏎')}")
        lines.append('')
    (W / 'classify_input.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('n_texts', 'n_pool_forwards', 'seconds', 'self_check')}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print('待分类单元', len(todo), '自动规则', sum(1 for r in rows_out if r.get('cause_source', '').startswith('自动')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
