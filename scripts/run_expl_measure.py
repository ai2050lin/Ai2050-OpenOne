# -*- coding: utf-8 -*-
"""EXPL-v1 前向测量与评估（research/interventions/EXPL-v1.prereg.json）。

输入：冻结的单元清单、解释（explain_*.json）、测试文本（tests_*.json）；在 Qwen3-4B 上逐段前向：
固定中性前缀 + 文本，取目标单元在文本部分 token 上 A = silu(G)⊙V 的最大值，除以普查最大值得到 s。
对照：同一层内按种子 20261024 错位配对，取配对单元的 5 段 new 文本在本单元上测量。
按预注册判据给出每个单元 win / tie / lose 与汇总；另做前向口径自检（普查正极片段重测）。

用法：python scripts/run_expl_measure.py --work <scratchpad/expl> --out research/interventions/EXPL-v1-results.json
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

PRE = ROOT / 'research/interventions/EXPL-v1.prereg.json'
UNITS = ROOT / 'research/interventions/EXPL-v1.units.json'
PREFIX = '以下是一段文字。The following is a short text.\n'
KINDS = ('new', 'near_miss', 'distant')


def file_sha_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main(argv=None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args(argv)
    if file_sha_lf(PRE) != PRE.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:
        raise SystemExit('EXPL-v1 预注册文件哈希不符')
    pre = json.loads(PRE.read_text(encoding='utf-8'))
    if hashlib.sha256(UNITS.read_bytes()).hexdigest() != pre['units']['sha256_raw'] and \
            hashlib.sha256(UNITS.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != pre['units']['sha256_raw']:
        raise SystemExit('单元清单哈希不符')
    units = {u['code']: u for u in json.loads(UNITS.read_text(encoding='utf-8'))['units']}
    work = Path(args.work)
    expl = {it['code']: it for f in sorted(glob.glob(str(work / 'explain_*.json')))
            for it in json.loads(Path(f).read_text(encoding='utf-8'))['items']}
    tests = {it['code']: it for f in sorted(glob.glob(str(work / 'tests_*.json')))
             for it in json.loads(Path(f).read_text(encoding='utf-8'))['items']}
    t0 = time.time()

    # ---- 对照配对：同一层内错位（种子 20261024） ----
    rng = np.random.default_rng(20261024)
    by_layer = {}
    for c, u in sorted(units.items()):
        by_layer.setdefault(u['block'], []).append(c)
    partner = {}
    for b, codes in sorted(by_layer.items()):
        with_new = [c for c in codes if len((tests.get(c) or {}).get('new') or []) > 0]
        perm = list(rng.permutation(len(codes)))
        for i, c in enumerate(codes):
            cands = [codes[(perm[i] + k) % len(codes)] for k in range(1, len(codes) + 1)]
            partner[c] = next((p for p in cands if p != c and p in with_new), None)

    # ---- 待测文本：(文本, [(单元 code, 类别, 序号)]) ----
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

    # 自检：随机 20 个单元的普查正极首条片段（左侧上下文 + 命中 + 右侧）
    from server.neuron_census_service import unit_card
    srng = np.random.default_rng(7)
    check_codes = list(srng.choice(sorted(units), 20, replace=False))
    check_ref = {}
    for c in check_codes:
        u = units[c]
        p = unit_card(model_key=pre['model']['model_key'], block=u['block'], unit=u['unit'], run=None, n_poles=1)['census']['poles']['pos'][0]
        text = p['left'] + p['hit'] + p['right']
        check_ref[c] = p['a']
        add(text, c, 'selfcheck', 0)

    tok = AutoTokenizer.from_pretrained(str(ROOT / pre['model']['path']))
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / pre['model']['path']), dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    layers = list(model.get_decoder().layers)
    need = {}
    for c, u in units.items():
        need.setdefault(u['block'], set()).add(u['unit'])
    need = {b: torch.tensor(sorted(s), device='cuda') for b, s in need.items()}
    col = {b: {int(j): k for k, j in enumerate(need[b].tolist())} for b in need}
    cap = {}

    def make_hooks(b):
        def hg(mod, a, out):
            cap[('g', b)] = out[..., need[b]].float()

        def hv(mod, a, out):
            cap[('v', b)] = out[..., need[b]].float()
        return hg, hv
    for b in need:
        hg, hv = make_hooks(b)
        layers[b].mlp.gate_proj.register_forward_hook(hg)
        layers[b].mlp.up_proj.register_forward_hook(hv)
    n_pre = len(tok(PREFIX, add_special_tokens=False)['input_ids'])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    results = {}                                     # (code, kind, i) → 最大激活
    texts = list(jobs)
    with torch.inference_mode():
        for s in range(0, len(texts), args.batch):
            chunk = texts[s:s + args.batch]
            enc = [tok(PREFIX + t, add_special_tokens=False)['input_ids'] for t in chunk]
            W = max(len(e) for e in enc)
            ids = torch.full((len(enc), W), pad, dtype=torch.long)
            mask = torch.zeros((len(enc), W), dtype=torch.long)
            for r, e in enumerate(enc):
                ids[r, :len(e)] = torch.tensor(e)
                mask[r, :len(e)] = 1
            model.model(input_ids=ids.cuda(), attention_mask=mask.cuda())
            valid = mask.cuda().bool().clone()
            valid[:, :n_pre] = False                 # 只看文本部分
            for b in need:
                A = torch.nn.functional.silu(cap[('g', b)]) * cap[('v', b)]       # (B, T, n_units_b)
                A = A.masked_fill(~valid[..., None], float('-inf'))
                amax = A.max(1).values.cpu().numpy()
                for r, t in enumerate(chunk):
                    for (c, kind, i) in jobs[t]:
                        u = units[c]
                        if u['block'] == b:
                            results[(c, kind, i)] = float(amax[r, col[b][u['unit']]])
            cap.clear()

    # ---- 评估（判据见预注册 unit_outcome） ----
    per_unit = []
    for c in sorted(units):
        u = units[c]
        e = expl.get(c) or {}
        cm = u['census_max']
        rec = {'code': c, 'block': u['block'], 'unit': u['unit'], 'census_max': cm, 'explanation': e.get('explanation'),
               'confidence': e.get('confidence'), 'partner': partner.get(c)}
        if not e or e.get('confidence') == '无法解释':
            rec.update(outcome='lose', reason='无法解释' if e else '缺少解释')
            per_unit.append(rec)
            continue
        if cm <= 0:
            rec.update(outcome='unscorable', reason='普查最大值 ≤ 0')
            per_unit.append(rec)
            continue
        s = {k: [results[(c, k, i)] / cm for i in range(5) if (c, k, i) in results] for k in KINDS + ('control',)}
        rec['s'] = s
        rec['missing'] = {k: 5 - len(s[k]) for k in KINDS}
        strong = {k: (float(np.mean(np.array(v) >= 0.3)) if v else None) for k, v in s.items()}
        rec['strong_rate'] = strong
        med = {k: (float(np.median(v)) if v else None) for k, v in s.items()}
        rec['median'] = med
        rec['blind_max'] = max(s['new']) if s['new'] else None
        n_strong_new = int(np.sum(np.array(s['new']) >= 0.3)) if s['new'] else 0
        c1 = n_strong_new >= 2
        c2 = med['new'] is not None and med['near_miss'] is not None and med['new'] - med['near_miss'] >= 0.10
        c3 = strong['control'] is not None and strong['new'] is not None and strong['new'] - strong['control'] >= 0.2
        rec['criteria'] = {'new_strong_ge_2': c1, 'new_minus_near_ge_0.10': c2, 'new_minus_control_strong_ge_0.2': c3}
        if c1 and c2 and c3:
            rec['outcome'] = 'win'
        elif med['new'] is None or med['near_miss'] is None or med['new'] <= med['near_miss']:
            rec['outcome'] = 'lose'
        else:
            rec['outcome'] = 'tie'
        per_unit.append(rec)

    def share(rows, key='outcome', val='win'):
        return float(np.mean([r.get(key) == val for r in rows])) if rows else None
    n = len(per_unit)
    seg = {name: [r for r in per_unit if lo <= r['block'] / 36 < hi] for name, lo, hi in (('浅', 0, 1 / 3), ('中', 1 / 3, 2 / 3), ('深', 2 / 3, 1.01))}
    scored = [r for r in per_unit if 's' in r]
    summary = {
        'n_units': n,
        'outcomes': {k: sum(r['outcome'] == k for r in per_unit) for k in ('win', 'tie', 'lose', 'unscorable')},
        'win_share': share(per_unit),
        'cannot_explain': sum(r.get('reason') == '无法解释' for r in per_unit),
        'by_segment': {k: {'n': len(v), 'win_share': share(v)} for k, v in seg.items()},
        'by_confidence': {k: {'n': len([r for r in per_unit if r.get('confidence') == k]),
                              'win_share': share([r for r in per_unit if r.get('confidence') == k])} for k in ('高', '中', '低', '无法解释')},
        'blind_max_ge_1': float(np.mean([r['blind_max'] is not None and r['blind_max'] >= 1.0 for r in scored])) if scored else None,
        'distant_any_strong': float(np.mean([(r['strong_rate']['distant'] or 0) > 0 for r in scored])) if scored else None,
        'distant_strong_rate_all': float(np.mean([x >= 0.3 for r in scored for x in r['s']['distant']])) if scored else None,
        'near_miss_strong_rate_all': float(np.mean([x >= 0.3 for r in scored for x in r['s']['near_miss']])) if scored else None,
        'new_strong_rate_all': float(np.mean([x >= 0.3 for r in scored for x in r['s']['new']])) if scored else None,
        'control_strong_rate_all': float(np.mean([x >= 0.3 for r in scored for x in r['s']['control']])) if scored else None,
    }
    w = summary['win_share']
    summary['decision'] = ('原子层采用原生神经元' if w >= 0.6 else
                           '原生神经元作为原子，需补充子空间 / 字典节点并对 lose 单元做原因分类' if w >= 0.3 else
                           '原子层不采用单个神经元，转向残差子空间或字典特征')
    sc = [abs(results[(c, 'selfcheck', 0)] - check_ref[c]) / max(abs(check_ref[c]), 1e-6) for c in check_codes if (c, 'selfcheck', 0) in results]
    out = {'schema': 'explanation_test_result.v1', 'prereg_id': pre['id'], 'prereg_sha256': file_sha_lf(PRE),
           'units_sha256_raw': hashlib.sha256(UNITS.read_bytes()).hexdigest(), 'prefix': PREFIX, 'n_prefix_tokens': n_pre,
           'n_texts': len(texts), 'seconds': round(time.time() - t0, 1),
           'self_check': {'n': len(sc), 'rel_err_median': float(np.median(sc)) if sc else None, 'rel_err_max': float(np.max(sc)) if sc else None,
                          'note': '普查片段原为 512 窗口中的位置，这里接在固定前缀后重测，位置与上文不同，只报告不判定'},
           'summary': summary, 'units': per_unit}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('n_texts', 'seconds', 'self_check')}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
