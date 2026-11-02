# -*- coding: utf-8 -*-
"""FU-v2：确认「槽位条件化的专家乘积」——固定语法槽位后，风格与逻辑对下一词的作用近似相加；交互主要是「因子 × 槽位」。

FU-v1（Phase 280）事后发现：同一槽位内风格 × 逻辑可加份额 0.89–0.95（输出），但风格操纵弱（主效应 3%）、只有英文两种槽位。
本版：32 个新事件；正式风格同时更换前提措辞与连接词（consequently / nevertheless；因此 / 然而）；三种槽位（动词 / 名词 / 形容词）；英文与中文。
本版没有拟合参数，全部事件都用于判定。读出：末位下一词对数概率（各格前 20 候选的并集，按格中心化）与全部层隐状态。
  H0  行为门（每种语言，各槽位合并）：槽位内风格与逻辑各自占槽位内平方和 ≥ 0.1。
  H1  输出：槽位内可加份额 (SS_s + SS_l) / (SS_s + SS_l + SS_sl) ≥ 0.8，在（3 槽位 × 2 语言）6 格中至少 5 格。
  H2  最终层隐状态：同一口径，6 格中至少 5 格。
  H3  输出（每种语言）：SS(风格×槽位) + SS(逻辑×槽位) ≥ 3 × [SS(风格×逻辑) + SS(三阶)]。
  描述：逐层槽位内可加份额；风格 / 逻辑效应方向在不同槽位之间的 cos（同一内部方向还是不同方向）。
被测模型（bf16，不使用量化）：Qwen3-4B、Gemma-3-4B-it。两模型分别判定。
用法：python scripts/fusion_v2.py prereg | run --model M | analyze --model M
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

from shared.dist_protocol import fusion as F  # noqa: E402
import fu_materials_v2 as MAT  # noqa: E402
import fusion_v1 as FU1  # noqa: E402

PREREG = ROOT / 'research/fusion/FU-v2.prereg.json'
K = 3
LANGS = ['en', 'zh']


def prereg_doc():
    texts = [MAT.prompt(it, lang, *c) for lang in LANGS for it in MAT.ITEMS for c in MAT.cells()]
    return {'id': 'FU-v2', 'schema': 'fusion_prereg.v2', 'created': '2026-11-02',
            'basis': 'FU-v1（Phase 280）事后：槽位内风格与逻辑近似相加，交互集中在「因子 × 槽位」；本版作为主假设预注册',
            'models': {'qwen3-4b': 'bf16', 'gemma-3-4b-it': 'bf16'}, 'precision_rule': '不使用任何量化',
            'materials': {'source': 'scripts/fu_materials_v2.py', 'n_events': len(MAT.ITEMS), 'n_prompts': len(texts),
                          'sha256': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
                          'example_en': [MAT.prompt(MAT.ITEMS[0], 'en', *c) for c in MAT.cells()[:6]],
                          'example_zh': [MAT.prompt(MAT.ITEMS[0], 'zh', *c) for c in MAT.cells()[:6]]},
            'power_note': '连续量份额阈值；每种语言 32 个事件 × 12 格',
            'doc': __doc__, 'criteria': F.CRITERIA_V2}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（FU-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path):
    d = ROOT / 'research/fusion' / f'FU-v2-{Path(model_path).name}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, upd):
    p = report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def do_run(model_path):
    import torch
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = FU1.snapshot(model_path)
    od = FU1.NATIVE / snap[:16] / 'FU-v2'
    od.mkdir(parents=True, exist_ok=True)
    tok, model = FU1.load(model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    arrays, tops = {}, {}
    for lang in LANGS:
        H = []
        for ii, it in enumerate(MAT.ITEMS):
            seqs = [tok(MAT.prompt(it, lang, *c)).input_ids for c in MAT.cells()]
            n = max(map(len, seqs))
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.as_tensor(s)
                att[b, :len(s)] = 1
            o = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            ar = torch.arange(len(seqs), device='cuda')
            last = torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
            H.append(torch.stack([h[ar, last].float() for h in o.hidden_states], 1).cpu().numpy())
            lp = torch.log_softmax(o.logits[ar, last].float(), -1)
            top = lp.topk(crit['top_k_per_cell'], -1).indices
            cand = torch.unique(top.flatten())
            arrays[f'{lang}_cand{ii}'] = cand.cpu().numpy()
            arrays[f'{lang}_lp{ii}'] = lp[:, cand].cpu().numpy()
            if ii == 0:
                tops[lang] = [[tok.decode([int(t)]) for t in row[:5]] for row in top]
            del o
        np.save(od / f'states_{lang}.npy', np.stack(H))
    np.savez(od / 'logp.npz', **arrays)
    man = {'model': model_path, 'model_snapshot': snap, 'precision': 'bf16', 'seconds': round(time.time() - t0, 1),
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip()}
    save(model_path, {**man, 'top5_first_event': tops})
    print(json.dumps(man, ensure_ascii=False))


def do_analyze(model_path):
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = FU1.snapshot(model_path)
    od = FU1.NATIVE / snap[:16] / 'FU-v2'
    z = np.load(od / 'logp.npz')
    res, cells_out, cells_state = {}, {}, {}
    for lang in LANGS:
        H = np.load(od / f'states_{lang}.npy').astype(np.float64)        # [n_item, 12, L+1, d]
        n, _, L1, _ = H.shape
        fo = {e: 0.0 for e in F.EFFECTS + ['total']}
        wo = {g: {'s': 0.0, 'l': 0.0, 'sl': 0.0} for g in range(K)}
        for i in range(n):
            Y = F.center_cells(z[f'{lang}_lp{i}'])
            for e, v in F.factorial_ss(Y, K).items():
                fo[e] += v
            for g, w in F.within_slot_ss(Y, K).items():
                for e in w:
                    wo[g][e] += w[e]
        ws = []
        for l in range(L1):
            acc = {g: {'s': 0.0, 'l': 0.0, 'sl': 0.0} for g in range(K)}
            for i in range(n):
                for g, w in F.within_slot_ss(H[i, :, l], K).items():
                    for e in w:
                        acc[g][e] += w[e]
            ws.append(acc)
        within_out = {MAT.SLOTS[g]: F.within_additive(wo[g]) for g in range(K)}
        within_state = [{MAT.SLOTS[g]: F.within_additive(ws[l][g]) for g in range(K)} for l in range(L1)]
        tot_w = {e: sum(wo[g][e] for g in range(K)) for e in ('s', 'l', 'sl')}
        sw = sum(tot_w.values())
        h0 = {'style_share': tot_w['s'] / sw, 'logic_share': tot_w['l'] / sw}
        h3 = {'slot_inter': (fo['sg'] + fo['lg']) / fo['total'], 'sl_inter': (fo['sl'] + fo['slg']) / fo['total']}
        h3['ratio'] = h3['slot_inter'] / max(h3['sl_inter'], 1e-12)
        # 风格 / 逻辑效应方向在不同槽位之间的 cos（中层）
        mid = round((L1 - 1) / 2)
        Hm = H[:, :, mid].reshape(n, 2, 2, K, -1)
        sdir = [(Hm[:, 1, :, g] - Hm[:, 0, :, g]).mean((0, 1)) for g in range(K)]
        ldir = [(Hm[:, :, 1, g] - Hm[:, :, 0, g]).mean((0, 1)) for g in range(K)]
        cosf = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        cos = {'style': {f'{MAT.SLOTS[a]}-{MAT.SLOTS[b]}': cosf(sdir[a], sdir[b]) for a in range(K) for b in range(a + 1, K)},
               'logic': {f'{MAT.SLOTS[a]}-{MAT.SLOTS[b]}': cosf(ldir[a], ldir[b]) for a in range(K) for b in range(a + 1, K)},
               'style_vs_logic': {MAT.SLOTS[g]: cosf(sdir[g], ldir[g]) for g in range(K)}, 'layer': mid}
        res[lang] = {'factorial_output_shares': {e: fo[e] / fo['total'] for e in F.EFFECTS}, 'within_output': within_out,
                     'within_state_final': within_state[-1], 'within_state_by_layer': within_state, 'H0': h0, 'H3': h3, 'direction_cos': cos}
        for g in MAT.SLOTS:
            cells_out[f'{lang}/{g}'] = within_out[g]
            cells_state[f'{lang}/{g}'] = within_state[-1][g]
    n1 = sum(v >= crit['H1_within_additive_min'] for v in cells_out.values())
    n2 = sum(v >= crit['H2_state_additive_min'] for v in cells_state.values())
    verdict = {'H0': bool(all(res[l]['H0']['style_share'] >= crit['H0_within_share_min'] and res[l]['H0']['logic_share'] >= crit['H0_within_share_min'] for l in LANGS)),
               'H1': bool(n1 >= crit['H1_min_cells']), 'H2': bool(n2 >= crit['H1_min_cells']),
               'H3': bool(all(res[l]['H3']['ratio'] >= crit['H3_slot_over_sl'] for l in LANGS))}
    rep = save(model_path, {'criteria': crit, 'results': res, 'cells_output': cells_out, 'cells_state_final': cells_state,
                            'counts': {'H1': n1, 'H2': n2}, 'verdict': verdict})
    p = write_result(f'fu_v2_{Path(model_path).name}', rep, model_path=model_path, seed=crit['seed'], tool='scripts/fusion_v2.py')
    print(json.dumps(verdict), '\nout', {k_: round(v, 3) for k_, v in cells_out.items()}, '\nstate', {k_: round(v, 3) for k_, v in cells_state.items()},
          '\nH0', {l: {k_: round(v, 3) for k_, v in res[l]['H0'].items()} for l in LANGS},
          '\nH3', {l: {k_: round(v, 3) for k_, v in res[l]['H3'].items()} for l in LANGS},
          '\ncos', {l: res[l]['direction_cos'] for l in LANGS}, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
