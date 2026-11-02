# -*- coding: utf-8 -*-
"""AM-v2：用「槽位条件化的专家乘积」（Phase 281 预注册确认）预测未见组合的下一词分布。

材料：FU-v2 的 32 个事件 × 风格 × 逻辑 × 槽位（中 / 英），重新采集末位全词表对数概率。按事件 sha256 切分发现 16 / 确认 16。
  P1 事件内补全（全部事件，无拟合）：同一事件同一槽位，用另外三格加法补全第四格 Ŷ[s,l] = Y[1−s,l] + Y[s,1−l] − Y[1−s,1−l]；
     对照：相邻两格平均。
  P2 跨事件迁移（确认集事件；效应只在发现集估计）：新事件只看到 (日常, 因果) 格 Y00，预测其余 9 格
     Ŷ[s,l,g] = Y00[g] + s·Δs_g + l·Δl_g；对照：只用 Y00、效应不分槽位、加交互项、发现集同格平均（总体先验）、均匀。
  KL：被预测格在该事件 12 个真实格各自前 20 的并集上，真分布与预测各自重新归一化。
  H1  P1：加法补全 KL ≤ 0.5 × 相邻平均 KL，在（语言 × 槽位）6 格中至少 5 格。
  H2  P2：恢复比例 1 − KL_专家乘积 / KL_只用 Y00 ≥ 0.5（每种语言）。
  H3  P2：分槽位的效应比不分槽位的 KL 低 ≥ 10%（每种语言）——槽位条件化有预测价值。
  H4  P2：加交互项只使 KL 降低 ≤ 10%（每种语言）——可加已足够。
被测模型（bf16，不使用量化）：Qwen3-4B、Gemma-3-4B-it，分别判定。
用法：python scripts/assembly_v2.py prereg | run --model M | analyze --model M
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

from shared.dist_protocol import assembly as S  # noqa: E402
import fu_materials_v2 as MAT  # noqa: E402
import fusion_v1 as FU1  # noqa: E402

PREREG = ROOT / 'research/assembly/AM-v2.prereg.json'
NATIVE = Path('D:/ai2050_native/_assembly')
LANGS = ['en', 'zh']
K = 3


def split():
    order = sorted(range(len(MAT.ITEMS)), key=lambda i: hashlib.sha256(f'AM-v2|{MAT.ITEMS[i][0]}'.encode()).hexdigest())
    return {'conf': sorted(order[:16]), 'disc': sorted(order[16:])}


def prereg_doc():
    return {'id': 'AM-v2', 'schema': 'assembly_prereg.v2', 'created': '2026-11-02',
            'basis': 'Phase 281（FU-v2）：槽位条件化的专家乘积在两模型、中英、三槽位上预注册确认；本版检验它能否预测未见组合',
            'models': {'qwen3-4b': 'bf16', 'gemma-3-4b-it': 'bf16'}, 'precision_rule': '不使用任何量化',
            'materials': 'scripts/fu_materials_v2.py（FU-v2 的同一批提示）', 'split': split(),
            'note_reuse': '提示与 FU-v2 相同（FU-v2 判定的是可加份额，不涉及预测）；本版重新采集全词表对数概率',
            'power_note': 'P1：每种语言 32 事件 × 3 槽位 × 4 格；P2：16 个确认事件 × 9 格；连续量阈值',
            'doc': __doc__, 'criteria': S.CRITERIA_V2}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（AM-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path):
    d = ROOT / 'research/assembly' / f'AM-v2-{Path(model_path).name}'
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
    snap = FU1.snapshot(model_path)
    od = NATIVE / snap[:16] / 'AM-v2'
    od.mkdir(parents=True, exist_ok=True)
    tok, model = FU1.load(model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    for lang in LANGS:
        out = []
        for it in MAT.ITEMS:
            seqs = [tok(MAT.prompt(it, lang, *c)).input_ids for c in MAT.cells()]
            n = max(map(len, seqs))
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.as_tensor(s)
                att[b, :len(s)] = 1
            o = model(input_ids=ids.cuda(), attention_mask=att.cuda())
            ar = torch.arange(len(seqs), device='cuda')
            last = torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
            out.append(torch.log_softmax(o.logits[ar, last].float(), -1).to(torch.float16).cpu().numpy())
            del o
        np.save(od / f'logp_{lang}.npy', np.stack(out))                  # [n_event, 12, V]
    man = {'model': model_path, 'model_snapshot': snap, 'precision': 'bf16', 'seconds': round(time.time() - t0, 1),
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip()}
    save(model_path, man)
    print(json.dumps(man, ensure_ascii=False))


def do_analyze(model_path):
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = FU1.snapshot(model_path)
    od = NATIVE / snap[:16] / 'AM-v2'
    sp = split()
    res, p1cells = {}, {}
    for lang in LANGS:
        A = np.load(od / f'logp_{lang}.npy').astype(np.float32)          # [n, 12, V]
        n, _, V = A.shape
        Y = A.reshape(n, 2, 2, K, V)                                     # 格序 6s + 3l + g
        cand = [np.unique(np.argsort(-A[i], 1)[:, :crit['top_k_cand']]) for i in range(n)]
        # P1
        p1 = {}
        for g in range(K):
            ka, kn = [], []
            for i in range(n):
                Z = Y[i, :, :, g]
                for s_ in (0, 1):
                    for l_ in (0, 1):
                        ka.append(S.kl_restricted(Z[s_, l_], S.additive_completion(Z, s_, l_), cand[i]))
                        kn.append(S.kl_restricted(Z[s_, l_], S.neighbor_mean(Z, s_, l_), cand[i]))
            p1[MAT.SLOTS[g]] = {'kl_additive': float(np.mean(ka)), 'kl_neighbor': float(np.mean(kn)),
                                'ratio': float(np.mean(ka) / np.mean(kn))}
            p1cells[f'{lang}/{MAT.SLOTS[g]}'] = p1[MAT.SLOTS[g]]['ratio']
        # P2
        D = sp['disc']
        eff = {g: S.slot_effects(Y[D, :, :, g]) for g in range(K)}
        ds_all = np.mean([eff[g][0] for g in range(K)], 0)
        dl_all = np.mean([eff[g][1] for g in range(K)], 0)
        pop = Y[D].mean(0)                                               # [2, 2, K, V]
        kls = {k_: [] for k_ in ('base', 'poe_slot', 'poe_noslot', 'poe_interaction', 'population', 'uniform')}
        by_cell = {}
        for i in sp['conf']:
            for g in range(K):
                y00 = Y[i, 0, 0, g]
                ds, dl, dsl = eff[g]
                for s_ in (0, 1):
                    for l_ in (0, 1):
                        if s_ == 0 and l_ == 0:
                            continue
                        t = Y[i, s_, l_, g]
                        preds = {'base': y00, 'poe_slot': y00 + s_ * ds + l_ * dl, 'poe_noslot': y00 + s_ * ds_all + l_ * dl_all,
                                 'poe_interaction': y00 + s_ * ds + l_ * dl + s_ * l_ * dsl, 'population': pop[s_, l_, g],
                                 'uniform': np.zeros(V, np.float32)}
                        for k_, pr in preds.items():
                            v = S.kl_restricted(t, pr, cand[i])
                            kls[k_].append(v)
                            by_cell.setdefault(f'{MAT.SLOTS[g]}/s{s_}l{l_}', {}).setdefault(k_, []).append(v)
        m = {k_: float(np.mean(v)) for k_, v in kls.items()}
        p2 = {'kl': m, 'recovered_vs_base': 1 - m['poe_slot'] / m['base'], 'recovered_vs_uniform': 1 - m['poe_slot'] / m['uniform'],
              'slot_gain': (m['poe_noslot'] - m['poe_slot']) / m['poe_noslot'], 'interaction_gain': (m['poe_slot'] - m['poe_interaction']) / m['poe_slot'],
              'by_cell': {c: {k_: float(np.mean(v)) for k_, v in d.items()} for c, d in by_cell.items()}}
        res[lang] = {'P1': p1, 'P2': p2}
        print(lang, 'P1', {g: round(v['ratio'], 3) for g, v in p1.items()}, 'P2', {k_: round(v, 3) for k_, v in m.items()},
              'rec_base', round(p2['recovered_vs_base'], 3), 'slot_gain', round(p2['slot_gain'], 3), 'int_gain', round(p2['interaction_gain'], 3), flush=True)
    n1 = sum(v <= crit['H1_ratio_max'] for v in p1cells.values())
    verdict = {'H1': bool(n1 >= crit['H1_min_cells']),
               'H2': bool(all(res[l]['P2']['recovered_vs_base'] >= crit['H2_recovered_min'] for l in LANGS)),
               'H3': bool(all(res[l]['P2']['slot_gain'] >= crit['H3_slot_gain_min'] for l in LANGS)),
               'H4': bool(all(res[l]['P2']['interaction_gain'] <= crit['H4_interaction_gain_max'] for l in LANGS))}
    rep = save(model_path, {'criteria': crit, 'split': sp, 'results': res, 'P1_cells': p1cells, 'counts': {'H1': n1}, 'verdict': verdict})
    p = write_result(f'am_v2_{Path(model_path).name}', rep, model_path=model_path, seed=crit['seed'], tool='scripts/assembly_v2.py')
    print(json.dumps(verdict), 'H1 cells', {k_: round(v, 3) for k_, v in p1cells.items()}, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
