# -*- coding: utf-8 -*-
"""FU-v1：风格 × 逻辑 × 语法在下一词预测中怎样融合——是各自独立相加（专家乘积），还是相互作用？

材料：scripts/fu_materials_v1.py，40 个事件 × 风格（日常 / 正式前提）× 逻辑（so / but）× 语法槽位（主语后接动词 / 所有格 ' 后接名词），
英文；按事件切分发现 24 / 确认 16，判定只用确认集（本版没有需要拟合的参数，发现集只作描述）。
读出：末位（区分全部三个因子的文本都已出现）的下一词对数概率与全部层的隐状态。
  H0  行为门：输出上三个因子各自的主效应份额（确认集合并）≥ 0.05。
  H1  输出可加：候选 token（8 个格各自前 20 的并集）的按格中心化对数概率，三个主效应占格间平方和 ≥ 0.8。
  H2  状态可加：中层段 [L/3, 2L/3] 各层隐状态「去掉语法主效应后」的可加份额 (s + l) / (总 − g) 平均 ≥ 0.8
      （末位 token 只随语法因子不同，语法主效应主要是 token 身份，不计入）。
  H3  融合在后段：同一口径下，最终层隐状态的交互份额 ≥ 中层段平均交互份额的 1.5 倍。
  描述：逐层可加份额曲线、交互在各因子对上的分配、承载交互最多的候选 token。
被测模型（全部 bf16，不使用任何量化）：Qwen3-4B（主）；Gemma-3-4B-it（复现，不同厂家）。
用法：python scripts/fusion_v1.py prereg | run --model M | analyze --model M
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
import fu_materials_v1 as MAT  # noqa: E402

PREREG = ROOT / 'research/fusion/FU-v1.prereg.json'
NATIVE = Path('D:/ai2050_native/_fusion')


def prereg_doc():
    texts = [MAT.prompt(it, *c) for it in MAT.ITEMS for c in MAT.cells()]
    return {'id': 'FU-v1', 'schema': 'fusion_prereg.v1', 'created': '2026-11-02',
            'basis': '用户方案：把词嵌入、风格、逻辑、语法融合，破解整体编码；风格线（Phase 275–279）之后的融合检验',
            'models': {'qwen3-4b': 'bf16（主）', 'gemma-3-4b-it': 'bf16（复现）'},
            'precision_rule': '不使用任何量化（用户 2026-11-02 指示）',
            'materials': {'source': 'scripts/fu_materials_v1.py', 'split': MAT.split(), 'n_prompts': len(texts),
                          'sha256': hashlib.sha256(json.dumps(texts).encode()).hexdigest()},
            'tokenization': '按各 tokenizer 的默认 add_special_tokens（Gemma 加 BOS，Qwen 不加）；读出位置为最后一个 token',
            'readout_note': '末位 token 只随语法因子不同（主语名词 vs 撇号）；风格与逻辑只能经上下文影响末位，hs[0] 上只有语法主效应',
            'power_note': '连续量份额阈值；确认集 16 个事件 × 8 格',
            'doc': __doc__, 'criteria': F.CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（FU-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def load(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    return tok, model


def snapshot(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    return model_snapshot(model_path)['sha256']


def report_path(model_path):
    d = ROOT / 'research/fusion' / f'FU-v1-{Path(model_path).name}'
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
    snap = snapshot(model_path)
    od = NATIVE / snap[:16] / 'FU-v1'
    od.mkdir(parents=True, exist_ok=True)
    tok, model = load(model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    H, CAND, LP, TOPS = [], [], [], []
    for it in MAT.ITEMS:
        seqs = [tok(MAT.prompt(it, *c)).input_ids for c in MAT.cells()]
        n = max(map(len, seqs))
        ids = torch.full((8, n), pad, dtype=torch.long)
        att = torch.zeros((8, n), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.as_tensor(s)
            att[b, :len(s)] = 1
        o = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
        last = torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
        ar = torch.arange(8, device='cuda')
        H.append(torch.stack([h[ar, last].float() for h in o.hidden_states], 1).cpu().numpy())   # [8, L+1, d]，float32（Gemma 激活超出 float16 范围）
        lp = torch.log_softmax(o.logits[ar, last].float(), -1)
        top = lp.topk(crit['top_k_per_cell'], -1).indices
        cand = torch.unique(top.flatten())
        CAND.append(cand.cpu().numpy())
        LP.append(lp[:, cand].cpu().numpy())
        TOPS.append([[tok.decode([int(t)]) for t in row[:5]] for row in top])
        del o
    H = np.stack(H)
    np.save(od / 'states.npy', H)
    np.savez(od / 'logp.npz', **{f'cand{i}': c for i, c in enumerate(CAND)}, **{f'lp{i}': x for i, x in enumerate(LP)})
    man = {'model': model_path, 'model_snapshot': snap, 'precision': 'bf16', 'n_layers_plus_emb': int(H.shape[2]), 'hidden': int(H.shape[3]),
           'seconds': round(time.time() - t0, 1), 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip()}
    (od / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    save(model_path, {**man, 'top5_examples': {MAT.ITEMS[i][0]: TOPS[i] for i in range(3)}})
    print(json.dumps(man, ensure_ascii=False))


def do_analyze(model_path):
    from provenance import write_result
    from transformers import AutoTokenizer
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = snapshot(model_path)
    od = NATIVE / snap[:16] / 'FU-v1'
    H = np.load(od / 'states.npy').astype(np.float64)
    z = np.load(od / 'logp.npz')
    tok = AutoTokenizer.from_pretrained(model_path)
    sp = MAT.split()
    L = H.shape[2] - 1
    res = {}
    for name, idx in sp.items():
        outd = [F.decompose(F.center_cells(z[f'lp{i}'])) for i in idx]
        out = F.pooled(outd)
        layers = [F.pooled([F.decompose(H[i, :, l]) for i in idx]) for l in range(L + 1)]
        res[name] = {'output': out, 'layers': layers}
    C = res['conf']
    band = list(range(round(L / 3), round(2 * L / 3) + 1))
    band_add = float(np.mean([C['layers'][l]['additive_ex_g'] for l in band]))
    band_int = 1 - band_add
    fin_int = C['layers'][L]['interaction_ex_g']
    h0 = {f: C['output'][f] for f in ('s', 'l', 'g')}
    verdict = {'H0': bool(all(v >= crit['H0_main_share_min'] for v in h0.values())),
               'H1': bool(C['output']['additive_share'] >= crit['H1_out_additive_min']),
               'H2': bool(band_add >= crit['H2_state_additive_min']),
               'H3': bool(fin_int >= crit['H3_interaction_growth'] * band_int)}
    # 描述：交互最大的候选 token（确认集）
    tok_int = {}
    K = F.contrasts()
    for i in sp['conf']:
        Y = F.center_cells(z[f'lp{i}'])
        Yc = Y - Y.mean(0)
        est = K @ Yc / 8.0
        inter = 8 * (est[3:] ** 2).sum(0)
        for t, v in zip(z[f'cand{i}'], inter):
            s = tok.decode([int(t)])
            tok_int[s] = tok_int.get(s, 0.0) + float(v)
    top_int = sorted(tok_int.items(), key=lambda kv: -kv[1])[:15]
    upd = {'criteria': crit, 'band': [band[0], band[-1]], 'output_conf': C['output'], 'output_disc': res['disc']['output'],
           'state_additive_by_layer_conf': [x['additive_share'] for x in C['layers']],
           'state_additive_ex_g_by_layer_conf': [x['additive_ex_g'] for x in C['layers']],
           'state_effects_by_layer_conf': C['layers'], 'band_additive': band_add, 'final_interaction': fin_int,
           'H0_main_shares': h0, 'top_interaction_tokens_conf': top_int, 'verdict': verdict}
    rep = save(model_path, upd)
    p = write_result(f'fu_v1_{Path(model_path).name}', rep, model_path=model_path, seed=crit['seed'], tool='scripts/fusion_v1.py')
    print(json.dumps(verdict), '\noutput conf', {k_: round(v, 3) for k_, v in C['output'].items()},
          '\nstate additive by layer', [round(x['additive_share'], 3) for x in C['layers']],
          f'\nband {band[0]}-{band[-1]} additive {band_add:.3f}; final interaction {fin_int:.3f}',
          '\ntop interaction tokens', [(k_, round(v, 1)) for k_, v in top_int[:10]], '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
