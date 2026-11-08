# -*- coding: utf-8 -*-
"""主线 H3（MA-H3）：冻结载体与预测规则（ResearchPlan v6 第 8 节，切片 A，中文主线）。

问法 v2  H2 发现两种范围的答案前缀末 token 不同（“是”/“通常是”），范围效应被词身份混淆。H3 起改为等长最小对：
        资料：“问：在这份资料里，X是什么颜色的？\n答：X是”；现实：“问：在现实世界里，X是什么颜色的？\n答：X是”。
        两句 token 数相同、只在“这份资料 / 现实世界”处不同，答案前缀完全一致。实体、划分、资料正文与草稿不变。
步骤  check     discovery 上用问法 v2 重做 32 格行为检查（强制选择，同 MA-H1），未过 0.8 的格排除
      discover  discovery 按实体再分 fit / eval 两半。成对材料：干净句 = 资料范围，反事实 = 同一条的现实范围；
                指标 m = logit(K) − logit(P)（答案位置首 token）。在 fit 半上用 act_diff、dla、atp、atp_ig、exact 给 1,188 个
                组件（36 × 32 个头 + 36 个 MLP 块）排序（复用 MB-v1 的引擎与实现）
      evalfit   在 eval 半上同题比较：各方法在 10 个预算下的保留曲线 f（cf 替换）、移除曲线，随机与分层匹配随机对照
      freeze    按预注册规则选方法与预算，冻结载体集合 C 与预测规则，写 frozen.json（含 sha256）
      confirm   开封 confirm（一次），在 84 个新实体上检验冻结的预测，报告未解释部分
判据  CRITERIA（随结果返回，看到结果后不改）。
用法  python scripts/mainline_h3.py prereg|check|discover|evalfit|freeze|confirm --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

import arbitration_v1 as CF  # noqa: E402
import mainline_h0 as V3  # noqa: E402
import mainline_h0_v5 as V5  # noqa: E402
import mainline_h1 as H1  # noqa: E402
from shared.dist_protocol import mech_bench as B  # noqa: E402

OUT = ROOT / 'research/mainline/H3'
VERSION = 'v4'
PREREG = OUT / f'MA-H3-{VERSION}.prereg.json'
# v4 与 v3 问法相同，只把批大小 8 → 2（v3 的 discover 在 16 GB 卡上显存溢出到系统内存，14 小时无进度），行为检查沿用 check_v3
CHECK = OUT / 'check_v3.json'
FROZEN = OUT / 'frozen.json'
SEED = 'MA-H1'           # 材料构造与 MA-H1 相同
SPLIT_SEED = 'MA-H3'
# v1（MA-H3.prereg.json）：范围词只在问题里，答案前缀为“X是”。discovery 现实范围准确率 0.14，16 个现实格全部未过（check_v1.json）
Q_V1 = {'doc': '问：在这份资料里，{X}是什么颜色的？\n答：{X}是', 'real': '问：在现实世界里，{X}是什么颜色的？\n答：{X}是'}
# v2：答案前缀也带范围词，两种范围仍等长、末 token 相同
Q_V2 = {'doc': '问：在这份资料里，{X}是什么颜色的？\n答：在这份资料里，{X}是', 'real': '问：在现实世界里，{X}是什么颜色的？\n答：在现实世界里，{X}是'}
# v2 现实范围准确率 0.30（check_v2.json），仍全部未过。与 MA-H0-v5（0.97）的差别只剩“通常”。
# v3：两种范围的问题与答案前缀都带“通常”，仍等长、末 token 相同，唯一差别是“这份资料 / 现实世界”
Q_V3 = {'doc': '问：在这份资料里，{X}通常是什么颜色的？\n答：在这份资料里，{X}通常是', 'real': '问：在现实世界里，{X}通常是什么颜色的？\n答：在现实世界里，{X}通常是'}
Q_CUR = {'v2': Q_V2, 'v3': Q_V3, 'v4': Q_V3}[VERSION]
METHODS = B.METHODS
BUDGETS = B.BUDGETS
H1_EXCLUDED = [('real', 'last', 3, 0, 'pair'), ('real', 'last', 3, 1, 'pair')]

CRITERIA = {
    'gate_cell_acc': 0.8, 'gate_rule': '问法 v2 在 discovery 上逐格强制选择准确率；未过的格与 MA-H1 排除的两格一起排除，成对材料只用两种范围都可用的格',
    'pair_gate': 'MB-v1 行为门：干净句 m > 0 的比例 ≥ 0.8，且干净与反事实的平均 m 相差 ≥ 1 nat',
    'n_fit': 256, 'n_evalfit': 320, 'batch': 2,
    'vram_fraction': 0.95, 'vram_rule': '缓存分配器上限为显存的 95%，超出时报错停止，不溢出到系统内存',
    'method_rule': '在 discovery-eval 半上，cf 保留曲线的 auc_f 最高的方法；并列取 auc_f 区间下界更高者',
    'budget_rule': '所选方法在 discovery-eval 半上平均 f ≥ 0.8 的最小预算；没有则取 10%，并记“未达 0.8”',
    'confirm_predictions': {
        'P1_sufficiency': '只保留 C（其余换成现实范围的激活）时 f(C) ≥ 0.7，且比同预算的 3 组随机集合与 3 组分层匹配随机集合中最高者高 ≥ 0.3，实体 bootstrap 95% 区间下界 > 0',
        'P2_necessity': '只把 C 换成现实范围的激活时，资料范围相对现实范围的差距保留比例 f_remove ≤ 0.5',
        'P3_replication': '|f_confirm(C) − f_disc_eval(C)| ≤ 0.15',
    },
    'multiple_comparison': 'P1–P3 用 Holm 校正时按三项通过 / 未通过计；P1 的区间按 Holm 调整后的水平 α/3 报告',
    'residual': '未解释部分 = 1 − f(C)，按记录形式、目标位置、k、草稿拆分报告',
    'dictionary': '字典方法只在第 30 块有 transcoder，不能与全模型组件同题比较，本阶段不纳入，记为“未纳入”',
    'bootstrap': 2000,
}


def prereg_doc():
    return {'id': f'MA-H3-{VERSION}', 'revision_note': 'v1（sha256 c2c35194…）只在问题里放范围词，答案前缀为“X是”；discovery 现实范围准确率 0.14，16 个现实格全部未过，按预注册无法组成成对材料。'
                                                      'v2（951c5a1b…）在答案前缀里也放范围词，现实范围准确率 0.30，仍全部未过。v3 在两种范围的问题与答案前缀里都加“通常”（等长、末 token 相同），其余不变。'
                                                      'v4 问法、材料、方法与判据同 v3，行为检查沿用 check_v3；只把批大小 8 → 2，并限制显存上限：v3 的 discover 在 16 GB 卡上显存溢出到系统内存，运行 14 小时无进度输出后停止，未产生结果',
            'question_v1': Q_V1, 'question_v2': Q_V2, 'question_v3': Q_V3, 'schema': 'mainline_prereg.v1', 'created': '2026-11-08' if VERSION == 'v4' else '2026-11-07', 'stage': 'H3',
            'basis': 'Phase 324（MA-H2）：答案前缀混淆、候选读颜色头；MB-v1 的定位方法与引擎', 'question': Q_CUR,
            'materials': 'MA-H1 的实体与划分不变；discovery 按 sha256(MA-H3|id) 在每类内对半分为 fit / eval；confirm 在冻结后开封一次',
            'methods': METHODS, 'budgets': BUDGETS, 'criteria': CRITERIA,
            'h1_power': 'MA-H1 的 T1–T3 针对 H4 / H5 的配对二值检验；本阶段的 f 曲线不是 McNemar 口径，按 MB-v1 的实体 bootstrap 报告区间'}


def do_prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    d = hashlib.sha256(text.encode()).hexdigest()
    PREREG.with_suffix('.sha256').write_text(d + '\n', encoding='utf-8', newline='\n')
    print('prereg', d)


def read_rows(split):
    if split == 'discovery':
        return H1.read_discovery()
    path = H1.SEALED / f'{split}.tsv'
    lines = path.read_text(encoding='utf-8').splitlines()
    head = lines[0].split('\t')
    return [{**dict(zip(head, l.split('\t'))), 'color': '', 'color_alt': []} for l in lines[1:]]


def priors(model, tok, rows, palette, forms):
    prior = {'_scores': {}}
    for r in rows:
        sc, _ = V3.score(model, tok, '问：{X}通常是什么颜色的？\n答：{X}通常是'.format(X=r['zh']), {c: forms['zh'][c] for c in palette})
        prior['_scores'][r['id']] = np.array([sc[c] for c in palette])
        prior[r['id']] = palette[int(np.argmax(prior['_scores'][r['id']]))]
    return prior


def build(rows, prior, palette, forms):
    """问法 v2 的全部 32 格条目。临时替换 V5 的问法模板，构造后恢复。"""
    saved = V5.TEXT
    V5.TEXT = copy.deepcopy(saved)
    V5.TEXT['zh']['q'] = Q_CUR
    V5.SEED = SEED
    try:
        return V5.build_items('zh', rows, rows, prior, palette, forms)
    finally:
        V5.TEXT = saved


def halves(rows):
    fit, ev = [], []
    for cat in sorted({r['cat'] for r in rows}):
        g = sorted((r for r in rows if r['cat'] == cat), key=lambda r: hashlib.sha256(f'{SPLIT_SEED}|{r["id"]}'.encode()).hexdigest())
        fit += g[0::2]
        ev += g[1::2]
    return {r['id'] for r in fit}, {r['id'] for r in ev}


def make_pairs(items, tok, forms, excluded):
    """干净 = 资料范围，反事实 = 同一条目的现实范围。"""
    by = {}
    for cell, eid, prompt, cands, lab in items:
        by[(eid,) + cell[1:], cell[0]] = (cell, prompt, lab)
    pairs = []
    for (key, scope), (cell, prompt, lab) in by.items():
        if scope != 'doc' or cell in excluded or ('real',) + cell[1:] in excluded:
            continue
        rcell, rprompt, rlab = by[key, 'real']
        ca, co = tok.encode(prompt, add_special_tokens=False), tok.encode(rprompt, add_special_tokens=False)
        a, b = tok.encode(forms['zh'][lab['K']][0].strip(), add_special_tokens=False)[0], tok.encode(forms['zh'][lab['P']][0].strip(), add_special_tokens=False)[0]
        if len(ca) != len(co) or ca[-1] != co[-1] or a == b:
            raise SystemExit(f'成对材料不等长或答案相同：{key}')
        pairs.append({'frame': '|'.join(map(str, cell[1:])), 'family': key[0], 'cell': list(cell[1:]), 'clean_ids': ca, 'corrupt_ids': co, 'a': a, 'b': b})
    return pairs


def balanced(pairs, n, salt):
    """按格轮流抽取，至多 n 条，确定性。"""
    groups = {}
    for p in sorted(pairs, key=lambda p: hashlib.sha256(f'{salt}|{p["family"]}|{p["frame"]}'.encode()).hexdigest()):
        groups.setdefault(p['frame'], []).append(p)
    out, i = [], 0
    while len(out) < min(n, len(pairs)):
        for g in sorted(groups):
            if i < len(groups[g]) and len(out) < n:
                out.append(groups[g][i])
        i += 1
    return out


class Progress:
    """定时进度日志（AGENTS.md 第 0 节第 9 条）：每 every 秒或结束时写一行，含已完成 / 总数、用时、预计剩余与显存。
    共享显存按“缓存分配器保留量 − 设备实际占用”估计，大于 0 说明溢出到了系统内存。"""

    def __init__(self, label, total, every=120):
        self.label, self.total, self.every = label, total, every
        self.done, self.t0, self.last = 0, time.time(), 0.0
        self.line()

    def vram(self):
        import torch
        free, tot = torch.cuda.mem_get_info()
        used, res = (tot - free) / 2**30, torch.cuda.memory_reserved() / 2**30
        return f'显存 设备占用 {used:.1f}/{tot / 2**30:.1f} GB · 分配器保留 {res:.1f} GB · 峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB · 共享≈{max(0.0, res - used):.1f} GB'

    def line(self):
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done) if self.done else float('nan')
        print(f'[{datetime.now():%H:%M:%S}] {self.label} {self.done}/{self.total} · 已用 {el / 60:.1f} 分 · 预计剩余 '
              f'{eta / 60:.1f} 分 · {self.vram()}', flush=True)
        self.last = time.time()

    def __call__(self, n):
        self.done += n
        if self.done >= self.total or time.time() - self.last >= self.every:
            self.line()


def setup(model_path):
    import torch
    import mech_bench_run as R
    if 'vram_fraction' in CRITERIA:
        torch.cuda.set_per_process_memory_fraction(CRITERIA['vram_fraction'])
    model, tok, dtype = R.load(model_path, 'cuda')
    eng = R.Engine(model, 'cuda')
    torch.set_grad_enabled(True)
    return R, eng, model, tok


def do_check(model_path):
    import torch
    from provenance import write_result
    snap, _ = CF.key_of(model_path)
    palette, forms = CF.palette_forms()
    tok, model = CF.load_model(model_path)
    rows = read_rows('discovery')
    prior = priors(model, tok, rows, palette, forms)
    items = build(rows, prior, palette, forms)
    recs = []
    t0 = time.time()
    for cell, eid, prompt, cands, lab in items:
        sc, _ = V3.score(model, tok, prompt, cands)
        recs.append({'cell': list(cell), 'entity': eid, 'scores': sc, 'choice': max(sc, key=sc.get), **lab})
    rep = V5.analyze_lang(recs, np.random.default_rng(0))
    failed = [tuple(c['cell'][k] for k in ('scope', 'pos', 'k', 'draft', 'form')) for c in rep['cells'] if not c['gate']]
    out = {'id': f'MA-H3-{VERSION}-check', 'split': 'discovery', 'question': VERSION, 'model_snapshot': snap, 'seconds': round(time.time() - t0, 1),
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'failed_cells': failed, 'report': rep}
    CHECK.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('check', write_result(f'MA-H3-{VERSION}-check', out, model_path=model_path, seed=0, tool='scripts/mainline_h3.py'),
          'gate', rep['gate_pass'], 'failed', failed, rep['scope_switch'], 'torch', torch.__version__)


def excluded_cells():
    chk = json.loads(CHECK.read_text(encoding='utf-8'))
    return {tuple(c) for c in chk['failed_cells']} | set(H1_EXCLUDED)


def discovery_pairs(model, tok, palette, forms):
    rows = read_rows('discovery')
    prior = priors(model, tok, rows, palette, forms)
    pairs = make_pairs(build(rows, prior, palette, forms), tok, forms, excluded_cells())
    fit, ev = halves(rows)
    return [p for p in pairs if p['family'] in fit], [p for p in pairs if p['family'] in ev]


def do_discover(model_path):
    R, eng, model, tok = setup(model_path)
    palette, forms = CF.palette_forms()
    fit, ev = discovery_pairs(model, tok, palette, forms)
    sub = balanced(fit, CRITERIA['n_fit'], 'fit')
    t0 = time.time()
    res = R.discover_task(eng, sub, CRITERIA['batch'], 'cuda', progress=Progress('discover', len(sub)))
    ranks = {m: R.rank(res['node'][m]) for m in METHODS}
    np.savez(OUT / 'rankings.npz', **{f'rank_{m}': ranks[m] for m in METHODS}, **{f'score_{m}': res['node'][m] for m in METHODS},
             signed_dla=res['signed_dla'])
    meta = {'n_fit_pairs': len(fit), 'n_used': len(sub), 'n_eval_pairs': len(ev), 'gate': res['gate'], 'seconds': round(time.time() - t0, 1),
            'top10': {m: [node_name(eng, int(i)) for i in ranks[m][:10]] for m in METHODS}}
    (OUT / 'discover.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print('discover', meta['gate'], meta['seconds'], 's', json.dumps(meta['top10'], ensure_ascii=False))


def node_name(eng, i):
    return f'L{i // eng.H}H{i % eng.H}' if i < eng.L * eng.H else f'MLP{i - eng.L * eng.H}'


def run_plan(R, eng, items, plan, label='plan'):
    import torch
    n = len(items)
    prog = Progress(label, n)
    raw = {k: np.zeros(n, dtype=np.float32) for k, _ in plan}
    raw['clean'], raw['corrupt'], raw['empty'] = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    none = torch.zeros(eng.n_nodes, dtype=torch.bool)
    with torch.no_grad():
        for idx in R.batches(items, CRITERIA['batch']):
            clean, corrupt, a, b = R.tensors(items, idx, 'cuda')
            cf = eng.capture(corrupt)
            raw['clean'][idx] = eng.ld(clean, a, b).cpu().numpy()
            raw['corrupt'][idx] = eng.ld(corrupt, a, b).cpu().numpy()
            eng.set_nodes(none, cf)
            raw['empty'][idx] = eng.ld(clean, a, b).cpu().numpy()
            for key, keep in plan:
                eng.set_nodes(keep, cf)
                raw[key][idx] = eng.ld(clean, a, b).cpu().numpy()
            eng.clear()
            prog(len(idx))
    return raw


def f_of(raw, key, idx=None):
    s = slice(None) if idx is None else idx
    return B.faithfulness(raw[key][s], raw['clean'][s], raw['empty'][s])


def boot_ci(raw, fn, clusters, n_boot):
    vals = [fn(ix) for ix in B.cluster_boot_indices(clusters, n_boot, 0)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def do_evalfit(model_path):
    import mech_bench_run as R
    _, eng, model, tok = setup(model_path)
    palette, forms = CF.palette_forms()
    _, ev = discovery_pairs(model, tok, palette, forms)
    items = balanced(ev, CRITERIA['n_evalfit'], 'evalfit')
    z = np.load(OUT / 'rankings.npz')
    n, groups, rng = eng.n_nodes, R.group_of(eng), np.random.default_rng(0)
    ks = dict(zip(BUDGETS, B.budget_counts(n, BUDGETS)))
    plan = []
    for b, k in ks.items():
        for m in METHODS:
            keep = R.keep_from(z[f'rank_{m}'], k, n)
            plan += [(f'keep|{m}|{b}', keep), (f'remove|{m}|{b}', ~keep)]
        for d in range(3):
            plan.append((f'keep|random{d}|{b}', R.keep_from(rng.permutation(n), k, n)))
    t0 = time.time()
    raw = run_plan(R, eng, items, plan, 'evalfit')
    clusters = [p['family'] for p in items]
    res = {'n_items': len(items), 'seconds': round(time.time() - t0, 1), 'gate': B.behavior_gate(raw['clean'], raw['corrupt']), 'methods': {}}
    for m in METHODS + ['random0', 'random1', 'random2']:
        curve = {b: raw[f'keep|{m}|{b}'] for b in BUDGETS}
        fk = [f_of(raw, f'keep|{m}|{b}') for b in BUDGETS]
        entry = {'f_keep': [round(v, 4) for v in fk], 'auc_f': B.auc_f(curve, raw['clean'], raw['empty'])}
        if m in METHODS:
            entry['f_remove'] = [round(f_of(raw, f'remove|{m}|{b}'), 4) for b in BUDGETS]
            entry['auc_ci'] = boot_ci(raw, lambda ix, c=curve: B.auc_f({b: v[ix] for b, v in c.items()}, raw['clean'][ix], raw['empty'][ix]),
                                      clusters, 500)
        res['methods'][m] = entry
    (OUT / 'evalfit.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez(OUT / 'evalfit_raw.npz', **{k.replace('|', '__'): v for k, v in raw.items()})
    print('evalfit', res['gate'], res['seconds'], 's', {m: (round(v['auc_f'], 3), v['f_keep'][3:7]) for m, v in res['methods'].items()})


def do_freeze(model_path):
    if FROZEN.exists():
        raise SystemExit('已冻结：不得重选')
    ev = json.loads((OUT / 'evalfit.json').read_text(encoding='utf-8'))
    ms = ev['methods']
    best = max(METHODS, key=lambda m: (round(ms[m]['auc_f'], 6), ms[m]['auc_ci'][0]))
    fk = ms[best]['f_keep']
    hit = [b for b, v in zip(BUDGETS, fk) if v >= 0.8]
    budget = hit[0] if hit else 0.1
    z = np.load(OUT / 'rankings.npz')
    n = len(z[f'rank_{best}'])
    k = B.budget_counts(n, [budget])[0]
    class _E:      # 只用于命名
        H, L = 32, 36
    carriers = [int(i) for i in z[f'rank_{best}'][:k]]
    doc = {'id': 'MA-H3-frozen', 'created': datetime.now().isoformat(timespec='seconds'), 'method': best, 'budget': budget, 'k': k,
           'reached_0p8': bool(hit), 'f_disc_eval': fk[BUDGETS.index(budget)], 'carriers': carriers,
           'carrier_names': [node_name(_E, i) for i in carriers], 'auc_f': {m: ms[m]['auc_f'] for m in METHODS},
           'prediction_rule': CRITERIA['confirm_predictions'], 'rankings_sha256': hashlib.sha256((OUT / 'rankings.npz').read_bytes()).hexdigest(),
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip()}
    text = json.dumps(doc, ensure_ascii=False, indent=1) + '\n'
    FROZEN.write_text(text, encoding='utf-8', newline='\n')
    FROZEN.with_suffix('.sha256').write_text(hashlib.sha256(text.encode()).hexdigest() + '\n', encoding='utf-8', newline='\n')
    print('frozen', best, budget, k, doc['carrier_names'][:20], 'f_eval', doc['f_disc_eval'])


def do_confirm(model_path):
    import mech_bench_run as R
    from provenance import write_result
    fz = json.loads(FROZEN.read_text(encoding='utf-8'))
    splits = json.loads(H1.SPLITS.read_text(encoding='utf-8'))
    if any(o.get('split') == 'confirm' for o in splits['custody']['opened']):
        raise SystemExit('confirm 已开封过：只评估一次')
    splits['custody']['opened'].append({'split': 'confirm', 'when': datetime.now().isoformat(timespec='seconds'),
                                        'by': 'Claude（用户 2026-11-07 授权“做 H3”）', 'purpose': 'H3 冻结后的一次性评估',
                                        'frozen_before': FROZEN.with_suffix('.sha256').read_text().strip()})
    H1.SPLITS.write_text(json.dumps(splits, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    _, eng, model, tok = setup(model_path)
    palette, forms = CF.palette_forms()
    rows = read_rows('confirm')
    prior = priors(model, tok, rows, palette, forms)
    items = make_pairs(build(rows, prior, palette, forms), tok, forms, excluded_cells())
    n, groups, rng = eng.n_nodes, R.group_of(eng), np.random.default_rng(1)
    z = np.load(OUT / 'rankings.npz')
    C = R.keep_from(np.array(fz['carriers']), fz['k'], n)
    plan = [('keep|C', C), ('remove|C', ~C)]
    for d in range(3):
        plan.append((f'keep|random{d}', R.keep_from(rng.permutation(n), fz['k'], n)))
        plan.append((f'keep|matched{d}', R.matched_keep(z[f'rank_{fz["method"]}'], fz['k'], groups, rng)))
    for m in METHODS:
        plan.append((f'keep|{m}', R.keep_from(z[f'rank_{m}'], fz['k'], n)))
    for b, k in zip(BUDGETS, B.budget_counts(n, BUDGETS)):
        plan.append((f'curve|{b}', R.keep_from(z[f'rank_{fz["method"]}'], k, n)))
    t0 = time.time()
    raw = run_plan(R, eng, items, plan, 'confirm')
    clusters = [p['family'] for p in items]
    nb = CRITERIA['bootstrap']
    fC = f_of(raw, 'keep|C')
    ctrl = {k2: f_of(raw, k2) for k2, _ in plan if k2.startswith('keep|random') or k2.startswith('keep|matched')}
    best_ctrl = max(ctrl, key=ctrl.get)
    margin = fC - ctrl[best_ctrl]
    m_ci = boot_ci(raw, lambda ix: f_of(raw, 'keep|C', ix) - max(f_of(raw, k2, ix) for k2 in ctrl), clusters, nb)
    f_rm = f_of(raw, 'remove|C')
    P1 = fC >= 0.7 and margin >= 0.3 and m_ci[0] > 0
    P2 = f_rm <= 0.5
    P3 = abs(fC - fz['f_disc_eval']) <= 0.15
    resid = {}
    for j, name in enumerate(('pos', 'k', 'draft', 'form')):
        for lv in sorted({p['cell'][j] for p in items}, key=str):
            ix = np.array([p['cell'][j] == lv for p in items])
            resid[f'{name}={lv}'] = {'n': int(ix.sum()), 'f_C': round(f_of(raw, 'keep|C', ix), 4), 'unexplained': round(1 - f_of(raw, 'keep|C', ix), 4)}
    rep = {'id': 'MA-H3-confirm', 'split': 'confirm', 'n_items': len(items), 'n_entities': len(set(clusters)), 'seconds': round(time.time() - t0, 1),
           'frozen_sha256': FROZEN.with_suffix('.sha256').read_text().strip(), 'method': fz['method'], 'budget': fz['budget'], 'k': fz['k'],
           'gate': B.behavior_gate(raw['clean'], raw['corrupt']),
           'P1_sufficiency': {'f_C': fC, 'controls': ctrl, 'best_control': best_ctrl, 'margin': margin, 'margin_ci95': m_ci, 'pass': bool(P1)},
           'P2_necessity': {'f_remove': f_rm, 'pass': bool(P2)},
           'P3_replication': {'f_disc_eval': fz['f_disc_eval'], 'f_confirm': fC, 'pass': bool(P3)},
           'methods_at_k': {m: f_of(raw, f'keep|{m}') for m in METHODS},
           'curve': {str(b): f_of(raw, f'curve|{b}') for b in BUDGETS}, 'residual': resid, 'criteria': CRITERIA}
    (OUT / 'confirm.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print('confirm', write_result('MA-H3-confirm', rep, model_path=model_path, seed=0, tool='scripts/mainline_h3.py'))
    print(json.dumps({k: rep[k] for k in ('gate', 'P1_sufficiency', 'P2_necessity', 'P3_replication', 'methods_at_k')}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'check', 'discover', 'evalfit', 'freeze', 'confirm'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'check': lambda: do_check(a.model), 'discover': lambda: do_discover(a.model),
     'evalfit': lambda: do_evalfit(a.model), 'freeze': lambda: do_freeze(a.model), 'confirm': lambda: do_confirm(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
