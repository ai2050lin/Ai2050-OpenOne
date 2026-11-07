# -*- coding: utf-8 -*-
"""主线 H1（MA-H1）：冻结材料与统计（ResearchPlan v6 第 8 节，切片 A，中文主线）。

依据  Phase 322：MA-H0-v5 中文行为门 32 / 32 通过；用户 2026-11-07 决定以中文为主线进入 H1，英文未过格排除在内部分析之外。
材料  新实体池 252 个（12 类 × 21，作者 Claude 编写，均未出现在既有词表与模板中，名称不含颜色字），只存节点本地：
      <AI2050_NATIVE_DIR>/_mainline/MA-H1/pool.tsv。
划分  按家族（实体）切分：每类内按 sha256(MA-H1|id) 排序后轮流分到 discovery / confirm / lockbox，各 84 个（每类 7 个）。
      干扰实体只从同一划分、同一类别中取，避免跨划分泄漏。另有留出句式（lockbox_frame），只与 lockbox 实体一起用一次。
封存  confirm 与 lockbox 词表写入节点本地 sealed/ 目录，仓库只提交其 sha256、数量与看管记录；discovery 词表进仓库。
统计  功效三栏用 tools/power_analysis.check_phase；主效应、交互、保护、排除与多重比较口径写入预注册。
检查  只用 discovery 做材料检查（与 MA-H0-v5 相同的 32 格行为门），不读取 confirm / lockbox。
用法  python scripts/mainline_h1.py split|prereg|check --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from power_analysis import check_phase  # noqa: E402

OUT = ROOT / 'research/mainline/H1'
LOCAL = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native')) / '_mainline' / 'MA-H1'
POOL = LOCAL / 'pool.tsv'
SEALED = LOCAL / 'sealed'
SPLITS = OUT / 'splits.json'
PREREG = OUT / 'MA-H1.prereg.json'
SEED = 'MA-H1'
SPLIT_NAMES = ('discovery', 'confirm', 'lockbox')

TEXT_MAIN = V5.TEXT['zh']
TEXT_HELDOUT = {   # lockbox_frame 专用：记录、提问与引导句都换写法，只与 lockbox 实体一起评估一次
    'intro': '以下是一份清单，各项按登记顺序列出。', 'sep': '', 'filler': TEXT_MAIN['filler'],
    'draft': '（旁注：有人先前写过“{X}是{D}的”，这条旁注未经核实。）',
    'record1': '编号{n}：{X}，记为{C}。',
    'record2': '编号{n}：{X}记为{C}，{Y}记为{D}。',
    'q': {'doc': '问：依照这份清单，{X}是什么颜色？\n答：依照这份清单，{X}是',
          'real': '问：现实生活中，{X}一般是什么颜色？\n答：现实生活中，{X}一般是'},
}

POWER_TESTS = [
    {'id': 'T1', 'phase': 'H4', 'test': '冻结载体的原生干预：资料范围逐题准确率（干预 vs 原始，McNemar 配对）',
     'delta': 0.15, 'p_d': 0.20, 'split': 'confirm', 'rationale': '只保留在 discovery 上效应 ≥ 0.15 的候选进入确认；翻转率按保守 0.20 估计'},
    {'id': 'T2', 'phase': 'H3–H5', 'test': '两因素条件下的逐题行为差（切片 B 加入第二因素后，与单因素同题配对）',
     'delta': 0.10, 'p_d': 0.10, 'split': 'confirm', 'rationale': 'v6 第 6 节两因素完整条件；AGENTS 第 13 节缺省翻转率'},
    {'id': 'T3', 'phase': 'H5', 'test': '留出句式推广：同一 lockbox 实体在主句式与留出句式下的逐题准确率差',
     'delta': 0.15, 'p_d': 0.20, 'split': 'lockbox', 'rationale': '只有差到 0.15 才算句式依赖；翻转率保守 0.20'},
]

STATS_PLAN = {
    'unit': '家族 = 实体；功效与 bootstrap 都按家族计数，同一家族的 32 格不算独立样本',
    'main_effects': ['范围（资料 / 现实）', '记录形式（单实体 / 双实体）', '目标记录位置（首 / 末）', '记录数 k（2 / 3）', '草稿（无 / 有）'],
    'interaction': 'I_fg = Y11 − Y10 − Y01 + Y00（v6 第 6 节），Y 为同一测量边界下的原生状态、写出量或输出评分；向量形式保留全部原生坐标',
    'protection': '每次干预同时报告：同模型在无关提示（自然文本）上的下一词 KL 与无上下文颜色问答准确率变化；KL 超过 discovery 上同剂量随机方向的 95 分位，记为“副作用过大”',
    'exclusion': ['discovery 材料检查未过 0.8 的格，在 confirm / lockbox 中不进入内部分析，结果单列',
                  '可用颜色不足（排除先验前 3 色后少于 5 种）的实体在构造时跳过并记录',
                  '英文：MA-H0-v5 未过的格（现实 · 双实体 · 末条 · k=3 · 有草稿）排除；英文为辅助语言，不进入主线确认'],
    'multiple_comparison': '确认集内全部预注册检验用 Holm 校正，α = 0.05',
    'selection': '候选层、头、单元与读出方向只在 discovery 上选；confirm 只评估一次；lockbox 与 lockbox_frame 在全部确认结论冻结后一次性开封',
}

CUSTODY = {
    'sealed_dir': '节点本地 <AI2050_NATIVE_DIR>/_mainline/MA-H1/sealed/',
    'rule': 'confirm、lockbox 词表与留出句式结果在相应阶段前不读取；开封要在研究记录里写明开封人、时间、用途与开封前已冻结的结论，并更新本文件',
    'opened': [],
}


def read_pool():
    lines = [l for l in POOL.read_text(encoding='utf-8').splitlines() if l and not l.startswith('#')]
    head = lines[0].split('\t')
    return [dict(zip(head, l.split('\t'))) for l in lines[1:]]


def assign(rows):
    out = {s: [] for s in SPLIT_NAMES}
    for cat in sorted({r['cat'] for r in rows}):
        group = sorted((r for r in rows if r['cat'] == cat), key=lambda r: hashlib.sha256(f'{SEED}|{r["id"]}'.encode()).hexdigest())
        for i, r in enumerate(group):
            out[SPLIT_NAMES[i % 3]].append(r)
    return out


def tsv(rows):
    return 'id\tcat\tzh\ten\n' + ''.join(f"{r['id']}\t{r['cat']}\t{r['zh']}\t{r['en']}\n" for r in rows)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def do_split():
    if SPLITS.exists():
        raise SystemExit('划分已建立并封存：不得重切')
    rows = read_pool()
    parts = assign(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    SEALED.mkdir(parents=True, exist_ok=True)
    texts = {s: tsv(parts[s]) for s in SPLIT_NAMES}
    (OUT / 'discovery.tsv').write_text(texts['discovery'], encoding='utf-8', newline='\n')
    for s in ('confirm', 'lockbox'):
        (SEALED / f'{s}.tsv').write_text(texts[s], encoding='utf-8', newline='\n')
    doc = {'id': 'MA-H1', 'created': datetime.now().isoformat(timespec='seconds'), 'pool_sha256': sha(POOL.read_text(encoding='utf-8')),
           'pool_size': len(rows), 'rule': '每类内按 sha256(MA-H1|id) 排序，依次轮流分到 discovery / confirm / lockbox',
           'splits': {s: {'n': len(parts[s]), 'sha256': sha(texts[s]), 'per_category': {c: sum(r['cat'] == c for r in parts[s]) for c in sorted({r['cat'] for r in rows})},
                          'location': 'research/mainline/H1/discovery.tsv' if s == 'discovery' else f'sealed/{s}.tsv（节点本地）'} for s in SPLIT_NAMES},
           'custody': CUSTODY}
    SPLITS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({s: doc['splits'][s]['n'] for s in SPLIT_NAMES}), 'sealed', SEALED)


def power_table():
    out = []
    splits = json.loads(SPLITS.read_text(encoding='utf-8'))['splits']
    for t in POWER_TESTS:
        r = check_phase(t['delta'], splits[t['split']]['n'], p_d=t['p_d'])
        out.append({**t, 'n_planned': splits[t['split']]['n'], 'n_required': r['n_required'], 'power': r['achieved_power'], 'pass': r['pass']})
    return out


def prereg_doc():
    return {'id': 'MA-H1', 'schema': 'mainline_prereg.v1', 'created': '2026-11-07', 'stage': 'H1',
            'basis': 'Phase 322（MA-H0-v5）；用户 2026-11-07 决定以中文为主线', 'language': '中文（主线）',
            'splits_sha256': sha(SPLITS.read_text(encoding='utf-8')), 'task': '与 MA-H0-v5 相同的长文属性绑定：32 格，候选完整答案强制选择',
            'cells': V5.CELLS, 'frames': {'main': TEXT_MAIN, 'lockbox_frame': TEXT_HELDOUT},
            'colour_rule': 'K：先验最低 5 色中、排除先验前 3 色后按 sha256(MA-H1|id|zh) 选 1；干扰色从“非 K、非先验前 3”的颜色中按 sha256 选，互不相同。'
                           '新实体没有词表颜色字段，只用模型先验排除合理颜色',
            'distractors': '干扰实体只取同一划分、同一类别的实体', 'prior': '先验色 = 无上下文问答“问：X通常是什么颜色的？答：X通常是”的 11 色完整答案对数概率（与 EA-v2 相同）',
            'power': power_table(), 'stats_plan': STATS_PLAN, 'custody': CUSTODY,
            'gate': {'cell_acc': 0.8, 'rule': '材料检查只在 discovery 上做；32 格全部 ≥ 0.8 才把材料视为可用'}}


def do_prereg():
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    digest = sha(text)
    PREREG.with_suffix('.sha256').write_text(digest + '\n', encoding='utf-8', newline='\n')
    print('prereg', digest)
    for t in json.loads(text)['power']:
        print(t['id'], t['split'], 'Δ', t['delta'], 'p_d', t['p_d'], 'n', t['n_planned'], 'need', t['n_required'], 'power', t['power'], t['pass'])


def read_discovery():
    lines = (OUT / 'discovery.tsv').read_text(encoding='utf-8').splitlines()
    head = lines[0].split('\t')
    return [{**dict(zip(head, l.split('\t'))), 'color': '', 'color_alt': []} for l in lines[1:]]


def do_check(model_path):
    import torch
    import transformers
    from provenance import write_result
    snap, _ = CF.key_of(model_path)
    palette, forms = CF.palette_forms()
    rows = read_discovery()
    tok, model = CF.load_model(model_path)
    t0 = time.time()
    prior = {'_scores': {}}
    q = '问：{X}通常是什么颜色的？\n答：{X}通常是'
    for r in rows:
        sc, _ = V3.score(model, tok, q.format(X=r['zh']), {c: forms['zh'][c] for c in palette})
        prior['_scores'][r['id']] = np.array([sc[c] for c in palette])
        prior[r['id']] = palette[int(np.argmax(prior['_scores'][r['id']]))]
    V5.SEED = SEED          # 本次构造用 MA-H1 的种子；V5.build_items 读模块级 SEED
    items = V5.build_items('zh', rows, rows, prior, palette, forms)
    recs = []
    for n, (cell, eid, prompt, cands, labels) in enumerate(items):
        sc, _ = V3.score(model, tok, prompt, cands)
        recs.append({'cell': list(cell), 'entity': eid, 'scores': sc, 'choice': max(sc, key=sc.get), **labels})
        if n % 400 == 0:
            print(f'@@progress {n}/{len(items)} {round(time.time() - t0)}s', flush=True)
    rep = V5.analyze_lang(recs, np.random.default_rng(0))
    out = {'id': 'MA-H1-discovery-check', 'split': 'discovery', 'n_entities': len(rows), 'model_snapshot': snap,
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'seconds': round(time.time() - t0, 1),
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
           'prior_colours': {c: sum(prior[r['id']] == c for r in rows) for c in palette}, 'report': rep}
    (OUT / 'discovery_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'discovery_check_records.json').write_text(json.dumps(recs, ensure_ascii=False), encoding='utf-8')
    path = write_result('MA-H1-discovery-check', out, model_path=model_path, seed=0, tool='scripts/mainline_h1.py')
    print('check', path, 'gate', rep['gate_pass'], 'failed', rep['cells_failed'], rep['scope_switch'])
    print('  baselines', {s: (round(b['model_acc'], 3), b['best'], round(b['heuristics'][b['best']], 3), b['beyond']) for s, b in rep['baselines'].items()})
    for c in rep['cells']:
        if not c['gate']:
            print('  fail', c['cell'], round(c['acc'], 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['split', 'prereg', 'check'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'split': do_split, 'prereg': do_prereg, 'check': lambda: do_check(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
