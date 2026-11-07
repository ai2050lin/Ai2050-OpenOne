# -*- coding: utf-8 -*-
"""主线 H0 第三版（MA-H0-v5）：修正 v4（Phase 322）的两处设计缺陷，换一批新实体。

修订  1. 干扰色排除目标实体在现实中也合理的颜色：模型无上下文先验前 3 色，以及词表登记的常见色与备选色。
         v4 现实范围的“绑定错误”里，英文 53 / 89、中文 24 / 79 选的是这类颜色，不能算错。
      2. 现实范围的正确答案就是模型常识色，“取常识色”启发式恒为 1.0，不是有效基线。改为：
         资料范围用全格同一规则的启发式（同 v4）；范围切换用“不看范围”的启发式（总取资料色 / 总取常识色），
         在两种范围合起来的全部格上比较。
      3. 实体取 v3、v4 都没用过的实体（每类 7 个）。
      4. 记录数 k 改为 2 / 3：排除合理颜色后，部分实体只剩 5 种可用颜色，排不下 k = 4 的双实体记录（需 7 种）。
其余（条件、读出、门槛、特征）与 v4 相同。
用法  python scripts/mainline_h0_v5.py prereg|run|analyze --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

import arbitration_v1 as CF  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import mainline_h0 as V3  # noqa: E402
import mainline_h0_v4 as V4  # noqa: E402

OUT = V4.OUT
PREREG = OUT / 'MA-H0-v5.prereg.json'
SEED = 'MA-H0-v5'
PER_CAT = 7
CELLS = list(itertools.product(('doc', 'real'), ('first', 'last'), (2, 3), (0, 1), ('single', 'pair')))
V4.CELLS = CELLS   # V4.analyze_lang 按模块级 CELLS 分格；本进程内改为 v5 的格
TEXT = V4.TEXT
CRITERIA = {**V4.CRITERIA,
            'baseline_rule': '资料范围：每种启发式（最后 / 第一 / 草稿否则最后）在资料范围全部格上用同一规则；'
                             '范围切换：不看范围的启发式（总取资料色 K / 总取常识色 P）在两种范围的全部格上计算。模型须分别高出 beyond_heuristic',
            'plausible_rule': '干扰色不得是目标实体的先验前 3 色或词表登记的常见色、备选色'}
CLAIM_UPDATE = {'stage': 'H0', 'version': 'v5', 'base': 'MA-H0-v4（Phase 322）', 'changes': ['干扰色排除现实中合理的颜色', '基线分资料范围与范围切换两套', '换一批实体', '记录数改为 2 / 3'],
                'features': 'E1、E2 与 v4 相同，在新实体上重新检验'}


def pick_entities():
    used = {r['id'] for r in V3.pick_entities()} | {r['id'] for r in V4.pick_entities()}
    rows = [r for r in V1.lexicon() if r['id'] not in used]
    out = []
    for cat in sorted({r['cat'] for r in rows}):
        group = sorted((r for r in rows if r['cat'] == cat), key=lambda r: hashlib.sha256(f'{SEED}|{r["id"]}'.encode()).hexdigest())
        out += group[:PER_CAT]
    return out


def prereg_doc():
    return {'id': 'MA-H0-v5', 'schema': 'mainline_prereg.v1', 'created': '2026-11-07', 'basis': 'Phase 322（MA-H0-v4）发现的两处设计缺陷',
            'claim_update': CLAIM_UPDATE, 'text': TEXT, 'cells': CELLS, 'filler_per_gap': V4.FILLER_PER_GAP,
            'entities': [r['id'] for r in pick_entities()], 'excluded_entities': 'MA-H0-v3、v4 用过的 98 个实体',
            'colour_rule': 'K：先验最低 5 色中按 sha256(MA-H0-v5|id|lang) 选 1；干扰色从“非 K、非先验前 3、非词表常见色与备选色”的颜色中按 sha256 选，互不相同',
            'pair_rule': V4.prereg_doc()['pair_rule'], 'draft_rule': V4.prereg_doc()['draft_rule'],
            'scoring': V3.prereg_doc()['scoring'], 'criteria': CRITERIA}


def do_prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


h = V3.h


def build_items(lang, ents, all_rows, prior, palette, forms):
    t = TEXT[lang]
    word = lambda c: forms[lang][c][0].strip()
    name = lambda r: r['zh'] if lang == 'zh' else r['en']
    items = []
    for r in ents:
        P = prior[r['id']]
        scores = prior['_scores'][r['id']]
        top3 = {palette[i] for i in np.argsort(scores)[::-1][:3]}
        plausible = top3 | {r['color']} | set(r.get('color_alt') or [])
        low = [palette[i] for i in np.argsort(scores)[:5] if palette[i] not in plausible]
        K = low[h(SEED, r['id'], lang) % len(low)]
        mates = sorted((m for m in all_rows if m['cat'] == r['cat'] and m['id'] != r['id']), key=lambda m: h(SEED, r['id'], m['id']))[:5]   # 最多需要 1 + 2 × 2 = 5 个干扰实体
        used, D = {K} | plausible, []
        for m in mates:
            free = [c for c in palette if c not in used]
            c = free[h(SEED, r['id'], m['id'], lang) % len(free)]
            used.add(c)
            D.append((m, c))
        slot_first = h(SEED, r['id'], 'slot') % 2 == 0
        fills_all = [t['filler'][(h(SEED, r['id'], lang, 'f') + j) % len(t['filler'])] for j in range(V4.FILLER_PER_GAP * 4)]
        for scope, pos, k, draft, form in CELLS:
            if form == 'single':
                others, partner = [[x] for x in D[:k - 1]], None
                target = [(r, K)]
            else:
                partner = D[0]
                target = [(r, K), partner] if slot_first else [partner, (r, K)]
                others = [D[1 + 2 * i:3 + 2 * i] for i in range(k - 1)]
            recs = [target] + others if pos == 'first' else others + [target]
            parts, mentioned = [t['intro']], []
            for j, rec in enumerate(recs):
                if len(rec) == 1:
                    (m, c), = rec
                    parts.append(t['record1'].format(n=j + 1, X=name(m), C=word(c)))
                else:
                    (m1, c1), (m2, c2) = rec
                    parts.append(t['record2'].format(n=j + 1, X=name(m1), C=word(c1), Y=name(m2), D=word(c2)))
                mentioned += [c for _, c in rec]
                parts += fills_all[j * V4.FILLER_PER_GAP:(j + 1) * V4.FILLER_PER_GAP]
            body = t['sep'].join(parts)
            dcol = partner[1] if partner else D[0][1]
            if draft:
                body += '\n' + t['draft'].format(X=name(r), E=name(r), D=word(dcol))
            prompt = body + '\n\n' + t['q'][scope].format(X=name(r))
            dlist = [c for c in mentioned if c != K]
            cands = {c: forms[lang][c] for c in [K, P] + dlist}
            labels = {'correct': K if scope == 'doc' else P, 'K': K, 'P': P, 'D': dlist, 'last': mentioned[-1], 'first': mentioned[0],
                      'prior': P, 'draft': dcol if draft else None, 'partner': partner[1] if partner else None}
            items.append(((scope, pos, k, draft, form), r['id'], prompt, cands, labels))
    return items


def do_run(model_path):
    import torch
    import transformers
    snap, key = CF.key_of(model_path)
    palette, forms = CF.palette_forms()
    T = CF.colour_targets(key)
    rows = V1.lexicon()
    idx = {r['id']: i for i, r in enumerate(rows)}
    ents = pick_entities()
    tok, model = CF.load_model(model_path)
    t0 = time.time()
    result = {'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'langs': {}}
    for lang in ('zh', 'en'):
        prior = {'_scores': {}}
        for r in rows:
            s = T['disc', lang][idx[r['id']]]
            prior[r['id']] = palette[int(np.argmax(s))]
            prior['_scores'][r['id']] = s
        items = build_items(lang, ents, rows, prior, palette, forms)
        recs, lengths = [], []
        for n, (cell, eid, prompt, cands, labels) in enumerate(items):
            sc, plen = V3.score(model, tok, prompt, cands)
            lengths.append(plen)
            recs.append({'cell': list(cell), 'entity': eid, 'scores': sc, 'choice': max(sc, key=sc.get), **labels})
            if n % 200 == 0:
                print(f'@@progress {lang} {n}/{len(items)} {round(time.time() - t0)}s', flush=True)
        result['langs'][lang] = {'records': recs, 'prompt_tokens': [int(np.min(lengths)), int(np.median(lengths)), int(np.max(lengths))],
                                 'example': items[-1][2]}
    result.update(seconds=round(time.time() - t0, 1), torch=torch.__version__, transformers=transformers.__version__,
                  attn_implementation=getattr(model.config, '_attn_implementation', ''))
    (OUT / 'run_v5.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    print('run done', result['seconds'], 's')


def analyze_lang(recs, rng):
    rep = V4.analyze_lang(recs, rng)
    acc = lambda rs: float(np.mean([r['choice'] == r['correct'] for r in rs]))
    blind = {'always_K': float(np.mean([r['K'] == r['correct'] for r in recs])), 'always_P': float(np.mean([r['P'] == r['correct'] for r in recs]))}
    best = max(blind, key=blind.get)
    rep['baselines'] = {'doc': rep['baselines']['doc'],
                        'scope_switch': {'model_acc': acc(recs), 'heuristics': blind, 'best': best,
                                         'beyond': acc(recs) - blind[best] >= CRITERIA['beyond_heuristic']}}
    return rep


def do_analyze(model_path):
    run = json.loads((OUT / 'run_v5.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(0)
    rep = {'id': 'MA-H0-v5', 'criteria': CRITERIA, 'model_snapshot': run['model_snapshot'], 'prereg_sha256': run['prereg_sha256'],
           'seconds': run['seconds'], 'versions': {k: run.get(k) for k in ('torch', 'transformers', 'attn_implementation')}, 'langs': {}}
    for lang, d in run['langs'].items():
        rep['langs'][lang] = {**analyze_lang(d['records'], rng), 'prompt_tokens': d['prompt_tokens'], 'example': d['example']}
    (OUT / 'report_v5.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    from provenance import write_result
    path = write_result('MA-H0-v5', rep, model_path=model_path, seed=0, tool='scripts/mainline_h0_v5.py')
    print('report', OUT / 'report_v5.json', path)
    for lang, d in rep['langs'].items():
        print(lang, 'gate', d['gate_pass'], 'failed', d['cells_failed'], 'acc', round(d['overall_acc'], 3), d['scope_switch'])
        print('  baselines', {s: (round(b['model_acc'], 3), b['best'], round(b['heuristics'][b['best']], 3), b['beyond']) for s, b in d['baselines'].items()})
        print('  features', {k: v if 'verdict' in v else (round(v['value'], 3), [round(x, 3) for x in v['ci95']], v['supported']) for k, v in d['features'].items()})
        for c in d['cells']:
            if not c['gate']:
                print('  fail', c['cell'], round(c['acc'], 3), 'K', round(c['chose_K'], 3), 'bind', round(c['bind_error'], 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
