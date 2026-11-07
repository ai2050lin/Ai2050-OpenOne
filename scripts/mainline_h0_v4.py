# -*- coding: utf-8 -*-
"""主线 H0 第二版（MA-H0-v4）：在 v3（Phase 321）基础上按主张卡的决定表修订。

修订  1. 记录号键退出主线首批：只用实体名键（v3 中记录号键在现实范围大面积失败，原因待单独查）。
      2. 基线改为每种启发式在全部格上用同一规则计算（v3 按格取最大值恒为 1.0，判据无信息）。
      3. 资料范围加难：新增“双实体记录”形式，每条记录写两个实体各自的颜色，目标可在记录的前或后一槽；
         这样“取最后一条 / 第一条”与“取同记录另一个颜色”都会出错，绑定错误才能被看到。
      4. 换一批实体（同词表中 v3 未用的实体，每类 7 个），使 v3 事后看到的现象可以在新实体上检验。
      5. 预注册新特征：英文带草稿把现实范围拉回资料色（v3 事后观察），以及资料范围绑定错误的类型。
条件  范围（资料 / 现实）× 目标记录位置（首条 / 末条）× 记录数 k（2 / 4）× 草稿（无 / 有）× 记录形式（单实体 / 双实体），32 格；中英文分别校准。
读出  与 v3 相同：候选完整答案对数概率（表面形式取最大）强制选择；候选 = K、P、全部干扰色。
用法  python scripts/mainline_h0_v4.py prereg|run|analyze --model models/hf/qwen3-4b
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
import mainline_h0 as V3  # noqa: E402  复用打分、bootstrap 与 v3 的实体选择

OUT = ROOT / 'research/mainline/H0'
PREREG = OUT / 'MA-H0-v4.prereg.json'
SEED = 'MA-H0-v4'
PER_CAT = 7

CRITERIA = {
    'gate_cell_acc': 0.8,
    'gate_rule': '每种语言 32 格全部 ≥ gate_cell_acc 才算行为门通过；未过的格单列',
    'baseline_rule': '每种启发式在同一范围的全部格上用同一规则计算准确率；模型在该范围的准确率须比最好的启发式高至少 beyond_heuristic',
    'beyond_heuristic': 0.2,
    'effect_min': 0.1,
    'bootstrap': 2000,
    'error_type_min_errors': 20,          # 资料范围绑定错误少于此数时，错误类型记为“无法判定”
    'error_type_share': 2 / 3,            # 某一类错误占比达到此值才判定该类型
}

TEXT = {
    'zh': {
        'intro': V3.TEXT['zh']['intro'], 'filler': V3.TEXT['zh']['filler'], 'draft': V3.TEXT['zh']['draft'], 'sep': '',
        'record1': '第{n}条记录：{X}是{C}的。',
        'record2': '第{n}条记录：{X}是{C}的，{Y}是{D}的。',
        'q': {'doc': '问：按这份资料，{X}是什么颜色的？\n答：按这份资料，{X}是',
              'real': '问：在现实世界里，{X}通常是什么颜色的？\n答：在现实世界里，{X}通常是'},
    },
    'en': {
        'intro': V3.TEXT['en']['intro'], 'filler': V3.TEXT['en']['filler'], 'draft': V3.TEXT['en']['draft'], 'sep': ' ',
        'record1': 'Record {n}: the {X} is {C}.',
        'record2': 'Record {n}: the {X} is {C}, and the {Y} is {D}.',
        'q': {'doc': 'Q: According to this document, what color is the {X}?\nA: According to this document, the {X} is',
              'real': 'Q: In the real world, what color is the {X} usually?\nA: In the real world, the {X} is usually'},
    },
}
CELLS = list(itertools.product(('doc', 'real'), ('first', 'last'), (2, 4), (0, 1), ('single', 'pair')))
FILLER_PER_GAP = 3

CLAIM_UPDATE = {
    'stage': 'H0', 'version': 'v4', 'base': 'MA-H0-v3（Phase 321）的主张卡与假设 A1、A2、R1、R2、D1 不变',
    'changes': ['记录号键退出主线首批', '基线改为全格同一规则', '加双实体记录使资料范围离开天花板', '换一批实体', '新增两项预注册特征'],
    'new_features': [
        {'id': 'E1', 'name': '草稿拉回资料色', 'test': '现实范围中，有草稿时选资料色 K 的比例减去无草稿时的比例 ≥ effect_min 且区间不含 0（每种语言分别判定）',
         'decision': '成立则“错误草稿 × 现实范围”作为切片 B 的第一组冲突；不成立则 v3 的英文现象记为未复现'},
        {'id': 'E2', 'name': '绑定错误的类型', 'test': '资料范围的绑定错误中，选“同一记录另一实体的颜色”与选“资料中最后出现的颜色”各占多少',
         'decision': '同记录错误占多数支持“先定位记录、再在记录内取错槽位”（与 A1 相容）；最后颜色占多数支持 A2；错误太少则无法判定'},
    ],
}


def pick_entities():
    used = {r['id'] for r in V3.pick_entities()}
    rows = [r for r in V1.lexicon() if r['id'] not in used]
    out = []
    for cat in sorted({r['cat'] for r in rows}):
        group = sorted((r for r in rows if r['cat'] == cat), key=lambda r: hashlib.sha256(f'{SEED}|{r["id"]}'.encode()).hexdigest())
        out += group[:PER_CAT]
    return out


def prereg_doc():
    return {'id': 'MA-H0-v4', 'schema': 'mainline_prereg.v1', 'created': '2026-11-07',
            'basis': 'Phase 321（MA-H0-v3）的结果与主张卡决定表；ResearchPlan v6 第 8、9 节', 'claim_update': CLAIM_UPDATE,
            'text': TEXT, 'cells': CELLS, 'filler_per_gap': FILLER_PER_GAP, 'entities': [r['id'] for r in pick_entities()],
            'excluded_entities': 'MA-H0-v3 用过的 49 个实体',
            'colour_rule': 'K：先验最低 5 色中按 sha256(MA-H0-v4|id|lang) 选 1；干扰实体取同类别实体，颜色按 sha256 从剩余颜色中选，与 K、P 及其他干扰色都不同',
            'pair_rule': '双实体记录：目标记录中目标在前槽或后槽按 sha256 交替；其他记录各放两个干扰实体',
            'draft_rule': '单实体形式草稿色取第一个干扰色；双实体形式取目标同记录伙伴的颜色',
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
        low = [palette[i] for i in np.argsort(scores)[:5] if palette[i] != P]
        K = low[h(SEED, r['id'], lang) % len(low)]
        mates = sorted((m for m in all_rows if m['cat'] == r['cat'] and m['id'] != r['id']), key=lambda m: h(SEED, r['id'], m['id']))[:7]
        used, D = {K, P}, []
        for m in mates:
            free = [c for c in palette if c not in used]
            c = free[h(SEED, r['id'], m['id'], lang) % len(free)]
            used.add(c)
            D.append((m, c))
        slot_first = h(SEED, r['id'], 'slot') % 2 == 0
        fills_all = [t['filler'][(h(SEED, r['id'], lang, 'f') + j) % len(t['filler'])] for j in range(FILLER_PER_GAP * 4)]
        for scope, pos, k, draft, form in CELLS:
            if form == 'single':
                others = [[x] for x in D[:k - 1]]
                target = [(r, K)]
                partner = None
            else:
                partner = D[0]
                target = [(r, K), partner] if slot_first else [partner, (r, K)]
                others = [D[1 + 2 * i:3 + 2 * i] for i in range(k - 1)]
            recs = [target] + others if pos == 'first' else others + [target]
            parts = [t['intro']]
            mentioned = []
            for j, rec in enumerate(recs):
                if len(rec) == 1:
                    (m, c), = rec
                    parts.append(t['record1'].format(n=j + 1, X=name(m), C=word(c)))
                else:
                    (m1, c1), (m2, c2) = rec
                    parts.append(t['record2'].format(n=j + 1, X=name(m1), C=word(c1), Y=name(m2), D=word(c2)))
                mentioned += [c for _, c in rec]
                parts += fills_all[j * FILLER_PER_GAP:(j + 1) * FILLER_PER_GAP]
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
    (OUT / 'run_v4.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    print('run done', result['seconds'], 's')


def analyze_lang(recs, rng):
    C = CRITERIA
    acc = lambda rs: float(np.mean([r['choice'] == r['correct'] for r in rs])) if rs else float('nan')
    cells = []
    for cell in CELLS:
        rs = [r for r in recs if tuple(r['cell']) == cell]
        a = acc(rs)
        cells.append({'cell': dict(zip(('scope', 'pos', 'k', 'draft', 'form'), cell)), 'n': len(rs), 'acc': a,
                      'bind_error': float(np.mean([r['choice'] in r['D'] for r in rs])), 'chose_P': float(np.mean([r['choice'] == r['P'] for r in rs])),
                      'chose_K': float(np.mean([r['choice'] == r['K'] for r in rs])), 'gate': a >= C['gate_cell_acc']})
    baselines = {}
    for scope in ('doc', 'real'):
        rs = [r for r in recs if r['cell'][0] == scope]
        heur = {b: float(np.mean([r[b] == r['correct'] for r in rs])) for b in ('last', 'first', 'prior')}
        heur['draft_else_last'] = float(np.mean([(r['draft'] or r['last']) == r['correct'] for r in rs]))
        best = max(heur, key=heur.get)
        baselines[scope] = {'model_acc': acc(rs), 'heuristics': heur, 'best': best,
                            'beyond': acc(rs) - heur[best] >= C['beyond_heuristic']}
    doc = [r for r in recs if r['cell'][0] == 'doc']
    real = [r for r in recs if r['cell'][0] == 'real']
    err = lambda rs: float(np.mean([r['choice'] in r['D'] for r in rs])) if rs else 0.0
    recency = lambda rs: err([r for r in rs if r['cell'][1] == 'first']) - err([r for r in rs if r['cell'][1] == 'last'])
    pair_cost = lambda rs: acc([r for r in rs if r['cell'][4] == 'single']) - acc([r for r in rs if r['cell'][4] == 'pair'])
    drafted = lambda rs: (float(np.mean([r['choice'] == r['draft'] for r in rs if r['cell'][3]]))
                          - float(np.mean([r['choice'] == (r['partner'] or r['D'][0]) for r in rs if not r['cell'][3]])))
    pull_k = lambda rs: (float(np.mean([r['choice'] == r['K'] for r in rs if r['cell'][3]]))
                         - float(np.mean([r['choice'] == r['K'] for r in rs if not r['cell'][3]])))
    feats = {}
    for fname, fn, rs in (('recency_A2', recency, doc), ('pair_cost', pair_cost, doc), ('draft_copy_D1', drafted, recs), ('draft_pull_K_E1', pull_k, real)):
        v, ci = fn(rs), V3.boot(fn, rs, C['bootstrap'], rng)
        feats[fname] = {'value': v, 'ci95': ci, 'supported': v >= C['effect_min'] and ci[0] > 0}
    errs = [r for r in doc if r['choice'] in r['D']]
    same = sum(r['partner'] is not None and r['choice'] == r['partner'] for r in errs)
    last = sum(r['choice'] == r['last'] and r['choice'] != r['partner'] for r in errs)
    if len(errs) < C['error_type_min_errors']:
        etype = '无法判定（错误太少）'
    elif same / len(errs) >= C['error_type_share']:
        etype = '同记录槽位错误为主'
    elif last / len(errs) >= C['error_type_share']:
        etype = '最后颜色为主（A2）'
    else:
        etype = '混合'
    feats['error_type_E2'] = {'errors': len(errs), 'same_record': same, 'last_colour_other': last, 'verdict': etype}
    return {'cells': cells, 'gate_pass': all(c['gate'] for c in cells), 'cells_failed': sum(not c['gate'] for c in cells),
            'overall_acc': acc(recs), 'baselines': baselines, 'features': feats,
            'scope_switch': {'doc_acc': acc(doc), 'real_acc': acc(real)}}


def do_analyze(model_path):
    run = json.loads((OUT / 'run_v4.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(0)
    rep = {'id': 'MA-H0-v4', 'criteria': CRITERIA, 'model_snapshot': run['model_snapshot'], 'prereg_sha256': run['prereg_sha256'],
           'seconds': run['seconds'], 'versions': {k: run.get(k) for k in ('torch', 'transformers', 'attn_implementation')}, 'langs': {}}
    for lang, d in run['langs'].items():
        rep['langs'][lang] = {**analyze_lang(d['records'], rng), 'prompt_tokens': d['prompt_tokens'], 'example': d['example']}
    (OUT / 'report_v4.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    from provenance import write_result
    path = write_result('MA-H0-v4', rep, model_path=model_path, seed=0, tool='scripts/mainline_h0_v4.py')
    print('report', OUT / 'report_v4.json', path)
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
