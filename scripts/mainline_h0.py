# -*- coding: utf-8 -*-
"""主线 H0（MA-H0）：长文属性绑定的主张卡、行为门与简单基线（ResearchPlan v6 第 8、9 节，切片 A）。

问题  一份含多条记录的资料里，每条记录给一个实体配一种颜色（与模型常识相反）。提问指定范围（按资料 / 现实世界）
      和选择键（直接说实体名 / 只说第几条记录），并可附一条写错的草稿答案。模型能否按范围和键取对颜色？
材料  实体取 EA-v1 发现集词表（已在 Phase 264–271 暴露，只作 H0 校准，不进入 H1 的确认与锁箱）；每类 7 个，共 49 个。
      目标颜色 K = 该实体先验最低的 5 种颜色中按 sha256 选 1 种；干扰记录取同类别的其他实体，颜色与 K、P 都不同；
      记录之间插入固定的中性句，使资料成为几百 token 的长文。
条件  范围 scope（资料 / 现实）× 目标位置 pos（首条 / 末条）× 记录数 k（2 / 4）× 草稿 draft（无 / 有）× 选择键 key（实体名 / 记录号），
      共 32 格；中英文分别校准。
读出  答案位置之后各候选完整答案的对数概率（表面形式取最大）。候选 = K、P 与全部干扰色 D。强制选择：取最高者。
      正确答案：资料范围为 K，现实范围为 P（模型自己的无上下文首选色）。
判据  CRITERIA（随结果返回，看到结果后不改）。
用法  python scripts/mainline_h0.py prereg|run|analyze --model models/hf/qwen3-4b
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

import arbitration_v1 as CF  # noqa: E402  复用：调色板与表面形式、EA-v2 先验色、模型加载与快照
import entity_atlas_run as V1  # noqa: E402

OUT = ROOT / 'research/mainline/H0'
PREREG = OUT / 'MA-H0-v3.prereg.json'   # v1、v2 冻结后、运行前发现中性句含颜色字（发黄 / yellow；空白页的“白”），作废并升版本
SEED = 'MA-H0'
PER_CAT = 7

CRITERIA = {
    'gate_cell_acc': 0.8,                 # 每格强制选择准确率下限
    'gate_rule': '每种语言 32 格全部 ≥ gate_cell_acc 才算行为门通过；未过的格单列，不进入内部分析',
    'beyond_heuristic': 0.2,              # 模型准确率须比该格最好的单一启发式高出至少这么多，才记为“超出启发式”
    'effect_min': 0.1,                    # 三项行为特征的最小效应（比例差），且按实体 bootstrap 的 95% 区间不含 0
    'bootstrap': 2000,
}

TEXT = {
    'zh': {
        'intro': '下面是一份资料，记录按收到的先后排列。',
        'record': '第{n}条记录：{X}是{C}的。',
        'filler': ['这份资料由档案室统一整理。', '整理时没有改动原始说法。', '各条记录之间没有先后依赖。', '资料里的描述只适用于这份资料。',
                   '记录员每天下午核对一次。', '部分记录来自口头转述。', '资料的纸张已经有些旧了。', '核对时使用的是同一支笔。',
                   '这一页的页码在右下角。', '装订线位于左侧。', '有两条记录的字迹比较潦草。', '资料最后附有一页备注。'],
        'draft': '（之前的草稿答案写的是：{X}是{D}的，尚未核对。）',
        'q': {('doc', 'name'): '问：按这份资料，{X}是什么颜色的？\n答：按这份资料，{X}是',
              ('doc', 'num'): '问：按这份资料，第{n}条记录里的东西是什么颜色的？\n答：按这份资料，它是',
              ('real', 'name'): '问：在现实世界里，{X}通常是什么颜色的？\n答：在现实世界里，{X}通常是',
              ('real', 'num'): '问：第{n}条记录里的东西，在现实世界里通常是什么颜色的？\n答：在现实世界里，它通常是'},
        'sep': '',
    },
    'en': {
        'intro': 'Below is a document; the records are listed in the order they were received.',
        'record': 'Record {n}: the {E} is {C}.',
        'filler': ['The archive office compiled this document.', 'No original wording was changed during compilation.',
                   'The records do not depend on each other.', 'The descriptions apply only to this document.',
                   'The clerk checks the records every afternoon.', 'Some records come from spoken reports.',
                   'The paper has become slightly worn.', 'The same pen was used for checking.',
                   'The page number is in the lower right corner.', 'The binding is on the left side.',
                   'Two records are written in messy handwriting.', 'A page of notes is attached at the end.'],
        'draft': '(An earlier draft answer said: the {E} is {D}; it has not been checked.)',
        'q': {('doc', 'name'): 'Q: According to this document, what color is the {E}?\nA: According to this document, the {E} is',
              ('doc', 'num'): 'Q: According to this document, what color is the thing in record {n}?\nA: According to this document, it is',
              ('real', 'name'): 'Q: In the real world, what color is the {E} usually?\nA: In the real world, the {E} is usually',
              ('real', 'num'): 'Q: In the real world, what color is the thing in record {n} usually?\nA: In the real world, it is usually'},
        'sep': ' ',
    },
}
CELLS = list(itertools.product(('doc', 'real'), ('first', 'last'), (2, 4), (0, 1), ('name', 'num')))
FILLER_PER_GAP = 3

CLAIM = {
    'stage': 'H0',
    'slice': 'A 共同底座：长文属性绑定',
    'claim': '模型按提问指定的范围与选择键，从长文中取出与正确实体绑定的属性；范围决定读资料还是读常识，选择键决定读哪条记录。',
    'model': 'Qwen3-4B bf16（主线首个模型；之后在 Qwen3-14B、GLM4-9B 复验）',
    'factors': {
        '语言': '输入与输出同为中文或同为英文，两种语言分别校准；本阶段不做跨语言输出',
        '风格与体裁': '固定为说明性资料体，本阶段不变',
        '上下文逻辑': '范围（资料 / 现实）、选择键（实体名 / 记录号）、错误草稿（无 / 有）',
        '语法与符号': '记录句式固定为“第 n 条记录：X 是 C 的。”，标点与括号固定；只在选择键上改变指代方式（名词 / “它”）',
    },
    'roles': {'目标实体': 'X', '目标属性': 'K（资料色）', '常识属性': 'P（模型无上下文首选色）', '干扰属性': 'D（同资料中其他实体的颜色）',
              '选择键': '实体名或记录号', '范围词': '按这份资料 / 在现实世界里', '草稿': '写错的旧答案，颜色取某个 D'},
    'readout': '答案前缀末位（“是”/“is”）之后的候选完整答案对数概率；因果可见性：范围、键和全部记录都在读出位置之前',
    'hypotheses': [
        {'id': 'A1', 'name': '按名检索绑定', 'mechanism': '答案位置用实体名作检索键，从资料中找到该实体所在记录并读出其颜色',
         'behavior': '绑定错误少且与目标位置无关；记录号作键需要多一步“号→记录”的对应，准确率下降',
         'later': 'H2–H4：注意力从答案位置读向目标实体名或其颜色 token；修补实体名位置可交换答案'},
        {'id': 'A2', 'name': '位置或近因启发', 'mechanism': '答案位置偏向读取最近或最显著的颜色，不依赖实体与颜色的绑定',
         'behavior': '目标在首条时选干扰色的比例明显高于目标在末条时；选择键类型影响小',
         'later': 'H2–H4：注意力集中在最后一条记录的颜色 token，与实体名无关'},
        {'id': 'R1', 'name': '范围早门控', 'mechanism': '范围词在读取前决定读资料还是读常识，未选中的一路不进入答案位置',
         'behavior': '与 R2 在行为上相同（都答对）；需要逐层读出区分', 'later': 'H2：现实范围下资料色在任何层都不领先'},
        {'id': 'R2', 'name': '晚仲裁', 'mechanism': '两路都被计算，晚层再按范围选择', 'behavior': '与 R1 相同',
         'later': 'H2：现实范围下资料色在中层先领先、晚层被改写'},
        {'id': 'D1', 'name': '草稿被照抄', 'mechanism': '近处的现成答案被直接复制', 'behavior': '有草稿时选草稿色的比例明显升高',
         'later': '把“错误草稿”作为切片 B 的冲突因素'},
    ],
    'decisions': [
        ['行为门大面积未过（多数格 < 0.8）', '先区分是模型做不到还是材料与问法问题；后者改材料升版本（MA-H0-v2），不进入 H1'],
        ['只有记录号键的格未过', '记录号键退出主线首批，H1 只冻结实体名键；记录号键单列为待改问法'],
        ['门通过，且模型准确率不比最好的单一启发式高 0.2', '不能说模型在做绑定；先改材料让启发式失效，再进入 H1'],
        ['门通过，目标首条时的绑定错误明显更多（A2 特征）', 'H2 优先测位置性注意力，A1 降为备择'],
        ['门通过，记录号键明显更难、位置无影响（A1 特征）', 'H2 优先测按名检索的头，A2 降为备择'],
        ['草稿使选草稿色的比例明显升高（D1 特征）', '错误草稿进入切片 B 作为冲突因素；否则记为“可同时满足”'],
        ['现实范围在有资料时仍答常识色', '复现 Phase 271 的范围切换；R1 与 R2 留到 H2 用逐层读出区分'],
    ],
    'baselines': {
        'last': '取资料中最后一条记录的颜色', 'first': '取第一条记录的颜色', 'prior': '取模型无上下文首选色 P',
        'draft': '有草稿时取草稿色（无草稿时不定义）',
    },
    'exposure': '实体词表在 Phase 264–271 已暴露；本阶段只作行为校准，H1 另建新实体与新句式的 discovery / confirm / lockbox',
}


def pick_entities():
    rows = V1.lexicon()
    out = []
    for cat in sorted({r['cat'] for r in rows}):
        group = sorted((r for r in rows if r['cat'] == cat), key=lambda r: hashlib.sha256(f'{SEED}|{r["id"]}'.encode()).hexdigest())
        out += group[:PER_CAT]
    return out


def prereg_doc():
    return {'id': 'MA-H0-v3', 'revision_note': 'v1（sha256 f8f8e319…）、v2（9965f300…）冻结后、任何运行前先后发现中性句含颜色字（“发黄 / yellow”“空白页”的“白”），会泄漏颜色；改为不含颜色字的句子并升版本。v1、v2 均未运行', 'schema': 'mainline_prereg.v1', 'created': '2026-11-07', 'basis': 'ResearchPlan v6 第 8 节 H0、第 9 节切片 A；Phase 271（CF-v2）',
            'claim_card': CLAIM,
            'text': {lang: {**t, 'q': {f'{s}/{k}': v for (s, k), v in t['q'].items()}} for lang, t in TEXT.items()}, 'cells': CELLS, 'filler_per_gap': FILLER_PER_GAP,
            'entities': [r['id'] for r in pick_entities()], 'colour_rule': 'K：先验最低 5 色中按 sha256(MA-H0|id|lang) 选 1；D：同类实体，颜色按 sha256 从剩余颜色中选，与 K、P、其他 D 不同',
            'scoring': '候选完整答案对数概率，表面形式取最大；英文形式带前导空格，中文直接接在“是”之后',
            'criteria': CRITERIA}


def do_prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def h(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest(), 16)


def build_items(lang, ents, all_rows, prior, palette, forms):
    """返回 [(cell, entity_id, prompt, candidates{name: [forms]}, labels)]。"""
    t = TEXT[lang]
    word = lambda c: forms[lang][c][0].strip()
    name = lambda r: r['zh'] if lang == 'zh' else r['en']
    items = []
    for r in ents:
        P = prior[r['id']]
        scores = prior['_scores'][r['id']]
        low = [palette[i] for i in np.argsort(scores)[:5] if palette[i] != P]
        K = low[h(SEED, r['id'], lang) % len(low)]
        mates = sorted((m for m in all_rows if m['cat'] == r['cat'] and m['id'] != r['id']), key=lambda m: h(SEED, r['id'], m['id']))[:3]
        used, Ds = {K, P}, []
        for m in mates:
            free = [c for c in palette if c not in used]
            c = free[h(SEED, r['id'], m['id'], lang) % len(free)]
            used.add(c)
            Ds.append((m, c))
        for scope, pos, k, draft, key in CELLS:
            others = Ds[:k - 1]
            recs = [(r, K)] + others if pos == 'first' else others + [(r, K)]
            n_target = 1 if pos == 'first' else k
            fills = [t['filler'][(h(SEED, r['id'], lang, 'f') + j) % len(t['filler'])] for j in range(FILLER_PER_GAP * k)]
            parts = [t['intro']]
            for j, (m, c) in enumerate(recs):
                parts.append(t['record'].format(n=j + 1, X=name(m), E=name(m), C=word(c)))
                parts += fills[j * FILLER_PER_GAP:(j + 1) * FILLER_PER_GAP]
            body = t['sep'].join(parts)
            dcol = others[0][1]
            if draft:
                body += '\n' + t['draft'].format(X=name(r), E=name(r), D=word(dcol))
            q = t['q'][(scope, key)].format(X=name(r), E=name(r), n=n_target)
            prompt = body + '\n\n' + q
            cands = {c: forms[lang][c] for c in [K, P] + [c for _, c in others]}
            labels = {'correct': K if scope == 'doc' else P, 'K': K, 'P': P, 'D': [c for _, c in others],
                      'last': recs[-1][1], 'first': recs[0][1], 'prior': P, 'draft': dcol if draft else None}
            items.append(((scope, pos, k, draft, key), r['id'], prompt, cands, labels))
    return items


def score(model, tok, prompt, cands):
    import torch
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    seqs, owner = [], []
    for c, fs in cands.items():
        for f in fs:
            seqs.append(tok(prompt + f, add_special_tokens=False).input_ids)
            owner.append(c)
    L = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, device='cuda')
    att = torch.zeros((len(seqs), L), dtype=torch.long, device='cuda')
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.as_tensor(s)
        att[i, :len(s)] = 1
    lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
    out = {}
    for i, s in enumerate(seqs):
        start = len(p_ids)
        if s[:start] != p_ids:                     # 分词跨越边界时退回公共前缀
            start = next(j for j in range(min(len(s), len(p_ids))) if s[j] != p_ids[j])
        v = float(sum(lp[i, j - 1, s[j]] for j in range(start, len(s))))
        out[owner[i]] = max(out.get(owner[i], -1e9), v)
    return out, len(p_ids)


def do_run(model_path):
    import torch
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
            sc, plen = score(model, tok, prompt, cands)
            lengths.append(plen)
            recs.append({'cell': list(cell), 'entity': eid, 'scores': sc, 'choice': max(sc, key=sc.get), **labels})
            if n % 200 == 0:
                print(f'@@progress {lang} {n}/{len(items)} {round(time.time() - t0)}s', flush=True)
        result['langs'][lang] = {'records': recs, 'prompt_tokens': [int(np.min(lengths)), int(np.median(lengths)), int(np.max(lengths))],
                                 'example': items[-1][2]}
    result['seconds'] = round(time.time() - t0, 1)
    result['torch'] = torch.__version__
    import transformers
    result['transformers'] = transformers.__version__
    result['attn_implementation'] = getattr(model.config, '_attn_implementation', '')
    (OUT / 'run.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    print('run done', result['seconds'], 's')


def boot(fn, recs, n, rng):
    ents = sorted({r['entity'] for r in recs})
    by = {e: [r for r in recs if r['entity'] == e] for e in ents}
    vals = []
    for _ in range(n):
        pick = rng.choice(len(ents), len(ents))
        vals.append(fn([r for i in pick for r in by[ents[i]]]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def analyze_lang(recs, rng):
    C = CRITERIA
    acc = lambda rs: float(np.mean([r['choice'] == r['correct'] for r in rs])) if rs else float('nan')
    cells = []
    for cell in CELLS:
        rs = [r for r in recs if tuple(r['cell']) == cell]
        heur = {b: float(np.mean([r[b] == r['correct'] for r in rs])) for b in ('last', 'first', 'prior')}
        if cell[3]:
            heur['draft'] = float(np.mean([r['draft'] == r['correct'] for r in rs]))
        best = max(heur.values())
        a = acc(rs)
        cells.append({'cell': dict(zip(('scope', 'pos', 'k', 'draft', 'key'), cell)), 'n': len(rs), 'acc': a,
                      'bind_error': float(np.mean([r['choice'] in r['D'] for r in rs])), 'chose_P': float(np.mean([r['choice'] == r['P'] for r in rs])),
                      'chose_K': float(np.mean([r['choice'] == r['K'] for r in rs])), 'heuristics': heur, 'best_heuristic': best,
                      'gate': a >= C['gate_cell_acc'], 'beyond_heuristic': a - best >= C['beyond_heuristic']})
    doc = [r for r in recs if r['cell'][0] == 'doc']
    err = lambda rs: float(np.mean([r['choice'] in r['D'] for r in rs])) if rs else 0.0
    recency = lambda rs: err([r for r in rs if r['cell'][1] == 'first']) - err([r for r in rs if r['cell'][1] == 'last'])
    keycost = lambda rs: acc([r for r in rs if r['cell'][4] == 'name']) - acc([r for r in rs if r['cell'][4] == 'num'])
    drafted = lambda rs: (float(np.mean([r['choice'] == r['draft'] for r in rs if r['cell'][3]]))
                          - float(np.mean([r['choice'] == r['D'][0] for r in rs if not r['cell'][3]])))
    feats = {}
    for name, fn, rs in (('recency_A2', recency, doc), ('key_cost_A1', keycost, doc), ('draft_copy_D1', drafted, recs)):
        ci = boot(fn, rs, C['bootstrap'], rng)
        v = fn(rs)
        feats[name] = {'value': v, 'ci95': ci, 'supported': v >= C['effect_min'] and ci[0] > 0}
    return {'cells': cells, 'gate_pass': all(c['gate'] for c in cells), 'cells_failed': sum(not c['gate'] for c in cells),
            'overall_acc': acc(recs), 'features': feats,
            'scope_switch': {'doc_acc': acc(doc), 'real_acc': acc([r for r in recs if r['cell'][0] == 'real'])}}


def do_analyze(model_path):
    run = json.loads((OUT / 'run.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(0)
    rep = {'id': 'MA-H0', 'criteria': CRITERIA, 'model_snapshot': run['model_snapshot'], 'prereg_sha256': run['prereg_sha256'],
           'seconds': run['seconds'], 'versions': {k: run.get(k) for k in ('torch', 'transformers', 'attn_implementation')}, 'langs': {}}
    for lang, d in run['langs'].items():
        rep['langs'][lang] = {**analyze_lang(d['records'], rng), 'prompt_tokens': d['prompt_tokens'], 'example': d['example']}
    (OUT / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    from provenance import write_result
    path = write_result('MA-H0', rep, model_path=model_path, seed=0, tool='scripts/mainline_h0.py')
    print('report', OUT / 'report.json', path)
    for lang, d in rep['langs'].items():
        print(lang, 'gate', d['gate_pass'], 'failed', d['cells_failed'], 'acc', round(d['overall_acc'], 3), d['scope_switch'],
              {k: (round(v['value'], 3), [round(x, 3) for x in v['ci95']], v['supported']) for k, v in d['features'].items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': lambda: do_prereg(), 'run': lambda: do_run(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
