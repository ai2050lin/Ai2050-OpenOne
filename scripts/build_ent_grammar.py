# -*- coding: utf-8 -*-
"""生成纠缠谱模式族 ENT-XF 的文法（template_grammar.v1）并做材料检查。

    python scripts/build_ent_grammar.py                      # 写 shared/dist_protocol/grammars/ENT-XF.json
    python scripts/build_ent_grammar.py --check-tokenizer models/hf/qwen3-4b   # 另做分词检查（只用 tokenizer，不跑模型）

设计（完全交叉、平衡）：
    名词（词表 shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv，7 类 × 30）
    × 语言 lang ∈ {zh, en} × 极性 polarity ∈ {aff, neg} × 任务线索 task ∈ {color, size, none} × 句式变体 v1/v2
    = 每个名词 24 个句式，全部名词进入全部句式。

句式骨架（任务线索与极性都在名词之前，因此在 target 与 final 读出位置都因果可见）：
    zh v1 「至于{线索}，我真的相信 / 我不相信{X}很特别。」   zh v2 「说到{线索}，我真的觉得 / 我不觉得{X}挺常见。」
    en v1 "As for {cue}, I really do think / I do not think the {E} is special."
    en v2 "Speaking of {cue}, I would truly say / I would not say the {E} is common."
    线索词：颜色 / 大小 / 这个；color / size / that（Qwen3 分词下都是 1 个 token）。
    同一语言 × 变体内，肯定与否定、三种线索的名词前缀 token 数相同（--check-tokenizer 核对）；句末 token 在同一语言内相同。
行为 probe：task=color 的句式问颜色、task=size 的句式问大小（完整答案强制选择，对照集为全部其他颜色 / 另一种大小）；
task=none 的句式没有 probe（无线索对照条件）。逐名词的行为门由 entangle.gate_nouns 执行（退出的名词整体退出，
不按格排除），保证剩余设计仍完全交叉，避免按格排除造成的不平衡虚增交互。

grammar_version 2（Phase 250：看过 v1 行为结果后、v2 运行前的重新校准；不回头套用到 v1 结果）：
    颜色：可接受答案 = 主颜色 + 词表 color_alt（至多 2 个备选色）的全部表面形式；对照集 = 其余调色板颜色的全部表面形式。
        表面形式：黄 = 黄色 / 金黄色（yellow / golden），棕 = 棕色 / 褐色，粉 = 粉色 / 粉红色，灰 = gray / grey；
        transparent（透明）与 amber（琥珀色）只作可接受答案，不进入对照集。分析标签 attrs.color 仍是主颜色。
        提示改为问答式「问：{X}通常是什么颜色的？\\n答：{X}通常是」/「Q: What color is the {E} usually?\\nA: The {E} is
        usually」：v1 提示在被测模型的自由回答中常续写成设问（「红色的，对吗？」）或试题格式（「____ A. Red」）。
        备选色规则：R1 只加入指称对象真实、常见的颜色（品种 / 成熟度 / 常见形态），不加入环境或联想色（海→蓝、植物→绿）
        与图示色；R2 每个名词至多 2 个备选色，中英文相同；R3 对全部名词审阅，不只审 v1 失败项；R4 典型颜色在调色板外时
        允许调色板外的可接受答案。候选颜色来自现实复核与被测模型的自由回答（scripts/calibrate_ent_freeresp.py）；
        自由回答只用于提出候选、修正标注与扩充可接受集，不用于挑选或剔除名词。逐项理由见 lexicon/ENT-XF_nouns.v2.changes.tsv。
    大小：zh 提示改为「……的个头通常」+ 更大 / 更高 对 更小 / 更矮。v1 的「……通常更」+「大」是非规范分词
        （Qwen3 把「更大」合成 1 个 token，「更小」是 更 + 小），v1 中 zh 的 L 名词 116 条全错、间隔 −9 到 −16。
        en 增加 larger / taller（L）与 shorter（S）。
    行为门：per_attribute（预注册）——含颜色属性的因子对只用颜色 probe、含大小属性的只用大小 probe，其余因子对仍要求
        全部 probe 通过（与 v1 的 all_probes 相同）。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import grammar  # noqa: E402

LEXICON = ROOT / 'shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv'
OUT = ROOT / 'shared/dist_protocol/grammars/ENT-XF.json'

COLOR_ZH = {'red': '红', 'yellow': '黄', 'green': '绿', 'blue': '蓝', 'white': '白', 'black': '黑', 'brown': '棕',
            'orange': '橙', 'purple': '紫', 'pink': '粉', 'gray': '灰'}
# 每种颜色的表面形式（可接受答案与对照集都用全部形式；transparent 只作可接受答案，不进入对照集）
COLOR_FORMS_ZH = {'red': ['红色'], 'yellow': ['黄色', '金黄色'], 'green': ['绿色'], 'blue': ['蓝色'], 'white': ['白色'],
                  'black': ['黑色'], 'brown': ['棕色', '褐色'], 'orange': ['橙色'], 'purple': ['紫色'],
                  'pink': ['粉色', '粉红色'], 'gray': ['灰色'], 'transparent': ['透明'], 'amber': ['琥珀色']}
COLOR_FORMS_EN = {c: [f' {c}'] for c in COLOR_ZH}
COLOR_FORMS_EN.update({'yellow': [' yellow', ' golden'], 'gray': [' gray', ' grey'], 'transparent': [' transparent'],
                       'amber': [' amber']})
NON_CONTRAST = {'transparent', 'amber'}      # 调色板外的可接受答案（R4），不进入任何对照集
# 颜色 probe 提示（v2 改为问答式）：v1 的「{X}的颜色通常是」在 Qwen3-4B 的自由回答中 105/210 续写成「X色的，对吗？」式
# 设问（常先说错再纠正），另 33/210 转成「怎样的？」提问；问答式两种语言的设问 / 提问续写均为 0/210（Phase 250）
COLOR_PROMPT_ZH = '问：{X}通常是什么颜色的？\n答：{X}通常是'
COLOR_PROMPT_EN = 'Q: What color is the {E} usually?\nA: The {E} is usually'
N_ANS = 6               # 颜色可接受答案槽位数（主颜色 + 至多 2 个备选色的表面形式；不足时用第一个补齐，生成时去重）
SIZE_FORMS = {('zh', 'L'): ['更大', '更高'], ('zh', 'S'): ['更小', '更矮'],
              ('en', 'L'): [' bigger', ' larger', ' taller'], ('en', 'S'): [' smaller', ' shorter']}
N_SIZE = 3
COLOR_WORDS_ZH = sorted(set(COLOR_ZH.values()) | {'褐', '金', '银', '青', '朱', '翠'})
COLOR_WORDS_EN = sorted(set(COLOR_ZH) | {'grey', 'gold', 'golden', 'silver', 'violet'})

CUE_ZH = {'color': '颜色', 'size': '大小', 'none': '这个'}
CUE_EN = {'color': 'color', 'size': 'size', 'none': 'that'}
TEXTS = {
    ('zh', 'v1', 'aff'): '至于{cue}，我真的相信{X}很特别。',
    ('zh', 'v1', 'neg'): '至于{cue}，我不相信{X}很特别。',
    ('zh', 'v2', 'aff'): '说到{cue}，我真的觉得{X}挺常见。',
    ('zh', 'v2', 'neg'): '说到{cue}，我不觉得{X}挺常见。',
    ('en', 'v1', 'aff'): 'As for {cue}, I really do think the {E} is special.',
    ('en', 'v1', 'neg'): 'As for {cue}, I do not think the {E} is special.',
    ('en', 'v2', 'aff'): 'Speaking of {cue}, I would truly say the {E} is common.',
    ('en', 'v2', 'neg'): 'Speaking of {cue}, I would not say the {E} is common.',
}


def _contrast(forms):
    return sorted({f for c, fs in forms.items() if c not in NON_CONTRAST for f in fs})


PROBES = {
    ('zh', 'color'): {'prompt': COLOR_PROMPT_ZH, 'answer': '{Cz1}', 'answers': [f'{{Cz{k}}}' for k in range(1, N_ANS + 1)],
                      'contrast': _contrast(COLOR_FORMS_ZH)},
    ('en', 'color'): {'prompt': COLOR_PROMPT_EN, 'answer': '{Ce1}',
                      'answers': [f'{{Ce{k}}}' for k in range(1, N_ANS + 1)], 'contrast': _contrast(COLOR_FORMS_EN)},
    ('zh', 'size'): {'prompt': '跟成年人相比，{X}的个头通常', 'answer': '{Sz1}',
                     'answers': [f'{{Sz{k}}}' for k in range(1, N_SIZE + 1)],
                     'contrast': sorted(SIZE_FORMS[('zh', 'L')] + SIZE_FORMS[('zh', 'S')])},
    ('en', 'size'): {'prompt': 'Compared with an adult human, the {E} is usually', 'answer': '{Se1}',
                     'answers': [f'{{Se{k}}}' for k in range(1, N_SIZE + 1)],
                     'contrast': sorted(SIZE_FORMS[('en', 'L')] + SIZE_FORMS[('en', 'S')])},
}
SOURCE = ('作者（Claude，2026-10-16）依常识编写：color = 最典型颜色（11 色调色板），size = 与成年人相比；'
          'v2（Phase 250）按 R1–R4 复核，加入至多 2 个备选可接受颜色；未经外部规范库核对，是否进入分析由逐名词行为门决定')


REVISION_NOTE = (
    'v2（Phase 250，复核 v1 行为门后升版本；句式、因子、划分盐值与预注册阈值不变）：'
    '① zh 大小 probe 原「……个头通常更」+「大 / 小」是非规范分词（Qwen3 中「更大」为 1 个 token，「更小」为 更+小），'
    'v1 中 zh 的 L 名词全错、间隔 −9 到 −16 → 改为「……个头通常」+ 更大 / 更高 对 更小 / 更矮；en 增加 larger / taller 与 shorter。'
    '② 颜色：v1 单一标签对真实多色名词过窄 → 词表 v2 加 color_alt（备选可接受颜色），可接受答案 = 主颜色 + 备选色的全部'
    '表面形式（黄色 / 金黄色、棕色 / 褐色、粉色 / 粉红色、gray / grey、yellow / golden），对照集 = 其余颜色的全部表面形式'
    '（与可接受集不相交）；透明、琥珀色只作可接受答案。颜色提示改为问答式（v1 提示在自由回答中 zh 105/210 续写成'
    '「X色的，对吗？」设问、33/210 转成提问，en 20/210 续写成试题格式；问答式均为 0/210）。'
    '备选色依据现实颜色（规则 R1–R4），并参考被测模型的自由回答'
    '（scripts/calibrate_ent_freeresp.py）复核标注；自由回答只用于修正标注与扩充可接受集，不用于挑选或剔除名词。'
    '逐项理由见 lexicon/ENT-XF_nouns.v2.changes.tsv。分析标签 attrs.color 仍为主颜色。'
    '③ 新预注册：行为门 per_attribute（含颜色的因子对只用颜色 probe，含大小的只用大小 probe，其余因子对仍要求全部 probe），'
    '不套用到 v1（Phase 249）结果。')


def _pad(forms, n):
    assert 0 < len(forms) <= n, forms
    return forms + [forms[0]] * (n - len(forms))


def answer_slots(r):
    """某名词的可接受答案槽位：颜色 = 主颜色 + 备选色的全部表面形式；大小 = 该大小的全部表面形式。"""
    cols = [r['color']] + r['color_alt']
    zh = _pad([f for c in cols for f in COLOR_FORMS_ZH[c]], N_ANS)
    en = _pad([f for c in cols for f in COLOR_FORMS_EN[c]], N_ANS)
    sz = _pad(SIZE_FORMS[('zh', r['size'])], N_SIZE)
    se = _pad(SIZE_FORMS[('en', r['size'])], N_SIZE)
    slots = {}
    for k in range(N_ANS):
        slots[f'Cz{k + 1}'], slots[f'Ce{k + 1}'] = zh[k], en[k]
    for k in range(N_SIZE):
        slots[f'Sz{k + 1}'], slots[f'Se{k + 1}'] = sz[k], se[k]
    return slots


def read_lexicon(path=LEXICON):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        nid, cat, cat_zh, sup, zh, en, color, color_alt, size = line.split('\t')
        rows.append(dict(id=nid, cat=cat, cat_zh=cat_zh, super=sup, zh=zh, en=en, color=color, color_alt=[c for c in color_alt.split(",") if c],
                         size=size))
    return rows


def build_spec(rows):
    frames = []
    for lang in ('zh', 'en'):
        for variant in ('v1', 'v2'):
            for pol in ('aff', 'neg'):
                for task in ('color', 'size', 'none'):
                    cue = (CUE_ZH if lang == 'zh' else CUE_EN)[task]
                    text = TEXTS[(lang, variant, pol)].replace('{cue}', cue)
                    fr = {'id': f'{lang}-{pol}-{task}-{variant}', 'variant': variant,
                          'levels': {'lang': lang, 'polarity': pol, 'task': task},
                          'text': text, 'target': 'X' if lang == 'zh' else 'E'}
                    if task != 'none':
                        fr['probe'] = json.loads(json.dumps(PROBES[(lang, task)]))
                    frames.append(fr)
    fillers = []
    for r in rows:
        fillers.append({
            'id': r['id'], 'levels': {'cat': r['cat']},
            'slots': {'X': r['zh'], 'E': r['en'], **answer_slots(r)},
            'attrs': {'color': r['color'], 'size': 'large' if r['size'] == 'L' else 'small', 'super': r['super'],
                      'taxonomy': ['实体', {'plant': '植物', 'animal': '动物', 'food': '食品'}[r['super']], r['cat_zh']]},
            'src': SOURCE})
    return {
        'schema_version': grammar.SCHEMA_VERSION,
        'family_id': 'ENT-XF',
        'grammar_version': 2,
        'revision_note': REVISION_NOTE,
        'module': 'ENT',
        'name': '纠缠谱 · 名词 × 语言 × 极性 × 任务线索（完全交叉）',
        'description': ('7 类 × 30 个名词（附英文对译、类别层级、典型颜色与大小）× 语言(zh/en) × 极性(肯定/否定) × 任务线索'
                        '(问颜色/问大小/无线索) × 2 个句式变体，每个名词进入全部 24 个句式。线索与极性都在名词之前：target（名词'
                        '最后一个 token）与 final（句末）两个读出位置上四个因子都因果可见；同一语言内句末 token 相同，'
                        '肯定 / 否定与三种线索的名词前缀 token 数相同。用于纠缠谱（entangle.v1）：函数型方差分解、交互谱、'
                        '因子子空间夹角、条件旋转与未见组合干扰，以及逐头 / 逐 MLP 的部件归因。'
                        'v2：颜色可接受答案含至多 2 个有现实依据的备选色，zh 大小 probe 改为规范分词，行为门按属性分开（预注册）。'),
        'factors': ['cat', 'lang', 'polarity', 'task'],
        'concepts': {'polarity=neg': '否定', 'polarity=aff': '肯定'},
        'layers': 'all',
        'readouts': ['target', 'final'],
        'components': {'readouts': ['target', 'final'], 'layers': 'all'},
        'suffstats_budget_bytes': 56 * 1024 * 1024,
        'shards': 4,
        'splits': {'unit': 'filler', 'fractions': dict(grammar.DEFAULT_FRACTIONS), 'salt': 'v1'},
        'prereg': {'min_behavior_acc': 0.8, 'min_probe_n': 3, 'min_cell_n': 8, 'min_lockbox_cos': 0.4},
        'entangle': {
            'attrs': ['color', 'size', 'super'],
            'task_alignment': {'color': 'task=color', 'size': 'task=size'},
            'pairs': [['cat', 'lang'], ['cat', 'polarity'], ['cat', 'task'], ['lang', 'polarity'], ['lang', 'task'],
                      ['polarity', 'task'], ['color', 'task'], ['size', 'task'], ['color', 'lang'], ['noun', 'lang'],
                      ['noun', 'task']],
            'noun_gate': 'per_attribute',
            'gate_probes': {'color': 'task=color', 'size': 'task=size'},
            'note': ('成对因子、线索对齐与行为门策略在冻结前写入（预注册）；判据见 shared/dist_protocol/entangle.py 的 CRITERIA。'
                     'v2 行为门 per_attribute：含 color 的因子对只要求该名词的颜色 probe（zh、en 全部）通过，含 size 的只要求'
                     '大小 probe 通过，其余因子对要求全部 probe 通过（同 v1 的 all_probes）。新预注册，不套用到 v1 结果')},
        'frames': frames,
        'fillers': fillers,
    }


def check_material(spec, rows):
    """与 tokenizer 无关的检查：返回 (errors, notes)。"""
    errors, notes = [], []
    errs = grammar.validate(spec)
    errors += errs
    if errs:
        return errors, notes
    cats = Counter(r['cat'] for r in rows)
    if len(set(cats.values())) != 1:
        errors.append(f'类别不平衡：{dict(cats)}')
    for key in ('zh', 'en', 'id'):
        dup = [k for k, v in Counter(r[key] for r in rows).items() if v > 1]
        if dup:
            errors.append(f'{key} 重复：{dup}')
    for r in rows:
        if any(c in r['zh'] for c in COLOR_WORDS_ZH):
            errors.append(f'名词含颜色字：{r["zh"]}')
        if any(w in r['en'].lower().split() or r['en'].lower().startswith(w) for w in COLOR_WORDS_EN):
            errors.append(f'英文名含颜色词：{r["en"]}')
    # 近重复：一个中文名是另一个的子串（跨划分时构成词汇泄漏）
    subs = [(a['zh'], b['zh']) for a in rows for b in rows if a is not b and a['zh'] in b['zh']]
    if subs:
        notes.append(f'中文名子串关系（同一词素，跨划分时注意）：{subs}')
    corpus = grammar.generate(spec, include_canary=False)
    vis = corpus['visibility']
    for r_, m in vis.items():
        bad = [f for f, ok in m.items() if not ok]
        if bad:
            errors.append(f'读出 {r_} 上因子不可见：{bad}')
    by_noun = defaultdict(set)
    for it in corpus['items']:
        by_noun[it['filler_id']].add(it['frame_id'])
    if len({len(v) for v in by_noun.values()}) != 1 or len(next(iter(by_noun.values()))) != len(spec['frames']):
        errors.append('不是完全交叉：存在名词没有进入全部句式')
    splits = defaultdict(set)
    for it in corpus['items']:
        splits[it['filler_id']].add(it['split'])
    if any(len(s) != 1 for s in splits.values()):
        errors.append('同一名词出现在多个划分（必须按名词整体切分）')
    per = Counter((next(iter(splits[r['id']])), r['cat']) for r in rows)
    notes.append('划分 × 类别名词数：' + ', '.join(f'{k[0]}/{k[1]}={v}' for k, v in sorted(per.items())))
    size = Counter((next(iter(splits[r['id']])), r['size']) for r in rows)
    notes.append('划分 × 大小名词数（L 远少于 S，size 因子的功效受此限制）：'
                 + ', '.join(f'{k[0]}/{k[1]}={v}' for k, v in sorted(size.items())))
    notes.append(f'有备选可接受颜色的名词：{sum(1 for r in rows if r["color_alt"])} / {len(rows)}'
                 f'（其中 2 个备选色：{sum(1 for r in rows if len(r["color_alt"]) == 2)}）')
    for r in rows:
        alts = r['color_alt']
        if r['color'] not in COLOR_FORMS_ZH or any(c not in COLOR_FORMS_ZH for c in alts) or r['color'] in alts \
                or len(set(alts)) != len(alts) or len(alts) > 2 or r['size'] not in ('L', 'S'):
            errors.append(f'颜色 / 大小标签异常：{r["id"]}')
    finals = defaultdict(set)
    for it in corpus['items']:
        finals[it['cond']['lang']].add(it['text'][-1])
    if any(len(v) != 1 for v in finals.values()):
        errors.append(f'同一语言内句末字符不一致：{dict(finals)}')
    for it in corpus['items']:
        p = it.get('probe')
        if p and (not p.get('contrast') or set(p['answers']) & set(p['contrast'])):
            errors.append(f'probe 答案集 / 对照集异常：{it["id"]}')
            break
    notes.append(f'样本 {len(corpus["items"])} 条；句式 {len(spec["frames"])}；名词 {len(rows)}；'
                 f'条件格 {len(set(grammar.cond_key(it["cond"]) for it in corpus["items"]))}')
    return errors, notes


def check_tokenizer(spec, rows, path):
    """分词检查（只加载 tokenizer）：名词前缀 token 数在 语言 × 变体 内是否一致、句末 token、名词末 token 碰撞。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    out = []
    groups = defaultdict(set)
    for fr in spec['frames']:
        slot = '{' + fr['target'] + '}'
        prefix = fr['text'][:fr['text'].index(slot)]
        n = len(tok(prefix, add_special_tokens=False)['input_ids'])
        groups[(fr['levels']['lang'], fr['variant'])].add(n)
        last = tok(fr['text'].replace(slot, rows[0]['zh'] if fr['target'] == 'X' else rows[0]['en']),
                   add_special_tokens=False)['input_ids'][-1]
        groups[('final', fr['levels']['lang'])].add(tok.decode([last]))
    for k, v in sorted(groups.items()):
        flag = '' if len(v) == 1 else '  ← 不一致'
        out.append(f'{k}: {sorted(v)}{flag}')
    for lang, key, frame_text in (('zh', 'zh', '至于这个，我真的相信{}很特别。'), ('en', 'en', 'As for that, I really do think the {} is special.')):
        pre = frame_text.split('{}')[0].rstrip(' ')     # 英文名词带前导空格并入名词 token
        n0 = len(tok(pre, add_special_tokens=False)['input_ids'])
        lasts, lens = defaultdict(list), Counter()
        for r in rows:
            ids = tok(frame_text.format(r[key]), add_special_tokens=False)['input_ids']
            tail = len(tok(frame_text.split('{}')[1], add_special_tokens=False)['input_ids'])
            noun_ids = ids[n0:len(ids) - tail]
            lens[len(noun_ids)] += 1
            if noun_ids:
                lasts[tok.decode([noun_ids[-1]])].append((r['cat'], r[key]))
        cross = {t: v for t, v in lasts.items() if len({c for c, _ in v}) > 1}
        within = {t: len(v) for t, v in lasts.items() if len(v) > 1 and len({c for c, _ in v}) == 1}
        out.append(f'{lang} 名词 token 数分布：{dict(sorted(lens.items()))}')
        out.append(f'{lang} 名词末 token 同类碰撞（类别中心语素，嵌入层基线会反映）：{within}')
        out.append(f'{lang} 名词末 token 跨类碰撞：{ {t: [w for _, w in v] for t, v in cross.items()} }')
    # probe 答案形式：提示与答案分别分词后拼接（runner 的 answer_logprobs），拼接结果应等于整句的规范分词
    for (lang, task), pr in sorted(PROBES.items()):
        name = rows[0]['zh'] if lang == 'zh' else rows[0]['en']
        prompt = pr['prompt'].replace('{X}', name).replace('{E}', name)
        p_ids = tok(prompt, add_special_tokens=False)['input_ids']
        extra = {f for c in NON_CONTRAST for f in (COLOR_FORMS_ZH if lang == 'zh' else COLOR_FORMS_EN)[c]}
        forms = sorted(set(pr['contrast']) | extra if task == 'color' else set(pr['contrast']))
        bad, lens = [], {}
        for f in forms:
            a_ids = tok(f, add_special_tokens=False)['input_ids']
            lens[f] = len(a_ids)
            if tok(prompt + f, add_special_tokens=False)['input_ids'] != p_ids + a_ids:
                bad.append(f)
        out.append(f'probe {lang}/{task} 答案 token 数：{lens}')
        out.append(f'probe {lang}/{task} 非规范拼接（提示 + 答案 ≠ 整句分词）：{bad or "无"}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-tokenizer', default=None)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    rows = read_lexicon()
    spec = build_spec(rows)
    errors, notes = check_material(spec, rows)
    for n in notes:
        print('[note]', n)
    if errors:
        for e in errors:
            print('[error]', e)
        return 1
    Path(args.out).write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'[ok] wrote {args.out}  grammar_sha256={grammar.spec_sha(spec)[:12]}')
    if args.check_tokenizer:
        for line in check_tokenizer(spec, rows, args.check_tokenizer):
            print('[tok]', line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
