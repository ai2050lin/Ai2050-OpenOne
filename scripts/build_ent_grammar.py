# -*- coding: utf-8 -*-
"""生成纠缠谱模式族 ENT-XF 的文法（template_grammar.v1）并做材料检查。

    python scripts/build_ent_grammar.py                      # 写 shared/dist_protocol/grammars/ENT-XF.json
    python scripts/build_ent_grammar.py --check-tokenizer models/hf/qwen3-4b   # 另做分词检查（只用 tokenizer，不跑模型）

设计（完全交叉、平衡）：
    名词（词表 shared/dist_protocol/lexicon/ENT-XF_nouns.v1.tsv，7 类 × 30）
    × 语言 lang ∈ {zh, en} × 极性 polarity ∈ {aff, neg} × 任务线索 task ∈ {color, size, none} × 句式变体 v1/v2
    = 每个名词 24 个句式，全部名词进入全部句式。

句式骨架（任务线索与极性都在名词之前，因此在 target 与 final 读出位置都因果可见）：
    zh v1 「至于{线索}，我真的相信 / 我不相信{X}很特别。」   zh v2 「说到{线索}，我真的觉得 / 我不觉得{X}挺常见。」
    en v1 "As for {cue}, I really do think / I do not think the {E} is special."
    en v2 "Speaking of {cue}, I would truly say / I would not say the {E} is common."
    线索词：颜色 / 大小 / 这个；color / size / that（Qwen3 分词下都是 1 个 token）。
    同一语言 × 变体内，肯定与否定、三种线索的名词前缀 token 数相同（--check-tokenizer 核对）；句末 token 在同一语言内相同。
行为 probe：task=color 的句式问颜色、task=size 的句式问大小（完整答案强制选择，对照集为全部其他颜色 / 另一种大小）；
task=none 的句式没有 probe（无线索对照条件）。逐名词的行为门由 entangle.gate_nouns 执行（某名词任一 probe 失败 → 整个名词
的全部句式退出分析），保证剩余设计仍完全交叉，避免按格排除造成的不平衡虚增交互。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import grammar  # noqa: E402

LEXICON = ROOT / 'shared/dist_protocol/lexicon/ENT-XF_nouns.v1.tsv'
OUT = ROOT / 'shared/dist_protocol/grammars/ENT-XF.json'

COLOR_ZH = {'red': '红', 'yellow': '黄', 'green': '绿', 'blue': '蓝', 'white': '白', 'black': '黑', 'brown': '棕',
            'orange': '橙', 'purple': '紫', 'pink': '粉', 'gray': '灰'}
COLOR_ZH_ALT = {'brown': '褐', 'pink': '粉红'}
COLOR_EN_ALT = {'gray': 'grey'}
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
PROBES = {
    ('zh', 'color'): {'prompt': '{X}的颜色通常是', 'answer': '{Cz}色', 'answers': ['{Cz}色', '{Cz_alt}色'],
                      'contrast': sorted({f'{c}色' for c in list(COLOR_ZH.values()) + list(COLOR_ZH_ALT.values())})},
    ('en', 'color'): {'prompt': 'The typical color of the {E} is', 'answer': ' {Ce}', 'answers': [' {Ce}', ' {Ce_alt}'],
                      'contrast': sorted({f' {c}' for c in list(COLOR_ZH) + list(COLOR_EN_ALT.values())})},
    ('zh', 'size'): {'prompt': '跟成年人相比，{X}的个头通常更', 'answer': '{Sz}', 'answers': ['{Sz}'], 'contrast': ['大', '小']},
    ('en', 'size'): {'prompt': 'Compared with an adult human, the {E} is usually', 'answer': ' {Se}', 'answers': [' {Se}'],
                     'contrast': [' bigger', ' smaller']},
}
SOURCE = ('作者（Claude，2026-10-16）依常识编写：color = 最典型颜色（11 色调色板），size = 与成年人相比；'
          '未经外部规范库核对，是否进入分析由逐名词行为门决定')


def read_lexicon(path=LEXICON):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        nid, cat, cat_zh, sup, zh, en, color, size = line.split('\t')
        rows.append(dict(id=nid, cat=cat, cat_zh=cat_zh, super=sup, zh=zh, en=en, color=color, size=size))
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
            'slots': {'X': r['zh'], 'E': r['en'],
                      'Cz': COLOR_ZH[r['color']], 'Cz_alt': COLOR_ZH_ALT.get(r['color'], COLOR_ZH[r['color']]),
                      'Ce': r['color'], 'Ce_alt': COLOR_EN_ALT.get(r['color'], r['color']),
                      'Sz': '大' if r['size'] == 'L' else '小', 'Se': 'bigger' if r['size'] == 'L' else 'smaller'},
            'attrs': {'color': r['color'], 'size': 'large' if r['size'] == 'L' else 'small', 'super': r['super'],
                      'taxonomy': ['实体', {'plant': '植物', 'animal': '动物', 'food': '食品'}[r['super']], r['cat_zh']]},
            'src': SOURCE})
    return {
        'schema_version': grammar.SCHEMA_VERSION,
        'family_id': 'ENT-XF',
        'grammar_version': 1,
        'module': 'ENT',
        'name': '纠缠谱 · 名词 × 语言 × 极性 × 任务线索（完全交叉）',
        'description': ('7 类 × 30 个名词（附英文对译、类别层级、典型颜色与大小）× 语言(zh/en) × 极性(肯定/否定) × 任务线索'
                        '(问颜色/问大小/无线索) × 2 个句式变体，每个名词进入全部 24 个句式。线索与极性都在名词之前：target（名词'
                        '最后一个 token）与 final（句末）两个读出位置上四个因子都因果可见；同一语言内句末 token 相同，'
                        '肯定 / 否定与三种线索的名词前缀 token 数相同。用于纠缠谱（entangle.v1）：函数型方差分解、交互谱、'
                        '因子子空间夹角、条件旋转与未见组合干扰，以及逐头 / 逐 MLP 的部件归因。'),
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
            'noun_gate': 'all_probes',
            'note': '成对因子与线索对齐在冻结前写入（预注册）；判据见 shared/dist_protocol/entangle.py 的 CRITERIA'},
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
