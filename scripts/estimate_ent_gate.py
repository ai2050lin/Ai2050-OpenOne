# -*- coding: utf-8 -*-
"""ENT-XF：用 v1 已保存的强制选择记录估计新答案集下的行为门（只是估计，不是结果）。

    python scripts/estimate_ent_gate.py --behavior ~/.ai2050_agent/tasks/T-*/outputs/behavior.json \
        --spec shared/dist_protocol/grammars/ENT-XF.json [--out x.json]

v1 只保存了最佳可接受答案与最强对照的对数概率，因此只能给区间（Phase 250）：
  通过（下界）：v1 通过，且 v1 的最佳答案仍可接受；新加入、v1 未评分的对照形式（如 金黄色 / golden）可能翻转它。
  可能通过：v1 失败，但 v1 的最强对照已进入新的可接受集（新答案得分 ≥ 原最强对照），次强对照未知。
  失败：v1 失败且最强对照仍在对照集中（新的可接受形式都曾在对照集里，得分不超过它），除非新形式是 v1 未评分的。
  不可估计：提示改变（zh 大小）。
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import grammar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--behavior', nargs='+', required=True)
    ap.add_argument('--spec', default=str(ROOT / 'shared/dist_protocol/grammars/ENT-XF.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    corpus = grammar.generate(spec, include_canary=False)
    new = {}
    for it in corpus['items']:
        if it.get('probe'):
            new[(it['filler_id'], it['cond']['lang'], it['cond']['task'])] = it['probe']
    old = {}
    for pat in args.behavior:
        for p in glob.glob(os.path.expanduser(pat)):
            for b in json.loads(Path(p).read_text(encoding='utf-8'))['items']:
                fr, noun = b['id'].split(':')[1:]
                lang, _, task, _ = fr.split('-')
                old[(noun, lang, task)] = b
    old_contrast_forms = {
        ('zh', 'color'): {'红色', '黄色', '绿色', '蓝色', '白色', '黑色', '棕色', '橙色', '紫色', '粉色', '灰色', '褐色', '粉红色'},
        ('en', 'color'): {' red', ' yellow', ' green', ' blue', ' white', ' black', ' brown', ' orange', ' purple',
                          ' pink', ' gray', ' grey'},
        ('en', 'size'): {' bigger', ' smaller'}}
    status = {}
    for key, pr in new.items():
        noun, lang, task = key
        b = old.get(key)
        if b is None:
            continue
        if (lang, task) == ('zh', 'size'):
            status[key] = 'not_estimable'
            continue
        unscored = (set(pr['answers']) | set(pr['contrast'])) - old_contrast_forms[(lang, task)]
        new_ans_unscored = set(pr['answers']) & unscored
        new_con_unscored = set(pr['contrast']) & unscored
        if b['correct']:
            ok = b['best_answer'] in pr['answers']
            status[key] = ('pass_unless_new_contrast' if new_con_unscored else 'pass') if ok else 'unknown'
        elif b['best_contrast'] in pr['answers']:
            status[key] = 'possible'
        else:
            status[key] = 'fail_unless_new_answer' if new_ans_unscored else 'fail'
    by_lt = defaultdict(Counter)
    for (noun, lang, task), s in status.items():
        by_lt[f'{lang}/{task}'][s] += 1
    nouns = sorted({k[0] for k in status})

    def gate(noun, keys, level):
        ss = [status[(noun, l, t)] for l, t in keys]
        if any(s == 'not_estimable' for s in ss):
            return 'not_estimable'
        lo = all(s in ('pass', 'pass_unless_new_contrast') for s in ss)
        hi = all(s not in ('fail',) for s in ss)
        return 'pass' if lo and level == 'lo' else ('pass' if hi and level == 'hi' else 'fail')

    color_keys = [('zh', 'color'), ('en', 'color')]
    est = {'color_gate_lower': sum(gate(n, color_keys, 'lo') == 'pass' for n in nouns),
           'color_gate_upper': sum(gate(n, color_keys, 'hi') == 'pass' for n in nouns),
           'en_size_lower': sum(status[(n, 'en', 'size')] in ('pass', 'pass_unless_new_contrast') for n in nouns),
           'en_size_upper': sum(status[(n, 'en', 'size')] != 'fail' for n in nouns),
           'n_nouns': len(nouns), 'zh_size': 'not_estimable（提示已改，v1 无对应评分）'}
    res = {'kind': 'estimate', 'note': '由 v1 强制选择记录推出的区间估计，不是 v2 行为门结果',
           'by_lang_task': {k: dict(v) for k, v in sorted(by_lt.items())}, 'nouns': est,
           'per_probe': {'|'.join(k): s for k, s in sorted(status.items())}}
    print(json.dumps({k: res[k] for k in ('by_lang_task', 'nouns')}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
