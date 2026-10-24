# -*- coding: utf-8 -*-
"""EXPL-v2 准备（research/interventions/EXPL-v2.prereg.json）：B 组解释卡片、测试者输入、检索池与标注者输入。

阶段：
  cards     B 组普查卡片（正极 top-20，左侧上下文 60 token），4 批
  tester    测试者输入（只含匿名编号与解释）；--arm A 用 v1 解释，--arm B 用 explain_B_*.json
  pools     两组每个单元 30 个检索位置（普查正极第 21–32 名 + 18 个随机位置，种子 20261026），真值信息只写进 pools.json
  labeler   标注者输入（只含解释与打乱后的 30 段位置文本），--arm A|B
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import neuron_census_service as S  # noqa: E402

MK, RUN = '6be1034f0b55a6df-bf16', '2ca5f0a756d36c8a-20000000'
UNITS = ROOT / 'research/interventions/EXPL-v2.units.json'
V1_EXPL = ROOT / 'research/interventions/EXPL-v1'
LEFT, RIGHT = 60, 10
BATCH_SIZES = (17, 16, 16, 16)


def ctx_text(tok, stream, off, left=LEFT, right=RIGHT):
    c = S.context(tok, stream, off)               # 命中段（处理被切开的多字节字符）
    a = max(0, off - left)
    while a < off and S._decode(tok, stream[a:a + 1]).startswith('�'):
        a += 1
    hs = off
    while hs > a and not c['hit'].startswith(S._decode(tok, stream[hs:off + 1])) and False:
        hs -= 1
    left_txt = S._decode(tok, stream[a:off])
    return (left_txt + '【' + c['hit'] + '】' + S._decode(tok, stream[off + 1:off + 1 + right])).replace('\n', '⏎')


def batches(items):
    out, s = [], 0
    for n in BATCH_SIZES:
        out.append(items[s:s + n])
        s += n
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['cards', 'tester', 'pools', 'labeler'])
    ap.add_argument('--arm', choices=['A', 'B'])
    ap.add_argument('--work', required=True)
    args = ap.parse_args(argv)
    W = Path(args.work)
    W.mkdir(parents=True, exist_ok=True)
    U = json.loads(UNITS.read_text(encoding='utf-8'))
    arms = {'A': sorted(U['arm_A']['units'], key=lambda u: u['code']), 'B': sorted(U['arm_B']['units'], key=lambda u: u['code'])}
    man = S.census_manifest(MK, RUN)
    cman, streams = S.corpus(man['corpus_fp'])
    tok = S.tokenizer(man['model_path'])

    def explanations(arm):
        files = sorted(glob.glob(str(V1_EXPL / 'explain_*.json'))) if arm == 'A' else sorted(glob.glob(str(W / 'explain_B_*.json')))
        return {it['code']: it for f in files for it in json.loads(Path(f).read_text(encoding='utf-8'))['items']}

    if args.stage == 'cards':
        for k, part in enumerate(batches(arms['B'])):
            lines = ['# 神经元普查卡片（每个单元只看本单元的信息）', '',
                     '说明：每个单元是 Qwen3-4B 某一层 MLP 中的一个中间单元。下面列出它在约 1900 万 token 中英文网页/百科语料上激活最强的 20 个位置。',
                     '每条片段格式为「相对强度｜左侧上下文【命中 token】右侧上下文」，相对强度 = 该位置激活 / 该单元在整个语料中的最大激活。',
                     '「直接读出」是该单元输出方向在词表上最推高 / 压低的 token（只是直接通路近似）。', '']
            for u in part:
                L = S.census_layer(MK, RUN, u['block'])
                j = u['unit']
                mx = float(L['amax'][j])
                card = S.unit_card(model_key=MK, block=u['block'], unit=j, run=RUN, n_poles=1)['census']
                fr = card['fire_rate'].get('gt_10pct_max', {}).get('per_page', '—') if card['fire_rate'] else '—'
                lines += [f"## {u['code']}（第 {u['block']} 层，共 36 层）",
                          f"- 强激活频率（>10% 最大值）：{fr}；语料最大激活 {mx:.3g}",
                          f"- 直接读出推高：{' | '.join(repr(t)[1:-1] for t in card['readout']['promotes'])}",
                          f"- 直接读出压低：{' | '.join(repr(t)[1:-1] for t in card['readout']['suppresses'])}", '- 正极片段：']
                for i in range(20):
                    g = int(L['top_idx'][j, i])
                    src, off = g >> S.SRC_SHIFT, g & ((1 << S.SRC_SHIFT) - 1)
                    lines.append(f"  - {float(L['top_val'][j, i]) / mx:.2f}｜{ctx_text(tok, streams[src], off)}")
                lines.append('')
            (W / f'cards_B_{k}.md').write_text('\n'.join(lines), encoding='utf-8')
        print('cards ok')

    if args.stage == 'tester':
        ex = explanations(args.arm)
        items = [ex[u['code']] for u in arms[args.arm] if u['code'] in ex and ex[u['code']]['confidence'] != '无法解释']
        n = len(items)
        cuts = np.linspace(0, n, 5).round().astype(int)
        for k in range(4):
            lines = ['# 待测试的神经元解释', '', '每条是一个神经元（匿名编号）的职能解释。请只根据解释文字写测试文本。', '']
            lines += [f"- {it['code']}：{it['explanation']}" for it in items[cuts[k]:cuts[k + 1]]]
            (W / f'tester_input_{args.arm}_{k}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('tester', args.arm, n)

    if args.stage == 'pools':
        rng = np.random.default_rng(20261026)
        sep = int(cman.get('separator_id', 151643))
        pools = {}
        for arm in ('A', 'B'):
            for u in arms[arm]:
                L = S.census_layer(MK, RUN, u['block'])
                j = u['unit']
                pos = [('pos', int(L['top_idx'][j, i])) for i in range(20, 32) if int(L['top_idx'][j, i]) >= 0]
                rnd = []
                while len(rnd) < 18:
                    src = int(rng.integers(len(streams)))
                    off = int(rng.integers(16, len(streams[src])))
                    if off % man['seq_len'] < man['skip_head'] or int(streams[src][off]) == sep:
                        continue
                    rnd.append(('rand', (src << S.SRC_SHIFT) | off))
                entries = pos + rnd
                order = rng.permutation(len(entries))
                rows = []
                for slot, oi in enumerate(order):
                    kind, g = entries[oi]
                    src, off = g >> S.SRC_SHIFT, g & ((1 << S.SRC_SHIFT) - 1)
                    rows.append({'slot': slot + 1, 'kind': kind, 'gidx': g, 'source': cman['sources'][src]['source'],
                                 'text': ctx_text(tok, streams[src], off)})
                pools[u['code']] = {'arm': arm, 'block': u['block'], 'unit': j, 'census_max': float(L['amax'][j]), 'rows': rows}
        data = json.dumps(pools, ensure_ascii=False, indent=1)
        (W / 'pools.json').write_text(data, encoding='utf-8')
        print('pools', len(pools), hashlib.sha256(data.encode()).hexdigest()[:16])

    if args.stage == 'labeler':
        ex = explanations(args.arm)
        pools = json.loads((W / 'pools.json').read_text(encoding='utf-8'))
        codes = [u['code'] for u in arms[args.arm] if u['code'] in ex and ex[u['code']]['confidence'] != '无法解释']
        nb = (len(codes) + 19) // 20
        cuts = np.linspace(0, len(codes), nb + 1).round().astype(int)
        for k in range(nb):
            lines = ['# 检索标注', '', '每个神经元给出一条职能解释，以及 30 段语料片段，每段有一个用【】标出的 token。',
                     '请只根据解释判断：该神经元会不会在【】处强烈激活。对每段回答「是」或「否」。', '']
            for c in codes[cuts[k]:cuts[k + 1]]:
                lines += [f"## {c}", f"解释：{ex[c]['explanation']}", '']
                lines += [f"{r['slot']}. {r['text']}" for r in pools[c]['rows']]
                lines.append('')
            (W / f'labeler_input_{args.arm}_{k}.md').write_text('\n'.join(lines), encoding='utf-8')
        print('labeler', args.arm, len(codes), nb)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
