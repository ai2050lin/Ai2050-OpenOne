# -*- coding: utf-8 -*-
"""Phase 2971 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2971\card_set_extension')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2971_card_set_extension.py')
LEDGER = os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMFILE = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
           r'\.workbuddy\memory\MEMORY.md')


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


e = json.load(io.open(os.path.join(OUTD, 'execution.json'),
                      encoding='utf-8'))
STAMP = e['created'].replace('T', ' ')[:16]

shas = {
    'execution.json': s8(os.path.join(OUTD, 'execution.json')),
    'result.json': s8(os.path.join(OUTD, 'result.json')),
    'primitive_cards.json': s8(os.path.join(OUTD,
                                            'primitive_cards.json')),
    'primitive_cards.md': s8(os.path.join(OUTD,
                                          'primitive_cards.md')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'primitive_cards_extended_chain_34'
assert r['anchors']['a1']['ok'] and r['anchors']['a2']['ok']
assert r['anchors']['a3']['ok'] and r['anchors']['a4']['ok']
assert r['tests']['T1']['ok'] and r['n_cards'] == 34

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2971_card_set_extension'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2971,
    'name': 'card_set_extension',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'primitive_cards.json': shas['primitive_cards.json'],
        'primitive_cards.md': shas['primitive_cards.md'],
        'script': shas['script'],
    },
})
l14 = [l for l in led['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model'][0]
assert meas_id not in l14['connects']
l14['connects'].append(meas_id)
new_sha = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_sha
json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('ledger: %d -> %d, L14 connects %d, sha %s -> %s'
      % (n_before, len(led['measurements']),
         len(l14['connects']), old_sha, new_sha))

sec = u"""
## Phase 2971: 机制链收官卡片扩充——34 卡入册（环24-32 + 四语言表注记） [%(STAMP)s]

**设计**：纯文档 Phase（ZERO forward），复刻 2961 规范：2962-2970 九环（词类/语言机制签名）按 2961 卡 schema（12 字段）扩入卡组，新增 **vocab_note 字段**登记 2887 四语言表结构注记（en/fr/de/es、同 concept 多 L 词、配对口径=峰词先行过滤 13 对）；每个数字 verbatim 可溯源至封存 source result.json 或其 MEMO 节。

**锚 4/4 + T1 过（run3 权威）**：
- **a1**：9 个源 result.json 二进制 sha256-8 与 MEMO 登记"result <h8>"全对账（9/9）。
- **a2**：MEMO 各节标题存在且 verdict 字符串在节内（9/9）。
- **a3**：卡 phase 集合 = {2936..2960} ∪ {2962..2970}（2961 是卡组 Phase 本身不出卡）——恰好 34 卡无缺口无重复。
- **a4**：全部 34 卡 key_numbers 逐串双源命中（源 result 原文或 MEMO 节文本）——预注册前预检（探针先行）避免 all_void，是纪律 10 的设计期应用。
- **T1**：schema 完整（14 字段非空、新卡 n_key_numbers>=3、环24-32 连续）。

**判决：`primitive_cards_extended_chain_34`**（run3 权威）。

**结论**：机制链正式收官为 **34 卡原语卡片集**——前置口径/方向审计（2936/2937）+ 环1-23（2938-2960，null 重编码全层涌现与单点操作化关闭）+ 环24-32（2962-2970，词类在承重带静态量、语言在瞬态峰位时序量、读出坐标层对两者皆盲、载体分布式共享）；阶段一（机制原语完型）+ 阶段二前半（词类/语言签名）闭环。卡片集是跨模型对齐（方案 v2 阶段三）的基线登记物。

**硬伤与勘误（run1→run3，两次 correction）**：① run1 **a3 判据不可达**（要求 2961 自身也有卡——卡组 Phase 不出卡，纪律 10 判据可达性复现）；② run2 Edit 工具幻影编辑再现（a3 修正报成功但未落盘，重跑仍 fail）——改用 Python 补丁 + Grep 复核（既有人账教训第三次复现，Windows 环境纪律）；③ run3 权威，锚 4/4。设计期另拦：预检探针发现 MEMO 负号用 U+2212（-4.0 检索 miss），key_numbers 采用 MEMO 原文字符串。

**产物**：`phase2971/card_set_extension/` execution {a1} / result {b1} / primitive_cards.json {c1} / primitive_cards.md {d1} / script {e1}。

**接续（2972 候选）**：A（主选）语言×词类双因子签名矩阵（n≥60 合并词表预注册：语言 2-4 类 × 词类 F/C，Freedman-Lane 双因子，2963 机器扩维）；B 延迟头群功能身份（top 延迟头消融，2965 机器，检验 359 头群的因果必要性）；C h8/h21 峰位词属性离线检验（2968 npz 离线，零前向）；D 卡片集跨模型对齐开题（方案 v2 阶段三：qwen3-4b 卡组 vs 第二模型同协议复制）。
""" % {'STAMP': STAMP, 'a1': shas['execution.json'],
       'b1': shas['result.json'], 'c1': shas['primitive_cards.json'],
       'd1': shas['primitive_cards.md'], 'e1': shas['script']}
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2971' not in wl:
    entry = (u"\n## Phase 2971（2026-09-20）机制链收官卡片扩充\n"
             u"- 判决 primitive_cards_extended_chain_34（锚 4/4 + T1，run3 权威）：2962-2970 九环按 2961 schema 扩入，34 卡入册，新增 vocab_note（2887 四语言表注记）。\n"
             u"- 两次 correction：a3 判据不可达（2961 不出卡）；Edit 幻影编辑再现→Python 补丁+Grep 复核。\n"
             u"- MEMORY.md 清理整合（3374→2936 字符，含幻影编辑补丁修复）。\n"
             u"- Ledger 110 条 / L14 78 / hash %s。\n" % new_sha)
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
if 'max=2970' in mem:
    mem = mem.replace('max=2970，下一个 **2971**（A 主选：机制链卡片扩充 2962-2970 九环入 2961 卡组，纯文档；B 双因子签名 n≥60；C 延迟头群消融；D h8/h21 峰位词属性）。',
                      'max=2971，下一个 **2972**（A 主选：语言×词类双因子签名矩阵 n≥60 预注册；B 延迟头群功能身份消融；C h8/h21 峰位词属性离线；D 卡片集跨模型对齐开题）。')
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
print('closeout done')
