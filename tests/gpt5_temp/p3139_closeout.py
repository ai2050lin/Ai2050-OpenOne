# -*- coding: utf-8 -*-
"""Phase 3139 closeout: five-write
(ledger / MEMO / wlog / workspace MEMORY
/ npz already on disk) + idempotent.
v2: no %-formatting inside the MEMO
section literal (placeholders only)."""
import io
import json
import hashlib
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D39 = (RDIR + r'\phase3139\omega_p137_'
       'idinteract_portconsume_rewrite_xmat')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = ROOT + r'\.workbuddy\memory' \
      r'\2026-09-28.md'
WMEM = ROOT + r'\.workbuddy\memory' \
       r'\MEMORY.md'

res_raw = io.open(D39 + r'\result.json',
                  'rb').read()
res_sha8 = hashlib.sha256(res_raw) \
    .hexdigest()[:8]
r39 = json.loads(res_raw.decode('utf-8'))
assert r39['phase'] == 3139
assert r39['smoke'] is False
VERDICT = r39['verdict']
print('res39 sha8=%s verdict ok'
      % res_sha8)

# ---------- 1. ledger (idempotent) ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
have = [e for e in led['measurements']
        if e.get('phase') == 3139]
if not have:
    entry = {
        'phase': 3139,
        'name': r39['name'],
        'date': time.strftime('%Y-%m-%d'),
        'kind': 'state_bank_followup',
        'verdict': VERDICT,
        'runtime_s': r39['runtime_s'],
        'hashes': {
            'result_sha256_8': res_sha8,
            'seal_sha256_8': r39['seal_sha8']},
        'anchors': {
            'res38_sha8': 'f7ef08be',
            'dvec_sha8': {'17': '5e4c3085',
                          '29': 'ee9484b2',
                          '33': '59fbe0d3',
                          '38': 'aced803b'},
            'xphase': r39['part_e']['xphase']},
        'summary': (
            'ish2 0.687-0.757 hard75 fail; '
            'r_retr=0; port B/I/C/own '
            '0.29/0.45/0.07/0.04 -> I '
            'dominant but own-row only '
            '3-9% of I (port-class again); '
            'cinj_active 0.18-0.29 (L26 '
            'peak) vs iinj L17 peak 0.328 '
            '-> opposite layer spectra; '
            'xmat AUC 1.0, retr_same_s '
            'chance')}
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = hashlib.sha256(
        json.dumps(led['measurements'],
                   sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
n = len(led2['measurements'])
e39 = [e for e in led2['measurements']
       if e.get('phase') == 3139]
assert len(e39) == 1
assert n == 276, n
led_sha = led2['ledger_sha256_8']
print('ledger n=%d sha8=%s'
      % (n, led_sha))

# ---------- 2. MEMO (idempotent) ------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3139:' not in memo:
    hhmm = time.strftime('%H:%M')
    sec = """
## Phase 3139: 身份交互分解与端口消费（T4 第22 Phase）[__HHMM__]

**判决**：`__VERDICT__`（正式跑 2700.3s 一次通过，xphase=1.0，bank 8 分片直接复用 3138 免重跑 capture；result sha8=__RESSHA__，ledger n=276 sha8=__LEDSHA__）

### §1 交互分解（Part C）
- ish2（同一行集 T01 vs T23 模板组、全局分解基去 B+组模板均差）：KEY_L P 方向 0.755/0.687/0.706/0.754（L17/29/33/38），全层中位 0.751——hard75 fail / soft85 fail：即使剥离模板公共成分，身份跨模板组稳定性仍仅 ~0.75，非稳定部分 ~25 pct
- i_templ_share：模板方向 C[t]-Cbar 张成 3 维子空间只能解释身份能量的 3.4/11.2/10.0/7.2 pct——ish2 不足不是模板方向泄漏，而是与 4 个 C 方向正交的交互残差（WR 行均差）
- r_retr：交互残差的行级 top-1 检索 = 0（chance 0.0015）——交互残差不携带可检索的行身份

### §2 端口消费测试（Part D）—— 本 Phase 核心发现（×3）
冻结 3135 dvec（sha 锚定 drift=0）在 Gram-Schmidt 正交化 B/C/I 子空间的能量占比（672 行 med）：
- L17/29/33/38：B 0.289/0.306/0.265/0.308 | I 0.385/0.446/0.474/0.458 | C 0.068/0.071/0.050/0.078 | 残差 0.258/0.168/0.199/0.148
- 发现 1（×3）：行为端口的主消费对象是身份子空间（45 pct），远超模板（7 pct）；但 C 份额 5-8 pct 远超 4/4096 约 0.1 pct 随机基线——模板方向与 dvec 有真实重叠
- 发现 2（×3）：own_over_I 仅 0.087/0.043/0.033/0.085——dvec 落在身份空间但不特异于本行身份（本行只占身份投影的 3-9 pct）。行为端口读出的是"身份子空间的整体方向结构"，不是"单行专属指纹坐标"——3109-3136 端口类/功能等价理论的第 4 次独立确证，直接否定"每 token 一条专属坐标指纹"假说

### §3 重写窗口因果（Part E）
- cinj_active：C(T2) 注入（allstep）行为 chg L17/26/29/32 = 0.188/0.289/0.258/0.242（d2.0），d1.0 L29 0.180——模板成分有独立行为通路，且 L26 峰与 3138 检索谷底（L26-32 重写窗口）重合
- iinj（错位身份 roll-1 注入）：L17/26/29/32 = 0.328/0.102/0.109/0.078——L17 峰
- 发现 3（×3）：身份注入与模板注入的层位谱相反——身份到 L17（身份稳定层+行为主端口）最有效、重写窗口层被部分重写；模板到 L26-32 重写层最有效。两类成分走不同的层位通路

### §4 跨材料前哨（Part F）
- 84 个未见 (s,o) 对 × 2 方向：P/A1 方向 AUC L17/29/38 = 1.000/1.000/1.000——方向信息完全泛化到新材料
- retr_same_s 0.060/0.036/0.060 约等于 chance 0.036、retr_pair = 0——点态身份检索不泛化（新材料 distractor 结构不同即失效）：指纹检索绑定表面模板结构，非纯实体身份

### §5 综合与 3140 预注册
三层判决合并：(a) 端口消费层级 I(45 pct) >> C(7 pct)，但 own-row 特异性缺失——"身份子空间"是读出单位而非"行指纹坐标"；(b) 身份/模板两成分行为通路层位谱相反（L17 vs L26-32）；(c) 方向泛化（AUC 1.0）与点态检索不泛化（chance）并存。

3140（Ω-P138）预注册：
1. 层位谱分离正式化：身份注入（iinj）vs 模板注入（cinj）× 层位全谱 L17-38（步长 2，16 层 × 各 2 剂量）——绘制"身份读出层 vs 模板重写层"层位拓扑，检验 L17/L26 双峰结构
2. own 不特异性深挖：dvec 在 I 空间投影的逐行能量衰减曲线（own vs 邻行 ±1/±2 vs 全体 med）+ 与 3136 co36/co50 坐标注入结果对照——量化"端口读方向不读坐标"
3. WR 行为通路：交互残差 WR 主成分方向注入（L26/L29，dose 扫描）——WR 是否有独立行为效应（r_retr=0 但可能有行为读出）
4. 检索失败归因：新材料行改用与 bank 相同 distractor 模板重建 prompt 重测 retr_same_s——分离"实体身份"与"表面结构"贡献

关键数字：port I 0.385/0.446/0.474/0.458，own_over_I 0.087/0.043/0.033/0.085；cinj 0.188/0.289/0.258/0.242；iinj 0.328/0.102/0.109/0.078；ish2 0.755/0.687/0.706/0.754；xmat AUC 1.0。
"""
    sec = (sec
           .replace('__HHMM__', hhmm)
           .replace('__VERDICT__', VERDICT)
           .replace('__RESSHA__', res_sha8)
           .replace('__LEDSHA__', led_sha))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
memo2 = io.open(MEMO, encoding='utf-8').read()
i = memo2.rfind('## Phase 3139:')
assert i > 0
sec2 = memo2[i:]
for frag in ('port_I_dominant', 'cinj_active',
             'own_over_I', res_sha8, led_sha,
             '3140（Ω-P138）预注册'):
    assert frag in sec2, frag
print('memo ok (len=%d)' % len(sec2))

# ---------- 3. wlog (idempotent) ------
wl = io.open(WLOG, encoding='utf-8').read()
if '闭环：正式跑 2700' not in wl:
    line = ("- Phase 3139 (Ω-P137) 闭环：正式跑 2700.3s 一次通过"
            "（bank 复用 3138；SMOKE 四修：锚值须从 result.json 精确"
            "提取勿用舍入数组心算 index、ish2/r_retr 行集构造错误→"
            "全行集 T01/T23 组间比较）；verdict=" + VERDICT
            + "；端口消费 I 45pct/own 3-9pct=端口类第 4 次确证；"
              "cinj_active L26 峰 vs iinj L17 峰层位谱相反；"
              "xmat AUC 1.0/retr chance；closeout 五写（ledger "
              "n=276 sha8=" + led_sha + "）；3140 预注册。\n")
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        f.write(line)
wl2 = io.open(WLOG, encoding='utf-8').read()
assert '闭环：正式跑 2700' in wl2
print('wlog ok')

# ---------- 4. workspace MEMORY -------
mm = io.open(WMEM, encoding='utf-8').read()
changed = False
if '3139（T4）' not in mm:
    old_next = "- max=3138，下一 3139"
    if old_next in mm:
        mm = mm.replace(
            old_next,
            "- max=3139，下一 3140")
    anchor_line = "- 3138（T4）"
    i = mm.find(anchor_line)
    assert i >= 0
    j = mm.find('\n', i)
    new_line = ("- 3139（T4）：ish2 0.687–0.757 fail（非模板交互残差"
                "所致，i_templ 仅 3–11%）；r_retr=0；端口消费 I "
                "0.39–0.47 主导 / own 仅 3–9% = 端口读方向不读行坐标"
                "（端口类第 4 次确证）；cinj_active 0.18–0.29（L26 峰"
                "=重写窗）vs iinj L17 峰 0.328=层位谱相反；xmat AUC "
                "1.0 / retr_same_s chance（方向泛化、点态检索不泛化）。")
    mm = mm[:j] + '\n' + new_line + mm[j:]
    changed = True
if changed:
    with io.open(WMEM, 'w',
                 encoding='utf-8') as f:
        f.write(mm)
mm2 = io.open(WMEM, encoding='utf-8').read()
assert '3139（T4）' in mm2
assert 'max=3139，下一 3140' in mm2
print('wmem ok')

print('CLOSEOUT DONE ledger_n=%d '
      'ledger_sha8=%s res_sha8=%s'
      % (n, led_sha, res_sha8))
