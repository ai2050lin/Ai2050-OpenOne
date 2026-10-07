# -*- coding: utf-8 -*-
"""补丁 5：
 gen_memo_phase20.py
  (m1) §5 配对表 `q(k,'xhalf')` 取到 dict ⇒ 改取 `max_abs_dxh`。
  (m2) §7 第 3 条引用 `XH_12_by_site`（Phase 20 内**不存在**该量）⇒ 未验证的结论句，删去并换成真实第三源
       （独立重实现 disk_verify）+ 把探针↔生产写成 Panel [B] 的可检验形式。
  (m3) §6 确认集 `com_B_conf` 可能为 None ⇒ 加守卫。
  (m4) §12 `V['A0_nf4'] and EX[...]['nb']` 取巧写法 ⇒ 直取臂序首臂 nb。
 gen_present_phase20.py
  (p1) §F「逐位」容差 5e-7 对 GPU 非确定性过严 ⇒ 放宽到 1e-6 并改标签/文案。
"""
import io

G = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\gen_memo_phase20.py'
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\gen_present_phase20.py'
n = 0


def rep(path, pairs):
    global n
    s = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        c = s.count(old)
        assert c == 1, '%s :: count=%d :: %r' % (path, c, old[:70])
        s = s.replace(old, new); n += 1
    io.open(path, 'w', encoding='utf-8', newline='\n').write(s)
    return s


s = rep(G, [
    # (m1)
    ("      + q(k, 'xhalf') + ' | [' + f(p['J']['ratio_min'], 3) + ', ' + f(p['J']['ratio_max'], 3) + '] |')",
     "      + f(p['xhalf']['max_abs_dxh'], 4) + ' | [' + f(p['J']['ratio_min'], 3) + ', ' + f(p['J']['ratio_max'], 3) + '] |')"),
    # (m2)
    ("A('1. **探针 ↔ 生产**：A0 的 bf16 腿由 **两个独立进程**（`PROBE=1` 与生产）在同一实现下产出 —— '\n"
     "  '两者共用 `执行码`，生产再跑一次以确认可复现（探针结果见 `_probe20_A0_*.json`）。')\n"
     "A('2. **跨 Phase 三锚**：校准臂现场读入 P18/P16/P17 result 并**逐位断言**（上方 `ANCHOR_ALL_OK`）。')\n"
     "A('3. **P16 ↔ P12**：A0 的 bf16 `xhalf` 另有 **Phase 12 冻结值**（`XH_12_by_site`，18 位点）可作第三方比对；')\n"
     "A('   本 Phase 的 A0·bf16 自算值与之只差 **' + q(PID, 'xhalf') + '**（legacy 子域内）⇒ 两个独立 bf16 实现互证。')",
     "A('1. **探针 ↔ 生产（同实现、两条独立进程加载）**：A0 的两个口径各跑一遍 `PROBE=1`（**全量配对与实例**、'\n"
     "  '仅缩 α 网格）与生产（全尺度）。Panel [B] 的输入（`U_ℓ` / 配对集 / 实例）与 α 网格无关 ⇒ '\n"
     "  '`com_B` 族 / `comlayer_B_all` / `share_mlp_beh_nb` / `com_V` 在两遍之间应在容差 1e-6 内一致'\n"
     "  '（探针读数见 `_probe20_A0_{nf4,bf16}.json`，逐项比对见 `present_phase20.html` §F）。')\n"
     "A('2. **跨 Phase 三锚**：校准臂现场读入 P18/P16/P17 result 并**逐位断言**（上方 `Q2_joint` 与 §3 的 got/expected 表）。')\n"
     "A('3. **独立重实现**：`disk_verify_phase20.py` 用**独立代码**重算 `J(ℓ)` / `com_layer` / 置换零假设 / `xhalf`'\n"
     "  '并逐项比对本 result，判据走「重算 → 按主脚本文义导出标签 → 比对」（不写死预期），要求 **0 FAIL**。')"),
    # (m3)
    ("        A('%-8s ' % a + '  '.join(k + ': ' + f(abs(E(a, 'com_B_conf')[k] - E(a, 'com_B')[k]), 4)\n"
     "                                  for k in ['INC_ALL', 'INC_MLP', 'INC_ATTN']))",
     "        A('%-8s ' % a + '  '.join(\n"
     "            k + ': ' + (f(abs(E(a, 'com_B_conf')[k] - E(a, 'com_B')[k]), 4)\n"
     "                        if E(a, 'com_B_conf').get(k) is not None else 'NA')\n"
     "            for k in ['INC_ALL', 'INC_MLP', 'INC_ATTN']))"),
    # (m4)
    ("A('- **并列**：邻域宽度 ±2 敏感性（四臂 `nb` 恰都 ' + str(V['A0_nf4'] and EX['arms']['A0_nf4']['nb'])\n"
     "  + '）；P17 `P6` 的 MEMO 改判（承 P18 `P5`）。')",
     "A('- **并列**：邻域宽度 ±2 敏感性（四臂 `nb` 恰都 ' + str(EX['arms'][AO[0]]['nb'])\n"
     "  + '）；P17 `P6` 的 MEMO 改判（承 P18 `P5`）。')"),
    # (m5) 文件清单：勘误/补丁脚本也应登记
    ("  '`do_append_phase20.py`、`closeout_docs_phase20.py`、`disk_verify_phase20.py`、`gen_present_phase20.py`')",
     "  '`do_append_phase20.py`、`closeout_docs_phase20.py`、`disk_verify_phase20.py`、`gen_present_phase20.py`'\n"
     "  '（＋同轮补丁 `_patch_p20_{1..5}*.py`）')"),
])

s2 = rep(P, [
    ("            same = (pv is not None and rv is not None and abs(float(pv) - float(rv)) <= 5e-7)",
     "            same = (pv is not None and rv is not None and abs(float(pv) - float(rv)) <= 1e-6)"),
    ("<th>探针（全量配对 / 缩幅网格）</th><th>生产（全尺度）</th><th>逐位</th></tr>')",
     "<th>探针（全量配对 / 缩 α 网格）</th><th>生产（全尺度）</th><th>≤1e-6</th></tr>')"),
    ("'⇒ 这些数字逐位相同。<b>α 网格依赖量</b>",
     "'⇒ 这些数字在容差 1e-6 内一致。<b>α 网格依赖量</b>"),
])

for path, probes in ((G, ["p['xhalf']['max_abs_dxh'], 4", '**独立重实现**', "EX['arms'][AO[0]]['nb']", 'XH_12_by_site']),
                     (P, ["<= 1e-6)", '<th>≤1e-6</th>', '在容差 1e-6 内一致'])):
    t = io.open(path, encoding='utf-8').read()
    for pr in probes:
        print('  chk %-34s -> %d' % (pr[:34], t.count(pr)))
print('PATCHED %d spots' % n)
