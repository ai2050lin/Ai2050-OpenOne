# -*- coding: utf-8 -*-
"""补丁 8（口径补注 E-comv）：
主脚本 `com_V_recomputed = com_of_mass(W17['w_all'], RE_L)` 用的是 **P17 冻结 w_all 谱**（跨 Phase 可比口径），
因此**同一模型的两臂 com_V 按构造恒等** ⇒ 配对 Δcom_V ≡ 0 ⇒ 联合判据 Q3_com_V_stable 是**空检查**（不可作为跨精度证据）。
处理（**不改冻结脚本、不重跑臂**；只在报告/复核层增加一个已落盘量的聚合视图）：
  * present §C 增加 `Δcom_V(自谱)` 列 + 说明；
  * gen_memo §10 增加 `E-comv` 条目并现场渲染自谱位移；
  * disk_verify 增加 (a) 「冻结谱 com_V 两臂恒等」事实断言、(b) 自谱位移 ≤ 容差的**真实**跨精度断言；
  * closeout 的 Ledger rev_note 追加等价的英文披露条款。
"""
import io

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20'
n = 0


def rep(path, pairs):
    global n
    s = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        c = s.count(old)
        assert c == 1, '%s :: count=%d :: %r' % (path, c, old[:70])
        s = s.replace(old, new); n += 1
    io.open(path, 'w', encoding='utf-8', newline='\n').write(s)


# ---------------- present
rep(BASE + r'\gen_present_phase20.py', [
    ("H.append('<div class=\"card\"><table><tr><th>配对</th><th>Δcom_B(all)</th><th>Δcomlayer_B_all</th><th>Δcom_V</th>'\n"
     "         '<th>ρ(b_nf4,b_bf16)</th>",
     "H.append('<div class=\"card\"><table><tr><th>配对</th><th>Δcom_B(all)</th><th>Δcomlayer_B_all</th>'\n"
     "         '<th>Δcom_V<br><span class=\"note\">冻结谱·按构造=0</span></th><th>Δcom_V<br><span class=\"note\">自谱·真检验</span></th>'\n"
     "         '<th>ρ(b_nf4,b_bf16)</th>"),
    ("             % (k.replace('|', ' | '), f(_d('com_B_all')), f(_d('comlayer_B_all')), f(_d('com_V')),\n",
     "             % (k.replace('|', ' | '), f(_d('com_B_all')), f(_d('comlayer_B_all')), f(_d('com_V')),\n"
     "                f(R['arms'][k.split('|')[1]]['E10_summary']['com_V_own_spectrum']\n"
     "                  - R['arms'][k.split('|')[0]]['E10_summary']['com_V_own_spectrum']),\n"),
    ("H.append('<div class=\"note\">容差：<code>com_B/com_V/com_layer</code> ≤ %s 层；",
     "H.append('<div class=\"note\"><b>口径补注（E-comv）</b>：左列 <code>Δcom_V</code> 由 <b>P17 冻结 "
     "<code>w_all</code> 谱</b>重算（跨 Phase 可比口径）⇒ 同模型两臂<b>按构造恒等</b>，只证明「锚一致」，"
     "<b>不是</b>跨精度证据；<b>右列</b>用<b>本臂自谱</b> <code>com_V_own_spectrum</code> 才是真正的向量质心跨口径位移"
     "（描述性；向量侧的正式跨精度结论在 <b>Phase 19</b>）。容差：<code>com_B/com_V/com_layer</code> ≤ %s 层；"),
])

# ---------------- gen_memo
rep(BASE + r'\gen_memo_phase20.py', [
    ("A('- **P19**：本 Phase 是 P19 的**行为侧对偶**；两者合起来覆盖「向量 + 行为 + 剖面」三族量。')",
     "A('- **P19**：本 Phase 是 P19 的**行为侧对偶**；两者合起来覆盖「向量 + 行为 + 剖面」三族量。')\n"
     "A('- **[E-comv] 口径补注**：本 Phase 落盘的 `com_V` 取 **P17 冻结 `w_all` 谱**重算（为与 P17/P18/P19 跨 Phase 可比），'\n"
     "  '因此**同一模型的两臂 `com_V` 按构造恒等**（配对 Δ ≡ 0）⇒ 联合判据 `Q3_com_V_stable` 只是「锚一致性」，'\n"
     "  '**不可**当作跨精度证据。向量质心的跨精度证据仍由 **P19** 承担；本 Phase 另报**各臂自谱** '\n"
     "  '`com_V_own_spectrum` 的沿口径位移作**描述性**补充：Δ = '\n"
     "  + qd(PID, 'com_V') + ' ⇒ 自谱 Δ = '\n"
     "  + f(E(PID.split('|')[1], 'com_V_own_spectrum') - E(PID.split('|')[0], 'com_V_own_spectrum'), 4)\n"
     "  + '（A0）/ ' + f(E(PID2.split('|')[1], 'com_V_own_spectrum') - E(PID2.split('|')[0], 'com_V_own_spectrum'), 4)\n"
     "  + '（A1）层（见 `present_phase20.html` §C 与 `disk_verify_phase20.txt`）。')"),
])

# ---------------- disk_verify
rep(BASE + r'\disk_verify_phase20.py', [
    ("    for key, kk in (('com_B_all', 'com_B_all'), ('com_V', 'com_V')):\n"
     "        d = (V[ab][kk] - V[an][kk])\n"
     "        chk('G4d', '%s Δ%s 重算' % (pk, kk), abs(d - p[kk]['delta']) <= 1e-9,\n"
     "            round(d, 9), round(p[kk]['delta'], 9), 1e-9)\n",
     "    for key, kk in (('com_B_all', 'com_B_all'), ('com_V', 'com_V')):\n"
     "        d = (V[ab][kk] - V[an][kk])\n"
     "        chk('G4d', '%s Δ%s 重算' % (pk, kk), abs(d - p[kk]['delta']) <= 1e-9,\n"
     "            round(d, 9), round(p[kk]['delta'], 9), 1e-9)\n"
     "    # [E-comv] (a) 冻结谱 com_V 两臂按构造恒等（记录事实，防止被误读为跨精度证据）\n"
     "    chk('G4df', '%s com_V（P17 冻结谱）两臂按构造恒等' % pk, abs(V[ab]['com_V'] - V[an]['com_V']) <= 1e-12,\n"
     "        round(V[ab]['com_V'] - V[an]['com_V'], 12), 0.0)\n"
     "    # [E-comv] (b) 自谱质心的跨口径位移才是真检验（描述性，须在容差内）\n"
     "    _dvo = (Sa['com_V_own_spectrum'] - Sb['com_V_own_spectrum'])\n"
     "    chk('G4dw', '%s Δcom_V(自谱) 描述性 ≤ 容差' % pk, abs(_dvo) <= FL['QUANT_TOL_COMV'],\n"
     "        round(_dvo, 6), FL['QUANT_TOL_COMV'])\n"),
])

# ---------------- closeout (Ledger rev_note 披露)
rep(BASE + r'\closeout_phase20.py', [
    ("    'COVERAGE LIMIT: the bf16 leg is a TWO-MODEL leg",
     "    'NOTE (E-comv): the com_V reported here recomputes the interval-sum centroid of the Phase-17 FROZEN '\n"
     "    'w_all spectrum (kept for cross-Phase comparability), so within a model it is ARM-INVARIANT by '\n"
     "    'construction (paired delta = 0) and must NOT be read as cross-precision evidence; the cross-precision '\n"
     "    'evidence for the vector centroid remains Phases 19s. The per-arm OWN-spectrum centroid '\n"
     "    'com_V_own_spectrum is reported descriptively: delta = %(dcvo0)s (A0) / %(dcvo1)s (A1) layers. '\n"
     "    'COVERAGE LIMIT: the bf16 leg is a TWO-MODEL leg"),
    ("    nw=json.dumps(NW, ensure_ascii=False), nrows=NROWS,\n",
     "    dcvo0=('%.4f' % (RES['arms'][PID.split('|')[1]]['E10_summary']['com_V_own_spectrum']\n"
     "                      - RES['arms'][PID.split('|')[0]]['E10_summary']['com_V_own_spectrum'])),\n"
     "    dcvo1=('%.4f' % (RES['arms'][PID2.split('|')[1]]['E10_summary']['com_V_own_spectrum']\n"
     "                      - RES['arms'][PID2.split('|')[0]]['E10_summary']['com_V_own_spectrum'])),\n"
     "    nw=json.dumps(NW, ensure_ascii=False), nrows=NROWS,\n"),
])

print('PATCHED %d spots' % n)
