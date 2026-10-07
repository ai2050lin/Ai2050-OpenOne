# -*- coding: utf-8 -*-
"""Phase 16 收尾补记：向当日 wlog 追加「收尾补记」段（幂等）。
纪律：段内反引号由 .py 文件承载（不经 bash 内联），避免被命令替换吞掉。
"""
import io
import os
import hashlib

WLOG = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
MARK = '### 收尾补记（Phase 16 / N2h1-α-9）'

block = '''### 收尾补记（Phase 16 / N2h1-α-9）

- **独立磁盘复核 `disk_verify_phase16.py`：`TOTAL checks = 206 ; FAIL = 0`**（A 文件与哈希 / B seal·exec·amend1 一致 / C result 结构 / D 从 `E4_per_pair` 逐位点**独立重算** `xhalf`·`J` 与 `XH_RANGE` 全域+legacy / E 可达性 `ρ(ℓ)`·`REACH` 掩膜·`ell_reach` / F legacy 域旧量 / G 主域旧量 / H 新量 `com_layer`·`span_k` **双边分位逐位复现** / I 判决与 7 条预测复算 / J MEMO 完整性（含前缀对冻结基线 `sha256` 的**逐字节验证**、`k=0` 尾随 CRLF 反解） / K Ledger 299 条 + 自哈希重算 `28ef0f92` / L `memo_baseline` + wlog 反引号不变量）。
- **首跑 3 FAIL 全部是复核脚本自身缺陷（产物零缺陷）**：① `J5 追加体与 MEMO 尾部一致` 失败 —— 渲染件是 **LF**、MEMO 是 **CRLF**，须归一后再比；② `J6 前缀未变` 失败 —— 原文**无尾随换行**（`k=0`），循环从 1 起跳漏掉 0；③ `E5 窗下一位点 rho < UNREACH` 失败 —— 该断言编码的是**设计意图**（写入窗=REACH 左端点），而**实测 A2 的 REACH 左端点 = ℓ3（ρ=0.2428 ≥ 0.10）而 `ell_reach` = 4** ⇒ 断言必须只编码**预注册判据原口径**（P3 = 写入窗 **∈** REACH）。
- **⚠️ 同轮重写（v1 → v2）**：追加成功后我又改了渲染器（§4 表头 + 锚点补齐）并重渲染 ⇒「生成器 ≠ 留痕件 ≠ MEMO」。处置：以**冻结的追加前基线 `sha256`（`0bd7bfd0` / 389,885 B）逐位反解回滚**（`base_T + CRLF×k`，`k=0` 命中）→ v1 原文留痕 `memo_append_phase16_v1_asappended.md` → 修正版重追加（MEMO 389,885 → **411,257 B** / sha8 **12903cd5** / 16 标题 / Phase 16 @ **L3684**）→ 复核脚本两处口径修净 → 重跑 **0 FAIL**。v2 正文新增 `### 10 同轮勘误（v1 → v2）`，逐条写明三处缺陷与「其余逐字节相同」。
- **两处元数据缺陷（已冻结，以勘误为准）**：① `execution_phase16.json` 的 `bootstrap.seeds` 记 `new_x/new_j = SEED+41/+53`，**实现**用 `SEED+61/+67`（只影响**零假设分位复现**，不动观测/判决）⇒ 复核脚本把「确实不符」写成**显式断言 B11** 作为长期不变量。② §4 表头 v1 把 `sup_id_matches_ref` 列误标为「F1b 词表匹配」（`F1b_ok` 三臂其实均 True；A1 的 False 是 GLM4 词表不同导致的**预期**差异）。
- **技能同步**：`rdc-main-axis-probe` 新增**坑 56**（置换零假设**保留 jump 多重集** ⇒ 谱熵/`max÷mean`/参与比整族**结构性退化**（双边必 p=1）；改用**顺序敏感 + 定义在物理层号轴**上的 `com_layer`（单位「层」、网格不变量）与 `span_k`；可达性掩膜 `REACH`；`ℓ_reach == L*_own` 3/3 严格；**「写入窗 = 左端点」只是意图**（A2 反例）；种子须与实现核对；**P6 否证 = 位置量的跨模型稳健性是幻觉**）→ 65,963 → **71,290 B**（15 臂 + **56 坑**）。`rdc-phase-closeout` 新增**教训 26**（改渲染器后**必须重跑 `do_append`**；回滚点用冻结基线 `sha256` 反解、`k` 可能为 **0**；v1 留痕 + 追加 `同轮勘误` 节）与**教训 27**（复核脚本自身三类错配：**EOL 归一** / 断言只编码预注册判据而非设计意图 / 冻结元数据漂移写成显式断言；配套：wlog 追加幂等、`history` 排除同名自 tag、首冻断言加守卫、基线键集对齐）→ 37,419 → **42,022 B**（**27 教训**）。
- **MEMORY.md 更新**：Ledger n=**299**（`ledger_sha256_8 = 28ef0f92`）、基线 411,257 B / `12903cd5` / 16 标题、收尾链「**九次 P8–P16**」、铁律 **(ac)** 入册（插值型读数不得设单一硬门）、§2 新增 P16 条目、§3 新增「⑧ P16 限界」、§5 新增三条 Windows/幂等/MEMO 口径纪律、§7 死线改为 **P17 = 把「位置」接到「组件」（±2 层向量预算）+ `span_k` 体系化 + `xhalf` 可达域敏感性**、§8 技能计数更新。
- **展示页**：`present_phase16.html`（自包含、浅色、数据驱动；含 ρ 阶梯条、新旧量对照、P6 否证框、amend1 分层、同轮勘误与限界）。
'''

b0 = open(WLOG, 'rb').read()
t0 = b0.decode('utf-8')
if MARK in t0:
    print('ALREADY: wlog 已含收尾补记，跳过')
else:
    t1 = t0.rstrip('\r\n') + '\r\n\r\n' + block.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n') + '\r\n'
    open(WLOG, 'wb').write(t1.encode('utf-8'))
    b1 = open(WLOG, 'rb').read()
    print('wlog: %d -> %d B (%+d)' % (len(b0), len(b1), len(b1) - len(b0)))
    print('sha256 =', hashlib.sha256(b1).hexdigest())
b = open(WLOG, 'rb').read().decode('utf-8')
print('backticks in block:', b[b.find(MARK):].count('`'))
print('bare_lf:', open(WLOG, 'rb').read().count(b'\n') - open(WLOG, 'rb').read().count(b'\r\n'))
