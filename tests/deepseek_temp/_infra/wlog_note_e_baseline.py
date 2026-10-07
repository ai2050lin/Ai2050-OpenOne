# -*- coding: utf-8 -*-
"""把「MEMO 完整性事件 E-baseline + Phase 20 收尾前置加固」追加到当日 wlog（CRLF、UTF-8+BOM、append-only）。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'

SEC = [
    '## MEMO 完整性事件（E-baseline）与 Phase 20 收尾前置加固 [08:25]',
    '',
    '- **发现**：核对 `_infra/memo_baseline.json` 的 `post-append-phase19` 快照（469161 B / `2255b365`）与实盘 MEMO 不符 —— 实盘 **469271 B / `ec7be6b6`**（行数同为 4481、`bare_lf 0`、19 个 Phase 标题、无 Phase 20 节）。',
    '- **定性（三重独立证据）**：① **mtime** —— 快照冻结 07:39:49、P19 `disk_verify_phase19.txt` 07:40:59（该次复核读到 `469161 vs 469161` 判 **PASS**）、**MEMO 07:52:07 被就地改写**；② **字节算术** —— P10–P19 共 10 个标题由短形式 `[HH:MM]` 规范化为完整形式 `[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+110 B = 实测差**，残差 **0**，行数不变；③ **判别性键** —— `sections` 用 `l[:44]` 生成，只有 P11/P14/P15 的时间戳跨越第 44 字符，三处的快照键**逐字符等于短形式的前 44 字符**。',
    '- **影响**：`post-append-phase19` 的 bytes/sha256 锚与 3 项 `sections` 键陈旧；**正文无损失、研究结论与 Ledger 不受影响**。',
    '- **处置（铁律：不改写历史）**：`history` 标 `stale` + `note`；新基线内嵌 `drift_events`；`sections` 键口径由「行前 44 字符」改为**完整标题行**并加**碰撞断言**（`closeout_docs_phase20.py` 与 `closeout_phase20.py` 均已改，后者附 `sections_key_rule`）；MEMO §11 增 `E-baseline`、`present` §G 增同条、`disk_verify_phase20.py` 增 G10h–G10o；**不动**追加源留痕件（它们保留短形式，正是本案的原始证人）。审计件：`tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json`、`audit_memo_drift_phase19.{py,txt}`、`memo_integrity_20261002.md`。',
    '- **技能 / 记忆同步**：`rdc-phase-closeout` 新增**教训 32**（快照非终点 / 键必须内射 / 三对齐面对账），计数改 **32 教训**；`rdc-main-axis-probe` 计数改 **61 坑**；MEMORY.md 铁律 **(af)** 入册、§5 改「P16–P20 教训」、§8 计数同步。',
    '- **同轮前置加固**：新增 `tests/deepseek/Phase20/run_phase20_closeout.py`（7 步 fail-fast 顺序驱动）；**格式静态自检抓到补丁 9 的 3 处缺陷**（字符串缺闭合引号 1、占位符 9 ↔ 实参 10 且次序错位 1、`%d` 收到 `str` 1）—— 全部修正后三处渲染点试渲染通过（把「收尾时刻才炸」提前到「写入时刻」）。',
    '- **Phase 20 正式运行进度**：A0_nf4 **完成（534.6 s）** —— `FULL_SWAP=10.749740`、`com_B(all)=21.085`、`com_B(mlp)=23.807`、`share_mlp_beh(nb)=0.6632`、`comlayer_B_all=12.211`、`E9 锚复现 OK`、保真度 arch 1.617e-02 / blocks 3.591e-03、`U_ℓ` 秩 5、`nb=[26,28]`、`REACH n=18` ⇒ **P18/ P17 冻结锚逐位复现**；A0_bf16 进行中（其余 A1_nf4 / A1_bf16 + MERGE 待跑）。',
    '',
]

raw = open(P, 'rb').read()
t = raw.decode('utf-8')          # wlog 无 BOM（承 closeout_docs 的 encode('utf-8') 写盘约定）
assert not raw.startswith(b'\xef\xbb\xbf'), 'wlog 意外带 BOM —— 与既有约定不符，中止'
assert raw.count(b'\n') - raw.count(b'\r\n') == 0, 'wlog 出现裸 LF'
body = '\r\n'.join(SEC)
new = t.rstrip('\r\n') + '\r\n\r\n' + body + '\r\n'
out = new.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(P, 'wb').write(out)

# 落盘复核
b2 = open(P, 'rb').read()
t2 = b2.decode('utf-8')
print('bytes %d -> %d (%+d)' % (len(raw), len(b2), len(b2) - len(raw)))
print('bom=%s crlf=%d bare_lf=%d' % (b2[:3] == b'\xef\xbb\xbf', b2.count(b'\r\n'),
                                     b2.count(b'\n') - b2.count(b'\r\n')))
print('前缀未变 =', b2.startswith(raw))
print('含 E-baseline =', 'E-baseline' in t2, '| 含 教训 32 =', '教训 32' in t2)
