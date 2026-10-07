# -*- coding: utf-8 -*-
"""向当日 wlog 追加「Phase 20 收尾链前置预检」事件条目（无 BOM、CRLF，与既有文件一致）。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'

SEC = [
    '## 08:50 Phase 20 收尾链前置预检（续）',
    '',
    '- **锚点缺陷（已修）**：`do_append_phase20.py` 的追加锚点清单含字面量 `Qwen3-14B`，而 `gen_memo_phase20.py` 原文只写「A2-bf16 segfault」'
    '⇒ 追加预检（`assert not pre_miss`）会**中止**。已在 `gen_memo` §1 改为「A2 = `Qwen3-14B`（29.5 GB）的 bf16 腿 segfault」，'
    '`py_compile` 通过、grep 复核命中。注：`qwen3-4b`/`glm4-9b` 两锚由 `ARMS[a]["model"]` **现场渲染**（非字面量）⇒ 静态 grep 会误报缺失，运行时存在。',
    '- **PROBE 缩幅网格语义已确认**：`PROBE=1` 用 **7 档 α × 子集位点**（`ALPHAS=[0,0.15,0.3,0.45,0.6,0.8,1.0]`、`PROFILE=[1..5]+range(6,35,2)`），'
    '故探针 `[P]` 面板 `com_layer_x=22.675 ≠ 锚 23.008`、`com_layer_j=12.563 ≠ 锚 8.839`、`P2=False` 且标签 `ANCHOR_NA_TREATMENT` 属**设计内**'
    '（`FULL_SCALE=(not SMOKE)and(not PROBE)` ⇒ `E9_anchor.applies=False`，标注 N/A、不可比）。生产臂**全尺度**复现 `com_layer(x)=23.008 / com_layer(J)=8.839`。',
    '- **配对与键核对**：MERGE 对每个 `*_nf4` 与同名 `*_bf16` 配对（`quant_pair_stats`）⇒ 生产将产出 **A0 与 A1 两对**（探针只有 A0 一对）。'
    '`closeout_phase20.py` 依赖键全部实存：`verdict[*]`（`com_B_all/com_B_mlp/comlayer_B_all/share_mlp_beh_nb/gap/com_V/spearman_wall_ball/scheme/Q2_label/Q1_arch_max/Q1_blk_max`）、'
    '`quant_pairs[*]`（`com_B_all/com_V/comlayer_B_all/com_layer_x/com_layer_j/rho_b_all/share_mlp_beh_nb/xhalf/J`）、`E10_summary[*].com_V_own_spectrum`。',
    '- **进度**：A0_nf4 534.6 s、A0_bf16 432.4 s、A1_nf4 557.4 s 均完成并逐位复现冻结锚；A1_bf16（CPU offload，14.0 GB）进行中，其后 MERGE。',
    '',
]

raw = open(P, 'rb').read()
assert raw[:3] != b'\xef\xbb\xbf', '该 wlog 期望无 BOM'
t = raw.decode('utf-8')
body = '\r\n'.join(SEC)
new = t.rstrip('\r\n') + '\r\n\r\n' + body + '\r\n'
out = new.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(P, 'wb').write(out)

b2 = open(P, 'rb').read()
t2 = b2.decode('utf-8')
print('bytes %d -> %d (%+d)' % (len(raw), len(b2), len(b2) - len(raw)))
print('bom=%s crlf=%d bare_lf=%d' % (b2[:3] == b'\xef\xbb\xbf', b2.count(b'\r\n'),
                                     b2.count(b'\n') - b2.count(b'\r\n')))
print('前缀未变 =', b2.startswith(raw))
print('含 前置预检 =', '收尾链前置预检' in t2, '| 含 Qwen3-14B =', 'Qwen3-14B' in t2)
