# -*- coding: utf-8 -*-
import os, time, hashlib
ROOT = r'D:\AI2050\Ai2050-OpenOne'
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

wlog_add = """

## N2 归因 Phase（N1 预注册的 N2；2026-10-01 05:11 完成）
- 目标：cloze is-a（`{W}是一种`）的"重建源"定位——逐层零消融 + 逐头零消融 + 三对照 + 跨模板存活。
- 模型：qwen3-4b / qwen2.5-3b-instruct（tie=True）、glm4-9b-chat-hf（tie=False，仅 n2d）。单卡 RTX 5080 串行，合计约 27 min GPU，零 OOM。
- **核心裁决**：全局**单头 share_max = 0.030** ≪ 0.30 → **分布化重构**（N2 预注册判据命中）。
- **交互指数 $I_\\ell=|\\Delta_{all}|/\\sum_h|\\Delta_h|$ 跨越 3 个数量级**：L24 0.007 / L9 0.010（**强补偿**：逐头 Σ1.07 有效但整层 0.008 近乎无害）↔ L6 **5.15**（**超可加**：整层 −6.01 vs 逐头之和 1.17）。⇒ **逐头归因在本系统内不是合法算子**。
- **三对照**：仅 attn L6（−6.014 vs noise −0.170±0.593）与 mlp L0（−5.108 vs −0.540±0.665）显著；attn L34/L35/L29 弱有效；mlp L9/L24/L29 **不显著**。L24–35 单层全部不显著，**只有联合消融才显现**（attn joint −3.887 vs Σsingle −2.355，指数 1.65；mlp joint −3.752 vs −1.386，指数 2.71）。
- **跨模板自检（本轮最重要）**：N1c 的 "L31–33 重建窗" **不跨模板**——best rank short 5 / mid 86 / long 28（qwen3-4b），rank≤10 在 mid/long **从未出现**。L6 效应 short −6.014 → mid −0.041 → long −0.207（**死**）。
- **跨模板存活件**：qwen3-4b = attn L29/L34 + mlp L0/L29；qwen2.5-3b = attn L0/L1/L31/L34/L35（**L34 −0.706/−0.681/−0.787 最稳**，min|Δ|=0.681）。glm4-9b 最大层 L3（−2.44）。
- **承诺层模型特异**：patch 曲线 qwen3-4b **L6 +10.111** 一次性锁死（其后 29 层 ±10.1 恒定）；qwen2.5-3b **L1 +9.625** 已锁死（无 L6 台阶）。
- **注意力落点**：三模型读位注意力被 **sink p0 / 标点**占据（0.88–0.92），长句主语质量 **<0.17** ⇒ 不存在"取主语头"。
- **行为 vs 内部脱钩**：long 模板下 苹果→水果 P=0.612（高于 short 0.375），但 logit-lens rank 轨迹从未进 top-10。
- 修 4 个 bug：o_proj **输入 4096 / 输出 2560** 维度混淆（capture 改前置钩子）；`self_attn` forward_hook 取"注意力权重"实为 attn_output（改 `output_attentions=True`+`attn_implementation='eager'`）；`head(norm(h))` 缺 `detach()`；N2 Stage C 噪声用输出范数注入输入空间。
- 产物：脚本 7（`n2_reconstruction_source.py`/`n2b_robustness.py`/`n2c_slot_commitment.py`/`n2d_attn_write.py`/`n2e_template_robustness.py`/`n2f_topk_diag.py`/`n2g_critical_layers.py`）+ 报告 13，全在 `tests/gpt5_temp/`。
- 记录：`research/deepseek/docs/AGI_DEEPSEEK_MEMO.md` 追加 `## Phase 5： 探索性探针 N2`（51298 → 73615 bytes，sha8 `bafbc150`→`07b208a3`，**前缀逐字节未变**，BOM/CRLF 一致，标题唯一）。**用户指令（2026-10-01）：本对话所有记录只写该文件，不新增其他记录类文件。**
- 下一步：**N2-h1 置换向量消融**（只换类别分量，决定承诺曲线是否几何伪影）→ N2-h2 单变量距离臂 → N2-h3 GLM4 补齐 → N2-h4 L31–35 块内双头联合 → N2-h5 面板 24→41+held-out → N2-h6 sink 扣减。
"""

mem_add = """
## 归因纪律（N2，2026-10-01，3 模型：qwen3-4b / qwen2.5-3b / glm4-9b）
- **逐头归因不是合法算子**：交互指数 $I_\\ell=|\\Delta_{all}|/\\sum_h|\\Delta_h|$ 实测跨 0.007（L9/L24 强补偿）↔ 5.15（L6 超可加）。**任何"某头负责某功能"的表述必须先报 $I_\\ell$**；单头 share 必须与整层 Δ 并报。
- **消融必须三对照**：zero / mean / **matched-norm noise(5 seeds)**；判据 = 真实效应超 noise 均值+2sd。qwen3-4b 实测噪声地板：attn ≈ −0.02±0.19（L34），mlp L0 −0.54±0.67。**未过地板的量不得进机制链**。
- **必须跨模板存活检验**：short `{W}是一种`(T=2) / mid(T=5) / long(T=11)，判据三模板均 Δ<−0.3。qwen3-4b 的 L6（short −6.01, w=1.00）在 mid/long 崩塌到 −0.04/−0.21 ⇒ **短句"关键层"是退化产物**。存活件：qwen3-4b = attn L29/L34 + mlp L0/L29；qwen2.5-3b = attn L0/L1/L31/L34/L35（L34 最稳）。
- **承诺层模型特异**：全状态 patch 曲线 qwen3-4b 在 **L6** 一次性锁死（+10.11，其后 29 层恒定），qwen2.5-3b 在 **L1**。全状态替换天然把"该层状态一旦充分即接管"当结果 ⇒ **要判定机制必须做只换类别分量的置换向量消融**（N2-h1）。
- **读位注意力被 sink 占据**：三模型三模板一致，读位注意力指向 p0 或标点（0.88–0.92），主语质量 <0.17 ⇒ **不存在"注意力取主语"的路径**；所有"注意力搬运语义"的叙事必须先扣 sink。
- **N1c 降级**："L31–33 把 is-a 从 rank 518 拉到 1"**只在 2-token 最小句成立**，自然句 best rank 86/28 且从未进 top-10 ⇒ 降级为**模板特异的可读性现象**，禁作为通用机制。
- **行为 ≠ 内部收敛**：long 模板 苹果→水果 P=0.612（高于 short 0.375）但 rank 轨迹从未进 top-10 ⇒ 不得用行为正确率反证内部存在收敛窗。
- **接口陷阱（本机）**：`o_proj` 输入 = `n_heads*head_dim`（qwen3-4b: 4096）≠ 输出 hidden（2560）——mean 消融 capture 必须用 **forward_pre_hook**；取注意力权重必须 `output_attentions=True` + `attn_implementation='eager'`（sdpa 下 `output_attentions` 不可靠）；`head(norm(h))` 必加 `detach()`。
- 记录落点：本对话所有研究记录只写 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（用户 2026-10-01 指令）。
"""

for path, add, tag in [(WLOG, wlog_add, 'wlog'), (MEM, mem_add, 'memory')]:
    b0 = open(path, 'rb').read()
    h0 = hashlib.sha256(b0).hexdigest()
    txt = open(path, encoding='utf-8').read()
    nl = '\r\n' if '\r\n' in txt else '\n'
    blob = add.replace('\r\n', '\n').replace('\n', nl).encode('utf-8')
    with open(path, 'ab') as f:
        f.write(blob)
    b1 = open(path, 'rb').read()
    print('%s before %d after %d delta %d prefix_identical %s' %
          (tag, len(b0), len(b1), len(b1) - len(b0), b1.startswith(b0)))
    print('   newline_style %r  sha8 %s -> %s' % (nl, h0[:8], hashlib.sha256(b1).hexdigest()[:8]))
    print('   marker_present %s' % (('## N2 归因 Phase' in open(path, encoding='utf-8').read())
                                    if tag == 'wlog' else ('## 归因纪律（N2' in open(path, encoding='utf-8').read())))
