# -*- coding: utf-8 -*-
"""MEMORY.md 末次压缩：去除与 MEMO 冗余的细节，目标 < 7000 chars。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
s = io.open(P, encoding='utf-8').read()
REPS = []

REPS.append((
    "- **P4–P7（主轴三段）**：嵌入=一级词典 / 层=开关 / **权重绑定决定 is-a 落点**；单头 share_max **3.0%** ⇒ 无「某头负责某功能」；读位槽 = **G−1=5 维类别子空间**充分必要（随机 5 维 170–6000×）；跨 L2 复现 3/3 但**跨族近乎正交 ⇒ 无通用类别算子**；写入窗族特异。**R1 纠错**：K_d 原判 FAIL → 降级 post-hoc P-N3b。",
    "- **P4–P7（主轴三段）**：嵌入=一级词典 / 层=开关 / **权重绑定决定 is-a 落点**；单头 share_max **3.0%** ⇒ 无「某头负责某功能」；读位槽 = **G−1=5 维类别子空间**充分必要（随机 5 维 170–6000×）；跨 L2 复现 3/3、**跨族近正交 ⇒ 无通用类别算子**。**R1 纠错**：K_d FAIL → 降级 post-hoc。",
))

REPS.append((
    "⑥ **P14/P15**：集中度判据在 n≤18 上**无区分力** ⇒ 结论须同报 null95 与裕度并声明**坐标 + 模型**；位置通道空度只在 T=2 测得；⑦ **P15**：A1/A2 **家族与规模混杂**；nf4 是**替换口径**（保真证据仅 A0）；`argmax_w_j` 在量化噪声下换窗（nf4 0 / bf16 1）⇒ 跨 Phase 层索引须在原口径重测；⑧ **P16**：`com_layer` 是**描述性位置量** ⇒ 能说「变化在 L24 附近」**不能说「L24 做了什么」**（接组件预算是下一死线）；置换零假设只检验「位置是否非随机」；「写入窗 = REACH 左端点」只是设计意图，实测 **A0/A1 成立、A2 不成立**（A2 左端点 ℓ=3 而 `ell_reach`=4）⇒ 判据写「写入窗 ∈ REACH」；exec `bootstrap.seeds` 的 `new_x/new_j` **元数据错记**（记 +41/+53，实为 +61/+67），只影响零假设分位复现；⑨ **P17**：`com_V`（向量质量质心）与 `com_layer`（行为质心）是**两件事**（A2 判别臂 `min_d`=17.53）；「深端写入无效」是**相关性**陈述（因果需逐层组件**行为**预算 = 下一死线）；`w_ℓ` 绝对量级只在臂内可比（跨臂只比质心位置与份额）。",
    "⑥ **P14/P15**：集中度判据在 n≤18 **无区分力** ⇒ 须同报 null95 与裕度并声明**坐标 + 模型**；⑦ **P15**：A1/A2 **家族与规模混杂**；nf4 是**替换口径**（保真仅 A0）⇒ 跨 Phase 层索引须在原口径重测；⑧ **P17**：`com_V`（向量质量质心）与 `com_layer`（行为质心）是**两件事**（A2 判别臂 `min_d`=17.53）；「深端写入无效」是**相关性**（因果需逐层组件**行为**预算）；`w_ℓ` 绝对量级只在臂内可比；⑨ **P16**：「写入窗 = REACH 左端点」实测 A2 不成立 ⇒ 判据写「写入窗 ∈ REACH」；exec `bootstrap.seeds` 的 `new_x/new_j` **元数据错记**（记 +41/+53 实为 +61/+67）。",
))

REPS.append((
    "- 3151 k3_only；3152 k1_not_triggered（k\\* 定律：B4@k\\* 0.046/0.008/0.081 vs 读出层 0.39/0.33/0.40）；3153 fingerprint_consistent_coverage_partial。3105–3150：真值=记录级一阶矩广播；写入头组 L20–28 主写 / L32 擦除；**判决符号=rev-3151b（负=优）**。路线：层相关写入算子族 {W_ℓ} + 层无关读出算子 P；K2→**3154 已预注册**。",
    "- 3151 k3_only；3152 k1_not_triggered（k\\* 定律：B4@k\\* 0.046/0.008/0.081 vs 读出层 0.39/0.33/0.40）；3153 fingerprint_consistent_coverage_partial。3105–3150：真值=记录级一阶矩广播；写入头组 L20–28 主写 / L32 擦除；**判决符号=rev-3151b（负=优）**。路线：层相关写入算子族 {W_ℓ} + 层无关读出算子 P；**3154 已预注册**。",
))

REPS.append((
    "append-only：锚点**先对追加源文件预检**（断言用**不变量**）；`do_append_*` 配 `rollback_*` 且**幂等**；**前缀锚只能是「写前读到的原始 bytes」**（`BOM+rstrip(CRLF)` 短 2 字节 ⇒ 假失败且发生在写盘后）。复核脚本用前缀锚、写完**先空跑**。`%` 转义 `%%`。关键写入后 Grep 复核真实磁盘；GPU 逐模型防 OOM。项目 Python：`.venv\\Scripts\\python.exe`。",
    "append-only：锚点**先对追加源预检**（断言用**不变量**）；`do_append_*` **幂等** + 配 `rollback_*`；**前缀锚只能 = 写前原始 bytes**（`BOM+rstrip(CRLF)` 短 2 字节 ⇒ 写盘后假失败）。写完**先空跑**；`%` 转义 `%%`。关键写入后 Grep 复核真实磁盘；GPU 逐模型防 OOM。Python：`.venv\\Scripts\\python.exe`。",
))

REPS.append((
    "(文书) 文档数字由 result 生成器渲染，禁手工转录；结论句走 verdict 分支，禁写死散文。",
    "(文书) 结论句走 verdict 分支，禁写死散文。",
))

for i, (old, new) in enumerate(REPS):
    n = s.count(old)
    print('REP[%d] count=%d' % (i, n))
    assert n == 1, 'REP[%d] expect 1 got %d' % (i, n)
    s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(P, encoding='utf-8').read()
assert 'P17 = 写入向量位置与效力' in t and 'P18 最高' in t and '(ad)' in t and '(ae)' in t
print('MEMORY.md final: %d B / %d chars / %d lines' % (len(t.encode('utf-8')), len(t), len(t.splitlines())))
