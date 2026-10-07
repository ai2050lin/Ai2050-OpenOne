# -*- coding: utf-8 -*-
# p3154_prereg.py: MEMO 追加重定向说明 + Phase 3154 新预注册（观测前冻结）
import os, hashlib, shutil, io

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
SNAP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'snapshot_AGI_GPT5_MEMO_pre3154.md')

before = open(MEMO, 'rb').read()
sha_before = hashlib.sha256(before).hexdigest()
if not os.path.exists(SNAP):
    shutil.copyfile(MEMO, SNAP)
print('memo bytes before:', len(before), 'sha8:', sha_before[:8])

txt = before.decode('utf-8')
if txt.startswith('\ufeff'):
    txt = txt[1:]
    bom = True
else:
    bom = False

marker = '### 预注册 Phase 3154（重定向）：G1-P4 多因素混杂分解'
assert marker not in txt, 'already appended'

block = (
    "\n### 预注册 Phase 3154（重定向）：G1-P4 多因素混杂分解（2026-10-01 观测前冻结）\n"
    "> 用户 2026-10-01 指令重定向：原预注册 3154（G2-P1 多关系族与算子可分离性）**顺延为 Phase 3155**；"
    "新 3154 = 对用户方法论问题（上下文中语言/风格/逻辑/距标点距离多维权如何在神经元中分解）的直接落地。\n\n"
    "**问题**：隐状态中语言(L: zh/en)、风格(S: formal/casual)、逻辑一致(G: con/contra)、标点距离(D 连续协变量) 四因素的方差份额能否分离、逻辑方向是否因果可读？\n\n"
    "**材料（冻结，脚本 tests/glm5/phase3154_mfd_multifactor_disentangle.py 内 32 条元组逐字一致）**："
    "6 主题(天气/运动/饮食/学习/植物/市场)×2 语言×2 风格×2 逻辑×4 同字数移逗号变体=192 句/模型"
    "（zh 按字 cut∈{20%,45%,70%,90%}，en 按词边界，逗号仅在 premise 内移位、组内字符数守恒）；"
    "con/contra 为等长最小对（中文字数相等、英文词数相等，fail-fast 断言）；"
    "held-out=16 句（T7 睡眠/T8 冷链，cut=45%）；SMOKE=主题[:2]×cuts[:2]+T7=40 行。\n\n"
    "**采集**：三模型（qwen3-4b NL36/D2560、qwen3-14b NL40/D5120、glm4-9b NL40/D4096）last-token 全隐层 H fp16 "
    "（208×(NL+1)×D），collect.npz sha8 登记；确定性锚=3 行重前向位级比对；D 协变量=n_tok(prompt)−n_tok(无逗号前缀)，"
    "组内随 cut 单调递减（违例率门≤0.5）。\n\n"
    "**分析（冻结）**：① 序贯投影 ANOVA [L→S→content(24 块 dummy)→G→D→resid]，SVD 正交基预计算一次、份额和=1（闭合断言），"
    "KOUT/KOUT/embedding 全表+全层 G/L/D 曲线；② 指纹门=KOUT 6 维份额向量三模型两两 Pearson≥0.8（3 对，summary 模式），"
    "任一对<0.8→`g1p4_fingerprint_inconsistent_material_method_descriptive`；③ 几何干预=ŵ_G（contra−con 组均值差）投影，"
    "flip=con 行注入 +0.5×分离度 后越 0 率 vs 100 随机方向 null（stat=分离度/组内 sd，p<0.05=逻辑轴显著）；"
    "④ held-out 门=ŵ_G 于 16 句符号准确率≥0.75。\n\n"
    "**判决格式**：`g1p4_<model>|G_..|D_..|resid_..|flip_.._null_.._p_..|hoacc_..`；"
    "summary=`g1p4_fingerprint_consistent|fpmin_..` / `g1p4_fingerprint_inconsistent_material_method_descriptive|fpmin_..|bad_pairs_..`。\n"
    "**诚实边界**：3153 已证模板内高秩散布 57–60% 不可被本设计硬拆（resid 份额预期仍主导）；本 Phase 只回答可控四因素的各自可分离份额与逻辑轴因果可读性。\n"
)

txt = txt.rstrip('\n') + '\n' + block
blob = txt.encode('utf-8')
if bom:
    blob = b'\ufeff' + blob
with open(MEMO, 'wb') as f:
    f.write(blob)

after = open(MEMO, 'rb').read()
ok = marker in after.decode('utf-8', errors='replace')
print('memo bytes after:', len(after), 'sha8:', hashlib.sha256(after).hexdigest()[:8])
print('marker on disk:', ok)
print('appended chars:', len(blob) - len(before))
