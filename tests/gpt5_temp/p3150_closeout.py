# -*- coding: utf-8 -*-
"""Phase 3150 closeout: five writes (ledger + MEMO +
daily log + MEMORY.md + verify). Idempotent."""

import hashlib
import io
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LED = os.path.join(ROOT, 'research', 'gpt5',
                   'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5',
                    'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory',
                     '2026-10-01.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory',
                   'MEMORY.md')
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913',
                   'phase3150',
                   'p0_freeze_carrierdecode',
                   'result.json')

RES_SHA8 = 'a2158ea5'
SEAL = '1c6eb2f9'

# 0) verify result on disk
raw = io.open(RES, 'rb').read()
assert hashlib.sha256(raw).hexdigest()[:8] == RES_SHA8
r = json.loads(raw.decode('utf-8'))
assert r['verdict'].startswith('a_3149_ok')
print('result verify ok')

now = time.strftime('%H:%M')

# 1) ledger append (n=287)
led = json.load(io.open(LED, encoding='utf-8'))
if not any(m.get('phase') == 3150
           for m in led['measurements']):
    led['measurements'].append({
        'phase': 3150, 'name': 'p0_freeze_carrierdecode',
        'seal_sha8': SEAL, 'result_sha8': RES_SHA8,
        'evidence_level': 'statistical',
        'model_scope': 'glm4-9b', 'n_rows': None,
        'prereg_id': 'p3150', 'superseded_by': None,
        'verdict': r['verdict'],
        'created': time.strftime(
            '%Y-%m-%d %H:%M:%S'),
        'runtime_s': r['runtime_s']})
    with io.open(LED, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print('ledger -> n=%d' % len(led['measurements']))
else:
    print('ledger already has 3150')

# 2) MEMO append
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3150:' not in memo:
    section = u'''
## Phase 3150: P0制度冻结+载体解码（T4 第33 Phase）[%s]

### 范式转移长文裁决（全文 `PARADIGM_SHIFT_VERDICT_v1.md`）
用户提供长文主张"停止微观修补、转向 TDA/SAE/Neural ODE"。**总裁决：诊断大体正确、历史叙事错误、三条处方一真二偏**。逐条 15 项判定（4✅/8⚠️/6❌）：①"还原论陷阱"对——但正是我们 3109/3138/3139/3140/3148/3149 先确证；②"3000 Phase 全在找齿轮"前提错——TESTPLAN v1（2026-09-30）已完成同等裁决并预注册 3150 P0（死线 K1-K3）；③TDA"环面/贝蒂数"❌数学病态（672 行 State Bank n<d，VR 复形高维 Betti 数=采样伪象），Mapper 降级采纳为 G5 等价类检验（死线 K-G5: ARI<0.3 弃）；④SAE ✅采纳为 G4（接口现成：3138 B/I/C=手工字典、3149 载体 token 族=模型自稀疏特征种子；成本被低估：65536 特征×10³ 激活/特征=6.55e7 需求，State Bank 差 305×，须 5e8 激活预算 margin 8×；工具 dictionary_learning 优先，sae_lens 需 GLM4 适配；死线 K-G4: G4.2 特征-载体族对齐 top-8 share<0.3 → 降级）；⑤ODE 李雅普诺夫 ❌（40 层离散=插值依赖）但逐层 Jacobian 谱 ✅降级采纳 G6（descriptive 级）；⑥"吸引盆分岔"仅弱形式可用（3148 d2.5 倒 U+交叉=真实分岔样行为的重述）。**执行决定：不是换船，是 TESTPLAN v1 原航线+三修正支线（G4 3161+ 候选前置 G1 通过）。**

### P0 制度冻结（F1-F7 全过，零 GPU）
F1+F2+F7 `metric_dict.json`（8 指标含 G4/G5/G6 新量+判决分级+bit 锚定位声明+端口替换纪律+绑定二分规则）；F3 `gate_precheck.py` 自测（mde=0.0866@n128）；F4 `counterexample_grep.py` 自测（kx_interchangeable 引用扫描）；F5 ledger v3 回填 286/286（evidence_level/model_scope/n_rows/prereg_id/superseded_by；3108→3109 superseded）；F6-min `FIRST_PRINCIPLES_3090_3149.md` 60 Phase 一行洞察。

### PART D 载体 token 身份解码：carrier_skeleton_class（关键新发现）
GLM4 tokenizer 解码 25 token：**NEG 13 tok=多语言词素碎片族**（'ato'/' pneum'/'aviours'/'plevel'/'岸'(CJK)/'وى'(阿拉伯)/'ędzy'/'tku'(波兰)/' бъ'(西里尔)/' inw'+2×U+FFFD 替换符，skeleton 类仅 1/13）；**POS 9 tok=文档骨架族**（'http'/'bold'/'FunctionFlags'/'HostException'(驼峰代码)/'this'/'The'/'X'/' entries'，skeleton 类 8/9）。SPEC 131401=' preoc' 不在 POS 共同集→佐证 3149 非主载体。**载体的本体不是内容词也不是坐标/方向，而是模板纹理的最小离散原子（骨架+碎片）**——"纹理与排列组合"直觉的首次可操作兑现；trail 假说（3125-3127）获 token 级支撑；G4.2 的检验对象由此现成。

### PART E (k,d) 等值线：kx_product_superlinear（细化 3149）
chg=0.15 线 k*d：top5 6.80 < top10 8.88 < full50 25.83；chg=0.25 线：10.0 < 12.8 < 34.7——**两条等值线 k*d 严格随 k 单调增** → 互换不对称：低剂量段广度冗余（top5 两个坐标点等效率）、高剂量段广度必要（top10 在 chg=0.85 不可达域内无法补足 full50）。3149 "乘积近似守恒"细化为"仅在低 chg 段局部近似"。

### PART F v1 偏置阈值：v3_threshold_located
sym=0.95 首次穿越 α*=0.1085；饱和拟合 sym=s_inf+(1−s_inf)/(1+(α/ac)^p)：**ac=0.15, s_inf=0.55, p=3.0**（sse 0.0158）。与 3147 v1_sym_mixed(0.63/0.58@0.25/0.5) 完全衔接——承重轴读出在微幅度（α≤0.10）方向对称，偏置是幅度阈值后的饱和效应。

### 锚与判决
verdict=a_3149_ok|link_ledger_ok|p_freeze_ok|carrier_skeleton_class|kx_product_superlinear|v3_threshold_located|sae_audit_feasible。res sha8=**a2158ea5**，seal sha8=**1c6eb2f9**，ledger n=**287**。产物：`phase3150\\p0_freeze_carrierdecode\\`（result.json/execution.json/run_log/carrier_tokens.npz）+ `metric_dict.json` + `FIRST_PRINCIPLES_3090_3149.md` + `PARADIGM_SHIFT_VERDICT_v1.md`。

### 预注册 Phase 3151：G1 组合验收基准（原案 TESTPLAN §4.1）
**问题**：LLM 组合能力可加还是真组合？**材料**：状态库 672×2×4 + 实体A×关系R×模板T 笛卡尔积按组合切分（防近重复泄漏）。**验收（唯一）**：未见组合上打败四基线（B1 独立加性/B2 记忆检索/B3 单坐标最大/B4 全加性）。**判决（可能否证整条主线）**：若 3 模型未见组合误差>5%% 且不优于 B4 → 弃"算子代数"（死线 K1）→ G4 SAE 提前。工具：gate_precheck 前置（MDE 校验 n 组合设计），判决分级 bit_anchored|statistical 强制标注。
''' % now
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write(section)
    print('MEMO appended')

# 3) daily log append
daily = io.open(DAILY, encoding='utf-8').read()
if 'Phase 3150 (gpt5 线)' not in daily:
    with io.open(DAILY, 'a', encoding='utf-8') as f:
        f.write(u'''
## Phase 3150 (gpt5 线) [%s]
- 范式转移长文裁决：诊断对/叙事错/SAE 采纳 G4、TDA 降级 G5、ODE 部分否决 G6；全文 PARADIGM_SHIFT_VERDICT_v1.md（15 项逐条判定）。
- P0 制度冻结 F1-F7 全过：metric_dict.json（8 指标+元规则）、gate_precheck/counterexample_grep 工具、ledger v3 回填 286/286、F6-min 60 节。
- 载体解码 carrier_skeleton_class：NEG=多语言碎片族 13 tok / POS=文档骨架族 9 tok（http/bold/驼峰代码）——载体=模板纹理最小离散原子；131401 佐证非主载体。
- 等值线 kx_product_superlinear（低剂量广度冗余/高剂量必要）；v1 阈值 α*=0.109 ac=0.15；SAE 审计 margin 8×。
- 锚：res a2158ea5 seal 1c6eb2f9 ledger n=287。3151=G1 组合验收基准（TESTPLAN §4.1）。
''' % now)
    print('daily appended')

# 4) MEMORY.md update
mem = io.open(MEM, encoding='utf-8').read()
if '3150：P0 完成' not in mem:
    mem = mem.replace(
        u'- 3149：carrier_common_found',
        u'- 3150：P0 完成（F1-F7+ledger v3 n=287）；载体=骨架/碎片 token 族（carrier_skeleton_class）；kx_product_superlinear；v1 阈值 α*≈0.11；SAE 采纳 G4（K-G4/K-G5 死线，PARADIGM_SHIFT_VERDICT_v1.md）；res 9a8d43cc seal 1c6eb2f9。\n- 3149：carrier_common_found')
    mem = mem.replace(
        u'- 排期：**3150 P0 制度冻结（F1-F7 零 GPU）** → 3151-3153 G1',
        u'- 排期：3150 P0 ✅（2026-10-01）→ 3151-3153 G1')
    with io.open(MEM, 'w', encoding='utf-8',
                 newline='') as f:
        f.write(mem)
    print('MEMORY updated')

print('CLOSEOUT DONE')
