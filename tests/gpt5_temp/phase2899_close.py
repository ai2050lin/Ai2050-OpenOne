# -*- coding: utf-8 -*-
"""Phase 2899 close-out: Ledger entries + MEMO append."""
import hashlib
import io
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
RES = os.path.join(BASE, 'phase2899', 'qwen_readout_robustness')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d = json.load(io.open(LEDGER, encoding='utf-8'))

assert not any(b['block_id'] == 'B_readout_robust_qwen'
               for b in d['blocks'])
d['blocks'].append({
    'block_id': 'B_readout_robust_qwen',
    'axis_id': 'language',
    'kind': 'multi_draw_triple_condition_injection',
    'model': 'qwen3-4b',
    'shape': [57, 10],
    'src': {'path': 'phase2899/qwen_readout_robustness/'
                    'qwen_readout_robustness.npz',
            'sha256_8': sha8(os.path.join(
                RES, 'qwen_readout_robustness.npz')),
            'phase': 2899, 'key': 'B_stale_s2896'},
    'notes': 'W=[26,36); stale(d18)/eigen(d(li))/orth triple '
             'injection x 8 null-draws, 2887 vocab verbatim 57 '
             'words, directions zero-forward from 2886 S_last; '
             '24/24 cell-draws dual-flag positive'})

assert not any(m['meas_id'] == 'M2899_qwen_readout_robustness'
               for m in d['measurements'])
d['measurements'].append({
    'meas_id': 'M2899_qwen_readout_robustness',
    'type': 'multi_draw_robustness',
    'model': 'qwen3-4b',
    'verdict': 'subspace_readout_tolerant_robust: 8 seeds, mlp '
               'only. ALL THREE conditions robust: stale '
               'acc_flag 8/8 (acc 0.70-0.88) margin 8/8 positive '
               '(+0.20..+0.26); eigen acc_flag 8/8 (0.79-0.93) '
               'margin 8/8 (+0.18..+0.23); orth acc_flag 8/8 '
               '(0.74-0.89) margin 8/8 (+0.18..+0.25); concept '
               'control ~0 throughout. Within-run determinism: '
               'seed 2896 reproduces M2896 accs exactly '
               '(0.8246/0.8421/0.8246). qwen mlp readout '
               'tolerance is draw-stable, contrasting glm4 attn '
               'stale fragility (M2898) - spectrum hierarchy '
               'confirmed under margin-first criteria',
    'source': {'path': 'phase2899/qwen_readout_robustness/'
                       'result.json',
               'sha256_8': sha8(os.path.join(RES, 'result.json')),
               'phase': 2899}})

assert not any(g.get('growth_id') == 'G_margin_first_validated_qwen'
               for g in d['growth_curve'])
d['growth_curve'].append({
    'growth_id': 'G_margin_first_validated_qwen',
    'phase': 2899,
    'model': 'qwen3-4b',
    'axis': 'language',
    'finding': 'margin-first multi-draw protocol (G_multi_draw_'
               'lesson) applied to a STRONG cell family: qwen mlp '
               'three-condition tolerance passes 24/24 dual-flag '
               'with margins 6-8x null p95 - robust positive '
               'counterpart to M2898 fragile-negative; '
               'subspace_readout_tolerant (M2896) upgraded to '
               'draw-stable status'})

for lk in d['linkage']:
    if lk.get('link_id') == 'L14_readout_spectrum_cross_model':
        lk['connects'] = lk['connects'] + \
            ['M2899_qwen_readout_robustness']
        lk['notes'] = (lk['notes'] + ' | confirmed by M2899: '
                       'qwen mlp tier draw-stable under 8-draw '
                       'margin-first protocol (24/24 dual-flag)')
        break

json.dump(d, io.open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
# round-trip check
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert d2['blocks'][-1]['block_id'] == 'B_readout_robust_qwen'
print('ledger ok: blocks=%d meas=%d growth=%d link=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['linkage'])))

memo = """
## Phase 2899: qwen mlp 读出三条件多抽稳健性 [2026-09-18 23:48]

### 原理
G_multi_draw_lesson（2898）确立：n=78 单抽 acc>p95 flag 复测信度低，margin 统计量 draw-stable。M2896 的 subspace_readout_tolerant 判决建立在单抽（SEED=2896）之上，需按新方法论升级为多抽 margin-first 检验。本 Phase 将 2898 多抽协议移植到 qwen mlp 通道三条件（stale/eigen/orth），锚条件取 eigen（M2896 中 margin 最强 0.1799），v2 锚门改用 margin_flag rate（margin-first）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 be3a0d79）
- 窗口 W=[26,36)；57 词 verbatim 2887；方向零前向自 2886 S_last（labels i%%2 断言）。
- SEEDS=[2896,2901..2907]（含历史抽 8 抽）；每抽：null_tids 自 rng(s)（word-tid 排除，VOCAB=151936），200 label perms 自 rng2(s)。
- v1（每抽）：mlp recompute rel err < 1e-6 且 hook-vs-call < 1e-4（shape-matched [1,2,2560]，2896 run-2 约定），否则该抽 void。
- v2 锚门（margin-first）：eigen margin_flag rate >= 0.8，否则 all void。
- 判决映射：stale margin_flag >= 0.8 且 acc_flag >= 0.5 => qwen_stale_robust；margin_flag = 0 => qwen_stale_margin_absent；三条件全 robust => subspace_readout_tolerant_robust；stale margin_flag = 0 => subspace_tolerant_stale_fragile。

### 结果（运行 69.4s，n_valid=8，v2 锚门通过 rate=1.0）
- **判决：subspace_readout_tolerant_robust —— 8 抽 x 3 条件 = 24/24 cell-draw 双 flag 全阳性**。
- stale：acc 0.7018-0.8772（flag 8/8），margin +0.2009..+0.2582（8/8，mp95 0.029-0.043）。
- eigen：acc 0.7895-0.9298（flag 8/8），margin +0.1769..+0.2341（8/8，mp95 0.030-0.049）。
- orth：acc 0.7368-0.8947（flag 8/8），margin +0.1831..+0.2502（8/8，mp95 0.040-0.051）。
- 概念对照全程 ~0（0.0000-0.0702）。margin 为 null p95 的 5-8 倍，远离阈值区。
- 确定性验证：seed=2896 精确复现 M2896 三 acc（0.8246/0.8421/0.8246）。
- v1 全 8 抽 recompute=0.0、hook=0.0。

### 硬伤与混杂
- 无新增。锚门本次以 margin_flag 定义（与 2898 的 acc_flag 锚门不同）——2898 时锚（orth acc_flag 5/8）不过门而 all-void；若 2899 用 acc_flag 锚门（eigen 8/8=1.0）同样过门，故判决对锚门定义不敏感（两种定义下均通过，已核对）。
- 单模型单通道；glm4 侧对应多抽仅 attn 双条件（2898），glm4 mlp eigen 格的多抽确认仍是 open（弱阳性 margin 0.0198 贴阈值）。

### 结论
1. **qwen mlp 读出容忍升级为 draw-stable**：subspace_readout_tolerant（M2896）经 8 抽 margin-first 复核无一翻转——qwen 的三条件全阳是稳固事实，非 null-draw 伪影。
2. **谱系层级在 margin-first 准则下成立**：qwen mlp（margin ~0.2，24/24）>> glm4 attn（eigen+orth ~0.12，8/8 margin 正但 acc_flag 5/8）>> glm4 mlp（仅 eigen，且其多抽确认 open）。
3. **方法论闭环**：G_multi_draw_lesson 同时找到脆弱阴性（glm4 attn stale，0/8）与稳固阳性（qwen mlp，24/24）两个对照端点——协议升级完成。

### 接续
- 2891 候选：A glm4 mlp eigen 格多抽确认（阈值邻近，主选）/ B norm-matched 对照（2892 L1 混杂，零前向）/ C Gemma4/Qwen2-7B 下载决策（待用户）。

### 文件
- 脚本 tests/glm5/phase2899_qwen_readout_robustness.py（be3a0d79）
- 产物 phase2899/qwen_readout_robustness/：execution.json 6667e6ac / result.json b4416f00 / qwen_readout_robustness.npz 2af9ba4d
- Ledger：B_readout_robust_qwen + M2899_qwen_readout_robustness + G_margin_first_validated_qwen + L14 精化（blocks 23 / measurements 38 / growth 24 / linkage 14）
"""
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo)
print('memo appended')
