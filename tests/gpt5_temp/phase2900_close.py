# -*- coding: utf-8 -*-
"""Phase 2900 close-out: Ledger entries + MEMO append."""
import hashlib
import io
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
RES = os.path.join(BASE, 'phase2900', 'glm4_mlp_eigen_robustness')
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

assert not any(m.get('meas_id') == 'M2900_glm4_mlp_eigen_robustness'
               for m in d['measurements'])
d['measurements'].append({
    'meas_id': 'M2900_glm4_mlp_eigen_robustness',
    'type': 'multi_draw_robustness',
    'model': 'glm4-9b-chat-hf',
    'verdict': 'glm4_mlp_eigen_mixed: 8 seeds, mlp only, 3 '
               'conditions. EIGEN (primary): acc_flag 7/8 (acc '
               '0.5641-0.7436, only s2904 False) but margin_flag '
               '1/8 (margins 0.0166-0.0601, 7/8 positive yet '
               'below own p95 ~0.032-0.055; only s2901 passes). '
               'Threshold-adjacent weak-positive cell: signal '
               'real but ~10x smaller than qwen mlp (M2899 '
               'margins 0.18-0.26). STALE control: acc_flag 1/8, '
               'margin_flag 3/8 - no robust stale signal, '
               'consistent with M2898. ORTH: acc_flag 4/8, '
               'margin_flag 1/8 - borderline. Reproduction check '
               'passed: seeds 2893/2897 reproduce M2893/M2897 '
               'eigen accs exactly (0.6923/0.6795). M2893/M2897 '
               'acc-flags STAND; their margin-absence is '
               'confirmed as the true reading',
    'source': {'path': 'phase2900/glm4_mlp_eigen_robustness/'
                       'result.json',
               'sha256_8': sha8(os.path.join(RES, 'result.json')),
               'phase': 2900}})

assert not any(g.get('growth_id') == 'G_dual_flag_divergence'
               for g in d['growth_curve'])
d['growth_curve'].append({
    'growth_id': 'G_dual_flag_divergence',
    'phase': 2900,
    'model': 'glm4-9b-chat-hf',
    'axis': 'language',
    'finding': 'weak cells split acc-flag vs margin-flag: glm4 '
               'mlp eigen is acc-positive 7/8 but margin-'
               'positive-vs-own-p95 only 1/8 - acc retrieves '
               'above-chance while same-language geometry barely '
               'exceeds its tighter null. Margin magnitude, not '
               'binary flag, is the quantitative carrier '
               'strength: spectrum margins qwen mlp ~0.2 >> glm4 '
               'attn ~0.12 >> glm4 mlp ~0.02-0.03'})

for lk in d['linkage']:
    if lk.get('link_id') == 'L14_readout_spectrum_cross_model':
        lk['connects'] = lk['connects'] + \
            ['M2900_glm4_mlp_eigen_robustness']
        lk['notes'] = (lk['notes'] + ' | M2900 quantifies glm4 '
                       'mlp eigen tier: acc_flag 7/8 but margin '
                       '~0.02-0.03 (1/8 over own p95) - real but '
                       'weak, ~10x below qwen; spectrum stands as '
                       'margin-magnitude hierarchy')
        break

json.dump(d, io.open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert d2['measurements'][-1]['meas_id'] == \
    'M2900_glm4_mlp_eigen_robustness'
print('ledger ok: blocks=%d meas=%d growth=%d link=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['linkage'])))

memo = """
## Phase 2900: glm4 mlp eigen 格多抽判别 [2026-09-18 23:54]

### 原理
L14 谱系中 glm4 mlp eigen 格是唯一未多抽的阳性格：M2893/M2897 单抽 acc-flag 阳（0.6923/0.6795 > p95）但 margin-flag 阴（0.0313/0.0198 < 自身 p95）。按 G_multi_draw_lesson（margin-first 多抽），8 抽判别该格性质；stale 作期望阴性内部对照、orth 作描述性第三条件。glm4 mlp 无已知 robust 阳性条件，故不设阳性锚门，程序稳定性由 v1 确定性 + 历史种子精确复现（reproduction_check）守卫。

### 预注册（冻结于 execution.json，脚本 SHA256-8 2cde5111）
- 窗口 W=[28,40)；78 词 verbatim 2890；方向零前向自 2890 npz。
- SEEDS=[2893,2897,2901..2906]；null-draw 规则与 2893/2897 完全一致（历史种子精确复现其 null 集）。
- v1（每抽）：mlp recompute rel err < 1e-6，否则该抽 void。
- 判决映射：eigen acc_flag_rate >= 0.8 且 margin_flag_rate >= 0.5 => glm4_mlp_eigen_robust；margin_flag_rate = 0 => acc_only_fragile；否则 mixed。

### 结果（运行 1320.4s，n_valid=8，v1 全过，reproduction_check 双通过）
- **判决：glm4_mlp_eigen_mixed**
- eigen：acc 0.5641-0.7436，acc_flag **7/8**（仅 s2904 False）；margin 0.0166-0.0601（7/8 为正），margin_flag 仅 **1/8**（s2901 0.0601 > 0.0349）。
- stale 对照：acc_flag 1/8、margin_flag 3/8——无 robust 信号，与 M2898 一致。
- orth：acc_flag 4/8、margin_flag 1/8——边缘。
- 概念对照全程 ~0。复现锚：s2893→0.6923（=M2893）、s2897→0.6795（=M2897）精确。

### 硬伤与混杂
- margin null 分布紧（p95 0.030-0.059），margin_flag 对弱信号判别力有限——margin 量级（效应大小）比二元 flag 更本质，已入 G_dual_flag_divergence。
- 单抽 acc 与 p95 的差在 0.006-0.13 间波动，s2904 acc 0.5641 显示弱格 acc-flag 也不稳。

### 结论
1. **glm4 mlp eigen 是真实但弱的信号**：acc 一致高于 null（7/8），margin 7/8 为正但量级 ~0.02-0.03，比 qwen mlp（~0.2）低约一个数量级——M2893/M2897 的 acc-flag 判决维持，margin-缺失确认为其真实读法。
2. **谱系定量化为 margin 量级层级**：qwen mlp ~0.2 >> glm4 attn ~0.12 >> glm4 mlp ~0.02-0.03——载体强度连续谱而非二分，G_dual_flag_divergence 入账。
3. **acc-flag 与 margin-flag 在弱格分歧**：acc 检索超Chance 而同语言几何勉强超其更紧的 null——未来弱格判读以 margin 量级为准。

### 接续
- 2901 候选：A GLM4 attn eigen 格多抽确认（M2897 margin 0.1213 强阳但仅单抽；attn/margin 系谱系第二层，主选）/ B norm-matched 对照（2892 L1 混杂，零前向）/ C Gemma4/Qwen2-7B 下载决策（待用户）。

### 文件
- 脚本 tests/glm5/phase2900_glm4_mlp_eigen_robustness.py（2cde5111）
- 产物 phase2900/glm4_mlp_eigen_robustness/：execution.json bc364ec7 / result.json 31af8120 / glm4_mlp_eigen_robustness.npz 8b3e27a8
- Ledger：M2900_glm4_mlp_eigen_robustness + G_dual_flag_divergence + L14 精化（measurements 39 / growth 25 / linkage 14）
"""
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo)
print('memo appended')
