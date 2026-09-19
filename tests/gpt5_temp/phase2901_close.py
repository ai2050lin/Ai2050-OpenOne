# -*- coding: utf-8 -*-
"""Phase 2901 close-out: Ledger entries + MEMO append."""
import hashlib
import io
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
RES = os.path.join(BASE, 'phase2901', 'glm4_attn_eigen_robustness')
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

assert not any(m.get('meas_id') == 'M2901_glm4_attn_eigen_robustness'
               for m in d['measurements'])
d['measurements'].append({
    'meas_id': 'M2901_glm4_attn_eigen_robustness',
    'type': 'multi_draw_robustness',
    'model': 'glm4-9b-chat-hf',
    'verdict': 'glm4_attn_eigen_robust: 8 seeds (same set as '
               'M2898), attn only, 3 conditions. EIGEN (primary): '
               'acc_flag 8/8 (acc 0.6538-0.7308) margin_flag 7/8 '
               '(margins 0.0342-0.1213, only s2906 below own '
               'p95). STALE control: margin 0/8 positive '
               '(-0.0011..-0.0176) - confirms M2898. ORTH: '
               'margin 8/8 positive (0.0317-0.1222) - '
               'strengthens M2898 descriptive. Reproduction '
               'checks 4/4 pass (s2891 stale=M2891 0.5897; '
               's2897 stale=M2897 0.6795 & margin=-0.0161=M2898; '
               's2897 eigen=M2897 0.7308). Spectrum second tier '
               'now multi-draw confirmed: glm4 attn eigen+orth '
               'draw-stable, stale absent',
    'source': {'path': 'phase2901/glm4_attn_eigen_robustness/'
                       'result.json',
               'sha256_8': sha8(os.path.join(RES, 'result.json')),
               'phase': 2901}})

for lk in d['linkage']:
    if lk.get('link_id') == 'L14_readout_spectrum_cross_model':
        lk['connects'] = lk['connects'] + \
            ['M2901_glm4_attn_eigen_robustness']
        lk['notes'] = (lk['notes'] + ' | M2901 confirms spectrum '
                       'second tier under multi-draw: glm4 attn '
                       'eigen acc 8/8 margin 7/8, orth margin '
                       '8/8, stale 0/8 - full spectrum now '
                       'margin-first adjudicated')
        break

json.dump(d, io.open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert d2['measurements'][-1]['meas_id'] == \
    'M2901_glm4_attn_eigen_robustness'
print('ledger ok: blocks=%d meas=%d growth=%d link=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['linkage'])))

memo = """
## Phase 2901: glm4 attn eigen 格多抽判别 [2026-09-19 00:54]

### 原理
L14 谱系第二层（glm4 attn）仅 M2897 单抽：eigen margin 0.1213 强阳。M2898 已对同种子集（[2891,2897,2901..2906]）跑过 attn stale+orth（冻结 v2 acc-锚门判 all-void，描述性 rates 成立）。本 Phase 以相同种子集补 eigen 主判条件，三条件全跑，margin-first 判决；无阳性锚门（理由同 2900），程序稳定性由 v1 + 4 项 reproduction_check（M2891/M2897/M2898）守卫。

### 预注册（冻结于 execution.json，脚本 SHA256-8 6f37b87c）
- 窗口 W=[28,40)；78 词 verbatim 2890；方向零前向自 2890 npz；null-draw 规则与 2891/2897 一致。
- v1（每抽）：attn recompute rel err < 1e-6。
- 判决映射：eigen acc_flag_rate >= 0.8 且 margin_flag_rate >= 0.5 => glm4_attn_eigen_robust；margin_flag_rate = 0 => acc_only_fragile；否则 mixed。
- stale/orth：描述性 + 与 M2898 跨运行对照。

### 结果（运行 1315.9s，n_valid=8，v1 全过，reproduction_check 4/4）
- **判决：glm4_attn_eigen_robust**
- eigen：acc 0.6538-0.7308，acc_flag **8/8**；margin 0.0342-0.1213，margin_flag **7/8**（仅 s2906 0.0342 < p95 0.0434）。
- stale 对照：margin **0/8 正**（-0.0011..-0.0176）——M2898 结论在独立重跑中精确再现。
- orth：margin **8/8 正**（0.0317-0.1222）——强化 M2898 描述性 rates。
- 复现锚 4/4：s2891 stale 0.5897=M2891；s2897 stale 0.6795=M2897 且 margin −0.0161=M2898；s2897 eigen 0.7308=M2897。

### 硬伤与混杂
- s2906 eigen margin 0.0342 低于自身 p95 0.0434（margin_flag 7/8 非 8/8）——弱于 qwen（24/24）的残余波动，量级仍属第二层（~0.06-0.12）。
- margin p95 在 glm4 attn 侧波动较大（0.040-0.061），小 margin 抽的 flag 判读需谨慎（G_dual_flag_divergence 教训沿用）。

### 结论
1. **谱系第二层多抽确证**：glm4 attn eigen+orth draw-stable（margin 7/8、8/8 正），stale 缺席（0/8）跨运行精确再现——M2898 的 stale+ 撤回与 orth 部分容忍双双定谳。
2. **全谱系 margin-first 判决完成**：qwen mlp（24/24，~0.2）>> glm4 attn（eigen 7/8+orth 8/8，~0.06-0.12）>> glm4 mlp（eigen acc 7/8 但 margin 1/8，~0.02-0.03）——三层数量级分离稳固。
3. **读出容忍谱系成为载体定律的定量形式**：per-model 变异 = 通道读出容忍梯度，现已全部经 8 抽 margin-first 检验，无未判格。

### 接续
- 2902 候选：A GLM4 attn/mlp margin 量级差的结构根源（通道几何分析，零前向或低前向，主选）/ B norm-matched 对照（2892 L1 混杂）/ C Gemma4/Qwen2-7B 下载决策（待用户）。

### 文件
- 脚本 tests/glm5/phase2901_glm4_attn_eigen_robustness.py（6f37b87c）
- 产物 phase2901/glm4_attn_eigen_robustness/：execution.json 897df5fb / result.json 789083bf / glm4_attn_eigen_robustness.npz 3ded0dca
- Ledger：M2901_glm4_attn_eigen_robustness + L14 精化（measurements 40 / linkage 14）
"""
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo)
print('memo appended')
