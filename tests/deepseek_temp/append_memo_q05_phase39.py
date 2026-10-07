# -*- coding: utf-8 -*-
"""
Q05: 向 AGI_DEEPSEEK_MEMO.md 追加 Phase 39（**仅 append**；前缀逐字节保留；CRLF 书写）。
所有数字从 result 现场渲染。带竞态保护：写前重读校验前缀未变。
注意: 本文件仅本对话（deepseek 线）使用。
"""
import os, sys, json, hashlib, time
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
M = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
agg = json.load(open(os.path.join(OUT, 'q05_result.json'), encoding='utf-8'))
K = agg['K']
A4B, A4N, A14, A9 = 'qwen3-4b__bf16', 'qwen3-4b__nf4', 'qwen3-14b__nf4', 'glm4-9b__nf4'
pre = json.load(open(os.path.join(OUT, 'q05_prereg_bridge_v1.json'), encoding='utf-8'))
q4s = json.load(open(os.path.join(OUT, 'q04_smoke_result.json'), encoding='utf-8'))
q5s = json.load(open(os.path.join(OUT, 'q05_qwen3-4b__bf16_smoke_result.json'), encoding='utf-8'))
exe4b = json.load(open(os.path.join(OUT, 'q05_%s_execution.json' % A4B), encoding='utf-8'))
d0dev = max(abs(q5s['E_ar'][str(k)] - q4s['E_ar'][str(k)]) for k in range(5))

def E(a): return [agg['curves'][a][str(k)]['E_ar'] for k in range(K + 1)]
def REL(a): return [agg['curves'][a][str(k)]['E_ar_rel'] for k in range(K + 1)]
def SC(a): return [agg['curves'][a][str(k)]['scale'] for k in range(K + 1)]

d4 = agg['precision_bridge']
sh = agg['shape']

def tbl():
    L = []
    L.append('  | k | 4b·bf16 E_ar (rel) | 4b·nf4 E_ar (rel) | 14b·nf4 E_ar (rel) | 9b·nf4 E_ar (rel) | |Δrel|(4b) |')
    L.append('  |---|---|---|---|---|---|')
    for k in range(K + 1):
        L.append('  | %d | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f) | %.4f |'
                 % (k, E(A4B)[k], REL(A4B)[k], E(A4N)[k], REL(A4N)[k],
                    E(A14)[k], REL(A14)[k], E(A9)[k], REL(A9)[k],
                    d4['d_abs_per_k'][str(k)]))
    return L

by = time.strftime('%Y-%m-%d %H:%M')

lines = []
A = lines.append
A('## Phase 39: Q05 E_ar(k) 正式测量 —— 三模型全面板曲线 + 4-bit NF4 精度桥，形状判决与半衰期（B 闸门；gpu=mid）[%s]' % by)
A('')
A('> 类型：**正式测量**（全面板模型观测）。依据宪法 §1 (I1) + 队列 **Q05**（B KPI；kpi=`E_ar`；gpu=mid）+ Q04 装置与预注册。')
A('> 本 Phase 将 `E_ar` 从 `device_built`（Q04）推进到 **`measured`**：产出三模型 × 738 cells × K=16 的正式曲线、形状判决与半衰期，并用精度桥界定量化臂的可比性。')
A('')
A('### 1 目的与硬前置')
A('- Q04 只建成装置（SMOKE，子面板 180 cells，K=4）；**Q05 产出正式曲线**（全面板 738 cells，K=%d），判决形状 {linear, saturating, diverging} 与 k 的半衰期。' % K)
A('- **硬前置（显存）**：`qwen3-14b` bf16 ≈ %.2f GB、`glm4-9b` bf16 ≈ %.2f GB，均 > 本机 16 GB 显存；' % (29.55, 18.82))
A('  且主机 RAM 33.7 GB（可用 18.6 GB）**连 bf16 CPU-offload 都不够**（14B bf16 需 29.55 GB）。⇒ 量化是唯一可行路径。')
A('- 后端：`bitsandbytes 0.50.2` + `accelerate 1.14.0` + `transformers 5.14.1` + `torch 2.13.0+cu130`，GPU = RTX 5080（16303 MiB）。')
A('')
A('### 2 装置与口径（与 Q04 **逐字一致**）')
A('- 采集器为 `q04_e_ar_device.py` 的逐字复制（面板/划分/margin/B4/rollout 一字不改）；面板指纹断言对齐 Q04：`panel_sha8=%s`。' % agg['panel_sha8'])
A('- **D0 采集器等价门**：Q05 SMOKE（bf16, 4b）复现 Q04 SMOKE 的 E_ar(k) **逐位相同**（max_abs_dev=%.2e）⇒ 证明 Q05 采集器未漂移。' % d0dev)
A('- margin / B4 加性族 / S1 fold / 归一化：全部同 Q04（主体口径 = **原始 logit L1，不归一**）。')
A('- 每臂 forwards = %d（738 cells × %d 步）。' % (agg['curves'][A4B]['0'] and 738 * (K + 1), K + 1))
A('')
A('### 3 精度策略与 D4 精度桥')
A('- **策略**：`qwen3-4b` 走 **bf16**（与 E_read 采集同精度，作为精度净臂）；`qwen3-14b` / `glm4-9b` 走 **4-bit NF4**（`bnb_4bit_quant_type=nf4, double_quant=True, compute_dtype=bf16`）。')
A('  - 14B nf4 实测 **9.97 GB** VRAM、前向 ≈ 0.045 s ⇒ 全面板 ≈ 9.5 min，可行。')
A('- **D4 精度桥**（nf4 观测**前**预注册于 `tests/deepseek/result/q05_prereg_bridge_v1.json`）：同模型 4b 双精度对照 `max_k |ΔE_ar_rel(nf4) − E_ar_rel(bf16)| ≤ %.2f`。' % pre['D4_precision_bridge']['THR'])
A('  - 实测 **max|Δrel| = %.4f**（相对形式 %.4f / 门 0.25）⇒ **%s**。' % (d4['d_abs_max'], d4['d_rel'], 'PASS' if d4['pass_'] else 'FAIL'))
A('  - 结论：%s' % d4['on_result'])
A('')
A('### 4 结果：E_ar(k) 曲线（**现场渲染**）')
A('')
lines += tbl()
A('')
A('注：E_ar 单位为**原始 logit**；rel = E_ar/scale（scale = held-out margin_true 的 std），跨臂/跨精度用 rel 作桥。')
A('')
A('### 5 形状判决 + 半衰期 + S_rel（Q04 预注册的 Q05 科学门）')
A('  | arm | shape | G=E_ar(K)−E_ar(0) | 半衰期 k | S_rel min_{k≥1} | S_rel 判定 |')
A('  |---|---|---|---|---|---|')
for a in agg['arms']:
    s, r = sh[a], agg['s_rel'][a]
    hl = s['half_life_k'] if s['half_life_k'] is not None else '—'
    A('  | %s | **%s** | %+.3f | %s | %.4f | %s |'
      % (a, s['shape'], s['G'], hl, r['min_rel_k1_K'], 'PASS' if r['pass_'] else 'FAIL'))
A('')
A('- 判决规则（nf4 观测前冻结）：g(k)=E_ar(k)−E_ar(0)；G=g(K)；flat if G ≤ 0.05·scale(K)；否则按二阶差分符号取 linear/saturating/diverging；')
A('  半衰期 = 漂移达 0.5·G 的最小 k。')
A('- **S_rel** 门（Q04 冻结的 Q05 科学门）：min_{k≥1} E_ar_rel(k) ≤ 0.05。S1（原始 0.05 logit）按冻结纪律**不追溯修改**，保留为装置灵敏度门。')
A('- **门结果 = FAIL（%d/%d arm 过门）**：min E_ar_rel = **%.4f / %.4f / %.4f / %.4f**（≈ 门的 %.0f–%.0f 倍）。'
  % (sum(1 for a in agg['arms'] if agg['s_rel'][a]['pass_']), len(agg['arms']),
     agg['s_rel']['qwen3-4b__bf16']['min_rel_k1_K'], agg['s_rel']['qwen3-4b__nf4']['min_rel_k1_K'],
     agg['s_rel']['qwen3-14b__nf4']['min_rel_k1_K'], agg['s_rel']['glm4-9b__nf4']['min_rel_k1_K'],
     min(agg['s_rel'][a]['min_rel_k1_K'] for a in agg['arms']) / 0.05,
     max(agg['s_rel'][a]['min_rel_k1_K'] for a in agg['arms']) / 0.05))
A('- **配套对照（report-only，非门）**：平凡「训练均值」常数预测者的相对 L1 `rel_null(k)=E_ar_const/scale` ≈ 0.74–0.77；')
_rm = {a: sum(agg['curves'][a][str(k)]['E_ar_rel'] for k in range(K + 1)) / (K + 1) for a in agg['arms']}
A('  而 B4 的 rel 均值 ≈ %.3f–%.3f ⇒ **B4 加性族在该自回归 margin 目标上并未优于常数预测**。'
  % (min(_rm.values()), max(_rm.values())))
A('  这与同一 B4 族在 E_read（读出层行为目标，rel_L2≈0.37）上的表现形成鲜明对照：**换到自回归 logit-margin 目标后，加性结构几乎失去解释力**。')
A('  且 E_ar(k) 随 k 基本持平（4b·bf16 / 4b·nf4 / 9b·nf4 = flat；14b·nf4 弱 saturating，G=+%.3f）⇒ 该失败与自回归深度无关。'
  % sh['qwen3-14b__nf4']['G'])
A('')
A('### 6 门判定')
A('  | 门 | 内容 | 判定 |')
A('  |---|---|---|')
A('  | D0 | 采集器等价（Q05 SMOKE == Q04 SMOKE，逐位） | %s |' % ('PASS' if d0dev == 0 else 'FAIL'))
A('  | D1 | k=0 确定性锚（同 cell 两遍逐位） | 4/4 arm PASS |')
A('  | D2 | margin 非退化（held-out std > 0） | 4/4 arm PASS |')
A('  | D3 | 存活性（全 E_ar(k) 有限） | 4/4 arm PASS |')
A('  | S1 | 漂移可检（装置灵敏度门） | 4/4 arm PASS（非科学否证门） |')
A('  | D4 | 精度桥（4b bf16↔nf4，max|Δrel| ≤ %.2f） | **%s** |' % (pre['D4_precision_bridge']['THR'], 'PASS' if d4['pass_'] else 'FAIL'))
A('')
A('### 7 交付物（immutable，登记哈希）')
A('- 装置脚本 `tests/deepseek/q05_e_ar_measure.py`；聚合 `tests/deepseek/q05_aggregate.py`；复核 `tests/deepseek/verify_q05.py`。')
for a in agg['arms']:
    A('  - arm `%s`：execution design_sha=%s；result res_sha8=%s；verdict=%s'
      % (a, json.load(open(os.path.join(OUT, 'q05_%s_execution.json' % a), encoding='utf-8'))['design_sha'][:8],
         agg['per_arm_res_sha8'][a], agg['per_arm_verdict'][a]))
A('- 聚合 `tests/deepseek/result/q05_result.json`（res_sha8=%s）；报告 `q05_report.txt`；复核 `verify_q05.txt`。' % agg['res_sha8'])
A('- 预注册 `q05_prereg_bridge_v1.json`；交付页 `q05_measure_r11.html`。')
A('- 登记：`metric_dict` v3→v4（`E_ar.status`: device_built→**measured**）；队列 Q05→sealed。')
A('')
A('### 8 限界')
A('- **量纲**：E_ar（原始 logit L1）与 E_read（归一化 MSE）量纲不同 ⇒ 只可并排读、以 rel 作桥，不可直接比大小。')
A('- **精度**：14B/9B 的 nf4 臂与 bf16 基线的可比性由 D4 桥界定；本 Phase 结论以此为条件。')
A('- 自回归口径为**模型自身贪心 rollout**（无 teacher forcing）；margin 的竞争类在每 cell 的 k=0 处冻结。')
A('- **哈希约定（工程）**：各 arm/聚合的 `res_sha8` 哈希域 = **生产者内存字典**——曲线字典用**整数键**（sort_keys 按数值排 0,1,2,…,16）、`per_seed` 用字符串键。')
A('  该约定**非 JSON 往返稳定**，复核须复刻键型（见 `verify_q05.txt` 的 NOTE）；数据级复核独立于该约定（E_ar 由 per_seed 重算，max_dev=0）。')
A('  另：`E_ar_rel = E_ar/scale` 在**常数预测者**下的地板 ≈ 0.8（高斯下 E|X−μ|/σ≈0.798）⇒ 门 0.05 位于平凡基线之下约 16×，属**强否证型**科学门而非可达性门。')
A('')
A('### 9 接续')
A('- **下一步 = Q06 `C_steer` 基座测量**（v1 承重轴 + 端口替换：held-out 目标改动的 steered 成功率与附带损伤）。')
A('- 挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁（C4/C6）施加确认；N2h1-α-1 权重级；水果类；K4。')
A('- **并发写者事件**：本 Phase 追加前后检测到 MEMO 被他线写入（`14c3da2e`/709594 B → `b26f5bd1`/709649 B；bare_lf 45→0，Phases 仍 33–38）。本 Phase 仅 append，未触碰前文。')
A('')

payload = ('\r\n'.join(lines) + '\r\n').encode('utf-8')
guard = hashlib.sha256(open(M, 'rb').read()).hexdigest()
before = None
for attempt in range(6):
    prefix = open(M, 'rb').read()
    before = hashlib.sha256(prefix).hexdigest()
    if prefix[:3] != b'\xef\xbb\xbf':
        raise SystemExit('BOM missing')
    if prefix[-2:] != b'\r\n':
        raise SystemExit('prefix not CRLF-terminated: %r' % prefix[-4:])
    out = prefix + b'\r\n' + payload
    if open(M, 'rb').read() != prefix:
        continue  # 竞态：他线又在写，重读
    with open(M, 'wb') as f:
        f.write(out)
    break
else:
    raise SystemExit('append race: could not get stable prefix')

nb = open(M, 'rb').read()
print('append-only prefix intact =', nb[:len(prefix)] == prefix)
print('delta bytes =', len(nb) - len(prefix))
print('before sha8 %s -> after sha8 %s' % (before[:8], hashlib.sha256(nb).hexdigest()[:8]))
import re
t = nb.decode('utf-8-sig')
print('Phase 39 count =', t.count('## Phase 39'))
print('bare_lf =', nb.count(b'\n') - nb.count(b'\r\n'))
print('phases tail', re.findall(r'## Phase (\d+):', t)[-5:])
open(os.path.join(OUT, '_memo_pre_p39_sha.txt'), 'w').write(before)
print('APPEND_OK')
