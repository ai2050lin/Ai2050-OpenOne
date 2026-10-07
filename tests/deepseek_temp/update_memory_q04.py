# -*- coding: utf-8 -*-
"""更新工作区长期记忆 MEMORY.md（索引）+ 追加当日 wlog（Q04 / Phase 38）。"""
import os, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-03.md')

# ---------------- MEMORY.md 定点替换 ----------------
raw = open(MEM, 'rb').read()
t = raw.decode('utf-8')
b0 = hashlib.sha256(raw).hexdigest()[:8]

REPL = [
    ('（**唯一研究日志**，Phase 1–37）', '（**唯一研究日志**，Phase 1–38）'),
    ('**P37 = Q03 E_read 基线（R9）**。现 `df987b1f`（700044 B）。',
     '**P37 = Q03 E_read 基线（R9）**；**P38 = Q04 E_ar(k) 装置（R10）**。现 `14c3da2e`（709594 B）。'),
    ('- **下一步 = Q04 `E_ar(k)` 装置**（gpu=mid，SMOKE 先通）→ Q05/Q06 `C_steer`。挂账：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4。',
     '- **✅ Q04 `E_ar(k)` 装置建成（R10）**：design_sha `33ddf69d`（幂等）／res_sha8 `04ad1af3`（三跑恒等）；SMOKE qwen3-4b（180 cells / 900 fwd）**装置门 4/4**；独立复核 **14/0**。口径登记 `metric_dict` **v2→v3**（content `5ce974ee`→`72a3d2c4`），E_ar.status=`device_built`。队列 `d0d2e208`→`631c1ba3`（Q04=`device_built`，sealed_items 仍 6）。**E_ar 与 E_read 量纲不同**（logit L1 vs 归一化 MSE）⇒ 只可并排读、以 `rel` 作桥。\n'
     '- **下一步 = Q05 `E_ar(k)` 正式测量**（三模型全面板 738 行 × k=0..16）+ Q06 `C_steer`。**Q05 硬前置**：14B/9B bf16 显存超限（29.6/18.8 GB > 16 GB）⇒ 先定量化/offload 并与 E_read 的 bf16 精度差异声明。Q05 科学门已预注册 `S_rel: min E_ar/scale ≤ 0.05`。挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4；P36 段 45 裸 LF 规范化。'),
    ('`rdc-phase-closeout`（**43 教训**）', '`rdc-phase-closeout`（**46 教训**）'),
]
for old, new in REPL:
    assert t.count(old) == 1, 'count=%d for %r' % (t.count(old), old[:40])
    t = t.replace(old, new)
open(MEM, 'wb').write(t.encode('utf-8'))
rb = open(MEM, 'rb').read()
print('MEMORY.md %d/%s -> %d/%s' % (len(raw), b0, len(rb), hashlib.sha256(rb).hexdigest()[:8]))

# ---------------- wlog 追加（EOL 自适应） ----------------
wraw = open(WLOG, 'rb').read()
eol = b'\r\n' if wraw.count(b'\r\n') > (wraw.count(b'\n') - wraw.count(b'\r\n')) else b'\n'
BLOCK = [
    '',
    '## R10 — Q04 `E_ar(k)` 装置建造（B 闸门，Phase 38）',
    '- 交付：`tests/deepseek/q04_e_ar_device.py`（装置）+ `q04_smoke_execution.json`（预注册 design_sha `33ddf69d`）+ `q04_smoke_result.json`（res_sha8 `04ad1af3`）+ `q04_smoke_report.txt`；独立复核 `verify_q04.txt` **14 PASS / 0 FAIL**。',
    '- 口径（登记 metric_dict v3）：`E_ar(k)=mean_cells|margin_pred−margin_true|`，margin=`logit(t_target)−logit(t_competitor)`；margin_true=模型自身 k 步贪心回喂后第 k 步 last-pos margin（无 teacher forcing）；margin_pred=与 E_read 同 B4 加性族 held-out 预测。K=16。面板与 E_read 同一 held-out 族（复核 V3 逐项相同指纹）。',
    '- SMOKE（qwen3-4b，PAIRS[:60]×3 模板=180 cells，K=4，900 forwards，~40 s）：E_ar(k)=3.2823/4.4031/3.3877/3.6703/3.1203；null(const)=5.5381/6.8469/5.1154/4.7581/4.0671。装置门 D1/D2/D3/S1 **全 PASS**，verdict=`SMOKE_DEVICE_OK|S1_DRIFT_DETECTED`。三跑 res_sha8 恒等 ⇒ 装置确定性成立。',
    '- ⚠ 口径发现：S1 阈 0.05 以**原始 logit** 为单位被超 88× ⇒ 门形式上可失败、实则必过。**不追溯改门**（防事后调门），改把 S1 降级为"装置灵敏度门"，并**在 Q05 观测前**预注册相对门 `S_rel: min E_ar_rel ≤ 0.05`（E_ar_rel=E_ar/scale），随 result 封存。',
    '- ⚠ SMOKE 局限：子面板仅前 10 实体 ⇒ 31 个 entity one-hot 为零，E_ar 偏大、per-seed 方差大（2.50/4.22/3.13 @k=0）；**数字不可科学解读**。正式曲线属 Q05。',
    '- ⚠ Q05 资源前置：qwen3-14b bf16≈29.6 GB、glm4-9b≈18.8 GB，均 > 本机 16 GB 显存；Q05 必须先定量化/offload 并声明与 E_read（bf16）的精度差异。',
    '- 工程教训（并入技能 44–46）：① 预注册哈希域**不得含时间戳**（首版因 `created` 入哈希而 DESIGN DRIFT）；② 新 KPI 首建装置必做**阈值量纲体检**（门是否真可失败）；③ 截断哈希比对宽度要一致（64 位 vs 8 位伪报 self-hash 不符）。',
    '- memo：700044 → 709594 B（`df987b1f` → `14c3da2e`），追加 Phase 38；前缀逐字节不变、bare_lf 仍 45（未动他线）。队列 `d0d2e208` → `631c1ba3`。技能 `rdc-phase-closeout` 43 → 46 教训（`a0fc65b3` → `bdf7da98`）。',
    '',
]
body = (eol.join(x.encode('utf-8') for x in BLOCK))
if not wraw.endswith(eol):
    body = eol + body
open(WLOG, 'ab').write(body)
wr = open(WLOG, 'rb').read()
print('wlog %d -> %d (%s)  R10block=%s' % (len(wraw), len(wr), 'CRLF' if eol == b'\r\n' else 'LF',
                                           b'## R10' in wr))
print('DONE')
