# -*- coding: utf-8 -*-
"""Q05: 追加当日工作日志 + 就地更新 MEMORY.md（跨轮索引）。数字现场渲染。"""
import os, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEM = os.path.join(ROOT, '.workbuddy', 'memory')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
MEMP = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')

agg = json.load(open(os.path.join(OUT, 'q05_result.json'), encoding='utf-8'))
K = agg['K']; A4B, A4N, A14, A9 = 'qwen3-4b__bf16', 'qwen3-4b__nf4', 'qwen3-14b__nf4', 'glm4-9b__nf4'
d4 = agg['precision_bridge']; sh = agg['shape']
vry = open(os.path.join(OUT, 'verify_q05.txt'), encoding='utf-8').read()
vpass = vry.strip().split(chr(10))[-2] if vry.strip().split(chr(10)) else '?'
vpass = [l for l in vry.split(chr(10)) if l.startswith('TOTAL')]
vpass = vpass[0] if vpass else '?'
mb = open(MEMP, 'rb').read()
memo_b, memo_sha = len(mb), hashlib.sha256(mb).hexdigest()[:8]
md = json.load(open(os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'metric_dict.json'), encoding='utf-8'))
q = json.load(open(os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'phase_queue_v1.json'), encoding='utf-8'))

def E(a): return ['%.3f' % agg['curves'][a][str(k)]['E_ar'] for k in range(K + 1)]
def R(a): return ['%.3f' % agg['curves'][a][str(k)]['E_ar_rel'] for k in range(K + 1)]

# ---------- 1. 当日工作日志（append-only） ----------
W = os.path.join(MEM, '2026-10-03.md')
blk = []
A = blk.append
A('')
A('## R11 — Q05 `E_ar(k)` 正式测量（B 闸门，Phase 39）')
A('- 交付：装置 `tests/deepseek/q05_e_ar_measure.py`（Q04 采集器**逐字复制**）+ 聚合 `q05_aggregate.py` + 复核 `verify_q05.py`；预注册 `q05_prereg_bridge_v1.json`。')
A('- **D0 采集器等价门**：Q05 SMOKE(bf16,4b) 与 Q04 SMOKE **逐位相同**（max_abs_dev=0）⇒ 采集器未漂移。')
A('- 四臂（全面板 738 cells × K=16，每臂 12,546 forwards）：4b·bf16（精度净臂）/ 4b·nf4（桥）/ 14b·nf4 / 9b·nf4。')
A('- **精度策略**：主机 RAM 33.7 GB(avail 18.6) ⇒ bf16 offload 亦不可行；14B(29.55 GB)/9B(18.82 GB) 走 **4-bit NF4**（14B 实测 9.97 GB VRAM、0.045 s/fwd）。')
A('- **D4 精度桥**（nf4 观测前预注册）：4b 同模型 bf16↔nf4 `max|ΔE_ar_rel|=%.4f`（门 %.2f）⇒ **%s**（%s）。'
  % (d4['d_abs_max'], d4['thr'], 'PASS' if d4['pass_'] else 'FAIL', d4['on_result']))
A('- **形状判决**：4b·bf16=**%s**、14b·nf4=**%s**、9b·nf4=**%s**（半衰期 k：%s / %s / %s）。'
  % (sh[A4B]['shape'], sh[A14]['shape'], sh[A9]['shape'],
     sh[A4B]['half_life_k'], sh[A14]['half_life_k'], sh[A9]['half_life_k']))
A('- **S_rel 门**（Q04 预注册）：min E_ar_rel = 4b %.4f / 14b %.4f / 9b %.4f。'
  % (agg['s_rel'][A4B]['min_rel_k1_K'], agg['s_rel'][A14]['min_rel_k1_K'], agg['s_rel'][A9]['min_rel_k1_K']))
A('- 曲线（E_ar 原始 logit L1；括号 rel）：')
A('  - 4b·bf16: %s' % ' '.join('%s(%s)' % (E(A4B)[k], R(A4B)[k]) for k in range(K + 1)))
A('  - 14b·nf4: %s' % ' '.join('%s(%s)' % (E(A14)[k], R(A14)[k]) for k in range(K + 1)))
A('  - 9b·nf4 : %s' % ' '.join('%s(%s)' % (E(A9)[k], R(A9)[k]) for k in range(K + 1)))
A('- 独立复核 `verify_q05.txt`：**%s**。聚合 `q05_result.json` res_sha8 `%s`。' % (vpass, agg['res_sha8']))
A('- 登记：`metric_dict` v3→v4（`E_ar.status`: device_built→**measured**，content `%s`）；队列 `%s`→`%s`（Q05 sealed，sealed_items %d 项）。'
  % (md['content_sha256_8'], '631c1ba3', hashlib.sha256(open(os.path.join(ROOT,'research','deepseek','atlas','phase_queue_v1.json'),'rb').read()).hexdigest()[:8], len(q['sealed_items'])))
A('- memo：追加 Phase 39 后当前 %d B（追加前 709649 B）；前缀逐字节不变、bare_lf 保持 0。' % memo_b)
A('- ⚠ 并发写者：本轮开始（`14c3da2e`/709594 B、bare_lf 45）与追加前（`b26f5bd1`/709649 B、bare_lf **0**）之间，**他线把 P36 裸 LF 全部规范化**；Phases 仍 33–38，未改判定。本线仅 append。')
A('- 工程教训（并入技能 47–49）：① 4-bit 量化臂必须配**同模型双精度桥**并在观测前预注册门；② bf16 常驻失败时先探 RAM——offload 可能同样不可行；③ 逐字复制采集器要用 **D0 位等价门**证明未漂移（不能只靠肉眼比对）。')
cur = open(W, encoding='utf-8').read()
if 'R11 — Q05' not in cur:
    with open(W, 'a', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(blk) + '\n')
    print('wlog appended')
else:
    print('wlog already has R11')

# ---------- 2. MEMORY.md 就地更新 ----------
mp = os.path.join(MEM, 'MEMORY.md')
t = open(mp, encoding='utf-8').read()
def rep(old, new):
    global t
    assert t.count(old) == 1, 'anchor count=%d for %r' % (t.count(old), old[:40])
    t = t.replace(old, new)

rep('（**唯一研究日志**，Phase 1–38）', '（**唯一研究日志**，Phase 1–39）')
rep('文件 free bare_lf = **45**（P36 引入）',
    '文件 free bare_lf = **0**（他线于 R11 期间把 P36 裸 LF 规范化；R11 前为 45）')
rep('**P38 = Q04 E_ar(k) 装置（R10）**。现 `14c3da2e`（709594 B）。',
    '**P38 = Q04 E_ar(k) 装置（R10）**；**P39 = Q05 E_ar(k) 正式测量（R11）**。现 `%s`（%d B）。' % (memo_sha, memo_b))
rep('- **下一步 = Q05 `E_ar(k)` 正式测量**（三模型全面板 738 行 × k=0..16）+ Q06 `C_steer`。**Q05 硬前置**：14B/9B bf16 显存超限（29.6/18.8 GB > 16 GB）⇒ 先定量化/offload 并与 E_read 的 bf16 精度差异声明。Q05 科学门已预注册 `S_rel: min E_ar/scale ≤ 0.05`。挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4；P36 段 45 裸 LF 规范化。',
    '- **✅ Q05 `E_ar(k)` 正式测量完成（R11）**：四臂 738×K16；精度策略=4b **bf16**、14B/9B **4-bit NF4**（bf16 offload 因 RAM 不足不可行）；**D4 桥** max|Δrel|=%.4f（门 %.2f）**%s**；形状 4b=%s/14b=%s/9b=%s；S_rel min=%.4f/%.4f/%.4f；独立复核 %s。口径登记 `metric_dict` v3→v4（E_ar.status=**measured**）；队列 Q05 sealed `%s`。**下一步 = Q06 `C_steer` 基座测量**（承重轴 + 端口替换：steered 成功率 + 附带损伤）。挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4。'
    % (d4['d_abs_max'], d4['thr'], 'PASS' if d4['pass_'] else 'FAIL', sh[A4B]['shape'], sh[A14]['shape'], sh[A9]['shape'],
       agg['s_rel'][A4B]['min_rel_k1_K'], agg['s_rel'][A14]['min_rel_k1_K'], agg['s_rel'][A9]['min_rel_k1_K'], vpass,
       hashlib.sha256(open(os.path.join(ROOT,'research','deepseek','atlas','phase_queue_v1.json'),'rb').read()).hexdigest()[:8]))

open(mp, 'w', encoding='utf-8', newline='\n').write(t)
print('MEMORY.md chars', len(t))
print('MEMORY_OK')
